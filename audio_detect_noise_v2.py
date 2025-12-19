"""
检测 WAV 文件结尾的 "chi" 杂音

该模块提供函数来检测音频文件结尾是否存在 "chi" 这样的杂音。
"""

import numpy as np
import librosa
import os
import io
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import json

# 尝试导入机器学习相关库
try:
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    joblib = None


def detect_chi_noise_core(y: np.ndarray,
                          sr: int,
                          tail_ms=200,           # 检测尾部窗口（毫秒）
                          ref_ms=300,            # 参考窗口（毫秒）
                          hi_band=(5000, 15000), # 高频频带范围（Hz）
                          ratio_db_thresh=5.5,   # 高频相对增益阈值（dB）
                          ratio_db_min=6.0,      # 高频相对增益最低要求（dB），用于过滤弱信号（降低以捕获弱杂音）
                          flux_db_thresh=3.0,    # 瞬态变化阈值（dB）
                          score_thresh=4.0,      # 综合评分阈值（提高以减少误检）
                          must_be_within_ms=200  # 杂音必须出现在结尾多少毫秒内（放宽以捕获边缘情况）
                          ):
    """
    检测音频数据结尾是否存在 "chi" 杂音（核心检测逻辑）
    
    Args:
        y: 音频数据数组（numpy array）
        sr: 采样率
        tail_ms: 检测尾部窗口长度（毫秒）
        ref_ms: 参考窗口长度（毫秒）
        hi_band: 高频频带范围（Hz），用于检测杂音特征
        ratio_db_thresh: 高频相对增益阈值（dB）
        ratio_db_min: 高频相对增益最低要求（dB），用于过滤弱信号
        flux_db_thresh: 瞬态变化阈值（dB）
        score_thresh: 综合评分阈值，超过此值判定为有杂音
        must_be_within_ms: 杂音必须出现在结尾多少毫秒内
    
    Returns:
        dict: 包含检测结果的字典
            - has_chi: bool, 是否检测到杂音
            - score: float, 综合评分
            - ratio_db_peak: float, 高频相对增益峰值（dB）
            - flux_db_peak: float, 瞬态变化峰值（dB）
            - details: dict, 详细检测信息
    """
    # 确保音频是单声道
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    
    # 计算采样点数
    N_tail = max(1, int(sr * tail_ms / 1000))
    N_ref = max(1, int(sr * ref_ms / 1000))
    
    # 提取尾部音频和参考音频
    tail = y[-N_tail:]
    ref_start = max(0, len(y) - N_tail - N_ref)
    ref = y[ref_start:len(y) - N_tail]
    
    # 如果参考窗口太短，使用音频前半部分
    if len(ref) < int(0.5 * N_tail):
        ref = y[:max(int(0.5 * N_tail), len(y)//2)]
    
    # STFT 分析
    n_fft = 1024 if sr < 32000 else 2048
    hop = n_fft // 4
    
    S_tail = librosa.stft(tail, n_fft=n_fft, hop_length=hop, window='hann')
    S_ref = librosa.stft(ref, n_fft=n_fft, hop_length=hop, window='hann')
    
    mag_tail = np.abs(S_tail)
    mag_ref = np.abs(S_ref)
    
    # 获取频率数组
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    hi_mask = (freqs >= hi_band[0]) & (freqs <= hi_band[1])
    
    # 计算高频能量
    hi_e_tail = np.maximum(1e-12, (mag_tail[hi_mask]**2).sum(axis=0))
    hi_e_ref = np.maximum(1e-12, (mag_ref[hi_mask]**2).sum(axis=0))
    
    # 使用中位数作为参考基准（更稳定）
    ref_med = float(np.median(hi_e_ref))
    
    # 计算相对增益（dB）
    ratio_db_seq = 10 * np.log10(hi_e_tail / (ref_med + 1e-12))
    ratio_db_peak = float(np.max(ratio_db_seq))
    
    # 计算瞬态变化（谱通量）
    flux_seq = np.maximum(0.0, np.diff(np.log(hi_e_tail + 1e-12)))
    flux_db_peak = float(10 * np.log10(1 + np.max(flux_seq))) if len(flux_seq) > 0 else 0.0
    
    # 计算频谱特征
    flat_tail = librosa.feature.spectral_flatness(S=mag_tail**2)
    flat_db = float(10 * np.log10(np.median(flat_tail) + 1e-12))
    
    rolloff_tail = librosa.feature.spectral_rolloff(S=mag_tail, sr=sr, roll_percent=0.95)
    rolloff_ratio = float(np.median(rolloff_tail) / (sr / 2))
    
    # 计算 RMS 能量比
    rms_tail = float(np.sqrt(np.mean(tail**2) + 1e-12))
    rms_ref = float(np.sqrt(np.mean(ref**2) + 1e-12))
    rms_ratio = rms_tail / (rms_ref + 1e-12)
    
    # 计算波峰因子
    peak_tail = float(np.max(np.abs(tail)))
    crest_db = float(20 * np.log10((peak_tail + 1e-9) / (rms_tail + 1e-9)))
    
    # 找到峰值位置
    k_peak = int(np.argmax(ratio_db_seq))
    t0_global = len(y) - N_tail + int(k_peak * hop)
    distance_from_end = len(y) - t0_global
    # 放宽边界判断，允许1个hop的误差
    threshold_samples = int((must_be_within_ms / 1000) * sr)
    near_end = distance_from_end <= threshold_samples + hop
    
    # 计算超阈值持续时间
    over_thresh = ratio_db_seq > ratio_db_thresh
    over_frames = int(np.sum(over_thresh))
    over_ms = over_frames * hop / sr * 1000.0
    
    # 综合评分
    score = 0.0
    
    # 主要指标（权重较高）
    if ratio_db_peak > ratio_db_thresh:
        score += 2.0  # 高频相对增益超过阈值
    elif ratio_db_peak > ratio_db_thresh - 2.0:
        score += 1.0  # 接近阈值
    
    # 如果ratio_db_peak刚好超过ratio_db_min，给予额外分数
    if ratio_db_peak >= ratio_db_min and ratio_db_peak < ratio_db_min + 1.0:
        score += 0.5  # 边缘情况额外加分
    
    if flux_db_peak > flux_db_thresh:
        score += 1.5  # 瞬态变化明显
    elif flux_db_peak > flux_db_thresh - 1.0:
        score += 0.5
    
    # 如果flux较低但crest很高，可能是瞬态杂音
    if flux_db_peak < flux_db_thresh and crest_db > 15.0:
        score += 0.5  # 补偿flux不足
    
    # 次要指标（权重较低）
    if flat_db > -10.0:  # 频谱较平坦（可能是噪声）
        score += 0.5
    
    # 如果flat_db太低（<-45），可能是语音而非杂音，降低分数
    # 但需要结合ratio_db_peak判断：如果ratio很高，即使flat_db较低也可能是杂音
    if flat_db < -45.0 and ratio_db_peak < 20.0:
        score *= 0.5  # 大幅降低分数，过滤语音信号（但ratio很高时保留）
    
    if rolloff_ratio > 0.75:  # 高频成分较多
        score += 0.5
    
    # 如果rolloff_ratio较低（<0.1），更可能是瞬态杂音
    if rolloff_ratio < 0.1:
        score += 0.3
    
    if crest_db > 12.0:  # 波峰因子较高（可能是瞬态杂音）
        score += 0.5
    
    if rms_ratio < 0.95:  # 尾部能量相对较低（符合杂音特征）
        score += 0.3
    
    if over_ms > 0 and over_ms < 150:  # 持续时间适中（不是持续噪声）
        score += 0.5
    
    # 如果rolloff_ratio中等（0.3-0.5）且rms_ratio较高（>0.4），可能是语音尾音而非杂音
    # 杂音通常rolloff_ratio很高（>0.6）或很低（<0.1），且rms_ratio较低（<0.3）
    if 0.3 <= rolloff_ratio <= 0.5 and rms_ratio > 0.4:
        score *= 0.6  # 降低分数，可能是语音尾音
    
    # 必须靠近结尾
    if not near_end:
        score *= 0.3  # 如果不在结尾附近，大幅降低分数
    
    # 对于ratio较低但其他指标较强的边缘情况，放宽限制
    # 如果ratio在5-6dB之间，但flux和crest都很高，可能是弱杂音
    if ratio_db_peak >= 5.0 and ratio_db_peak < ratio_db_min:
        if flux_db_peak > flux_db_thresh and crest_db > 15.0:
            # 弱杂音但特征明显，给予额外分数
            score += 1.0
    
    # 判定结果：必须满足最低 ratio 要求，且评分超过阈值，且靠近结尾
    # 对于边缘情况（ratio在5-6dB之间），如果score足够高且near_end，也可以判定为有杂音
    has_chi = (
        (ratio_db_peak >= ratio_db_min and score >= score_thresh and near_end) or
        (ratio_db_peak >= 5.0 and ratio_db_peak < ratio_db_min and 
         score >= score_thresh + 1.0 and near_end and flux_db_peak > 0.3)
    )
    
    return {
        "has_chi": bool(has_chi),
        "score": float(score),
        "ratio_db_peak": float(ratio_db_peak),
        "flux_db_peak": float(flux_db_peak),
        "flat_db": float(flat_db),
        "rolloff_ratio": float(rolloff_ratio),
        "crest_db": float(crest_db),
        "rms_ratio": float(rms_ratio),
        "over_ms": float(over_ms),
        "near_end": bool(near_end),
        "distance_from_end_ms": float(distance_from_end / sr * 1000),
        "details": {
            "tail_ms": tail_ms,
            "ref_ms": ref_ms,
            "hi_band": hi_band,
            "ratio_db_thresh": ratio_db_thresh,
            "ratio_db_min": ratio_db_min,
            "flux_db_thresh": flux_db_thresh,
            "score_thresh": score_thresh
        }
    }

def _extract_ml_features_v4_3(y: np.ndarray, sr: int, tail_ms=200, ref_ms=300, hi_band=(5000, 15000)):
    """
    提取音频特征（用于机器学习，V4.3版本，105个特征）
    
    此函数与 train_noise_detector_v4_3.py 中的 extract_features_v4_3 保持一致。
    优先从训练脚本导入，确保特征提取逻辑完全一致。
    
    Returns:
        np.ndarray: 特征向量（85个特征）
    """
    # 优先尝试从训练脚本导入（如果可用）
    try:
        import importlib.util
        import os
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_script_path = os.path.join(current_dir, "train_noise_detector_v4_3.py")
        
        if os.path.exists(train_script_path):
            spec = importlib.util.spec_from_file_location("train_v4_3", train_script_path)
            if spec and spec.loader:
                train_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(train_module)
                if hasattr(train_module, 'extract_features_v4_3'):
                    features = train_module.extract_features_v4_3(y, sr, tail_ms=tail_ms, ref_ms=ref_ms, hi_band=hi_band)
                    return np.array(features)
    except Exception as e:
        # 如果导入失败，抛出清晰的错误信息
        raise RuntimeError(
            f"无法从 train_noise_detector_v4_3.py 导入 extract_features_v4_3 函数。"
            f"错误: {e}\n"
            f"请确保 train_noise_detector_v4_3.py 文件存在且包含 extract_features_v4_3 函数。"
        )
    
    # 如果文件不存在，抛出错误
    raise FileNotFoundError(
        "train_noise_detector_v4_3.py 文件不存在。"
        "请确保该文件与 audio_detect_noise_v2.py 在同一目录下。"
    )


def detect_chi_noise_core_v4_3(y: np.ndarray,
                                sr: int,
                                model_path: Optional[Union[str, Path]] = None,
                                tail_ms=200,
                                ref_ms=300,
                                hi_band=(5000, 15000),
                                fallback_to_v1=True):
    """
    检测音频数据结尾是否存在 "chi" 杂音（基于机器学习的版本，V4.3）
    
    使用训练好的随机森林模型进行检测（V4.3版本，105个特征），相比硬编码阈值方法更加灵活和准确。
    模型会自动学习特征之间的复杂关系，无需手动调整阈值。
    
    Args:
        y: 音频数据数组（numpy array）
        sr: 采样率
        model_path: 模型文件路径，默认为 "./noise_detector_model_v4_3.pkl"
        tail_ms: 检测尾部窗口长度（毫秒）
        ref_ms: 参考窗口长度（毫秒）
        hi_band: 高频频带范围（Hz）
        fallback_to_v1: 如果模型加载失败，是否回退到v1版本
    
    Returns:
        dict: 包含检测结果的字典
            - has_chi: bool, 是否检测到杂音
            - probability: float, 模型预测的概率（0-1）
            - method: str, 使用的检测方法（"ml_v4.3", "v1_fallback"）
            - score: float, 综合评分（兼容v1格式，0-10）
            - ratio_db_peak: float, 高频相对增益峰值（dB）
            - flux_db_peak: float, 瞬态变化峰值（dB）
            - features: dict, 提取的所有特征值
            - details: dict, 详细检测信息
    """
    if not ML_AVAILABLE:
        if fallback_to_v1:
            result = detect_chi_noise_core(y, sr, tail_ms=tail_ms, ref_ms=ref_ms, hi_band=hi_band)
            result["method"] = "v1_fallback"
            result["probability"] = 1.0 if result["has_chi"] else 0.0
            return result
        else:
            raise ImportError("joblib not available. Install it with: pip install joblib scikit-learn")
    
    # 加载模型（V4.3版本）
    if model_path is None:
        model_path = Path("./noise_detector_model_v4_3.pkl")
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        if fallback_to_v1:
            result = detect_chi_noise_core(y, sr, tail_ms=tail_ms, ref_ms=ref_ms, hi_band=hi_band)
            result["method"] = "v1_fallback"
            result["probability"] = 1.0 if result["has_chi"] else 0.0
            return result
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        if fallback_to_v1:
            result = detect_chi_noise_core(y, sr, tail_ms=tail_ms, ref_ms=ref_ms, hi_band=hi_band)
            result["method"] = "v1_fallback"
            result["probability"] = 1.0 if result["has_chi"] else 0.0
            return result
        else:
            raise RuntimeError(f"Failed to load model: {e}")
    
    # 提取特征（V4.3版本使用105个特征）
    features = _extract_ml_features_v4_3(y, sr, tail_ms=tail_ms, ref_ms=ref_ms, hi_band=hi_band)
    features = features.reshape(1, -1)
    
    # 预测
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # 获取特征名称（V4.3版本，105个特征）
    feature_names = [
        'ratio_db_peak', 'ratio_db_mean', 'ratio_db_std', 'ratio_db_median',
        'flux_db_peak', 'flux_db_mean',
        'flat_db', 'flat_mean',
        'rolloff_ratio', 'rolloff_mean',
        'rms_ratio', 'crest_db', 'zcr_ratio', 'mfcc_mean',
        'over_ms', 'distance_from_end_ms', 'energy_concentration',
        'ref_med', 'ref_mean', 'ref_std',
        'peak_position_ratio', 'last_50ms_ratio', 'hi_e_peak_position',
        'spectral_centroid_mean', 'spectral_bandwidth_mean', 'hi_energy_ratio_mean',
        'peak_surrounding_ratio', 'tail_dynamic_range', 'hi_e_variance', 'rise_rate',
        'rms_ratio_last_100ms', 'peak_duration', 'peak_to_mean_ratio',
        'hi_e_concentration', 'decay_rate', 'spectral_contrast_mean',
        'last_20ms_ratio', 'peak_position_in_tail', 'energy_ratio_halfs',
        'sudden_energy_rise', 'energy_variance', 'sudden_cutoff_ratio',
        'zcr_change_rate', 'max_rms_gradient', 'rms_gradient_mean', 'rms_gradient_std',
        'tail_energy_concentration_ratio', 'spectral_change', 'envelope_change_ratio',
        'peak_rise_slope', 'peak_fall_slope', 'peak_window_concentration',
        'band1_ratio_mean', 'band2_ratio_mean', 'band3_ratio_mean',
        'ratio_db_skewness', 'ratio_db_kurtosis', 'peak_energy_ratio',
        'peak_position_normalized', 'last_10ms_ratio', 'peak_to_ref_ratio_db',
        'peak_duration_90', 'peak_duration_70', 'peak_surrounding_std',
        'energy_ratio_30_10', 'spectral_centroid_std', 'hi_e_peak_to_mean',
        'distance_from_end_normalized', 'peak_to_median_ratio',
        # V4.2新增16个特征名称
        'last_5ms_ratio', 'peak_ref_diff_db', 'energy_ratio_10_50',
        'hi_e_peak_pos_normalized', 'asymmetry', 'rms_change_rate',
        'peak_duration_50_ratio', 'hi_e_trend_slope', 'spectral_similarity',
        'peak_hi_e_linear', 'energy_ratio_last_1_percent', 'peak_width_ms',
        'tail_energy_entropy', 'peak_distance_from_end_abs', 'hi_e_peak_mean_diff',
        'peak_count',
        # V4.3新增20个特征名称
        'last_2ms_ratio', 'last_2ms_to_10ms_ratio', 'energy_ratio_5_95',
        'hi_e_accel_max', 'ratio_db_range', 'zcr_ratio_20_80',
        'hi_e_mean_ratio', 'post_peak_decay', 'tail_energy_std',
        'tail_energy_skewness', 'pre_post_diff', 'hi_e_p90_p50_ratio',
        'spectral_centroid_change_30ms', 'positive_ratio', 'hi_e_cv',
        'tail_autocorr', 'local_energy_ratio', 'rms_max_min_ratio',
        'hi_e_range_mean_ratio', 'last_15ms_ratio',
    ]
    
    # 构建特征字典
    features_dict = {name: float(val) for name, val in zip(feature_names, features[0])}
    
    # 计算一些额外的统计信息（兼容v1的输出格式）
    ratio_db_peak = features_dict['ratio_db_peak']
    flux_db_peak = features_dict['flux_db_peak']
    flat_db = features_dict['flat_db']
    rolloff_ratio = features_dict['rolloff_ratio']
    crest_db = features_dict['crest_db']
    rms_ratio = features_dict['rms_ratio']
    over_ms = features_dict['over_ms']
    distance_from_end_ms = features_dict['distance_from_end_ms']
    
    # 判断是否靠近结尾（200ms内）
    near_end = distance_from_end_ms <= 200.0
    
    return {
        "has_chi": bool(prediction == 1),
        "probability": float(probabilities[1]),  # 有杂音的概率
        "method": "ml_v4.3",
        "score": float(probabilities[1] * 10),  # 转换为0-10的分数，兼容v1格式
        "ratio_db_peak": ratio_db_peak,
        "flux_db_peak": flux_db_peak,
        "flat_db": flat_db,
        "rolloff_ratio": rolloff_ratio,
        "crest_db": crest_db,
        "rms_ratio": rms_ratio,
        "over_ms": over_ms,
        "near_end": bool(near_end),
        "distance_from_end_ms": distance_from_end_ms,
        "features": features_dict,
        "details": {
            "tail_ms": tail_ms,
            "ref_ms": ref_ms,
            "hi_band": hi_band,
            "model_path": str(model_path),
            "prediction": int(prediction),
            "probabilities": {
                "no_noise": float(probabilities[0]),
                "has_noise": float(probabilities[1])
            }
        }
    }


def detect_chi_noise_batch_core(audio_data_list: List[Tuple[np.ndarray, int, str]],
                                 verbose=True,
                                 **detect_kwargs):
    """
    批量检测音频数据（核心批量检测逻辑）
    
    使用 V4.3 模型进行检测，如果失败则回退到 V1 版本。
    
    Args:
        audio_data_list: 音频数据列表，每个元素为 (y, sr, name) 元组
            - y: 音频数据数组
            - sr: 采样率
            - name: 标识名称（用于输出）
        verbose: 是否打印详细信息
        **detect_kwargs: 传递给检测函数的其他参数
            - model_path: Optional[Union[str, Path]], 模型文件路径（V4.3模型，默认使用v4_3）
            - tail_ms: int, 检测尾部窗口长度（毫秒），默认200
            - ref_ms: int, 参考窗口长度（毫秒），默认300
            - hi_band: tuple, 高频频带范围（Hz），默认(5000, 15000)
            - fallback_to_v1: bool, 模型加载失败时是否回退到v1版本，默认True
            注意：v1特有的参数（ratio_db_thresh, ratio_db_min, flux_db_thresh, score_thresh, must_be_within_ms）
            会被自动过滤，不会传递给检测函数
    
    Returns:
        dict: 包含检测结果的字典
            - items_with_chi: list, 检测到杂音的项目名称列表
            - items_clean: list, 未检测到杂音的项目名称列表
            - results: dict, 每个项目的详细检测结果（key为name）
    """
    # 过滤掉v1特有的参数，只保留支持的参数
    supported_params = {'model_path', 'tail_ms', 'ref_ms', 'hi_band', 'fallback_to_v1'}
    detect_kwargs_filtered = {k: v for k, v in detect_kwargs.items() if k in supported_params}
    
    # 如果没有指定model_path，默认使用V4.3模型
    if 'model_path' not in detect_kwargs_filtered or detect_kwargs_filtered['model_path'] is None:
        detect_kwargs_filtered['model_path'] = Path("./noise_detector_model_v4_3.pkl")
    
    items_with_chi = []
    items_clean = []
    results = {}
    
    for y, sr, name in audio_data_list:
        try:
            # 优先使用 V4.3 模型，如果失败则直接回退到 V1
            try:
                result = detect_chi_noise_core_v4_3(y, sr, **detect_kwargs_filtered)
            except (FileNotFoundError, RuntimeError):
                # 如果V4.3模型不存在或加载失败，直接回退到V1
                if detect_kwargs_filtered.get('fallback_to_v1', True):
                    result = detect_chi_noise_core(y, sr, tail_ms=detect_kwargs_filtered.get('tail_ms', 200),
                                                  ref_ms=detect_kwargs_filtered.get('ref_ms', 300),
                                                  hi_band=detect_kwargs_filtered.get('hi_band', (5000, 15000)))
                    result["method"] = "v1_fallback"
                    result["probability"] = 1.0 if result["has_chi"] else 0.0
                else:
                    raise
            results[name] = result
            
            if result["has_chi"]:
                items_with_chi.append(name)
                if verbose:
                    method_info = f" [{result.get('method', 'unknown')}]" if 'method' in result else ""
                    print(f"⚠️  {name}: 检测到杂音 (score={result['score']:.2f}, "
                          f"ratio={result['ratio_db_peak']:.2f}dB, "
                          f"flux={result['flux_db_peak']:.2f}dB{method_info})")
            else:
                items_clean.append(name)
                if verbose:
                    method_info = f" [{result.get('method', 'unknown')}]" if 'method' in result else ""
                    print(f"✓  {name}: 正常 (score={result['score']:.2f}{method_info})")
        except Exception as e:
            print(f"✗  {name}: 检测失败 - {str(e)}")
            results[name] = {"error": str(e)}
    
    return {
        "items_with_chi": items_with_chi,
        "items_clean": items_clean,
        "results": results,
        "total_items": len(audio_data_list),
        "chi_count": len(items_with_chi),
        "clean_count": len(items_clean)
    }


def load_audio_from_path(audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """
    从文件路径加载音频数据
    
    Args:
        audio_path: 音频文件路径
    
    Returns:
        tuple: (y, sr) - 音频数据数组和采样率
    """
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    return y, sr


def load_audio_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    从字节数组加载音频数据
    
    Args:
        audio_bytes: WAV 格式的音频字节数据
    
    Returns:
        tuple: (y, sr) - 音频数据数组和采样率
    """
    with io.BytesIO(audio_bytes) as buffer:
        y, sr = librosa.load(buffer, sr=None, mono=True)
    return y, sr


def detect_chi_noise(audio_input: Union[str, Path, bytes], 
                     model_path: Optional[Union[str, Path]] = None,
                     tail_ms=200,           # 检测尾部窗口（毫秒）
                     ref_ms=300,            # 参考窗口（毫秒）
                     hi_band=(5000, 15000), # 高频频带范围（Hz）
                     fallback_to_v1=True    # 模型加载失败时是否回退到v1版本
                     ):
    """
    检测音频文件结尾是否存在 "chi" 杂音（封装函数，自动处理文件读取）
    
    使用基于机器学习的检测方法（v4.3版本），相比硬编码阈值方法更加灵活和准确。
    如果V4.3模型不可用，会自动回退到V1版本。
    
    Args:
        audio_input: 音频输入，可以是：
            - 文件路径（str 或 Path）
            - 字节数组（bytes，WAV 格式）
        model_path: 模型文件路径，默认为 "./noise_detector_model_v4_3.pkl"
        tail_ms: 检测尾部窗口长度（毫秒），默认200
        ref_ms: 参考窗口长度（毫秒），默认300
        hi_band: 高频频带范围（Hz），默认(5000, 15000)
        fallback_to_v1: 模型加载失败时是否回退到v1版本，默认True
    
    Returns:
        dict: 包含检测结果的字典
            - has_chi: bool, 是否检测到杂音
            - probability: float, 模型预测的概率（0-1）
            - method: str, 使用的检测方法（"ml_v4.3" 或 "v1_fallback"）
            - score: float, 综合评分（兼容v1格式，0-10）
            - ratio_db_peak: float, 高频相对增益峰值（dB）
            - flux_db_peak: float, 瞬态变化峰值（dB）
            - features: dict, 提取的所有特征值（仅机器学习方法）
            - details: dict, 详细检测信息
    """
    # 根据输入类型加载音频
    if isinstance(audio_input, bytes):
        y, sr = load_audio_from_bytes(audio_input)
    else:
        y, sr = load_audio_from_path(audio_input)
    
    # 如果没有指定model_path，默认使用V4.3模型
    if model_path is None:
        model_path = Path("./noise_detector_model_v4_3.pkl")
    
    # 优先使用 V4.3 模型，如果失败则直接回退到 V1
    try:
        return detect_chi_noise_core_v4_3(
            y, sr,
            model_path=model_path,
            tail_ms=tail_ms,
            ref_ms=ref_ms,
            hi_band=hi_band,
            fallback_to_v1=fallback_to_v1
        )
    except (FileNotFoundError, RuntimeError):
        # 如果V4.3模型不存在或加载失败，直接回退到V1
        if fallback_to_v1:
            result = detect_chi_noise_core(y, sr, tail_ms=tail_ms, ref_ms=ref_ms, hi_band=hi_band)
            result["method"] = "v1_fallback"
            result["probability"] = 1.0 if result["has_chi"] else 0.0
            return result
        else:
            raise


def detect_chi_noise_batch(directory, 
                           file_pattern="*.wav",
                           verbose=True,
                           **detect_kwargs):
    """
    批量检测目录中的音频文件（封装函数，自动处理文件读取）
    
    Args:
        directory: 目录路径
        file_pattern: 文件匹配模式，默认为 "*.wav"
        verbose: 是否打印详细信息
        **detect_kwargs: 传递给 detect_chi_noise_core 的其他参数
    
    Returns:
        dict: 包含检测结果的字典
            - files_with_chi: list, 检测到杂音的文件列表
            - files_clean: list, 未检测到杂音的文件列表
            - results: dict, 每个文件的详细检测结果（key为文件路径）
    """
    directory = Path(directory)
    files = sorted(directory.glob(file_pattern))
    
    # 读取所有音频文件
    audio_data_list = []
    for file_path in files:
        try:
            y, sr = load_audio_from_path(file_path)
            audio_data_list.append((y, sr, str(file_path)))
        except Exception as e:
            if verbose:
                print(f"✗  {file_path.name}: 读取失败 - {str(e)}")
    
    # 调用批量检测核心函数
    batch_result = detect_chi_noise_batch_core(
        audio_data_list,
        verbose=verbose,
        **detect_kwargs
    )
    
    # 转换返回格式以保持兼容性
    return {
        "files_with_chi": batch_result["items_with_chi"],
        "files_clean": batch_result["items_clean"],
        "results": batch_result["results"],
        "total_files": batch_result["total_items"],
        "chi_count": batch_result["chi_count"],
        "clean_count": batch_result["clean_count"]
    }


def test_detect_chi_noise_batch(test_dir: Union[str, Path] = "./audio_noise_case",
                                model_path: Optional[Union[str, Path]] = None,
                                tail_ms=200,
                                ref_ms=300,
                                hi_band=(5000, 15000),
                                fallback_to_v1=True,
                                verbose=True):
    """
    测试批量检测音频文件中的杂音
    
    Args:
        test_dir: 测试目录路径
        model_path: 模型文件路径，默认为 None（使用默认路径）
        tail_ms: 检测尾部窗口长度（毫秒）
        ref_ms: 参考窗口长度（毫秒）
        hi_band: 高频频带范围（Hz）
        fallback_to_v1: 模型加载失败时是否回退到v1版本
        verbose: 是否打印详细信息
    """
    test_dir = Path(test_dir)
    
    print("=" * 80)
    print("批量检测 WAV 文件结尾的 'chi' 杂音")
    print("=" * 80)
    print(f"检测目录: {test_dir}\n")
    
    # 批量检测
    batch_result = detect_chi_noise_batch(
        test_dir,
        verbose=verbose,
        model_path=model_path,
        tail_ms=tail_ms,
        ref_ms=ref_ms,
        hi_band=hi_band,
        fallback_to_v1=fallback_to_v1
    )
    
    print("\n" + "=" * 80)
    print("检测总结")
    print("=" * 80)
    print(f"总文件数: {batch_result['total_files']}")
    print(f"检测到杂音: {batch_result['chi_count']} 个")
    print(f"正常文件: {batch_result['clean_count']} 个")
    
    if batch_result['files_with_chi']:
        print("\n检测到杂音的文件:")
        for f in batch_result['files_with_chi']:
            print(f"  - {Path(f).name}")
    
    return batch_result


def test_detect_chi_noise(test_file: Union[str, Path],
                          model_path: Optional[Union[str, Path]] = None,
                          tail_ms=200,
                          ref_ms=300,
                          hi_band=(5000, 15000),
                          fallback_to_v1=True):
    """
    测试单个音频文件的杂音检测
    
    Args:
        test_file: 测试文件路径
        model_path: 模型文件路径，默认为 None（使用默认路径）
        tail_ms: 检测尾部窗口长度（毫秒）
        ref_ms: 参考窗口长度（毫秒）
        hi_band: 高频频带范围（Hz）
        fallback_to_v1: 模型加载失败时是否回退到v1版本
    
    Returns:
        dict: 检测结果
    """
    test_file = Path(test_file)
    
    if not test_file.exists():
        print(f"错误: 文件不存在 - {test_file}")
        return None
    
    print("=" * 80)
    print("单个文件杂音检测测试")
    print("=" * 80)
    print(f"测试文件: {test_file}\n")
    
    # 检测单个文件
    result = detect_chi_noise(
        test_file,
        model_path=model_path,
        tail_ms=tail_ms,
        ref_ms=ref_ms,
        hi_band=hi_band,
        fallback_to_v1=fallback_to_v1
    )
    
    # 打印结果
    print("检测结果:")
    print(f"  是否有杂音: {'是' if result['has_chi'] else '否'}")
    print(f"  检测方法: {result.get('method', 'unknown')}")
    print(f"  评分: {result['score']:.2f}")
    
    if 'probability' in result:
        print(f"  概率: {result['probability']:.4f}")
    
    print(f"  高频相对增益峰值: {result['ratio_db_peak']:.2f} dB")
    print(f"  瞬态变化峰值: {result['flux_db_peak']:.2f} dB")
    print(f"  距离结尾: {result.get('distance_from_end_ms', 0):.2f} ms")
    print(f"  是否靠近结尾: {'是' if result.get('near_end', False) else '否'}")
    
    if 'features' in result:
        print(f"\n特征数量: {len(result['features'])}")
    
    print("\n" + "=" * 80)
    
    return result


if __name__ == "__main__":
    # 测试配置
    test_dir = "./audio_noise_case"
    test_dir = "/Users/chen/Documents/zhetian/split/chapter_156/audio"
    
    # 测试单个文件路径（可以手动指定，或留空自动查找）
    test_file = "./audio_noise_case/chapter_13_audio_31.wav"  # 手动指定测试文件
    
    # # 先测试单个文件
    # if test_file:
    #     test_detect_chi_noise(test_file)
    
    # 然后运行批量检测
    print("\n" + "=" * 80)
    test_detect_chi_noise_batch(test_dir)

