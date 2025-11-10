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
from typing import List, Dict, Tuple, Union


def detect_chi_noise_core(y: np.ndarray,
                          sr: int,
                          tail_ms=200,           # 检测尾部窗口（毫秒）
                          ref_ms=300,            # 参考窗口（毫秒）
                          hi_band=(5000, 15000), # 高频频带范围（Hz）
                          ratio_db_thresh=5.5,   # 高频相对增益阈值（dB）
                          ratio_db_min=10.0,     # 高频相对增益最低要求（dB），用于过滤弱信号
                          flux_db_thresh=3.0,    # 瞬态变化阈值（dB）
                          score_thresh=4.0,      # 综合评分阈值（提高以减少误检）
                          must_be_within_ms=150  # 杂音必须出现在结尾多少毫秒内
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
    near_end = distance_from_end <= int((must_be_within_ms / 1000) * sr)
    
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
    
    if flux_db_peak > flux_db_thresh:
        score += 1.5  # 瞬态变化明显
    elif flux_db_peak > flux_db_thresh - 1.0:
        score += 0.5
    
    # 次要指标（权重较低）
    if flat_db > -10.0:  # 频谱较平坦（可能是噪声）
        score += 0.5
    
    if rolloff_ratio > 0.75:  # 高频成分较多
        score += 0.5
    
    if crest_db > 12.0:  # 波峰因子较高（可能是瞬态杂音）
        score += 0.5
    
    if rms_ratio < 0.95:  # 尾部能量相对较低（符合杂音特征）
        score += 0.3
    
    if over_ms > 0 and over_ms < 150:  # 持续时间适中（不是持续噪声）
        score += 0.5
    
    # 必须靠近结尾
    if not near_end:
        score *= 0.3  # 如果不在结尾附近，大幅降低分数
    
    # 判定结果：必须满足最低 ratio 要求，且评分超过阈值，且靠近结尾
    has_chi = (ratio_db_peak >= ratio_db_min and 
               score >= score_thresh and 
               near_end)
    
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


def detect_chi_noise_batch_core(audio_data_list: List[Tuple[np.ndarray, int, str]],
                                 verbose=True,
                                 **detect_kwargs):
    """
    批量检测音频数据（核心批量检测逻辑）
    
    Args:
        audio_data_list: 音频数据列表，每个元素为 (y, sr, name) 元组
            - y: 音频数据数组
            - sr: 采样率
            - name: 标识名称（用于输出）
        verbose: 是否打印详细信息
        **detect_kwargs: 传递给 detect_chi_noise_core 的其他参数
    
    Returns:
        dict: 包含检测结果的字典
            - items_with_chi: list, 检测到杂音的项目名称列表
            - items_clean: list, 未检测到杂音的项目名称列表
            - results: dict, 每个项目的详细检测结果（key为name）
    """
    items_with_chi = []
    items_clean = []
    results = {}
    
    for y, sr, name in audio_data_list:
        try:
            result = detect_chi_noise_core(y, sr, **detect_kwargs)
            results[name] = result
            
            if result["has_chi"]:
                items_with_chi.append(name)
                if verbose:
                    print(f"⚠️  {name}: 检测到杂音 (score={result['score']:.2f}, "
                          f"ratio={result['ratio_db_peak']:.2f}dB, "
                          f"flux={result['flux_db_peak']:.2f}dB)")
            else:
                items_clean.append(name)
                if verbose:
                    print(f"✓  {name}: 正常 (score={result['score']:.2f})")
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
                     tail_ms=200,           # 检测尾部窗口（毫秒）
                     ref_ms=300,            # 参考窗口（毫秒）
                     hi_band=(5000, 15000), # 高频频带范围（Hz）
                     ratio_db_thresh=5.5,   # 高频相对增益阈值（dB）
                     ratio_db_min=10.0,     # 高频相对增益最低要求（dB），用于过滤弱信号
                     flux_db_thresh=3.0,    # 瞬态变化阈值（dB）
                     score_thresh=4.0,      # 综合评分阈值（提高以减少误检）
                     must_be_within_ms=150  # 杂音必须出现在结尾多少毫秒内
                     ):
    """
    检测音频文件结尾是否存在 "chi" 杂音（封装函数，自动处理文件读取）
    
    Args:
        audio_input: 音频输入，可以是：
            - 文件路径（str 或 Path）
            - 字节数组（bytes，WAV 格式）
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
    # 根据输入类型加载音频
    if isinstance(audio_input, bytes):
        y, sr = load_audio_from_bytes(audio_input)
    else:
        y, sr = load_audio_from_path(audio_input)
    
    # 调用核心检测函数
    return detect_chi_noise_core(
        y, sr,
        tail_ms=tail_ms,
        ref_ms=ref_ms,
        hi_band=hi_band,
        ratio_db_thresh=ratio_db_thresh,
        ratio_db_min=ratio_db_min,
        flux_db_thresh=flux_db_thresh,
        score_thresh=score_thresh,
        must_be_within_ms=must_be_within_ms
    )


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


if __name__ == "__main__":
    # 测试检测函数
    import sys
    
    # 默认检测 outputs 目录
    test_dir = "./audio_noise_case"
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    
    print("=" * 80)
    print("检测 WAV 文件结尾的 'chi' 杂音")
    print("=" * 80)
    print(f"检测目录: {test_dir}\n")
    
    # 批量检测
    batch_result = detect_chi_noise_batch(
        test_dir,
        verbose=True,
        tail_ms=200,
        ref_ms=300,
        hi_band=(5000, 15000),
        ratio_db_thresh=5.5,
        ratio_db_min=10.0,  # 最低要求，过滤弱信号
        flux_db_thresh=3.0,
        score_thresh=4.0,  # 提高阈值以减少误检
        must_be_within_ms=150
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

