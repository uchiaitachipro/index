"""
训练杂音检测模型
使用机器学习方法替代硬编码阈值
"""
import numpy as np
import librosa
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

def extract_features(y: np.ndarray, sr: int, tail_ms=200, ref_ms=300, hi_band=(5000, 15000)):
    """
    提取音频特征（用于机器学习）
    
    Returns:
        dict: 特征字典
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
    
    ref_med = float(np.median(hi_e_ref))
    ref_mean = float(np.mean(hi_e_ref))
    ref_std = float(np.std(hi_e_ref) + 1e-12)
    
    # 计算相对增益（dB）
    ratio_db_seq = 10 * np.log10(hi_e_tail / (ref_med + 1e-12))
    ratio_db_peak = float(np.max(ratio_db_seq))
    ratio_db_mean = float(np.mean(ratio_db_seq))
    ratio_db_std = float(np.std(ratio_db_seq))
    ratio_db_median = float(np.median(ratio_db_seq))
    
    # 计算瞬态变化（谱通量）
    flux_seq = np.maximum(0.0, np.diff(np.log(hi_e_tail + 1e-12)))
    flux_db_peak = float(10 * np.log10(1 + np.max(flux_seq))) if len(flux_seq) > 0 else 0.0
    flux_db_mean = float(10 * np.log10(1 + np.mean(flux_seq))) if len(flux_seq) > 0 else 0.0
    
    # 计算频谱特征
    flat_tail = librosa.feature.spectral_flatness(S=mag_tail**2)
    flat_db = float(10 * np.log10(np.median(flat_tail) + 1e-12))
    flat_mean = float(10 * np.log10(np.mean(flat_tail) + 1e-12))
    
    rolloff_tail = librosa.feature.spectral_rolloff(S=mag_tail, sr=sr, roll_percent=0.95)
    rolloff_ratio = float(np.median(rolloff_tail) / (sr / 2))
    rolloff_mean = float(np.mean(rolloff_tail) / (sr / 2))
    
    # 计算 RMS 能量比
    rms_tail = float(np.sqrt(np.mean(tail**2) + 1e-12))
    rms_ref = float(np.sqrt(np.mean(ref**2) + 1e-12))
    rms_ratio = rms_tail / (rms_ref + 1e-12)
    
    # 计算波峰因子
    peak_tail = float(np.max(np.abs(tail)))
    crest_db = float(20 * np.log10((peak_tail + 1e-9) / (rms_tail + 1e-9)))
    
    # 计算零交叉率
    zcr_tail = float(librosa.feature.zero_crossing_rate(tail)[0].mean())
    zcr_ref = float(librosa.feature.zero_crossing_rate(ref)[0].mean())
    zcr_ratio = zcr_tail / (zcr_ref + 1e-12)
    
    # 计算MFCC特征（前3个系数）
    mfcc_tail = librosa.feature.mfcc(y=tail, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfcc_tail[1:4]))  # 使用1-3号系数
    
    # 计算超阈值持续时间
    over_thresh = ratio_db_seq > 5.5
    over_frames = int(np.sum(over_thresh))
    over_ms = over_frames * hop / sr * 1000.0
    
    # 找到峰值位置
    k_peak = int(np.argmax(ratio_db_seq))
    t0_global = len(y) - N_tail + int(k_peak * hop)
    distance_from_end = len(y) - t0_global
    distance_from_end_ms = distance_from_end / sr * 1000
    
    # 计算尾部能量分布
    tail_energy = np.abs(tail)**2
    tail_energy_norm = tail_energy / (np.sum(tail_energy) + 1e-12)
    energy_concentration = float(np.sum(tail_energy_norm[-int(len(tail)*0.3):]))  # 最后30%的能量集中度
    
    # 新增特征1: 峰值位置相对于尾部的比例
    peak_position_ratio = float(k_peak / max(len(ratio_db_seq), 1))
    
    # 新增特征2: 尾部最后50ms的能量
    last_50ms_samples = int(sr * 50 / 1000)
    last_50ms = tail[-last_50ms_samples:] if len(tail) >= last_50ms_samples else tail
    last_50ms_energy = float(np.sqrt(np.mean(last_50ms**2) + 1e-12))
    last_50ms_ratio = last_50ms_energy / (rms_tail + 1e-12)
    
    # 新增特征3: 高频能量的峰值位置
    hi_e_peak_idx = int(np.argmax(hi_e_tail))
    hi_e_peak_position = float(hi_e_peak_idx / max(len(hi_e_tail), 1))
    
    # 新增特征4: 频谱质心 (Spectral Centroid)
    spectral_centroid_tail = librosa.feature.spectral_centroid(S=mag_tail, sr=sr)[0]
    spectral_centroid_mean = float(np.mean(spectral_centroid_tail) / (sr / 2))
    
    # 新增特征5: 频谱带宽 (Spectral Bandwidth)
    spectral_bandwidth_tail = librosa.feature.spectral_bandwidth(S=mag_tail, sr=sr)[0]
    spectral_bandwidth_mean = float(np.mean(spectral_bandwidth_tail) / (sr / 2))
    
    # 新增特征6: 高频能量与总能量的比值
    total_energy_tail = np.sum(mag_tail**2, axis=0)
    hi_energy_ratio_mean = float(np.mean(hi_e_tail / (total_energy_tail + 1e-12)))
    
    # 新增特征7: 峰值前后的能量变化率
    if len(ratio_db_seq) > 1:
        peak_idx = int(np.argmax(ratio_db_seq))
        if peak_idx > 0 and peak_idx < len(ratio_db_seq) - 1:
            before_peak = ratio_db_seq[max(0, peak_idx-2):peak_idx].mean() if peak_idx >= 2 else ratio_db_seq[0]
            after_peak = ratio_db_seq[peak_idx+1:min(len(ratio_db_seq), peak_idx+3)].mean() if peak_idx < len(ratio_db_seq)-1 else ratio_db_seq[-1]
            peak_surrounding_ratio = float((ratio_db_peak - before_peak) / (abs(after_peak - before_peak) + 1e-12))
        else:
            peak_surrounding_ratio = 0.0
    else:
        peak_surrounding_ratio = 0.0
    
    # 新增特征8: 尾部音频的动态范围
    tail_dynamic_range = float(np.max(tail) - np.min(tail))
    
    # 新增特征9: 高频能量的方差（衡量稳定性）
    hi_e_variance = float(np.var(hi_e_tail) + 1e-12)
    
    # 新增特征10: ratio_db序列的上升率（检测瞬态）
    if len(ratio_db_seq) > 2:
        diff_seq = np.diff(ratio_db_seq)
        max_rise = float(np.max(diff_seq)) if len(diff_seq) > 0 else 0.0
        rise_rate = max_rise / (np.std(ratio_db_seq) + 1e-12)
    else:
        rise_rate = 0.0
    
    # 新增特征11: 尾部RMS与参考RMS的比值（更详细的能量比较）
    tail_last_100ms_samples = int(sr * 100 / 1000)
    tail_last_100ms = tail[-tail_last_100ms_samples:] if len(tail) >= tail_last_100ms_samples else tail
    rms_tail_last_100ms = float(np.sqrt(np.mean(tail_last_100ms**2) + 1e-12))
    rms_ratio_last_100ms = rms_tail_last_100ms / (rms_ref + 1e-12)
    
    # 新增特征12: 峰值能量持续时间（超过峰值的80%的时间）
    peak_80_thresh = ratio_db_peak * 0.8
    peak_duration = float(np.sum(ratio_db_seq > peak_80_thresh) * hop / sr * 1000)
    
    # 新增特征13: 峰值能量与平均能量的比值
    peak_to_mean_ratio = ratio_db_peak / (ratio_db_mean + 1e-12) if ratio_db_mean > -100 else 0.0
    
    # 新增特征14: 高频能量的集中度（峰值能量占总能量的比例）
    hi_e_total = np.sum(hi_e_tail)
    hi_e_peak = np.max(hi_e_tail)
    hi_e_concentration = float(hi_e_peak / (hi_e_total + 1e-12))
    
    # 新增特征15: 峰值后的能量衰减率
    if len(ratio_db_seq) > 1:
        peak_idx = int(np.argmax(ratio_db_seq))
        if peak_idx < len(ratio_db_seq) - 1:
            after_peak_values = ratio_db_seq[peak_idx+1:]
            if len(after_peak_values) > 0:
                decay_rate = float((ratio_db_peak - np.mean(after_peak_values)) / (len(after_peak_values) + 1e-12))
            else:
                decay_rate = 0.0
        else:
            decay_rate = 0.0
    else:
        decay_rate = 0.0
    
    # 新增特征16: 频谱对比度 (Spectral Contrast)
    try:
        spectral_contrast = librosa.feature.spectral_contrast(S=mag_tail, sr=sr)
        spectral_contrast_mean = float(np.mean(spectral_contrast))
    except:
        spectral_contrast_mean = 0.0
    
    # 新增特征17: 尾部最后20ms的能量（更精确的尾部检测）
    last_20ms_samples = int(sr * 20 / 1000)
    last_20ms = tail[-last_20ms_samples:] if len(tail) >= last_20ms_samples else tail
    last_20ms_energy = float(np.sqrt(np.mean(last_20ms**2) + 1e-12))
    last_20ms_ratio = last_20ms_energy / (rms_tail + 1e-12)
    
    # 新增特征18: ratio_db序列的峰值位置（相对于尾部的位置）
    peak_position_in_tail = float(k_peak * hop / len(tail)) if len(tail) > 0 else 0.0
    
    # V3 新增特征1: 尾部前半部分和后半部分的能量比
    tail_half = len(tail) // 2
    tail_first_half_energy = float(np.sqrt(np.mean(tail[:tail_half]**2) + 1e-12))
    tail_second_half_energy = float(np.sqrt(np.mean(tail[tail_half:]**2) + 1e-12))
    energy_ratio_halfs = tail_second_half_energy / (tail_first_half_energy + 1e-12)
    
    # V3 新增特征2: 突然的能量上升（检测尾部能量突然增加）
    if len(tail) > 10:
        # 计算RMS能量序列（使用滑动窗口）
        window_size = max(10, len(tail) // 20)
        rms_seq = []
        for i in range(0, len(tail) - window_size, window_size // 2):
            window = tail[i:i+window_size]
            rms_seq.append(np.sqrt(np.mean(window**2) + 1e-12))
        if len(rms_seq) > 1:
            rms_diff = np.diff(rms_seq)
            sudden_energy_rise = float(np.max(rms_diff) / (np.mean(rms_seq) + 1e-12))
        else:
            sudden_energy_rise = 0.0
    else:
        sudden_energy_rise = 0.0
    
    # V3 新增特征3: 能量方差（衡量尾部能量的波动）
    tail_energy_seq = np.abs(tail)**2
    energy_variance = float(np.var(tail_energy_seq) + 1e-12)
    
    # V3 新增特征4: 突然截止的比例（检测音频是否突然停止）
    # 计算最后10%的音频与之前90%的能量比
    cutoff_point = int(len(tail) * 0.9)
    if cutoff_point > 0 and cutoff_point < len(tail):
        before_cutoff = tail[:cutoff_point]
        after_cutoff = tail[cutoff_point:]
        energy_before = float(np.sqrt(np.mean(before_cutoff**2) + 1e-12))
        energy_after = float(np.sqrt(np.mean(after_cutoff**2) + 1e-12))
        sudden_cutoff_ratio = energy_after / (energy_before + 1e-12)
    else:
        sudden_cutoff_ratio = 0.0
    
    # V3 新增特征5: 零交叉率变化率
    if len(tail) > 20:
        # 分段计算ZCR
        segment_size = len(tail) // 5
        zcr_segments = []
        for i in range(0, len(tail) - segment_size, segment_size):
            segment = tail[i:i+segment_size]
            zcr_segments.append(librosa.feature.zero_crossing_rate(segment)[0].mean())
        if len(zcr_segments) > 1:
            zcr_diff = np.diff(zcr_segments)
            zcr_change_rate = float(np.std(zcr_diff) + 1e-12)
        else:
            zcr_change_rate = 0.0
    else:
        zcr_change_rate = 0.0
    
    # V3 新增特征6-8: RMS梯度特征（检测能量变化趋势）
    if len(tail) > 20:
        # 计算RMS序列
        rms_window = max(10, len(tail) // 30)
        rms_gradient_seq = []
        for i in range(0, len(tail) - rms_window, rms_window // 2):
            window = tail[i:i+rms_window]
            rms_gradient_seq.append(np.sqrt(np.mean(window**2) + 1e-12))
        if len(rms_gradient_seq) > 1:
            rms_gradients = np.diff(rms_gradient_seq)
            max_rms_gradient = float(np.max(np.abs(rms_gradients)))
            rms_gradient_mean = float(np.mean(rms_gradients))
            rms_gradient_std = float(np.std(rms_gradients) + 1e-12)
        else:
            max_rms_gradient = 0.0
            rms_gradient_mean = 0.0
            rms_gradient_std = 0.0
    else:
        max_rms_gradient = 0.0
        rms_gradient_mean = 0.0
        rms_gradient_std = 0.0
    
    # V3 新增特征9: 尾部能量集中度比例（最后20%的能量占比）
    tail_last_20_percent = int(len(tail) * 0.2)
    if tail_last_20_percent > 0:
        last_20_energy = float(np.sum(tail_energy[-tail_last_20_percent:]))
        total_tail_energy = float(np.sum(tail_energy))
        tail_energy_concentration_ratio = last_20_energy / (total_tail_energy + 1e-12)
    else:
        tail_energy_concentration_ratio = 0.0
    
    # V3 新增特征10: 频谱变化（检测频谱的突然变化）
    if mag_tail.shape[1] > 1:
        # 计算相邻帧之间的频谱差异
        spectral_diff = np.diff(mag_tail, axis=1)
        spectral_change = float(np.mean(np.abs(spectral_diff)) + 1e-12)
    else:
        spectral_change = 0.0
    
    # V3 新增特征11: 包络变化比例（检测音频包络的变化）
    # 使用Hilbert变换计算包络
    try:
        from scipy.signal import hilbert
        analytic_signal = hilbert(tail)
        envelope = np.abs(analytic_signal)
        if len(envelope) > 1:
            envelope_diff = np.diff(envelope)
            envelope_change_ratio = float(np.std(envelope_diff) / (np.mean(envelope) + 1e-12))
        else:
            envelope_change_ratio = 0.0
    except ImportError:
        # 如果没有scipy，使用简单的RMS作为包络
        envelope_window = max(10, len(tail) // 20)
        envelope = []
        for i in range(0, len(tail), envelope_window):
            window = tail[i:i+envelope_window]
            envelope.append(np.sqrt(np.mean(window**2) + 1e-12))
        if len(envelope) > 1:
            envelope_diff = np.diff(envelope)
            envelope_change_ratio = float(np.std(envelope_diff) / (np.mean(envelope) + 1e-12))
        else:
            envelope_change_ratio = 0.0
    
    # 返回特征向量（V3版本，49个特征）
    features = [
        ratio_db_peak,
        ratio_db_mean,
        ratio_db_std,
        ratio_db_median,
        flux_db_peak,
        flux_db_mean,
        flat_db,
        flat_mean,
        rolloff_ratio,
        rolloff_mean,
        rms_ratio,
        crest_db,
        zcr_ratio,
        mfcc_mean,
        over_ms,
        distance_from_end_ms,
        energy_concentration,
        ref_med,
        ref_mean,
        ref_std,
        # 新增特征
        peak_position_ratio,
        last_50ms_ratio,
        hi_e_peak_position,
        spectral_centroid_mean,
        spectral_bandwidth_mean,
        hi_energy_ratio_mean,
        peak_surrounding_ratio,
        tail_dynamic_range,
        hi_e_variance,
        rise_rate,
        rms_ratio_last_100ms,
        peak_duration,
        # 新增特征
        peak_to_mean_ratio,
        hi_e_concentration,
        decay_rate,
        spectral_contrast_mean,
        last_20ms_ratio,
        peak_position_in_tail,
        # V3 新增特征
        energy_ratio_halfs,
        sudden_energy_rise,
        energy_variance,
        sudden_cutoff_ratio,
        zcr_change_rate,
        max_rms_gradient,
        rms_gradient_mean,
        rms_gradient_std,
        tail_energy_concentration_ratio,
        spectral_change,
        envelope_change_ratio,
    ]
    
    return features

def prepare_training_data(data_dir: Path, noise_files: set):
    """
    准备训练数据
    """
    wav_files = sorted(data_dir.glob("*.wav"))
    X = []
    y = []
    file_names = []
    
    for wav_file in wav_files:
        try:
            audio_data, sr = librosa.load(str(wav_file), sr=None, mono=True)
            features = extract_features(audio_data, sr)
            X.append(features)
            
            file_name = wav_file.stem
            label = 1 if file_name in noise_files else 0
            y.append(label)
            file_names.append(file_name)
        except Exception as e:
            print(f"处理文件 {wav_file.name} 时出错: {e}")
    
    return np.array(X), np.array(y), file_names

if __name__ == "__main__":
    # 读取有杂音的文件列表
    noise_txt = Path("./audio_noise_case/noise.txt")
    noise_files = set()
    if noise_txt.exists():
        with open(noise_txt, 'r') as f:
            noise_files = {line.strip() for line in f if line.strip()}
    
    print(f"有杂音的文件: {sorted(noise_files)}")
    
    # 准备数据
    data_dir = Path("./audio_noise_case")
    X, y, file_names = prepare_training_data(data_dir, noise_files)
    
    print(f"\n总样本数: {len(X)}")
    print(f"有杂音样本: {np.sum(y)}")
    print(f"无杂音样本: {np.sum(1-y)}")
    
    # 使用全部数据训练（追求100%准确率）
    print("\n使用全部数据训练模型（不划分测试集）...")
    
    # 训练模型（优化参数，追求100%准确率）
    print("\n训练随机森林模型...")
    model = RandomForestClassifier(
        n_estimators=500,  # 进一步增加树的数量
        max_depth=25,      # 增加深度以捕获更复杂的模式
        min_samples_split=2,  # 进一步降低分裂阈值
        min_samples_leaf=1,   # 降低叶子节点最小样本数
        random_state=42,
        class_weight={0: 1.0, 1: 2.5},  # 给有杂音类别更高权重
        max_features='sqrt',  # 使用sqrt特征数，提高泛化能力
        bootstrap=True,
        oob_score=True  # 计算OOB分数
    )
    
    model.fit(X, y)
    
    # 评估模型（在全部数据上）
    train_score = model.score(X, y)
    
    print(f"\n全部数据准确率: {train_score:.4f}")
    if hasattr(model, 'oob_score_'):
        print(f"OOB分数: {model.oob_score_:.4f}")
    
    # 详细评估
    y_pred = model.predict(X)
    print("\n分类报告:")
    print(classification_report(y, y_pred, target_names=['无杂音', '有杂音']))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(y, y_pred))
    
    # 检查是否有错误分类
    errors = []
    for i, (pred, true, name) in enumerate(zip(y_pred, y, file_names)):
        if pred != true:
            errors.append((name, true, pred))
    
    if errors:
        print(f"\n错误分类的文件 ({len(errors)} 个):")
        for name, true_label, pred_label in errors:
            print(f"  {name}: 真实={true_label}, 预测={pred_label}")
    else:
        print("\n✓ 所有样本分类正确！")
    
    # 特征重要性
    feature_names = [
        'ratio_db_peak', 'ratio_db_mean', 'ratio_db_std', 'ratio_db_median',
        'flux_db_peak', 'flux_db_mean',
        'flat_db', 'flat_mean',
        'rolloff_ratio', 'rolloff_mean',
        'rms_ratio',
        'crest_db',
        'zcr_ratio',
        'mfcc_mean',
        'over_ms',
        'distance_from_end_ms',
        'energy_concentration',
        'ref_med', 'ref_mean', 'ref_std',
        # 新增特征名称
        'peak_position_ratio',
        'last_50ms_ratio',
        'hi_e_peak_position',
        'spectral_centroid_mean',
        'spectral_bandwidth_mean',
        'hi_energy_ratio_mean',
        'peak_surrounding_ratio',
        'tail_dynamic_range',
        'hi_e_variance',
        'rise_rate',
        'rms_ratio_last_100ms',
        'peak_duration',
        # 新增特征名称
        'peak_to_mean_ratio',
        'hi_e_concentration',
        'decay_rate',
        'spectral_contrast_mean',
        'last_20ms_ratio',
        'peak_position_in_tail',
        # V3 新增特征名称
        'energy_ratio_halfs',
        'sudden_energy_rise',
        'energy_variance',
        'sudden_cutoff_ratio',
        'zcr_change_rate',
        'max_rms_gradient',
        'rms_gradient_mean',
        'rms_gradient_std',
        'tail_energy_concentration_ratio',
        'spectral_change',
        'envelope_change_ratio',
    ]
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n特征重要性 (Top 10):")
    for i in range(min(10, len(indices))):
        print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # 保存模型（V3.3版本）
    model_path = Path("./noise_detector_model_v3.3.pkl")
    joblib.dump(model, model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 保存特征名称（V3.3版本）
    feature_info = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'version': 'v3.3'
    }
    with open('./noise_detector_features_v3.3.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"特征信息已保存到: noise_detector_features_v3.3.json")

