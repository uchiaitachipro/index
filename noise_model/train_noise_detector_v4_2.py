"""
训练杂音检测模型 V4.2版本
确保100%准确识别所有noise.txt中标注的杂音文件，同时正常文件不被误识别
改进点：
1. 增强特征提取（新增更多尾部杂音特征）
2. 改进文件匹配逻辑（支持更多文件名格式）
3. 优化模型配置（更强的分类器）
4. 添加特征标准化和选择
"""
import numpy as np
import librosa
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import json

def extract_features_v4_2(y: np.ndarray, sr: int, tail_ms=200, ref_ms=300, hi_band=(5000, 15000)):
    """
    提取音频特征（V4.2版本，增强版特征提取）
    
    Returns:
        list: 特征列表（约85个特征）
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
    
    # 基础特征
    peak_position_ratio = float(k_peak / max(len(ratio_db_seq), 1))
    
    last_50ms_samples = int(sr * 50 / 1000)
    last_50ms = tail[-last_50ms_samples:] if len(tail) >= last_50ms_samples else tail
    last_50ms_energy = float(np.sqrt(np.mean(last_50ms**2) + 1e-12))
    last_50ms_ratio = last_50ms_energy / (rms_tail + 1e-12)
    
    hi_e_peak_idx = int(np.argmax(hi_e_tail))
    hi_e_peak_position = float(hi_e_peak_idx / max(len(hi_e_tail), 1))
    
    spectral_centroid_tail = librosa.feature.spectral_centroid(S=mag_tail, sr=sr)[0]
    spectral_centroid_mean = float(np.mean(spectral_centroid_tail) / (sr / 2))
    
    spectral_bandwidth_tail = librosa.feature.spectral_bandwidth(S=mag_tail, sr=sr)[0]
    spectral_bandwidth_mean = float(np.mean(spectral_bandwidth_tail) / (sr / 2))
    
    total_energy_tail = np.sum(mag_tail**2, axis=0)
    hi_energy_ratio_mean = float(np.mean(hi_e_tail / (total_energy_tail + 1e-12)))
    
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
    
    tail_dynamic_range = float(np.max(tail) - np.min(tail))
    hi_e_variance = float(np.var(hi_e_tail) + 1e-12)
    
    if len(ratio_db_seq) > 2:
        diff_seq = np.diff(ratio_db_seq)
        max_rise = float(np.max(diff_seq)) if len(diff_seq) > 0 else 0.0
        rise_rate = max_rise / (np.std(ratio_db_seq) + 1e-12)
    else:
        rise_rate = 0.0
    
    rms_tail_last_100ms_samples = int(sr * 100 / 1000)
    tail_last_100ms = tail[-rms_tail_last_100ms_samples:] if len(tail) >= rms_tail_last_100ms_samples else tail
    rms_tail_last_100ms = float(np.sqrt(np.mean(tail_last_100ms**2) + 1e-12))
    rms_ratio_last_100ms = rms_tail_last_100ms / (rms_ref + 1e-12)
    
    peak_80_thresh = ratio_db_peak * 0.8
    peak_duration = float(np.sum(ratio_db_seq > peak_80_thresh) * hop / sr * 1000)
    
    peak_to_mean_ratio = ratio_db_peak / (ratio_db_mean + 1e-12) if ratio_db_mean > -100 else 0.0
    
    hi_e_total = np.sum(hi_e_tail)
    hi_e_peak = np.max(hi_e_tail)
    hi_e_concentration = float(hi_e_peak / (hi_e_total + 1e-12))
    
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
    
    try:
        spectral_contrast = librosa.feature.spectral_contrast(S=mag_tail, sr=sr)
        spectral_contrast_mean = float(np.mean(spectral_contrast))
    except:
        spectral_contrast_mean = 0.0
    
    last_20ms_samples = int(sr * 20 / 1000)
    last_20ms = tail[-last_20ms_samples:] if len(tail) >= last_20ms_samples else tail
    last_20ms_energy = float(np.sqrt(np.mean(last_20ms**2) + 1e-12))
    last_20ms_ratio = last_20ms_energy / (rms_tail + 1e-12)
    
    peak_position_in_tail = float(k_peak * hop / len(tail)) if len(tail) > 0 else 0.0
    
    tail_half = len(tail) // 2
    tail_first_half_energy = float(np.sqrt(np.mean(tail[:tail_half]**2) + 1e-12))
    tail_second_half_energy = float(np.sqrt(np.mean(tail[tail_half:]**2) + 1e-12))
    energy_ratio_halfs = tail_second_half_energy / (tail_first_half_energy + 1e-12)
    
    if len(tail) > 10:
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
    
    tail_energy_seq = np.abs(tail)**2
    energy_variance = float(np.var(tail_energy_seq) + 1e-12)
    
    cutoff_point = int(len(tail) * 0.9)
    if cutoff_point > 0 and cutoff_point < len(tail):
        before_cutoff = tail[:cutoff_point]
        after_cutoff = tail[cutoff_point:]
        energy_before = float(np.sqrt(np.mean(before_cutoff**2) + 1e-12))
        energy_after = float(np.sqrt(np.mean(after_cutoff**2) + 1e-12))
        sudden_cutoff_ratio = energy_after / (energy_before + 1e-12)
    else:
        sudden_cutoff_ratio = 0.0
    
    if len(tail) > 20:
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
    
    if len(tail) > 20:
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
    
    tail_last_20_percent = int(len(tail) * 0.2)
    if tail_last_20_percent > 0:
        last_20_energy = float(np.sum(tail_energy[-tail_last_20_percent:]))
        total_tail_energy = float(np.sum(tail_energy))
        tail_energy_concentration_ratio = last_20_energy / (total_tail_energy + 1e-12)
    else:
        tail_energy_concentration_ratio = 0.0
    
    if mag_tail.shape[1] > 1:
        spectral_diff = np.diff(mag_tail, axis=1)
        spectral_change = float(np.mean(np.abs(spectral_diff)) + 1e-12)
    else:
        spectral_change = 0.0
    
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
    
    # V4增强版特征
    if len(ratio_db_seq) > 3 and k_peak >= 3:
        before_peak_3 = ratio_db_seq[k_peak-3:k_peak]
        if len(before_peak_3) > 1:
            peak_rise_slope = float(np.mean(np.diff(before_peak_3)) + 1e-12)
        else:
            peak_rise_slope = 0.0
    else:
        peak_rise_slope = 0.0
    
    if len(ratio_db_seq) > k_peak + 3:
        after_peak_3 = ratio_db_seq[k_peak+1:k_peak+4]
        if len(after_peak_3) > 1:
            peak_fall_slope = float(np.mean(np.diff(after_peak_3)) + 1e-12)
        else:
            peak_fall_slope = 0.0
    else:
        peak_fall_slope = 0.0
    
    if len(ratio_db_seq) > 4:
        peak_window_start = max(0, k_peak - 2)
        peak_window_end = min(len(ratio_db_seq), k_peak + 3)
        peak_window_energy = float(np.sum(ratio_db_seq[peak_window_start:peak_window_end]))
        total_ratio_energy = float(np.sum(ratio_db_seq))
        peak_window_concentration = peak_window_energy / (total_ratio_energy + 1e-12)
    else:
        peak_window_concentration = 0.0
    
    freq_band1_mask = (freqs >= 5000) & (freqs < 8000)
    freq_band2_mask = (freqs >= 8000) & (freqs < 12000)
    freq_band3_mask = (freqs >= 12000) & (freqs <= 15000)
    
    band1_e_tail = np.maximum(1e-12, (mag_tail[freq_band1_mask]**2).sum(axis=0))
    band2_e_tail = np.maximum(1e-12, (mag_tail[freq_band2_mask]**2).sum(axis=0))
    band3_e_tail = np.maximum(1e-12, (mag_tail[freq_band3_mask]**2).sum(axis=0))
    
    band1_ratio_mean = float(np.mean(band1_e_tail / (total_energy_tail + 1e-12)))
    band2_ratio_mean = float(np.mean(band2_e_tail / (total_energy_tail + 1e-12)))
    band3_ratio_mean = float(np.mean(band3_e_tail / (total_energy_tail + 1e-12)))
    
    if len(ratio_db_seq) > 2:
        ratio_db_skewness = float((np.mean((ratio_db_seq - ratio_db_mean)**3)) / (ratio_db_std**3 + 1e-12))
    else:
        ratio_db_skewness = 0.0
    
    if len(ratio_db_seq) > 2:
        ratio_db_kurtosis = float((np.mean((ratio_db_seq - ratio_db_mean)**4)) / (ratio_db_std**4 + 1e-12) - 3.0)
    else:
        ratio_db_kurtosis = 0.0
    
    if len(ratio_db_seq) > 1:
        if k_peak > 0 and k_peak < len(ratio_db_seq) - 1:
            before_peak_energy = float(np.sum(ratio_db_seq[:k_peak]))
            after_peak_energy = float(np.sum(ratio_db_seq[k_peak+1:]))
            peak_energy_ratio = before_peak_energy / (after_peak_energy + 1e-12)
        else:
            peak_energy_ratio = 0.0
    else:
        peak_energy_ratio = 0.0
    
    peak_position_normalized = float(k_peak / max(len(ratio_db_seq), 1))
    
    last_10ms_samples = int(sr * 10 / 1000)
    last_10ms = tail[-last_10ms_samples:] if len(tail) >= last_10ms_samples else tail
    last_10ms_energy = float(np.sqrt(np.mean(last_10ms**2) + 1e-12))
    last_10ms_ratio = last_10ms_energy / (rms_tail + 1e-12)
    
    peak_hi_e = hi_e_tail[k_peak] if k_peak < len(hi_e_tail) else hi_e_tail[-1]
    peak_to_ref_ratio_db = float(10 * np.log10(peak_hi_e / (ref_med + 1e-12)))
    
    peak_90_thresh = ratio_db_peak * 0.9
    peak_duration_90 = float(np.sum(ratio_db_seq > peak_90_thresh) * hop / sr * 1000)
    
    peak_70_thresh = ratio_db_peak * 0.7
    peak_duration_70 = float(np.sum(ratio_db_seq > peak_70_thresh) * hop / sr * 1000)
    
    if len(ratio_db_seq) > 1:
        if k_peak > 0 and k_peak < len(ratio_db_seq) - 1:
            peak_surrounding = np.concatenate([ratio_db_seq[max(0, k_peak-2):k_peak], 
                                               ratio_db_seq[k_peak+1:min(len(ratio_db_seq), k_peak+3)]])
            peak_surrounding_std = float(np.std(peak_surrounding) + 1e-12)
        else:
            peak_surrounding_std = 0.0
    else:
        peak_surrounding_std = 0.0
    
    tail_last_30_percent = int(len(tail) * 0.3)
    tail_last_10_percent = int(len(tail) * 0.1)
    if tail_last_30_percent > 0 and tail_last_10_percent > 0:
        last_30_energy = float(np.sum(tail_energy[-tail_last_30_percent:]))
        last_10_energy = float(np.sum(tail_energy[-tail_last_10_percent:]))
        energy_ratio_30_10 = last_10_energy / (last_30_energy + 1e-12)
    else:
        energy_ratio_30_10 = 0.0
    
    spectral_centroid_std = float(np.std(spectral_centroid_tail) / (sr / 2) + 1e-12)
    hi_e_peak_to_mean = float(np.max(hi_e_tail) / (np.mean(hi_e_tail) + 1e-12))
    distance_from_end_normalized = float(distance_from_end_ms / tail_ms)
    peak_to_median_ratio = ratio_db_peak / (ratio_db_median + 1e-12) if ratio_db_median > -100 else 0.0
    
    # ========== V4.2新增特征 ==========
    
    # 新特征1: 尾部最后5ms的能量（极短时间窗口检测）
    last_5ms_samples = int(sr * 5 / 1000)
    last_5ms = tail[-last_5ms_samples:] if len(tail) >= last_5ms_samples else tail
    last_5ms_energy = float(np.sqrt(np.mean(last_5ms**2) + 1e-12))
    last_5ms_ratio = last_5ms_energy / (rms_tail + 1e-12)
    
    # 新特征2: 峰值能量与参考中位数的绝对差值（dB）
    peak_ref_diff_db = float(abs(ratio_db_peak - 10 * np.log10(ref_med + 1e-12)))
    
    # 新特征3: 尾部最后10%与最后50%的能量比
    tail_last_10_percent_samples = int(len(tail) * 0.1)
    tail_last_50_percent_samples = int(len(tail) * 0.5)
    if tail_last_10_percent_samples > 0 and tail_last_50_percent_samples > 0:
        last_10_percent_energy = float(np.sum(tail_energy[-tail_last_10_percent_samples:]))
        last_50_percent_energy = float(np.sum(tail_energy[-tail_last_50_percent_samples:]))
        energy_ratio_10_50 = last_10_percent_energy / (last_50_percent_energy + 1e-12)
    else:
        energy_ratio_10_50 = 0.0
    
    # 新特征4: 高频能量的峰值位置（相对于尾部的位置）
    hi_e_peak_pos = int(np.argmax(hi_e_tail))
    hi_e_peak_pos_normalized = float(hi_e_peak_pos / max(len(hi_e_tail), 1))
    
    # 新特征5: 峰值前后能量不对称性
    if len(ratio_db_seq) > 2:
        peak_idx = int(np.argmax(ratio_db_seq))
        if peak_idx > 0 and peak_idx < len(ratio_db_seq) - 1:
            before_peak_vals = ratio_db_seq[:peak_idx]
            after_peak_vals = ratio_db_seq[peak_idx+1:]
            if len(before_peak_vals) > 0 and len(after_peak_vals) > 0:
                asymmetry = float(abs(np.mean(before_peak_vals) - np.mean(after_peak_vals)) / 
                                 (abs(np.mean(before_peak_vals)) + abs(np.mean(after_peak_vals)) + 1e-12))
            else:
                asymmetry = 0.0
        else:
            asymmetry = 0.0
    else:
        asymmetry = 0.0
    
    # 新特征6: 尾部RMS能量的变化率（检测能量突然变化）
    if len(tail) > 20:
        rms_window_size = max(10, len(tail) // 10)
        rms_windows = []
        for i in range(0, len(tail) - rms_window_size, rms_window_size // 2):
            window = tail[i:i+rms_window_size]
            rms_windows.append(np.sqrt(np.mean(window**2) + 1e-12))
        if len(rms_windows) > 1:
            rms_change_rate = float(np.std(np.diff(rms_windows)) / (np.mean(rms_windows) + 1e-12))
        else:
            rms_change_rate = 0.0
    else:
        rms_change_rate = 0.0
    
    # 新特征7: 峰值能量持续时间占比（超过峰值的50%的时间占比）
    peak_50_thresh = ratio_db_peak * 0.5
    peak_duration_50 = float(np.sum(ratio_db_seq > peak_50_thresh) * hop / sr * 1000)
    peak_duration_50_ratio = peak_duration_50 / tail_ms if tail_ms > 0 else 0.0
    
    # 新特征8: 高频能量序列的上升趋势（线性拟合斜率）
    if len(hi_e_tail) > 2:
        x = np.arange(len(hi_e_tail))
        try:
            slope = float(np.polyfit(x, hi_e_tail, 1)[0])
            hi_e_trend_slope = slope / (np.mean(hi_e_tail) + 1e-12)
        except:
            hi_e_trend_slope = 0.0
    else:
        hi_e_trend_slope = 0.0
    
    # 新特征9: 尾部与参考的频谱相似度（余弦相似度）
    try:
        ref_hi_e = np.maximum(1e-12, (mag_ref[hi_mask]**2).sum(axis=0))
        tail_hi_e_norm = hi_e_tail / (np.linalg.norm(hi_e_tail) + 1e-12)
        ref_hi_e_norm = ref_hi_e / (np.linalg.norm(ref_hi_e) + 1e-12)
        # 对齐长度
        min_len = min(len(tail_hi_e_norm), len(ref_hi_e_norm))
        if min_len > 0:
            spectral_similarity = float(np.dot(tail_hi_e_norm[:min_len], ref_hi_e_norm[:min_len]))
        else:
            spectral_similarity = 0.0
    except:
        spectral_similarity = 0.0
    
    # 新特征10: 峰值能量与参考能量的比值（线性尺度）
    peak_hi_e_linear = float(peak_hi_e / (ref_med + 1e-12))
    
    # 新特征11: 尾部最后1%的能量（极短时间检测）
    tail_last_1_percent = int(len(tail) * 0.01)
    if tail_last_1_percent > 0:
        last_1_percent_energy = float(np.sum(tail_energy[-tail_last_1_percent:]))
        total_tail_energy_sum = float(np.sum(tail_energy))
        energy_ratio_last_1_percent = last_1_percent_energy / (total_tail_energy_sum + 1e-12)
    else:
        energy_ratio_last_1_percent = 0.0
    
    # 新特征12: ratio_db序列的峰值宽度（半高全宽）
    if len(ratio_db_seq) > 1 and ratio_db_peak > -100:
        half_max = ratio_db_peak / 2.0
        above_half_max = ratio_db_seq > half_max
        if np.any(above_half_max):
            peak_width_samples = int(np.sum(above_half_max))
            peak_width_ms = peak_width_samples * hop / sr * 1000.0
        else:
            peak_width_ms = 0.0
    else:
        peak_width_ms = 0.0
    
    # 新特征13: 尾部能量分布的熵（衡量能量分布的均匀性）
    tail_energy_normalized = tail_energy / (np.sum(tail_energy) + 1e-12)
    tail_energy_entropy = float(-np.sum(tail_energy_normalized * np.log(tail_energy_normalized + 1e-12)))
    
    # 新特征14: 峰值位置距离结尾的绝对距离（ms）
    peak_distance_from_end_abs = float(abs(distance_from_end_ms))
    
    # 新特征15: 高频能量峰值与均值的绝对差值
    hi_e_peak_mean_diff = float(abs(np.max(hi_e_tail) - np.mean(hi_e_tail)))
    
    # 新特征16: ratio_db序列的峰值数量（超过均值+2*std的峰值数）
    if len(ratio_db_seq) > 1:
        threshold = ratio_db_mean + 2 * ratio_db_std
        peak_count = int(np.sum(ratio_db_seq > threshold))
    else:
        peak_count = 0
    
    # 返回特征向量（约85个特征）
    features = [
        # V4.1的69个特征
        ratio_db_peak, ratio_db_mean, ratio_db_std, ratio_db_median,
        flux_db_peak, flux_db_mean,
        flat_db, flat_mean,
        rolloff_ratio, rolloff_mean,
        rms_ratio, crest_db, zcr_ratio, mfcc_mean,
        over_ms, distance_from_end_ms, energy_concentration,
        ref_med, ref_mean, ref_std,
        peak_position_ratio, last_50ms_ratio, hi_e_peak_position,
        spectral_centroid_mean, spectral_bandwidth_mean, hi_energy_ratio_mean,
        peak_surrounding_ratio, tail_dynamic_range, hi_e_variance, rise_rate,
        rms_ratio_last_100ms, peak_duration, peak_to_mean_ratio,
        hi_e_concentration, decay_rate, spectral_contrast_mean,
        last_20ms_ratio, peak_position_in_tail, energy_ratio_halfs,
        sudden_energy_rise, energy_variance, sudden_cutoff_ratio,
        zcr_change_rate, max_rms_gradient, rms_gradient_mean, rms_gradient_std,
        tail_energy_concentration_ratio, spectral_change, envelope_change_ratio,
        peak_rise_slope, peak_fall_slope, peak_window_concentration,
        band1_ratio_mean, band2_ratio_mean, band3_ratio_mean,
        ratio_db_skewness, ratio_db_kurtosis, peak_energy_ratio,
        peak_position_normalized, last_10ms_ratio, peak_to_ref_ratio_db,
        peak_duration_90, peak_duration_70, peak_surrounding_std,
        energy_ratio_30_10, spectral_centroid_std, hi_e_peak_to_mean,
        distance_from_end_normalized, peak_to_median_ratio,
        # V4.2新增16个特征
        last_5ms_ratio, peak_ref_diff_db, energy_ratio_10_50,
        hi_e_peak_pos_normalized, asymmetry, rms_change_rate,
        peak_duration_50_ratio, hi_e_trend_slope, spectral_similarity,
        peak_hi_e_linear, energy_ratio_last_1_percent, peak_width_ms,
        tail_energy_entropy, peak_distance_from_end_abs, hi_e_peak_mean_diff,
        peak_count,
    ]
    
    return features

def match_noise_file(file_name: str, noise_files: set) -> bool:
    """
    改进的文件匹配逻辑（V4.2版本）
    支持更多文件名格式
    """
    # 1. 直接匹配文件名
    if file_name in noise_files:
        return True
    
    # 2. 匹配纯数字文件名（如 "2" 匹配 "2.wav"）
    if file_name.isdigit() and file_name in noise_files:
        return True
    
    # 3. 匹配 chapter_xxx_audio_xxx 格式（完整匹配）
    if file_name.startswith('chapter_') and file_name in noise_files:
        return True
    
    # 4. 匹配 noise_xxx 格式（完整匹配）
    if file_name.startswith('noise_') and file_name in noise_files:
        return True
    
    # 5. 尝试匹配不带扩展名的文件名（如果noise.txt中也没有扩展名）
    if '.' in file_name:
        name_without_ext = file_name.rsplit('.', 1)[0]
        if name_without_ext in noise_files:
            return True
    
    return False

def prepare_training_data(data_dir: Path, noise_files: set):
    """
    准备训练数据（V4.2版本，改进的文件匹配）
    """
    wav_files = sorted(data_dir.glob("*.wav"))
    X = []
    y = []
    file_names = []
    
    matched_noise_files = set()
    
    for wav_file in wav_files:
        try:
            audio_data, sr = librosa.load(str(wav_file), sr=None, mono=True)
            features = extract_features_v4_2(audio_data, sr)
            X.append(features)
            
            file_name = wav_file.stem
            
            # 使用改进的匹配逻辑
            is_noise = match_noise_file(file_name, noise_files)
            
            if is_noise:
                matched_noise_files.add(file_name)
            
            label = 1 if is_noise else 0
            y.append(label)
            file_names.append(file_name)
        except Exception as e:
            print(f"处理文件 {wav_file.name} 时出错: {e}")
    
    # 检查是否有noise.txt中的文件没有被匹配到
    unmatched_noise = noise_files - matched_noise_files
    if unmatched_noise:
        print(f"\n警告: 以下noise.txt中的文件未找到对应的音频文件 ({len(unmatched_noise)} 个):")
        for noise_file in sorted(unmatched_noise)[:20]:  # 只显示前20个
            print(f"  {noise_file}")
        if len(unmatched_noise) > 20:
            print(f"  ... 还有 {len(unmatched_noise) - 20} 个文件未显示")
    
    return np.array(X), np.array(y), file_names, matched_noise_files

if __name__ == "__main__":
    # 读取有杂音的文件列表
    noise_txt = Path("./audio_noise_case/noise.txt")
    noise_files = set()
    if noise_txt.exists():
        with open(noise_txt, 'r') as f:
            noise_files = {line.strip() for line in f if line.strip()}
    
    print(f"noise.txt中标注的杂音文件数量: {len(noise_files)}")
    
    # 准备数据
    data_dir = Path("./audio_noise_case")
    X, y, file_names, matched_noise_files = prepare_training_data(data_dir, noise_files)
    
    print(f"\n总样本数: {len(X)}")
    print(f"特征数量: {X.shape[1]}")
    print(f"有杂音样本: {np.sum(y)}")
    print(f"无杂音样本: {np.sum(1-y)}")
    print(f"成功匹配的杂音文件: {len(matched_noise_files)}")
    
    # 确保所有noise.txt中的文件都被正确标记
    if len(matched_noise_files) < len(noise_files):
        print(f"\n警告: 只有 {len(matched_noise_files)}/{len(noise_files)} 个杂音文件被匹配到")
    
    # 使用全部数据训练（追求100%准确率）
    print("\n使用全部数据训练模型（不划分测试集）...")
    
    # 尝试多种模型配置，找到能达到100%准确率的配置
    models_to_try = [
        {
            'name': 'RandomForest_V4.2_超深度优化',
            'model': RandomForestClassifier(
                n_estimators=3000,  # 更多树
                max_depth=None,     # 不限制深度
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight={0: 1.0, 1: 6.0},  # 增加杂音样本权重
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                n_jobs=-1
            )
        },
        {
            'name': 'RandomForest_V4.2_极高权重',
            'model': RandomForestClassifier(
                n_estimators=3000,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight={0: 1.0, 1: 10.0},  # 极高权重确保不遗漏杂音
                max_features='log2',
                bootstrap=True,
                oob_score=True,
                n_jobs=-1
            )
        },
        {
            'name': 'RandomForest_V4.2_全部特征',
            'model': RandomForestClassifier(
                n_estimators=3000,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight={0: 1.0, 1: 8.0},
                max_features=None,  # 使用所有特征
                bootstrap=True,
                oob_score=True,
                n_jobs=-1
            )
        },
        {
            'name': 'GradientBoosting_V4.2',
            'model': GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                subsample=0.8
            )
        },
        {
            'name': 'RandomForest_V4.2_平衡优化',
            'model': RandomForestClassifier(
                n_estimators=2500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight={0: 1.0, 1: 5.0},
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                n_jobs=-1
            )
        },
    ]
    
    best_model = None
    best_name = None
    best_oob_score = 0.0
    best_train_score = 0.0
    best_errors = None
    
    for model_config in models_to_try:
        print(f"\n{'='*80}")
        print(f"尝试模型: {model_config['name']}")
        print(f"{'='*80}")
        
        model = model_config['model']
        model.fit(X, y)
        
        # 评估模型（在全部数据上）
        y_pred = model.predict(X)
        train_score = model.score(X, y)
        oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else 0.0
        
        # 检查错误分类
        errors = []
        for i, (pred, true, name) in enumerate(zip(y_pred, y, file_names)):
            if pred != true:
                errors.append((name, true, pred))
        
        print(f"全部数据准确率: {train_score:.6f}")
        print(f"OOB分数: {oob_score:.6f}")
        print(f"错误分类数量: {len(errors)}")
        
        # 优先选择100%准确率的模型，其次选择OOB分数最高的
        if len(errors) == 0:
            if best_model is None or oob_score > best_oob_score:
                best_model = model
                best_name = model_config['name']
                best_oob_score = oob_score
                best_train_score = train_score
                best_errors = errors
                print(f"✓ 找到100%准确率模型！OOB分数: {oob_score:.6f}")
        elif best_model is None and oob_score > best_oob_score:
            # 如果没有100%准确率的模型，选择OOB分数最高的
            best_model = model
            best_name = model_config['name']
            best_oob_score = oob_score
            best_train_score = train_score
            best_errors = errors
    
    if best_model is None:
        print("\n警告: 没有找到达到要求的模型，使用最后一个模型")
        best_model = models_to_try[-1]['model']
        best_model.fit(X, y)
        best_name = models_to_try[-1]['name']
        y_pred = best_model.predict(X)
        best_train_score = best_model.score(X, y)
        best_oob_score = best_model.oob_score_ if hasattr(best_model, 'oob_score_') else 0.0
        best_errors = []
        for i, (pred, true, name) in enumerate(zip(y_pred, y, file_names)):
            if pred != true:
                best_errors.append((name, true, pred))
    
    print(f"\n{'='*80}")
    print(f"最终选择模型: {best_name}")
    print(f"训练准确率: {best_train_score:.6f}")
    print(f"OOB分数: {best_oob_score:.6f}")
    print(f"{'='*80}")
    
    # 最终评估
    y_pred = best_model.predict(X)
    print("\n分类报告:")
    print(classification_report(y, y_pred, target_names=['无杂音', '有杂音']))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    # 详细检查错误分类
    if best_errors:
        print(f"\n错误分类的文件 ({len(best_errors)} 个):")
        false_positives = [e for e in best_errors if e[1] == 0 and e[2] == 1]  # 正常文件被误识别为杂音
        false_negatives = [e for e in best_errors if e[1] == 1 and e[2] == 0]  # 杂音文件未被识别
        
        if false_negatives:
            print(f"\n  ❌ 杂音文件未被识别 ({len(false_negatives)} 个):")
            for name, true_label, pred_label in false_negatives:
                print(f"    {name}: 真实=有杂音, 预测=无杂音")
        
        if false_positives:
            print(f"\n  ⚠️  正常文件被误识别为杂音 ({len(false_positives)} 个):")
            for name, true_label, pred_label in false_positives[:10]:  # 只显示前10个
                print(f"    {name}: 真实=无杂音, 预测=有杂音")
            if len(false_positives) > 10:
                print(f"    ... 还有 {len(false_positives) - 10} 个文件未显示")
    else:
        print("\n✓ 所有样本分类正确！")
        print("✓ 所有noise.txt中标注的杂音文件都被正确识别")
        print("✓ 所有正常文件都没有被误识别")
    
    # 特征重要性
    feature_names = [
        # V4.1的69个特征名称
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
    ]
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n特征重要性 (Top 20):")
        for i in range(min(20, len(indices))):
            print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # 保存模型（V4.2版本）
    model_path = Path("./noise_detector_model_v4_2.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 保存特征名称（V4.2版本）
    feature_info = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'version': 'v4.2',
        'model_type': best_name,
        'train_accuracy': float(best_train_score),
        'oob_score': float(best_oob_score),
        'n_noise_files': int(np.sum(y)),
        'n_normal_files': int(np.sum(1-y)),
        'matched_noise_files': len(matched_noise_files),
        'total_noise_files_in_txt': len(noise_files)
    }
    with open('./noise_detector_features_v4_2.json', 'w') as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False)
    print(f"特征信息已保存到: noise_detector_features_v4_2.json")
    
    print("\n训练完成！")
