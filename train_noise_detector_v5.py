"""
训练杂音检测模型 V5版本
改进点：
1. 使用XGBoost/LightGBM替代RandomForest，提升泛化能力
2. 添加深度学习方法（1D CNN + 梅尔频谱图特征）
3. 使用交叉验证评估真实泛化能力
4. 添加数据增强提升鲁棒性
5. 支持模型集成（Stacking/Voting）
6. 添加特征选择优化
"""
import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_JIT'] = '0'

import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# 尝试导入高级库
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("提示: 安装 xgboost 可获得更好性能: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("提示: 安装 lightgbm 可获得更好性能: pip install lightgbm")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("提示: 安装 pytorch 可使用深度学习方法: pip install torch")


def get_torch_device():
    """
    获取最佳的 PyTorch 设备
    优先级: CUDA > MPS (Mac) > CPU
    """
    if not PYTORCH_AVAILABLE:
        return None
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Mac Apple Silicon 的 MPS 加速
        return torch.device('mps')
    else:
        return torch.device('cpu')

# 尝试导入scipy用于峰值检测
try:
    from scipy.signal import find_peaks, hilbert
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 尝试导入tqdm用于进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ============== 特征名称定义 ==============
FEATURE_NAMES_V5 = [
    # V4.1的69个特征
    'ratio_db_peak', 'ratio_db_mean', 'ratio_db_std', 'ratio_db_median',
    'flux_db_peak', 'flux_db_mean', 'flat_db', 'flat_mean',
    'rolloff_ratio', 'rolloff_mean', 'rms_ratio', 'crest_db', 'zcr_ratio', 'mfcc_mean',
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
    # V4.2的16个特征
    'last_5ms_ratio', 'peak_ref_diff_db', 'energy_ratio_10_50',
    'hi_e_peak_pos_normalized', 'asymmetry', 'rms_change_rate',
    'peak_duration_50_ratio', 'hi_e_trend_slope', 'spectral_similarity',
    'peak_hi_e_linear', 'energy_ratio_last_1_percent', 'peak_width_ms',
    'tail_energy_entropy', 'peak_distance_from_end_abs', 'hi_e_peak_mean_diff',
    'peak_count',
    # V4.3的20个特征
    'last_2ms_ratio', 'last_2ms_to_10ms_ratio', 'energy_ratio_5_95',
    'hi_e_accel_max', 'ratio_db_range', 'zcr_ratio_20_80',
    'hi_e_mean_ratio', 'post_peak_decay', 'tail_energy_std',
    'tail_energy_skewness', 'pre_post_diff', 'hi_e_p90_p50_ratio',
    'spectral_centroid_change_30ms', 'positive_ratio', 'hi_e_cv',
    'tail_autocorr', 'local_energy_ratio', 'rms_max_min_ratio',
    'hi_e_range_mean_ratio', 'last_15ms_ratio',
    # V5新增特征（28个）
    'mel_mean', 'mel_std', 'mel_max', 'mel_min', 'mel_range',
    'mel_high_mean', 'mel_high_std', 'mel_temporal_change',
    'mfcc_delta_mean', 'mfcc_delta_std', 'mfcc_delta2_mean', 'mfcc_delta2_std',
    'mfcc_coef_0_mean', 'mfcc_coef_1_mean', 'mfcc_coef_2_mean', 'mfcc_coef_3_mean', 'mfcc_coef_4_mean',
    'mfcc_coef_0_std', 'mfcc_coef_1_std', 'mfcc_coef_2_std', 'mfcc_coef_3_std', 'mfcc_coef_4_std',
    'num_peaks', 'peak_heights_mean', 'peak_heights_std',
    'sc_diff_mean', 'sc_diff_max',
    'energy_env_diff_range',
]


def extract_features_v5(y: np.ndarray, sr: int, tail_ms=200, ref_ms=300, hi_band=(5000, 15000)):
    """
    提取音频特征（V5版本，在V4.3基础上增加新特征）
    
    新增特征：
    1. 梅尔频谱图统计特征
    2. MFCC差分特征
    3. 更多时域统计特征
    
    Returns:
        list: 特征列表（约133个特征）
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
    
    # 计算MFCC特征（前13个系数）
    mfcc_tail = librosa.feature.mfcc(y=tail, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfcc_tail[1:4]))
    
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
    energy_concentration = float(np.sum(tail_energy_norm[-int(len(tail)*0.3):]))
    
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
    
    # 包络变化率
    if SCIPY_AVAILABLE:
        try:
            analytic_signal = hilbert(tail)
            envelope = np.abs(analytic_signal)
            if len(envelope) > 1:
                envelope_diff = np.diff(envelope)
                envelope_change_ratio = float(np.std(envelope_diff) / (np.mean(envelope) + 1e-12))
            else:
                envelope_change_ratio = 0.0
        except:
            envelope_change_ratio = 0.0
    else:
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
        last_10_energy_val = float(np.sum(tail_energy[-tail_last_10_percent:]))
        energy_ratio_30_10 = last_10_energy_val / (last_30_energy + 1e-12)
    else:
        energy_ratio_30_10 = 0.0
    
    spectral_centroid_std = float(np.std(spectral_centroid_tail) / (sr / 2) + 1e-12)
    hi_e_peak_to_mean = float(np.max(hi_e_tail) / (np.mean(hi_e_tail) + 1e-12))
    distance_from_end_normalized = float(distance_from_end_ms / tail_ms)
    peak_to_median_ratio = ratio_db_peak / (ratio_db_median + 1e-12) if ratio_db_median > -100 else 0.0
    
    # V4.2特征
    last_5ms_samples = int(sr * 5 / 1000)
    last_5ms = tail[-last_5ms_samples:] if len(tail) >= last_5ms_samples else tail
    last_5ms_energy = float(np.sqrt(np.mean(last_5ms**2) + 1e-12))
    last_5ms_ratio = last_5ms_energy / (rms_tail + 1e-12)
    
    peak_ref_diff_db = float(abs(ratio_db_peak - 10 * np.log10(ref_med + 1e-12)))
    
    tail_last_10_percent_samples = int(len(tail) * 0.1)
    tail_last_50_percent_samples = int(len(tail) * 0.5)
    if tail_last_10_percent_samples > 0 and tail_last_50_percent_samples > 0:
        last_10_percent_energy = float(np.sum(tail_energy[-tail_last_10_percent_samples:]))
        last_50_percent_energy = float(np.sum(tail_energy[-tail_last_50_percent_samples:]))
        energy_ratio_10_50 = last_10_percent_energy / (last_50_percent_energy + 1e-12)
    else:
        energy_ratio_10_50 = 0.0
    
    hi_e_peak_pos = int(np.argmax(hi_e_tail))
    hi_e_peak_pos_normalized = float(hi_e_peak_pos / max(len(hi_e_tail), 1))
    
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
    
    peak_50_thresh = ratio_db_peak * 0.5
    peak_duration_50 = float(np.sum(ratio_db_seq > peak_50_thresh) * hop / sr * 1000)
    peak_duration_50_ratio = peak_duration_50 / tail_ms if tail_ms > 0 else 0.0
    
    if len(hi_e_tail) > 2:
        x = np.arange(len(hi_e_tail))
        try:
            slope = float(np.polyfit(x, hi_e_tail, 1)[0])
            hi_e_trend_slope = slope / (np.mean(hi_e_tail) + 1e-12)
        except:
            hi_e_trend_slope = 0.0
    else:
        hi_e_trend_slope = 0.0
    
    try:
        ref_hi_e = np.maximum(1e-12, (mag_ref[hi_mask]**2).sum(axis=0))
        tail_hi_e_norm = hi_e_tail / (np.linalg.norm(hi_e_tail) + 1e-12)
        ref_hi_e_norm = ref_hi_e / (np.linalg.norm(ref_hi_e) + 1e-12)
        min_len = min(len(tail_hi_e_norm), len(ref_hi_e_norm))
        if min_len > 0:
            spectral_similarity = float(np.dot(tail_hi_e_norm[:min_len], ref_hi_e_norm[:min_len]))
        else:
            spectral_similarity = 0.0
    except:
        spectral_similarity = 0.0
    
    peak_hi_e_linear = float(peak_hi_e / (ref_med + 1e-12))
    
    tail_last_1_percent = int(len(tail) * 0.01)
    if tail_last_1_percent > 0:
        last_1_percent_energy = float(np.sum(tail_energy[-tail_last_1_percent:]))
        total_tail_energy_sum = float(np.sum(tail_energy))
        energy_ratio_last_1_percent = last_1_percent_energy / (total_tail_energy_sum + 1e-12)
    else:
        energy_ratio_last_1_percent = 0.0
    
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
    
    tail_energy_normalized = tail_energy / (np.sum(tail_energy) + 1e-12)
    tail_energy_entropy = float(-np.sum(tail_energy_normalized * np.log(tail_energy_normalized + 1e-12)))
    
    peak_distance_from_end_abs = float(abs(distance_from_end_ms))
    hi_e_peak_mean_diff = float(abs(np.max(hi_e_tail) - np.mean(hi_e_tail)))
    
    if len(ratio_db_seq) > 1:
        threshold = ratio_db_mean + 2 * ratio_db_std
        peak_count = int(np.sum(ratio_db_seq > threshold))
    else:
        peak_count = 0
    
    # V4.3特征
    last_2ms_samples = max(1, int(sr * 2 / 1000))
    last_2ms = tail[-last_2ms_samples:] if len(tail) >= last_2ms_samples else tail
    last_2ms_energy = float(np.sqrt(np.mean(last_2ms**2) + 1e-12))
    last_2ms_ratio = last_2ms_energy / (rms_tail + 1e-12)
    last_2ms_to_10ms_ratio = last_2ms_energy / (last_10ms_energy + 1e-12)
    
    tail_last_5_percent = int(len(tail) * 0.05)
    if tail_last_5_percent > 0:
        last_5_percent_energy = float(np.sum(tail_energy[-tail_last_5_percent:]))
        first_95_percent_energy = float(np.sum(tail_energy[:-tail_last_5_percent]))
        energy_ratio_5_95 = last_5_percent_energy / (first_95_percent_energy + 1e-12)
    else:
        energy_ratio_5_95 = 0.0
    
    if len(hi_e_tail) > 2:
        hi_e_diff2 = np.diff(hi_e_tail, n=2)
        hi_e_accel_max = float(np.max(np.abs(hi_e_diff2))) if len(hi_e_diff2) > 0 else 0.0
    else:
        hi_e_accel_max = 0.0
    
    ratio_db_range = float(np.max(ratio_db_seq) - np.min(ratio_db_seq)) if len(ratio_db_seq) > 0 else 0.0
    
    tail_last_20_samples = int(len(tail) * 0.2)
    if tail_last_20_samples > 10:
        zcr_last_20 = float(librosa.feature.zero_crossing_rate(tail[-tail_last_20_samples:])[0].mean())
        zcr_first_80 = float(librosa.feature.zero_crossing_rate(tail[:-tail_last_20_samples])[0].mean())
        zcr_ratio_20_80 = zcr_last_20 / (zcr_first_80 + 1e-12)
    else:
        zcr_ratio_20_80 = 1.0
    
    hi_e_tail_mean = float(np.mean(hi_e_tail))
    hi_e_ref_mean = float(np.mean(hi_e_ref))
    hi_e_mean_ratio = hi_e_tail_mean / (hi_e_ref_mean + 1e-12)
    
    if len(ratio_db_seq) > k_peak + 5:
        post_peak = ratio_db_seq[k_peak+1:k_peak+6]
        if len(post_peak) > 1:
            post_peak_decay = float(np.mean(np.diff(post_peak)))
        else:
            post_peak_decay = 0.0
    else:
        post_peak_decay = 0.0
    
    tail_energy_std = float(np.std(tail_energy) + 1e-12)
    
    if len(tail_energy) > 2:
        tail_energy_mean_val = np.mean(tail_energy)
        tail_energy_std_val = np.std(tail_energy) + 1e-12
        tail_energy_skewness = float(np.mean(((tail_energy - tail_energy_mean_val) / tail_energy_std_val)**3))
    else:
        tail_energy_skewness = 0.0
    
    if len(ratio_db_seq) > k_peak + 3 and k_peak >= 3:
        pre_3 = ratio_db_seq[k_peak-3:k_peak]
        post_3 = ratio_db_seq[k_peak+1:k_peak+4]
        if len(pre_3) > 0 and len(post_3) > 0:
            pre_post_diff = float(np.mean(pre_3) - np.mean(post_3))
        else:
            pre_post_diff = 0.0
    else:
        pre_post_diff = 0.0
    
    hi_e_p90 = float(np.percentile(hi_e_tail, 90))
    hi_e_p50 = float(np.percentile(hi_e_tail, 50))
    hi_e_p90_p50_ratio = hi_e_p90 / (hi_e_p50 + 1e-12)
    
    last_30ms_samples = int(sr * 30 / 1000)
    if len(tail) >= last_30ms_samples and len(spectral_centroid_tail) > 1:
        n_frames_30ms = int(last_30ms_samples / hop)
        if n_frames_30ms > 1 and n_frames_30ms <= len(spectral_centroid_tail):
            sc_last_30ms = spectral_centroid_tail[-n_frames_30ms:]
            spectral_centroid_change_30ms = float(np.std(sc_last_30ms) / (np.mean(sc_last_30ms) + 1e-12))
        else:
            spectral_centroid_change_30ms = 0.0
    else:
        spectral_centroid_change_30ms = 0.0
    
    if len(ratio_db_seq) > 0:
        positive_ratio = float(np.sum(ratio_db_seq > 0) / len(ratio_db_seq))
    else:
        positive_ratio = 0.0
    
    hi_e_cv = float(np.std(hi_e_tail) / (np.mean(hi_e_tail) + 1e-12))
    
    if len(tail_energy) > 1:
        tail_energy_mean_val = np.mean(tail_energy)
        tail_energy_centered = tail_energy - tail_energy_mean_val
        autocorr_num = np.sum(tail_energy_centered[:-1] * tail_energy_centered[1:])
        autocorr_denom = np.sum(tail_energy_centered**2) + 1e-12
        tail_autocorr = float(autocorr_num / autocorr_denom)
    else:
        tail_autocorr = 0.0
    
    if len(ratio_db_seq) > 4:
        local_start = max(0, k_peak - 2)
        local_end = min(len(ratio_db_seq), k_peak + 3)
        local_energy = float(np.sum(np.abs(ratio_db_seq[local_start:local_end])))
        total_abs_energy = float(np.sum(np.abs(ratio_db_seq)))
        local_energy_ratio = local_energy / (total_abs_energy + 1e-12)
    else:
        local_energy_ratio = 0.0
    
    if len(tail) > 20:
        rms_win = max(10, len(tail) // 20)
        rms_vals = []
        for i in range(0, len(tail) - rms_win, rms_win // 2):
            rms_vals.append(np.sqrt(np.mean(tail[i:i+rms_win]**2) + 1e-12))
        if len(rms_vals) > 0:
            rms_max_min_ratio = float(max(rms_vals) / (min(rms_vals) + 1e-12))
        else:
            rms_max_min_ratio = 1.0
    else:
        rms_max_min_ratio = 1.0
    
    hi_e_range = float(np.max(hi_e_tail) - np.min(hi_e_tail))
    hi_e_range_mean_ratio = hi_e_range / (np.mean(hi_e_tail) + 1e-12)
    
    last_15ms_samples = int(sr * 15 / 1000)
    last_15ms = tail[-last_15ms_samples:] if len(tail) >= last_15ms_samples else tail
    last_15ms_energy = float(np.sqrt(np.mean(last_15ms**2) + 1e-12))
    last_15ms_ratio = last_15ms_energy / (rms_tail + 1e-12)
    
    # ========== V5新增特征 ==========
    
    # 1. 梅尔频谱图统计特征
    try:
        mel_spec = librosa.feature.melspectrogram(y=tail, sr=sr, n_mels=64)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = float(np.mean(mel_spec_db))
        mel_std = float(np.std(mel_spec_db))
        mel_max = float(np.max(mel_spec_db))
        mel_min = float(np.min(mel_spec_db))
        mel_range = mel_max - mel_min
        
        # 高频梅尔带（后32个）的能量
        mel_high_mean = float(np.mean(mel_spec_db[32:, :]))
        mel_high_std = float(np.std(mel_spec_db[32:, :]))
        
        # 时间维度的变化
        if mel_spec_db.shape[1] > 1:
            mel_temporal_diff = np.diff(mel_spec_db, axis=1)
            mel_temporal_change = float(np.mean(np.abs(mel_temporal_diff)))
        else:
            mel_temporal_change = 0.0
    except:
        mel_mean = mel_std = mel_max = mel_min = mel_range = 0.0
        mel_high_mean = mel_high_std = mel_temporal_change = 0.0
    
    # 2. MFCC差分特征
    try:
        mfcc_delta = librosa.feature.delta(mfcc_tail)
        mfcc_delta2 = librosa.feature.delta(mfcc_tail, order=2)
        
        mfcc_delta_mean = float(np.mean(np.abs(mfcc_delta)))
        mfcc_delta_std = float(np.std(mfcc_delta))
        mfcc_delta2_mean = float(np.mean(np.abs(mfcc_delta2)))
        mfcc_delta2_std = float(np.std(mfcc_delta2))
        
        # MFCC各系数的统计
        mfcc_coef_means = [float(np.mean(mfcc_tail[i])) for i in range(min(5, mfcc_tail.shape[0]))]
        mfcc_coef_stds = [float(np.std(mfcc_tail[i])) for i in range(min(5, mfcc_tail.shape[0]))]
        # 补齐到5个
        while len(mfcc_coef_means) < 5:
            mfcc_coef_means.append(0.0)
        while len(mfcc_coef_stds) < 5:
            mfcc_coef_stds.append(0.0)
    except:
        mfcc_delta_mean = mfcc_delta_std = mfcc_delta2_mean = mfcc_delta2_std = 0.0
        mfcc_coef_means = [0.0] * 5
        mfcc_coef_stds = [0.0] * 5
    
    # 3. 峰值检测特征
    if SCIPY_AVAILABLE:
        try:
            peaks, properties = find_peaks(np.abs(tail), height=rms_tail * 2)
            num_peaks = len(peaks)
            if num_peaks > 0:
                peak_heights_mean = float(np.mean(properties['peak_heights']))
                peak_heights_std = float(np.std(properties['peak_heights']))
            else:
                peak_heights_mean = peak_heights_std = 0.0
        except:
            num_peaks = 0
            peak_heights_mean = peak_heights_std = 0.0
    else:
        # 简化版峰值检测
        num_peaks = 0
        peak_heights_mean = peak_heights_std = 0.0
    
    # 4. 频谱质心的变化率
    if len(spectral_centroid_tail) > 1:
        sc_diff = np.diff(spectral_centroid_tail)
        sc_diff_mean = float(np.mean(np.abs(sc_diff)))
        sc_diff_max = float(np.max(np.abs(sc_diff)))
    else:
        sc_diff_mean = sc_diff_max = 0.0
    
    # 5. 能量包络的导数特征
    if len(tail_energy) > 10:
        energy_env_smooth = np.convolve(tail_energy, np.ones(10)/10, mode='valid')
        if len(energy_env_smooth) > 1:
            energy_env_diff = np.diff(energy_env_smooth)
            energy_env_diff_max = float(np.max(energy_env_diff))
            energy_env_diff_min = float(np.min(energy_env_diff))
            energy_env_diff_range = energy_env_diff_max - energy_env_diff_min
        else:
            energy_env_diff_range = 0.0
    else:
        energy_env_diff_range = 0.0
    
    # 返回特征向量
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
        # V4.2的16个特征
        last_5ms_ratio, peak_ref_diff_db, energy_ratio_10_50,
        hi_e_peak_pos_normalized, asymmetry, rms_change_rate,
        peak_duration_50_ratio, hi_e_trend_slope, spectral_similarity,
        peak_hi_e_linear, energy_ratio_last_1_percent, peak_width_ms,
        tail_energy_entropy, peak_distance_from_end_abs, hi_e_peak_mean_diff,
        peak_count,
        # V4.3的20个特征
        last_2ms_ratio, last_2ms_to_10ms_ratio, energy_ratio_5_95,
        hi_e_accel_max, ratio_db_range, zcr_ratio_20_80,
        hi_e_mean_ratio, post_peak_decay, tail_energy_std,
        tail_energy_skewness, pre_post_diff, hi_e_p90_p50_ratio,
        spectral_centroid_change_30ms, positive_ratio, hi_e_cv,
        tail_autocorr, local_energy_ratio, rms_max_min_ratio,
        hi_e_range_mean_ratio, last_15ms_ratio,
        # V5新增特征（28个）
        mel_mean, mel_std, mel_max, mel_min, mel_range,
        mel_high_mean, mel_high_std, mel_temporal_change,
        mfcc_delta_mean, mfcc_delta_std, mfcc_delta2_mean, mfcc_delta2_std,
        mfcc_coef_means[0], mfcc_coef_means[1], mfcc_coef_means[2], mfcc_coef_means[3], mfcc_coef_means[4],
        mfcc_coef_stds[0], mfcc_coef_stds[1], mfcc_coef_stds[2], mfcc_coef_stds[3], mfcc_coef_stds[4],
        num_peaks, peak_heights_mean, peak_heights_std,
        sc_diff_mean, sc_diff_max,
        energy_env_diff_range,
    ]
    
    return features


def augment_audio(y: np.ndarray, sr: int, augment_type: str = 'noise'):
    """
    数据增强函数
    
    Args:
        y: 音频数据
        sr: 采样率
        augment_type: 增强类型 ('noise', 'pitch', 'speed', 'volume')
    
    Returns:
        增强后的音频数据
    """
    if augment_type == 'noise':
        # 添加轻微噪声
        noise = np.random.randn(len(y)) * 0.005
        return y + noise
    elif augment_type == 'pitch':
        # 微调音调 (±1半音)
        n_steps = np.random.uniform(-1, 1)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    elif augment_type == 'speed':
        # 微调速度 (0.95-1.05)
        rate = np.random.uniform(0.95, 1.05)
        return librosa.effects.time_stretch(y, rate=rate)
    elif augment_type == 'volume':
        # 调整音量 (0.8-1.2)
        factor = np.random.uniform(0.8, 1.2)
        return y * factor
    return y


def match_noise_file(file_name: str, noise_files: set) -> bool:
    """改进的文件匹配逻辑"""
    if file_name in noise_files:
        return True
    if file_name.isdigit() and file_name in noise_files:
        return True
    if file_name.startswith('chapter_') and file_name in noise_files:
        return True
    if file_name.startswith('noise_') and file_name in noise_files:
        return True
    if '.' in file_name:
        name_without_ext = file_name.rsplit('.', 1)[0]
        if name_without_ext in noise_files:
            return True
    return False


def prepare_training_data(data_dir: Path, noise_files: set, use_augmentation=True, verbose=True):
    """
    准备训练数据（支持数据增强）
    
    Args:
        data_dir: 数据目录
        noise_files: 杂音文件名集合
        use_augmentation: 是否使用数据增强
        verbose: 是否打印详细信息
    
    Returns:
        X: 特征矩阵
        y: 标签数组
        file_names: 文件名列表
        matched_noise_files: 匹配到的杂音文件集合
    """
    wav_files = sorted(data_dir.glob("*.wav"))
    X = []
    y = []
    file_names = []
    
    matched_noise_files = set()
    
    total_files = len(wav_files)
    for idx, wav_file in enumerate(wav_files):
        if verbose and (idx + 1) % 50 == 0:
            print(f"  处理进度: {idx + 1}/{total_files}")
        
        try:
            audio_data, sr = librosa.load(str(wav_file), sr=None, mono=True)
            
            # 原始特征
            features = extract_features_v5(audio_data, sr)
            X.append(features)
            
            file_name = wav_file.stem
            is_noise = match_noise_file(file_name, noise_files)
            
            if is_noise:
                matched_noise_files.add(file_name)
            
            label = 1 if is_noise else 0
            y.append(label)
            file_names.append(file_name)
            
            # 数据增强（仅对有杂音的样本）
            if use_augmentation and is_noise:
                for aug_type in ['noise', 'volume']:
                    try:
                        aug_audio = augment_audio(audio_data, sr, aug_type)
                        aug_features = extract_features_v5(aug_audio, sr)
                        X.append(aug_features)
                        y.append(1)
                        file_names.append(f"{file_name}_aug_{aug_type}")
                    except Exception as e:
                        if verbose:
                            print(f"    增强失败 ({aug_type}): {e}")
                        
        except Exception as e:
            print(f"处理文件 {wav_file.name} 时出错: {e}")
    
    # 检查未匹配的文件
    unmatched_noise = noise_files - matched_noise_files
    if unmatched_noise and verbose:
        print(f"\n警告: 以下noise.txt中的文件未找到对应的音频文件 ({len(unmatched_noise)} 个):")
        for noise_file in sorted(unmatched_noise)[:10]:
            print(f"  {noise_file}")
        if len(unmatched_noise) > 10:
            print(f"  ... 还有 {len(unmatched_noise) - 10} 个未显示")
    
    return np.array(X), np.array(y), file_names, matched_noise_files


def evaluate_model_cv(model, X, y, cv=5, verbose=True):
    """
    使用交叉验证评估模型的真实泛化能力
    
    Args:
        model: 模型
        X: 特征矩阵
        y: 标签数组
        cv: 交叉验证折数
        verbose: 是否打印详细信息
    
    Returns:
        dict: 评估结果
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    
    # 计算ROC-AUC需要概率预测
    try:
        roc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    except:
        roc_scores = np.array([0.0] * cv)
    
    results = {
        'accuracy_mean': float(np.mean(accuracy_scores)),
        'accuracy_std': float(np.std(accuracy_scores)),
        'f1_mean': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'roc_auc_mean': float(np.mean(roc_scores)),
        'roc_auc_std': float(np.std(roc_scores)),
    }
    
    if verbose:
        print(f"  交叉验证准确率: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
        print(f"  交叉验证F1分数: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
        print(f"  交叉验证AUC-ROC: {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
    
    return results


# PyTorch模型定义（仅在PyTorch可用时）
if PYTORCH_AVAILABLE:
    class AudioNoiseCNN(nn.Module):
        """1D CNN模型用于杂音检测"""
        
        def __init__(self, input_size):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 2)
            self.dropout = nn.Dropout(0.3)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # x: (batch, features)
            x = x.unsqueeze(1)  # (batch, 1, features)
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool(x).squeeze(-1)  # (batch, 128)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x

    class PyTorchModelWrapper:
        """PyTorch模型包装器，使其兼容sklearn接口"""
        
        def __init__(self, model, scaler, device='cpu'):
            self.model = model
            self.scaler = scaler
            self.device = device
            self.classes_ = np.array([0, 1])
        
        def predict(self, X):
            self.model.eval()
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
        
        def predict_proba(self, X):
            self.model.eval()
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()


def train_pytorch_model(X_train, y_train, epochs=100, lr=0.001, verbose=True):
    """
    训练PyTorch深度学习模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        epochs: 训练轮数
        lr: 学习率
        verbose: 是否打印详细信息
    
    Returns:
        dict: 包含模型和scaler的字典
    """
    if not PYTORCH_AVAILABLE:
        return None
    
    device = get_torch_device()
    if verbose:
        device_name = str(device)
        if device.type == 'cuda':
            device_name = f"cuda ({torch.cuda.get_device_name(0)})"
        elif device.type == 'mps':
            device_name = "mps (Apple Silicon GPU)"
        print(f"  使用设备: {device_name}")
    
    # 根据设备类型调整参数
    # MPS 和 CUDA 使用更大的 batch_size，CPU 使用较小的
    if device.type in ('cuda', 'mps'):
        batch_size = 128  # GPU 用更大的 batch
        # MPS 优化：减少 epochs，因为收敛较快
        if device.type == 'mps':
            epochs = min(epochs, 50)  # MPS 最多50轮，避免过长训练
    else:
        batch_size = 32  # CPU 用较小 batch
    
    if verbose:
        print(f"  批次大小: {batch_size}, 最大轮数: {epochs}")
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 转换为张量
    X_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = AudioNoiseCNN(X_train.shape[1]).to(device)
    
    # 计算类别权重
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor([1.0, class_counts[0] / (class_counts[1] + 1)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # MPS/CUDA 使用更大学习率加速收敛
    if device.type in ('cuda', 'mps'):
        lr = 0.003  # 更大的学习率
        early_stop_patience = 10  # 更快的 early stopping
    else:
        early_stop_patience = 20
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # 训练
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_acc = 0.0
    best_model_state = None  # 保存最佳模型
    
    # 创建epoch进度条
    if verbose and TQDM_AVAILABLE:
        epoch_pbar = tqdm(range(epochs), desc="  训练进度", unit="epoch", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        epoch_pbar = range(epochs)
    
    for epoch in epoch_pbar:
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        scheduler.step(avg_loss)
        
        # 更新进度条信息
        if verbose and TQDM_AVAILABLE:
            epoch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2%}',
                'best': f'{best_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'
            })
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_acc = accuracy
            patience_counter = 0
            # 保存最佳模型状态
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    if TQDM_AVAILABLE:
                        epoch_pbar.close()
                    print(f"  Early stopping at epoch {epoch+1}, best loss: {best_loss:.4f}, best acc: {best_acc:.2%}")
                break
        
        # 没有tqdm时的简单进度显示
        if verbose and not TQDM_AVAILABLE and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2%}, LR: {optimizer.param_groups[0]['lr']:.1e}")
    
    # 恢复最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if verbose:
        print(f"  训练完成! 最终Loss: {best_loss:.4f}, 最终Acc: {best_acc:.2%}")
    
    return PyTorchModelWrapper(model, scaler, device)


def get_models_to_try():
    """获取要尝试的模型列表"""
    models = []
    
    # 1. XGBoost（推荐）
    if XGBOOST_AVAILABLE:
        models.append({
            'name': 'XGBoost',
            'model': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=5,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'needs_eval_set': True
        })
    
    # 2. LightGBM
    if LIGHTGBM_AVAILABLE:
        models.append({
            'name': 'LightGBM',
            'model': lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight={0: 1, 1: 5},
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'needs_eval_set': False
        })
    
    # 3. RandomForest（基准）
    models.append({
        'name': 'RandomForest_V5',
        'model': RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight={0: 1.0, 1: 6.0},
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        ),
        'needs_eval_set': False
    })
    
    # 4. RandomForest（高权重）
    models.append({
        'name': 'RandomForest_HighWeight',
        'model': RandomForestClassifier(
            n_estimators=1500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight={0: 1.0, 1: 10.0},
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        ),
        'needs_eval_set': False
    })
    
    return models


def train_and_evaluate_models(X, y, file_names, verbose=True):
    """
    训练和评估多个模型
    
    Args:
        X: 特征矩阵
        y: 标签数组
        file_names: 文件名列表
        verbose: 是否打印详细信息
    
    Returns:
        tuple: (best_model, best_name, results_list)
    """
    models_to_try = get_models_to_try()
    
    results = []
    best_model = None
    best_name = None
    best_cv_score = 0.0
    best_train_acc = 0.0
    
    for model_config in models_to_try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"训练模型: {model_config['name']}")
            print(f"{'='*60}")
        
        model = model_config['model']
        
        # 使用5折交叉验证评估泛化能力
        if verbose:
            print("进行5折交叉验证...")
        cv_results = evaluate_model_cv(model, X, y, cv=5, verbose=verbose)
        
        # 在全部数据上训练最终模型
        if verbose:
            print("在全部数据上训练最终模型...")
        
        if model_config.get('needs_eval_set') and XGBOOST_AVAILABLE:
            # XGBoost需要eval_set进行early stopping
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=42, stratify=y
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            # 重新在全部数据上训练（不使用early stopping）
            model_final = model.__class__(**{k: v for k, v in model.get_params().items() 
                                            if k != 'early_stopping_rounds'})
            model_final.fit(X, y)
            model = model_final
        else:
            model.fit(X, y)
        
        # 检查训练集准确率
        y_pred = model.predict(X)
        train_acc = float(np.mean(y_pred == y))
        
        # 检查错误分类
        errors = [(name, true, pred) for pred, true, name in zip(y_pred, y, file_names) if pred != true]
        
        if verbose:
            print(f"  训练集准确率: {train_acc:.4f}")
            print(f"  错误分类数量: {len(errors)}")
            
            if hasattr(model, 'oob_score_'):
                print(f"  OOB分数: {model.oob_score_:.4f}")
        
        results.append({
            'name': model_config['name'],
            'model': model,
            'cv_results': cv_results,
            'train_acc': train_acc,
            'errors': errors,
            'n_errors': len(errors)
        })
        
        # 选择最佳模型：优先选择100%准确率，其次选择交叉验证AUC-ROC最高
        if len(errors) == 0:
            # 100%准确率的模型
            if best_model is None or cv_results['roc_auc_mean'] > best_cv_score:
                best_cv_score = cv_results['roc_auc_mean']
                best_model = model
                best_name = model_config['name']
                best_train_acc = train_acc
                if verbose:
                    print(f"  ✓ 找到100%准确率模型！")
        elif best_model is None or (cv_results['roc_auc_mean'] > best_cv_score and best_train_acc < 1.0):
            best_cv_score = cv_results['roc_auc_mean']
            best_model = model
            best_name = model_config['name']
            best_train_acc = train_acc
    
    # 尝试PyTorch深度学习模型
    if PYTORCH_AVAILABLE:
        if verbose:
            print(f"\n{'='*60}")
            print("训练深度学习模型: PyTorch CNN")
            print(f"{'='*60}")
        
        try:
            pytorch_model = train_pytorch_model(X, y, epochs=100, verbose=verbose)
            if pytorch_model is not None:
                y_pred = pytorch_model.predict(X)
                train_acc = float(np.mean(y_pred == y))
                errors = [(name, true, pred) for pred, true, name in zip(y_pred, y, file_names) if pred != true]
                
                if verbose:
                    print(f"  训练集准确率: {train_acc:.4f}")
                    print(f"  错误分类数量: {len(errors)}")
                
                results.append({
                    'name': 'PyTorch_CNN',
                    'model': pytorch_model,
                    'cv_results': {'accuracy_mean': train_acc, 'accuracy_std': 0, 
                                  'f1_mean': 0, 'f1_std': 0, 'roc_auc_mean': 0, 'roc_auc_std': 0},
                    'train_acc': train_acc,
                    'errors': errors,
                    'n_errors': len(errors)
                })
        except Exception as e:
            if verbose:
                print(f"  PyTorch模型训练失败: {e}")
    
    # 尝试模型集成
    if len([r for r in results if 'PyTorch' not in r['name']]) >= 2:
        if verbose:
            print(f"\n{'='*60}")
            print("尝试模型集成 (Voting)")
            print(f"{'='*60}")
        
        try:
            # 只使用sklearn兼容的模型进行集成
            sklearn_results = [r for r in results if 'PyTorch' not in r['name']]
            estimators = [(r['name'], r['model']) for r in sklearn_results[:3]]  # 最多3个模型
            voting_clf = VotingClassifier(estimators=estimators, voting='soft')
            
            cv_results = evaluate_model_cv(voting_clf, X, y, cv=5, verbose=verbose)
            
            voting_clf.fit(X, y)
            y_pred = voting_clf.predict(X)
            train_acc = float(np.mean(y_pred == y))
            errors = [(name, true, pred) for pred, true, name in zip(y_pred, y, file_names) if pred != true]
            
            if verbose:
                print(f"  训练集准确率: {train_acc:.4f}")
                print(f"  错误分类数量: {len(errors)}")
            
            results.append({
                'name': 'Ensemble_Voting',
                'model': voting_clf,
                'cv_results': cv_results,
                'train_acc': train_acc,
                'errors': errors,
                'n_errors': len(errors)
            })
            
            # 检查集成模型是否更好
            if len(errors) == 0 and cv_results['roc_auc_mean'] > best_cv_score:
                if verbose:
                    print("  ✓ 集成模型效果更好，使用集成模型")
                best_model = voting_clf
                best_name = "Ensemble_Voting"
                best_cv_score = cv_results['roc_auc_mean']
        except Exception as e:
            if verbose:
                print(f"  集成模型训练失败: {e}")
    
    return best_model, best_name, results


def print_final_report(best_model, best_name, results, X, y, file_names):
    """打印最终报告"""
    print(f"\n{'='*80}")
    print(f"最终选择模型: {best_name}")
    print(f"{'='*80}")
    
    # 最终评估
    y_pred = best_model.predict(X)
    
    print("\n分类报告:")
    print(classification_report(y, y_pred, target_names=['无杂音', '有杂音']))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    # 详细检查错误分类
    errors = [(name, true, pred) for pred, true, name in zip(y_pred, y, file_names) if pred != true]
    if errors:
        print(f"\n错误分类的文件 ({len(errors)} 个):")
        false_negatives = [e for e in errors if e[1] == 1 and e[2] == 0]
        false_positives = [e for e in errors if e[1] == 0 and e[2] == 1]
        
        if false_negatives:
            print(f"\n  ❌ 杂音文件未被识别 ({len(false_negatives)} 个):")
            for name, true_label, pred_label in false_negatives[:10]:
                print(f"    {name}")
            if len(false_negatives) > 10:
                print(f"    ... 还有 {len(false_negatives) - 10} 个未显示")
        
        if false_positives:
            print(f"\n  ⚠️  正常文件被误识别为杂音 ({len(false_positives)} 个):")
            for name, true_label, pred_label in false_positives[:10]:
                print(f"    {name}")
            if len(false_positives) > 10:
                print(f"    ... 还有 {len(false_positives) - 10} 个未显示")
    else:
        print("\n✓ 所有样本分类正确！")
        print("✓ 所有杂音文件都被正确识别")
        print("✓ 所有正常文件都没有被误识别")
    
    # 模型比较总结
    print("\n" + "=" * 80)
    print("模型比较总结")
    print("=" * 80)
    for r in results:
        cv = r['cv_results']
        print(f"\n{r['name']}:")
        print(f"  - 交叉验证AUC-ROC: {cv['roc_auc_mean']:.4f} ± {cv['roc_auc_std']:.4f}")
        print(f"  - 交叉验证F1: {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}")
        print(f"  - 训练集准确率: {r['train_acc']:.4f}")
        print(f"  - 错误分类数: {r['n_errors']}")


def save_model_and_info(best_model, best_name, results, X, y, matched_noise_files, noise_files):
    """保存模型和特征信息"""
    # 保存模型
    model_path = Path("./noise_detector_model_v5.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 获取最佳结果
    best_result = next((r for r in results if r['name'] == best_name), results[0])
    
    # 保存特征信息
    feature_info = {
        'feature_names': FEATURE_NAMES_V5,
        'n_features': len(FEATURE_NAMES_V5),
        'version': 'v5',
        'model_type': best_name,
        'cv_auc_roc': float(best_result['cv_results']['roc_auc_mean']),
        'cv_f1': float(best_result['cv_results']['f1_mean']),
        'train_accuracy': float(best_result['train_acc']),
        'n_noise_files': int(np.sum(y)),
        'n_normal_files': int(np.sum(1-y)),
        'matched_noise_files': len(matched_noise_files),
        'total_noise_files_in_txt': len(noise_files),
        'n_errors': best_result['n_errors'],
        'improvements': [
            'XGBoost/LightGBM替代RandomForest（如已安装）',
            '交叉验证评估真实泛化能力',
            '数据增强提升鲁棒性',
            '新增梅尔频谱图特征',
            '新增MFCC差分特征',
            '新增峰值检测特征',
            '模型集成(Voting)',
            'PyTorch深度学习模型（如已安装）',
        ]
    }
    
    with open('./noise_detector_features_v5.json', 'w') as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False)
    print(f"特征信息已保存到: noise_detector_features_v5.json")
    
    return model_path


if __name__ == "__main__":
    print("=" * 80)
    print("杂音检测模型 V5 训练")
    print("=" * 80)
    
    # 检查可用的高级库
    print("\n可用库检查:")
    print(f"  - XGBoost: {'✓' if XGBOOST_AVAILABLE else '✗ (pip install xgboost)'}")
    print(f"  - LightGBM: {'✓' if LIGHTGBM_AVAILABLE else '✗ (pip install lightgbm)'}")
    print(f"  - PyTorch: {'✓' if PYTORCH_AVAILABLE else '✗ (pip install torch)'}")
    print(f"  - SciPy: {'✓' if SCIPY_AVAILABLE else '✗ (pip install scipy)'}")
    print(f"  - tqdm (进度条): {'✓' if TQDM_AVAILABLE else '✗ (pip install tqdm)'}")
    
    # 检查GPU加速设备
    if PYTORCH_AVAILABLE:
        device = get_torch_device()
        if device.type == 'cuda':
            print(f"  - GPU加速: ✓ CUDA ({torch.cuda.get_device_name(0)})")
        elif device.type == 'mps':
            print(f"  - GPU加速: ✓ MPS (Apple Silicon)")
        else:
            print(f"  - GPU加速: ✗ (使用CPU)")
    
    # 读取有杂音的文件列表
    noise_txt = Path("./audio_noise_case/noise.txt")
    noise_files = set()
    if noise_txt.exists():
        with open(noise_txt, 'r') as f:
            noise_files = {line.strip() for line in f if line.strip()}
    
    print(f"\nnoise.txt中标注的杂音文件数量: {len(noise_files)}")
    
    # 准备数据（启用数据增强）
    data_dir = Path("./audio_noise_case")
    print("\n正在提取特征（包含数据增强）...")
    X, y, file_names, matched_noise_files = prepare_training_data(
        data_dir, noise_files, use_augmentation=True, verbose=True
    )
    
    print(f"\n数据统计:")
    print(f"  总样本数: {len(X)} (含增强)")
    print(f"  特征数量: {X.shape[1]}")
    print(f"  有杂音样本: {np.sum(y)}")
    print(f"  无杂音样本: {np.sum(1-y)}")
    print(f"  成功匹配的杂音文件: {len(matched_noise_files)}")
    
    # 确保所有noise.txt中的文件都被正确标记
    if len(matched_noise_files) < len(noise_files):
        print(f"\n警告: 只有 {len(matched_noise_files)}/{len(noise_files)} 个杂音文件被匹配到")
    
    print("\n" + "=" * 80)
    print("模型训练与交叉验证评估")
    print("=" * 80)
    
    # 训练和评估模型
    best_model, best_name, results = train_and_evaluate_models(X, y, file_names, verbose=True)
    
    # 打印最终报告
    print_final_report(best_model, best_name, results, X, y, file_names)
    
    # 保存模型和信息
    save_model_and_info(best_model, best_name, results, X, y, matched_noise_files, noise_files)
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)