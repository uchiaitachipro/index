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
    
    # 返回特征向量
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
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, file_names, test_size=0.2, random_state=42, stratify=y
    )
    
    # 训练模型
    print("\n训练随机森林模型...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # 处理类别不平衡
    )
    
    model.fit(X_train, y_train)
    
    # 评估模型
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n训练集准确率: {train_score:.4f}")
    print(f"测试集准确率: {test_score:.4f}")
    
    # 详细评估
    y_pred = model.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['无杂音', '有杂音']))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
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
        'ref_med', 'ref_mean', 'ref_std'
    ]
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n特征重要性 (Top 10):")
    for i in range(min(10, len(indices))):
        print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # 保存模型
    model_path = Path("./noise_detector_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 保存特征名称
    feature_info = {
        'feature_names': feature_names,
        'n_features': len(feature_names)
    }
    with open('./noise_detector_features.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"特征信息已保存到: noise_detector_features.json")

