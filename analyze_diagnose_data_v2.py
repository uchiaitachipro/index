#!/usr/bin/env python3
"""
分析诊断数据压缩包，定位杂音产生的阶段（修正版）

说明：
- diagnose_data 里的所有数据都是有杂音的音频
- 对比 audio_noise_case 中不在 noise.txt 名单里的正常音频
"""

import os
import sys
import zipfile
import tempfile
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torchaudio
import librosa

from audio_detect_noise_v2 import detect_chi_noise_core


def load_normal_audio_list(audio_noise_case_dir: str) -> List[str]:
    """加载正常音频（不在 noise.txt 中的音频）列表"""
    noise_txt = Path(audio_noise_case_dir) / "noise.txt"
    
    # 读取杂音文件名单
    noise_names = set()
    if noise_txt.exists():
        with open(noise_txt, 'r') as f:
            for line in f:
                name = line.strip()
                if name:
                    # 去掉 .wav 后缀（如果有的话）
                    if name.endswith('.wav'):
                        name = name[:-4]
                    noise_names.add(name)
    
    # 获取所有 wav 文件
    all_wavs = list(Path(audio_noise_case_dir).glob("*.wav"))
    
    # 找出正常音频
    normal_wavs = []
    for wav in all_wavs:
        name = wav.stem  # 不含后缀的文件名
        if name not in noise_names:
            normal_wavs.append(str(wav))
    
    return normal_wavs


def analyze_wav_features(wav_path: str) -> Dict:
    """分析音频的尾部特征"""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    
    # 使用杂音检测
    noise_result = detect_chi_noise_core(y, sr)
    
    # 尾部200ms分析
    tail_samples = int(sr * 0.2)
    tail = y[-tail_samples:] if len(y) > tail_samples else y
    ref = y[:-tail_samples] if len(y) > tail_samples else y[:len(y)//2]
    
    # STFT 分析
    n_fft = 1024 if sr < 32000 else 2048
    hop = n_fft // 4
    
    S_tail = librosa.stft(tail, n_fft=n_fft, hop_length=hop)
    S_ref = librosa.stft(ref, n_fft=n_fft, hop_length=hop)
    
    mag_tail = np.abs(S_tail)
    mag_ref = np.abs(S_ref)
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # 高频能量分析 (5-15kHz)
    hi_mask = (freqs >= 5000) & (freqs <= 15000)
    hi_e_tail = np.mean(mag_tail[hi_mask] ** 2) + 1e-12
    hi_e_ref = np.mean(mag_ref[hi_mask] ** 2) + 1e-12
    hi_ratio_db = 10 * np.log10(hi_e_tail / hi_e_ref)
    
    return {
        "ratio_db_peak": noise_result["ratio_db_peak"],
        "flux_db_peak": noise_result["flux_db_peak"],
        "rolloff_ratio": noise_result["rolloff_ratio"],
        "flat_db": noise_result["flat_db"],
        "crest_db": noise_result["crest_db"],
        "hi_ratio_db": hi_ratio_db,
        "score": noise_result["score"],
        "has_chi": noise_result["has_chi"],
    }


def analyze_codes(codes_path: str) -> Dict:
    """分析 GPT 生成的 codes"""
    codes = torch.load(codes_path, weights_only=True, map_location='cpu')
    codes_np = codes.cpu().numpy().squeeze()
    
    silent_token = 52
    
    valid_len = len(codes_np)
    
    # 分析最后20个 token
    last_20 = codes_np[-20:] if len(codes_np) >= 20 else codes_np
    
    # 统计
    silent_count = int(np.sum(codes_np == silent_token))
    last_silent_count = int(np.sum(last_20 == silent_token))
    
    # 检查重复
    unique_last = len(np.unique(last_20))
    repetition_ratio = 1 - (unique_last / len(last_20)) if len(last_20) > 0 else 0
    
    return {
        "有效长度": valid_len,
        "尾部静音token数": last_silent_count,
        "token重复率": repetition_ratio,
        "最后20个token": last_20.tolist(),
    }


def analyze_mel(mel_path: str) -> Dict:
    """分析 CFM 输出的 mel 频谱"""
    mel = torch.load(mel_path, weights_only=True, map_location='cpu')
    mel_np = mel.cpu().numpy().squeeze()
    
    n_mels, n_frames = mel_np.shape
    
    # 分析最后50帧
    last_frames = min(50, n_frames // 4)
    if n_frames > last_frames:
        tail_mel = mel_np[:, -last_frames:]
        ref_mel = mel_np[:, :-last_frames]
    else:
        tail_mel = mel_np
        ref_mel = mel_np[:, :n_frames//2] if n_frames > 1 else mel_np
    
    # 能量计算
    tail_energy = np.mean(tail_mel ** 2)
    ref_energy = np.mean(ref_mel ** 2) if ref_mel.size > 0 else tail_energy
    
    # 高频部分
    hi_bins = min(40, n_mels // 2)
    tail_hi_energy = np.mean(tail_mel[-hi_bins:, :] ** 2)
    ref_hi_energy = np.mean(ref_mel[-hi_bins:, :] ** 2) if ref_mel.size > 0 else tail_hi_energy
    
    energy_ratio_db = 10 * np.log10(tail_energy / (ref_energy + 1e-12))
    hi_energy_ratio_db = 10 * np.log10(tail_hi_energy / (ref_hi_energy + 1e-12))
    
    # 能量跳变
    frame_energy = np.mean(mel_np ** 2, axis=0)
    if len(frame_energy) > 5:
        energy_diff = np.diff(frame_energy[-20:]) if len(frame_energy) >= 20 else np.diff(frame_energy)
        max_energy_jump = np.max(np.abs(energy_diff)) if len(energy_diff) > 0 else 0
    else:
        max_energy_jump = 0
    
    return {
        "尾部能量比_dB": float(energy_ratio_db),
        "尾部高频能量比_dB": float(hi_energy_ratio_db),
        "最大能量跳变": float(max_energy_jump),
    }


def analyze_zip_file(zip_path: str) -> Dict:
    """分析单个诊断 zip 文件（已知都是有杂音的）"""
    results = {
        "文件名": os.path.basename(zip_path),
        "GPT分析": {},
        "CFM分析": {},
        "音频分析": {},
        "杂音来源判断": "",
    }
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmp_dir)
        except Exception as e:
            results["错误"] = f"解压失败: {str(e)}"
            return results
        
        tmp_path = Path(tmp_dir)
        
        # 读取杂音检测结果
        noise_json_path = tmp_path / "noise_detection.json"
        if noise_json_path.exists():
            with open(noise_json_path, 'r', encoding='utf-8') as f:
                noise_info = json.load(f)
                results["文本"] = noise_info.get("text", "")[:50]
                results["说话人"] = noise_info.get("speaker", "")
        
        # 分析 codes
        codes_files = list(tmp_path.glob("seg*_codes.pt"))
        for codes_file in sorted(codes_files):
            try:
                results["GPT分析"][codes_file.stem] = analyze_codes(str(codes_file))
            except Exception as e:
                results["GPT分析"][codes_file.stem] = {"错误": str(e)}
        
        # 分析 CFM mel
        cfm_files = list(tmp_path.glob("seg*_CFM_output.pt"))
        for cfm_file in sorted(cfm_files):
            try:
                results["CFM分析"][cfm_file.stem] = analyze_mel(str(cfm_file))
            except Exception as e:
                results["CFM分析"][cfm_file.stem] = {"错误": str(e)}
        
        # 分析音频
        wav_files = list(tmp_path.glob("seg*_output.wav"))
        for wav_file in sorted(wav_files):
            try:
                results["音频分析"][wav_file.stem] = analyze_wav_features(str(wav_file))
            except Exception as e:
                results["音频分析"][wav_file.stem] = {"错误": str(e)}
        
        # 分析最终音频
        final_wav = tmp_path / "generated_audio.wav"
        if final_wav.exists():
            try:
                results["音频分析"]["final"] = analyze_wav_features(str(final_wav))
            except Exception as e:
                results["音频分析"]["final"] = {"错误": str(e)}
    
    return results


def analyze_normal_audio_sample(audio_path: str) -> Dict:
    """分析正常音频样本"""
    return analyze_wav_features(audio_path)


def compare_and_diagnose(diagnose_dir: str, audio_noise_case_dir: str, output_file: str = None):
    """对比分析杂音和正常音频"""
    
    print("\n" + "=" * 80)
    print("                      杂音阶段诊断分析")
    print("=" * 80)
    
    # 加载正常音频列表
    normal_wavs = load_normal_audio_list(audio_noise_case_dir)
    print(f"\n找到 {len(normal_wavs)} 个正常音频文件")
    
    # 采样分析正常音频（取前100个）
    normal_features = []
    sample_size = min(100, len(normal_wavs))
    print(f"采样分析 {sample_size} 个正常音频的特征...")
    
    for i, wav_path in enumerate(normal_wavs[:sample_size]):
        try:
            feat = analyze_normal_audio_sample(wav_path)
            normal_features.append(feat)
        except Exception as e:
            pass
    
    if normal_features:
        # 计算正常音频的特征统计
        normal_stats = {
            "ratio_db_peak": {
                "mean": np.mean([f["ratio_db_peak"] for f in normal_features]),
                "std": np.std([f["ratio_db_peak"] for f in normal_features]),
                "max": np.max([f["ratio_db_peak"] for f in normal_features]),
            },
            "flux_db_peak": {
                "mean": np.mean([f["flux_db_peak"] for f in normal_features]),
                "std": np.std([f["flux_db_peak"] for f in normal_features]),
                "max": np.max([f["flux_db_peak"] for f in normal_features]),
            },
            "rolloff_ratio": {
                "mean": np.mean([f["rolloff_ratio"] for f in normal_features]),
                "std": np.std([f["rolloff_ratio"] for f in normal_features]),
            },
            "flat_db": {
                "mean": np.mean([f["flat_db"] for f in normal_features]),
                "std": np.std([f["flat_db"] for f in normal_features]),
            },
            "hi_ratio_db": {
                "mean": np.mean([f["hi_ratio_db"] for f in normal_features]),
                "std": np.std([f["hi_ratio_db"] for f in normal_features]),
                "max": np.max([f["hi_ratio_db"] for f in normal_features]),
            },
        }
        
        print(f"\n📊 正常音频尾部特征统计:")
        print(f"   ratio_db_peak: 均值={normal_stats['ratio_db_peak']['mean']:.1f}, "
              f"标准差={normal_stats['ratio_db_peak']['std']:.1f}, "
              f"最大={normal_stats['ratio_db_peak']['max']:.1f}")
        print(f"   flux_db_peak:  均值={normal_stats['flux_db_peak']['mean']:.1f}, "
              f"标准差={normal_stats['flux_db_peak']['std']:.1f}")
        print(f"   hi_ratio_db:   均值={normal_stats['hi_ratio_db']['mean']:.1f}, "
              f"标准差={normal_stats['hi_ratio_db']['std']:.1f}, "
              f"最大={normal_stats['hi_ratio_db']['max']:.1f}")
    else:
        normal_stats = None
    
    # 分析杂音诊断数据
    diagnose_path = Path(diagnose_dir)
    zip_files = sorted(diagnose_path.glob("*.zip"))
    
    print(f"\n分析 {len(zip_files)} 个杂音诊断文件...")
    print("-" * 80)
    
    all_results = []
    cfm_hi_ratios = []
    cfm_jumps = []
    audio_ratios = []
    audio_fluxes = []
    
    for i, zip_file in enumerate(zip_files, 1):
        print(f"\n[{i}/{len(zip_files)}] {zip_file.name}")
        result = analyze_zip_file(str(zip_file))
        all_results.append(result)
        
        # 收集 CFM 特征
        for key, val in result.get("CFM分析", {}).items():
            if "错误" not in val:
                cfm_hi_ratios.append(val.get("尾部高频能量比_dB", 0))
                cfm_jumps.append(val.get("最大能量跳变", 0))
        
        # 收集音频特征
        for key, val in result.get("音频分析", {}).items():
            if "错误" not in val:
                audio_ratios.append(val.get("ratio_db_peak", 0))
                audio_fluxes.append(val.get("flux_db_peak", 0))
        
        # 打印关键信息
        text = result.get("文本", "")[:30]
        print(f"   文本: {text}...")
        
        # CFM 分析
        for key, val in result.get("CFM分析", {}).items():
            if "错误" not in val:
                print(f"   CFM {key}: 高频比={val.get('尾部高频能量比_dB', 0):.1f}dB, "
                      f"跳变={val.get('最大能量跳变', 0):.1f}")
        
        # 音频分析
        for key, val in result.get("音频分析", {}).items():
            if "错误" not in val:
                print(f"   音频 {key}: ratio={val.get('ratio_db_peak', 0):.1f}dB, "
                      f"flux={val.get('flux_db_peak', 0):.1f}dB, "
                      f"rolloff={val.get('rolloff_ratio', 0):.2f}")
    
    # 汇总分析
    print("\n" + "=" * 80)
    print("                      汇总分析")
    print("=" * 80)
    
    print(f"\n📊 杂音音频特征统计 ({len(zip_files)} 个样本):")
    if cfm_hi_ratios:
        print(f"   CFM尾部高频能量比: 均值={np.mean(cfm_hi_ratios):.1f}dB, "
              f"标准差={np.std(cfm_hi_ratios):.1f}, "
              f"范围=[{np.min(cfm_hi_ratios):.1f}, {np.max(cfm_hi_ratios):.1f}]")
    if cfm_jumps:
        print(f"   CFM能量跳变:       均值={np.mean(cfm_jumps):.1f}, "
              f"标准差={np.std(cfm_jumps):.1f}, "
              f"范围=[{np.min(cfm_jumps):.1f}, {np.max(cfm_jumps):.1f}]")
    if audio_ratios:
        print(f"   音频ratio_db_peak: 均值={np.mean(audio_ratios):.1f}dB, "
              f"标准差={np.std(audio_ratios):.1f}, "
              f"范围=[{np.min(audio_ratios):.1f}, {np.max(audio_ratios):.1f}]")
    if audio_fluxes:
        print(f"   音频flux_db_peak:  均值={np.mean(audio_fluxes):.1f}dB, "
              f"标准差={np.std(audio_fluxes):.1f}")
    
    # 对比分析
    if normal_stats:
        print(f"\n📈 杂音 vs 正常音频对比:")
        
        # ratio_db_peak 对比
        noise_ratio_mean = np.mean(audio_ratios) if audio_ratios else 0
        normal_ratio_mean = normal_stats["ratio_db_peak"]["mean"]
        ratio_diff = noise_ratio_mean - normal_ratio_mean
        print(f"   ratio_db_peak 差异: 杂音={noise_ratio_mean:.1f}dB vs 正常={normal_ratio_mean:.1f}dB "
              f"(差异={ratio_diff:+.1f}dB)")
        
        # CFM 高频能量比
        cfm_hi_mean = np.mean(cfm_hi_ratios) if cfm_hi_ratios else 0
        normal_hi_mean = normal_stats["hi_ratio_db"]["mean"]
        hi_diff = cfm_hi_mean - normal_hi_mean
        print(f"   CFM高频能量比 差异: 杂音={cfm_hi_mean:.1f}dB vs 正常={normal_hi_mean:.1f}dB "
              f"(差异={hi_diff:+.1f}dB)")
    
    # 诊断结论
    print("\n" + "=" * 80)
    print("                      诊断结论")
    print("=" * 80)
    
    # 分析各阶段问题
    cfm_issue_count = sum(1 for r in cfm_hi_ratios if r > 3)  # CFM 高频异常
    cfm_jump_issue_count = sum(1 for j in cfm_jumps if j > 30)  # CFM 跳变异常
    bigvgan_issue_count = sum(1 for r in audio_ratios if r > 30)  # BigVGAN 放大
    
    print(f"\n🔍 问题分布分析:")
    print(f"   CFM 尾部高频能量异常 (>3dB): {cfm_issue_count}/{len(cfm_hi_ratios)} ({cfm_issue_count/max(1,len(cfm_hi_ratios))*100:.0f}%)")
    print(f"   CFM 能量跳变异常 (>30):      {cfm_jump_issue_count}/{len(cfm_jumps)} ({cfm_jump_issue_count/max(1,len(cfm_jumps))*100:.0f}%)")
    print(f"   BigVGAN 高频放大 (>30dB):    {bigvgan_issue_count}/{len(audio_ratios)} ({bigvgan_issue_count/max(1,len(audio_ratios))*100:.0f}%)")
    
    # 判断主要来源
    print(f"\n🎯 杂音主要来源判断:")
    
    if normal_stats:
        # 比较 CFM 输出和最终音频的高频差异
        cfm_avg = np.mean(cfm_hi_ratios) if cfm_hi_ratios else 0
        audio_avg = np.mean(audio_ratios) if audio_ratios else 0
        
        if cfm_avg > 3:
            print(f"   → CFM 阶段: 尾部高频能量明显偏高 ({cfm_avg:.1f}dB)")
            print(f"     说明: CFM 扩散模型在收敛时产生了过多高频分量")
        
        if audio_avg > normal_stats["ratio_db_peak"]["mean"] + 2 * normal_stats["ratio_db_peak"]["std"]:
            amplification = audio_avg - cfm_avg
            if amplification > 15:
                print(f"   → BigVGAN 阶段: 显著放大了高频分量 (放大约 {amplification:.0f}dB)")
                print(f"     说明: BigVGAN 将 CFM 输出的轻微高频异常放大成明显杂音")
            else:
                print(f"   → BigVGAN 阶段: 边界处理产生高频噪声")
    
    print(f"\n💡 改进建议:")
    print(f"   1. 【最直接】对 BigVGAN 输出进行尾部淡出处理 (fade-out 最后20-50ms)")
    print(f"   2. 【从源头】增加 CFM diffusion_steps 提高收敛质量")
    print(f"   3. 【补充】在 mel 频谱阶段对尾部高频进行衰减")
    print(f"   4. 【可选】检查 GPT codes 是否在末尾自然过渡")
    
    # 保存结果
    if output_file:
        output_data = {
            "正常音频统计": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in normal_stats.items()} if normal_stats else None,
            "杂音数据统计": {
                "CFM高频能量比": {
                    "mean": float(np.mean(cfm_hi_ratios)) if cfm_hi_ratios else 0,
                    "std": float(np.std(cfm_hi_ratios)) if cfm_hi_ratios else 0,
                    "min": float(np.min(cfm_hi_ratios)) if cfm_hi_ratios else 0,
                    "max": float(np.max(cfm_hi_ratios)) if cfm_hi_ratios else 0,
                },
                "音频ratio_db_peak": {
                    "mean": float(np.mean(audio_ratios)) if audio_ratios else 0,
                    "std": float(np.std(audio_ratios)) if audio_ratios else 0,
                    "min": float(np.min(audio_ratios)) if audio_ratios else 0,
                    "max": float(np.max(audio_ratios)) if audio_ratios else 0,
                },
            },
            "详细结果": all_results,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n详细结果已保存到: {output_file}")
    
    print()


if __name__ == "__main__":
    diagnose_dir = "./diagnose_data"
    audio_noise_case_dir = "./audio_noise_case"
    output_file = "./diagnose_data_analysis_v2.json"
    
    compare_and_diagnose(diagnose_dir, audio_noise_case_dir, output_file)

