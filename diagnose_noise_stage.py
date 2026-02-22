"""
TTS杂音阶段诊断工具

用于定位TTS合成过程中哪个阶段产生了杂音。
通过保存各阶段的中间结果来分析问题。
"""

import os
import torch
import numpy as np
import librosa
import torchaudio
from pathlib import Path
import json
from datetime import datetime

# 导入杂音检测
from audio_detect_noise_v2 import detect_chi_noise, detect_chi_noise_core


def analyze_mel_spectrogram(mel: torch.Tensor, name: str = "mel"):
    """分析Mel频谱的统计特征"""
    mel_np = mel.cpu().numpy().squeeze()
    
    # 分析最后N帧
    last_frames = 50  # 约200ms
    if mel_np.shape[-1] > last_frames:
        tail_mel = mel_np[:, -last_frames:]
        ref_mel = mel_np[:, :-last_frames]
    else:
        tail_mel = mel_np
        ref_mel = mel_np
    
    stats = {
        "name": name,
        "shape": mel_np.shape,
        "全局均值": float(np.mean(mel_np)),
        "全局标准差": float(np.std(mel_np)),
        "尾部均值": float(np.mean(tail_mel)),
        "尾部标准差": float(np.std(tail_mel)),
        "参考均值": float(np.mean(ref_mel)),
        "参考标准差": float(np.std(ref_mel)),
        "尾部最大值": float(np.max(tail_mel)),
        "尾部最小值": float(np.min(tail_mel)),
        "尾部高频(后40bin)能量": float(np.mean(tail_mel[-40:, :]**2)),
        "参考高频(后40bin)能量": float(np.mean(ref_mel[-40:, :]**2) if ref_mel.shape[-1] > 0 else 0),
    }
    
    # 计算尾部与参考的能量比
    if stats["参考高频(后40bin)能量"] > 1e-12:
        stats["高频能量比(尾/参考)_dB"] = float(10 * np.log10(
            stats["尾部高频(后40bin)能量"] / stats["参考高频(后40bin)能量"]
        ))
    else:
        stats["高频能量比(尾/参考)_dB"] = 0.0
    
    return stats


def analyze_codes(codes: torch.Tensor, stop_token: int = 8193):
    """分析GPT生成的codes"""
    codes_np = codes.cpu().numpy().squeeze()
    
    # 检查stop token
    has_stop = stop_token in codes_np
    stop_position = np.where(codes_np == stop_token)[0]
    
    # 分析最后20个token
    last_20 = codes_np[-20:] if len(codes_np) >= 20 else codes_np
    
    stats = {
        "总长度": len(codes_np),
        "包含stop_token": bool(has_stop),
        "stop_token位置": stop_position.tolist() if has_stop else [],
        "最后20个token": last_20.tolist(),
        "token_52(静音)数量": int(np.sum(codes_np == 52)),
        "最后20个中token_52数量": int(np.sum(last_20 == 52)),
        "token值范围": [int(np.min(codes_np)), int(np.max(codes_np))],
    }
    
    return stats


def analyze_wav(wav: torch.Tensor, sr: int = 22050, name: str = "wav"):
    """分析生成的音频波形"""
    wav_np = wav.cpu().numpy().squeeze()
    
    # 使用杂音检测核心函数
    noise_result = detect_chi_noise_core(wav_np, sr)
    
    # 额外分析
    tail_samples = int(sr * 0.2)  # 最后200ms
    if len(wav_np) > tail_samples:
        tail = wav_np[-tail_samples:]
        ref = wav_np[:-tail_samples]
    else:
        tail = wav_np
        ref = wav_np
    
    stats = {
        "name": name,
        "采样率": sr,
        "总样本数": len(wav_np),
        "时长_秒": len(wav_np) / sr,
        "全局RMS": float(np.sqrt(np.mean(wav_np**2))),
        "尾部RMS": float(np.sqrt(np.mean(tail**2))),
        "参考RMS": float(np.sqrt(np.mean(ref**2))) if len(ref) > 0 else 0,
        "杂音检测结果": noise_result,
    }
    
    return stats


class TTS_Diagnostic_Hook:
    """
    TTS诊断钩子类
    用于拦截和保存各阶段的中间结果
    """
    
    def __init__(self, output_dir: str = "./diagnose_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
        self.current_segment = 0
        
    def save_codes(self, codes: torch.Tensor, segment_idx: int):
        """保存GPT生成的codes"""
        stats = analyze_codes(codes)
        stats["stage"] = "GPT_codes"
        stats["segment"] = segment_idx
        self.results.append(stats)
        
        # 保存原始数据
        torch.save(codes, self.output_dir / f"seg{segment_idx}_codes.pt")
        
        return stats
    
    def save_mel(self, mel: torch.Tensor, stage: str, segment_idx: int):
        """保存Mel频谱"""
        stats = analyze_mel_spectrogram(mel, stage)
        stats["stage"] = stage
        stats["segment"] = segment_idx
        self.results.append(stats)
        
        # 保存原始数据
        torch.save(mel, self.output_dir / f"seg{segment_idx}_{stage}.pt")
        
        return stats
    
    def save_wav(self, wav: torch.Tensor, sr: int, segment_idx: int, 
                 save_audio: bool = True):
        """保存音频波形"""
        stats = analyze_wav(wav, sr, f"segment_{segment_idx}")
        stats["stage"] = "BigVGAN_output"
        stats["segment"] = segment_idx
        self.results.append(stats)
        
        # 保存音频文件
        if save_audio:
            wav_path = self.output_dir / f"seg{segment_idx}_output.wav"
            torchaudio.save(str(wav_path), wav.cpu().type(torch.int16), sr)
        
        return stats
    
    def generate_report(self):
        """生成诊断报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_segments": self.current_segment,
            "stages": self.results,
            "summary": self._generate_summary()
        }
        
        # 保存报告
        report_path = self.output_dir / "diagnose_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return report
    
    def _generate_summary(self):
        """生成摘要"""
        noise_segments = []
        for r in self.results:
            if r.get("stage") == "BigVGAN_output":
                noise_result = r.get("杂音检测结果", {})
                if noise_result.get("has_chi", False):
                    noise_segments.append({
                        "segment": r.get("segment"),
                        "score": noise_result.get("score", 0),
                        "ratio_db_peak": noise_result.get("ratio_db_peak", 0),
                    })
        
        return {
            "检测到杂音的片段数": len(noise_segments),
            "杂音片段详情": noise_segments
        }


def diagnose_single_audio(audio_path: str, verbose: bool = True):
    """
    诊断单个有杂音的音频文件
    分析其频谱特征来推断杂音来源
    """
    print(f"\n{'='*60}")
    print(f"诊断音频: {audio_path}")
    print(f"{'='*60}")
    
    # 加载音频
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # 杂音检测
    result = detect_chi_noise(audio_path)
    
    print(f"\n📊 杂音检测结果:")
    print(f"  是否有杂音: {'是 ⚠️' if result['has_chi'] else '否 ✓'}")
    print(f"  检测方法: {result.get('method', 'unknown')}")
    print(f"  评分: {result['score']:.2f}")
    print(f"  高频增益峰值: {result['ratio_db_peak']:.2f} dB")
    print(f"  瞬态变化峰值: {result['flux_db_peak']:.2f} dB")
    print(f"  距离结尾: {result.get('distance_from_end_ms', 0):.2f} ms")
    
    # 频谱分析
    tail_ms = 200
    N_tail = int(sr * tail_ms / 1000)
    tail = y[-N_tail:]
    
    # STFT分析
    n_fft = 2048
    hop = n_fft // 4
    S = librosa.stft(tail, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # 分频段分析
    bands = [
        ("低频 (0-500Hz)", 0, 500),
        ("中频 (500-2kHz)", 500, 2000),
        ("中高频 (2-5kHz)", 2000, 5000),
        ("高频 (5-10kHz)", 5000, 10000),
        ("超高频 (10-15kHz)", 10000, 15000),
    ]
    
    print(f"\n📈 尾部频谱能量分布 (最后{tail_ms}ms):")
    for name, low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        energy = np.mean(mag[mask]**2) if np.any(mask) else 0
        energy_db = 10 * np.log10(energy + 1e-12)
        print(f"  {name}: {energy_db:.2f} dB")
    
    # 推断杂音来源
    print(f"\n🔍 杂音来源推断:")
    
    ratio_db = result['ratio_db_peak']
    flux_db = result['flux_db_peak']
    distance_ms = result.get('distance_from_end_ms', 0)
    
    if result['has_chi']:
        if ratio_db > 15 and flux_db > 4:
            print("  → 可能来源: CFM扩散模型收敛不完全")
            print("    特征: 高频能量突增明显，瞬态变化大")
            print("    建议: 增加diffusion_steps或调整inference_cfg_rate")
        elif ratio_db > 10 and distance_ms < 50:
            print("  → 可能来源: BigVGAN边界处理")
            print("    特征: 杂音出现在极末端")
            print("    建议: 对输出音频末端进行淡出处理")
        elif ratio_db > 8 and result.get('crest_db', 0) > 15:
            print("  → 可能来源: GPT生成异常token")
            print("    特征: 波峰因子高，可能有尖锐脉冲")
            print("    建议: 检查GPT生成的codes是否包含异常值")
        else:
            print("  → 可能来源: Length Regulator长度映射问题")
            print("    特征: 杂音特征不明显但仍被检测到")
            print("    建议: 检查code_lens到target_lengths的映射比例")
    else:
        print("  → 该音频未检测到杂音")
    
    return result


def batch_diagnose(audio_dir: str, pattern: str = "*.wav"):
    """批量诊断目录中的音频"""
    audio_dir = Path(audio_dir)
    files = sorted(audio_dir.glob(pattern))
    
    print(f"\n{'='*60}")
    print(f"批量诊断目录: {audio_dir}")
    print(f"找到 {len(files)} 个音频文件")
    print(f"{'='*60}")
    
    results = []
    noise_count = 0
    
    for f in files:
        try:
            result = detect_chi_noise(str(f))
            result['file'] = f.name
            results.append(result)
            
            if result['has_chi']:
                noise_count += 1
                print(f"⚠️  {f.name}: 有杂音 (score={result['score']:.2f}, "
                      f"ratio={result['ratio_db_peak']:.2f}dB)")
        except Exception as e:
            print(f"✗  {f.name}: 分析失败 - {e}")
    
    print(f"\n📊 统计:")
    print(f"  总文件数: {len(files)}")
    print(f"  有杂音: {noise_count}")
    print(f"  无杂音: {len(files) - noise_count}")
    
    # 分析杂音特征分布
    if noise_count > 0:
        noise_files = [r for r in results if r.get('has_chi')]
        ratio_values = [r['ratio_db_peak'] for r in noise_files]
        flux_values = [r['flux_db_peak'] for r in noise_files]
        
        print(f"\n📈 杂音特征分布:")
        print(f"  ratio_db_peak: 均值={np.mean(ratio_values):.2f}, "
              f"最小={np.min(ratio_values):.2f}, 最大={np.max(ratio_values):.2f}")
        print(f"  flux_db_peak: 均值={np.mean(flux_values):.2f}, "
              f"最小={np.min(flux_values):.2f}, 最大={np.max(flux_values):.2f}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # 默认分析 audio_noise_case 目录
    test_dir = "./audio_noise_case"
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            # 单文件诊断
            diagnose_single_audio(path)
        else:
            # 目录批量诊断
            batch_diagnose(path)
    else:
        # 从 audio_noise_case 随机选择几个文件进行详细诊断
        test_dir = Path(test_dir)
        wav_files = list(test_dir.glob("*.wav"))[:5]  # 取前5个
        
        if wav_files:
            print("🔬 对前5个杂音样本进行详细诊断...\n")
            for f in wav_files:
                diagnose_single_audio(str(f))
        else:
            print(f"在 {test_dir} 中未找到WAV文件")



