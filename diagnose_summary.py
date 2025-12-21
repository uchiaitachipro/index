#!/usr/bin/env python3
"""
读取诊断分析结果并生成可读的摘要报告
"""

import json
from pathlib import Path


def print_summary(analysis_file: str):
    """打印诊断分析摘要"""
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = data.get("统计", {})
    results = data.get("详细结果", [])
    
    print("\n" + "=" * 80)
    print("                      诊断数据分析报告")
    print("=" * 80)
    
    # 统计概览
    total = sum(stats.values())
    has_noise = total - stats.get("无杂音", 0)
    
    print(f"\n📊 总体统计:")
    print(f"   总样本数: {total}")
    print(f"   有杂音: {has_noise} ({has_noise/total*100:.1f}%)")
    print(f"   无杂音: {stats.get('无杂音', 0)} ({stats.get('无杂音', 0)/total*100:.1f}%)")
    
    # 杂音来源分布
    print(f"\n📈 杂音来源分布:")
    for source, count in sorted(stats.items(), key=lambda x: -x[1]):
        if source != "无杂音" and count > 0:
            pct = count / has_noise * 100 if has_noise > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"   {source}: {count} ({pct:.1f}%) {bar}")
    
    # 详细分析
    print(f"\n📋 各样本详细分析:")
    print("-" * 80)
    
    for r in results:
        filename = r.get("文件名", "unknown")
        source = r.get("杂音来源阶段", "未知")
        conclusion = r.get("诊断结论", "无")
        text = r.get("文本", "")[:30] + "..." if r.get("文本") else ""
        speaker = r.get("说话人", "")
        
        # 获取杂音检测分数
        score = 0
        ratio_db = 0
        for key, val in r.get("各阶段分析", {}).items():
            if val.get("阶段") == "最终输出":
                noise_det = val.get("杂音检测", {})
                score = noise_det.get("score", 0)
                ratio_db = noise_det.get("ratio_db_peak", 0)
                break
        
        # 获取 CFM 指标
        cfm_hi_ratio = 0
        cfm_jump = 0
        for key, val in r.get("各阶段分析", {}).items():
            if val.get("阶段") == "CFM":
                cfm_hi_ratio = val.get("尾部高频能量比_dB", 0)
                cfm_jump = val.get("最大能量跳变", 0)
                break
        
        if source == "无杂音":
            print(f"\n✅ {filename}")
            print(f"   说话人: {speaker} | 文本: {text}")
            print(f"   结论: 无杂音 (score={score:.1f})")
        else:
            print(f"\n⚠️  {filename}")
            print(f"   说话人: {speaker} | 文本: {text}")
            print(f"   杂音来源: {source}")
            print(f"   杂音指标: score={score:.1f}, ratio={ratio_db:.1f}dB")
            print(f"   CFM指标: 高频比={cfm_hi_ratio:.1f}dB, 跳变={cfm_jump:.1f}")
            print(f"   诊断结论: {conclusion}")
    
    # 结论与建议
    print("\n" + "=" * 80)
    print("                      结论与建议")
    print("=" * 80)
    
    print("\n🔍 主要发现:")
    
    # 按来源分类建议
    bigvgan_count = stats.get("BigVGAN放大", 0)
    cfm_hi_count = stats.get("CFM高频溢出", 0)
    cfm_edge_count = stats.get("CFM边界", 0)
    
    if bigvgan_count > 0:
        print(f"\n   1. BigVGAN放大边界效应 ({bigvgan_count}例):")
        print("      - CFM 输出的 mel 频谱尾部虽然看起来正常")
        print("      - 但 BigVGAN 将边界处的轻微高频分量放大成明显杂音")
        print("      - 建议: 对 BigVGAN 输出进行尾部淡出处理 (fade-out)")
    
    if cfm_hi_count > 0:
        print(f"\n   2. CFM高频溢出 ({cfm_hi_count}例):")
        print("      - CFM 扩散模型在生成结尾时产生了过多高频成分")
        print("      - 表现为 rolloff_ratio > 0.75 且频谱较平")
        print("      - 建议: 增加 diffusion_steps 或调整 inference_cfg_rate")
    
    if cfm_edge_count > 0:
        print(f"\n   3. CFM边界跳变 ({cfm_edge_count}例):")
        print("      - CFM 输出的 mel 频谱在结尾处有明显能量跳变")
        print("      - 建议: 检查 length_regulator 的长度映射是否精确")
    
    print("\n💡 通用改进建议:")
    print("   1. 对所有生成音频进行尾部淡出处理 (最后10-50ms)")
    print("   2. 增加 CFM 扩散步数以提高收敛质量")
    print("   3. 考虑在 mel 频谱阶段对尾部进行能量衰减")
    print("   4. 可以尝试在 GPT codes 末尾添加静音 token 来自然过渡")
    print()


if __name__ == "__main__":
    print_summary("./diagnose_data_analysis.json")

