#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexTTS2 字幕功能测试脚本
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indextts.infer_v2_subtitle import IndexTTS2SubTitle


def test_subtitle_basic():
    """测试基本的字幕生成功能"""
    print("=" * 80)
    print("测试 1: 基本字幕生成")
    print("=" * 80)
    
    # 初始化模型
    tts = IndexTTS2SubTitle(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_cuda_kernel=False,
        use_fp16=True
    )
    
    # 测试文本
    prompt_wav = "examples/voice_02.wav"
    text = "你好，这是一段测试文本。今天天气很好。"
    
    # 生成
    result = tts.infer(
        spk_audio_prompt=prompt_wav,
        text=text,
        output_path="outputs/test_subtitle.wav",
        verbose=False
    )
    
    # 验证返回值
    assert isinstance(result, dict), "返回值应该是字典"
    assert 'audio_path' in result, "返回值应包含 audio_path"
    assert 'subtitles' in result, "返回值应包含 subtitles"
    assert isinstance(result['subtitles'], list), "subtitles 应该是列表"
    
    print(f"✓ 返回值格式正确")
    print(f"✓ 生成了 {len(result['subtitles'])} 个字幕片段")
    
    # 验证字幕格式
    for i, subtitle in enumerate(result['subtitles']):
        assert 'text' in subtitle, f"字幕 {i} 缺少 text 字段"
        assert 'pronounce_text' in subtitle, f"字幕 {i} 缺少 pronounce_text 字段"
        assert 'time_begin' in subtitle, f"字幕 {i} 缺少 time_begin 字段"
        assert 'time_end' in subtitle, f"字幕 {i} 缺少 time_end 字段"
        assert subtitle['time_end'] > subtitle['time_begin'], f"字幕 {i} 结束时间应大于开始时间"
        
        # 打印字幕信息
        print(f"\n片段 {i+1}:")
        print(f"  文本: {subtitle['text']}")
        print(f"  时间: {subtitle['time_begin']:.2f}ms - {subtitle['time_end']:.2f}ms")
    
    print(f"\n✓ 所有字幕格式正确")
    
    # 验证时间连续性
    for i in range(len(result['subtitles']) - 1):
        current = result['subtitles'][i]
        next_sub = result['subtitles'][i + 1]
        # 下一个片段的开始时间应该在当前片段结束时间之后（考虑静音间隔）
        assert next_sub['time_begin'] >= current['time_end'], \
            f"字幕 {i+1} 开始时间 ({next_sub['time_begin']}) 应在字幕 {i} 结束时间 ({current['time_end']}) 之后"
    
    print(f"✓ 时间戳连续性正确")
    print(f"\n测试 1 通过! ✓")


def test_subtitle_long_text():
    """测试长文本的字幕生成"""
    print("\n" + "=" * 80)
    print("测试 2: 长文本字幕生成")
    print("=" * 80)
    
    tts = IndexTTS2SubTitle(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_cuda_kernel=False,
        use_fp16=True
    )
    
    prompt_wav = "examples/voice_02.wav"
    text = "离开神城已经一个多月了，重新回来，叶凡感受到了一种亲切。这是一个充满激情的巨城。平静归来，他以为没有人注意。可是刚进城不久，就有不少人眼神火热地迎了上来。"
    
    result = tts.infer(
        spk_audio_prompt=prompt_wav,
        text=text,
        output_path="outputs/test_subtitle_long.wav",
        max_text_tokens_per_segment=50,  # 使用较小的值强制分段
        verbose=False
    )
    
    print(f"✓ 长文本生成了 {len(result['subtitles'])} 个字幕片段")
    
    total_duration = result['subtitles'][-1]['time_end'] if result['subtitles'] else 0
    print(f"✓ 总时长: {total_duration/1000:.2f} 秒")
    
    # 打印所有片段
    for i, subtitle in enumerate(result['subtitles']):
        duration = subtitle['time_end'] - subtitle['time_begin']
        print(f"\n片段 {i+1}: ({duration/1000:.2f}s)")
        print(f"  {subtitle['text']}")
    
    print(f"\n测试 2 通过! ✓")


def test_subtitle_without_output_path():
    """测试不保存文件时的字幕生成"""
    print("\n" + "=" * 80)
    print("测试 3: 不保存文件的字幕生成")
    print("=" * 80)
    
    tts = IndexTTS2SubTitle(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_cuda_kernel=False,
        use_fp16=True
    )
    
    prompt_wav = "examples/voice_02.wav"
    text = "这是测试。"
    
    result = tts.infer(
        spk_audio_prompt=prompt_wav,
        text=text,
        output_path=None,  # 不保存文件
        verbose=False
    )
    
    assert isinstance(result, dict), "返回值应该是字典"
    assert 'audio_data' in result, "返回值应包含 audio_data"
    assert 'subtitles' in result, "返回值应包含 subtitles"
    
    print(f"✓ 不保存文件时也能正确返回字幕")
    print(f"✓ 音频数据格式: {type(result['audio_data'])}")
    print(f"✓ 字幕片段数: {len(result['subtitles'])}")
    
    print(f"\n测试 3 通过! ✓")


if __name__ == "__main__":
    try:
        test_subtitle_basic()
        test_subtitle_long_text()
        test_subtitle_without_output_path()
        
        print("\n" + "=" * 80)
        print("所有测试通过! ✓✓✓")
        print("=" * 80)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

