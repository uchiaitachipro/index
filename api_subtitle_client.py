#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexTTS2 API 客户端示例 - 字幕功能演示

演示如何调用 TTS API 并获取字幕信息
"""

import requests
import json
import base64
import os


def call_tts_api_with_subtitle(
    text: str,
    spk_audio_path: str,
    api_url: str = "http://localhost:6006/tts_url",
    save_audio_path: str = None,
    save_subtitle_path: str = None,
    **kwargs
):
    """
    调用 TTS API 并获取字幕
    
    参数:
        text: 要合成的文本
        spk_audio_path: 说话人参考音频路径
        api_url: API 服务器地址
        save_audio_path: 保存音频文件的路径（可选）
        save_subtitle_path: 保存字幕文件的路径（可选）
        **kwargs: 其他 TTS 参数
    
    返回:
        dict: 包含音频和字幕信息的字典
    """
    
    # 构建请求数据
    request_data = {
        "text": text,
        "spk_audio_path": spk_audio_path,
        "return_subtitle": True,  # 启用字幕返回
        **kwargs
    }
    
    print(f">> 发送请求到 {api_url}")
    print(f">> 文本: {text[:50]}..." if len(text) > 50 else f">> 文本: {text}")
    
    # 发送请求
    response = requests.post(api_url, json=request_data)
    
    if response.status_code != 200:
        print(f">> 错误: {response.status_code}")
        print(response.json())
        return None
    
    # 解析响应
    result = response.json()
    
    if result.get("status") != "success":
        print(f">> 错误: {result.get('error', '未知错误')}")
        return None
    
    print(f">> 成功接收响应")
    print(f">> 采样率: {result['sample_rate']} Hz")
    print(f">> 字幕片段数: {result['subtitle_count']}")
    
    # 保存音频（如果指定了路径）
    if save_audio_path:
        audio_base64 = result['audio']
        audio_bytes = base64.b64decode(audio_base64)
        
        os.makedirs(os.path.dirname(save_audio_path) or ".", exist_ok=True)
        with open(save_audio_path, 'wb') as f:
            f.write(audio_bytes)
        print(f">> 音频已保存到: {save_audio_path}")
    
    # 保存字幕（如果指定了路径）
    if save_subtitle_path and result.get('subtitles'):
        subtitles = result['subtitles']
        
        # 根据文件扩展名决定保存格式
        if save_subtitle_path.endswith('.srt'):
            # 保存为 SRT 格式
            save_srt(subtitles, save_subtitle_path)
            print(f">> SRT 字幕已保存到: {save_subtitle_path}")
        else:
            # 保存为 JSON 格式
            os.makedirs(os.path.dirname(save_subtitle_path) or ".", exist_ok=True)
            with open(save_subtitle_path, 'w', encoding='utf-8') as f:
                json.dump(subtitles, f, ensure_ascii=False, indent=2)
            print(f">> JSON 字幕已保存到: {save_subtitle_path}")
    
    return result


def call_tts_api_without_subtitle(
    text: str,
    spk_audio_path: str,
    api_url: str = "http://localhost:6006/tts_url",
    save_audio_path: str = None,
    **kwargs
):
    """
    调用 TTS API 不获取字幕（兼容旧版本）
    
    参数:
        text: 要合成的文本
        spk_audio_path: 说话人参考音频路径
        api_url: API 服务器地址
        save_audio_path: 保存音频文件的路径（可选）
        **kwargs: 其他 TTS 参数
    
    返回:
        bytes: 音频字节流
    """
    
    # 构建请求数据
    request_data = {
        "text": text,
        "spk_audio_path": spk_audio_path,
        "return_subtitle": False,  # 不返回字幕
        **kwargs
    }
    
    print(f">> 发送请求到 {api_url}")
    print(f">> 文本: {text[:50]}..." if len(text) > 50 else f">> 文本: {text}")
    
    # 发送请求
    response = requests.post(api_url, json=request_data)
    
    if response.status_code != 200:
        print(f">> 错误: {response.status_code}")
        return None
    
    audio_bytes = response.content
    print(f">> 成功接收音频 ({len(audio_bytes)} 字节)")
    
    # 保存音频（如果指定了路径）
    if save_audio_path:
        os.makedirs(os.path.dirname(save_audio_path) or ".", exist_ok=True)
        with open(save_audio_path, 'wb') as f:
            f.write(audio_bytes)
        print(f">> 音频已保存到: {save_audio_path}")
    
    return audio_bytes


def save_srt(subtitles: list, output_file: str):
    """保存为 SRT 字幕文件"""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, subtitle in enumerate(subtitles, 1):
            start = format_srt_time(subtitle['time_begin'])
            end = format_srt_time(subtitle['time_end'])
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle['text']}\n\n")


def format_srt_time(milliseconds):
    """转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
    seconds = int(milliseconds / 1000)
    ms = int(milliseconds % 1000)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def print_subtitles(subtitles: list):
    """打印字幕信息"""
    print("\n" + "=" * 80)
    print("字幕信息:")
    print("=" * 80)
    
    for i, subtitle in enumerate(subtitles, 1):
        duration = subtitle['time_end'] - subtitle['time_begin']
        print(f"\n片段 {i}:")
        print(f"  文本: {subtitle['text']}")
        print(f"  发音: {subtitle['pronounce_text']}")
        print(f"  时间: {subtitle['time_begin']:.0f}ms - {subtitle['time_end']:.0f}ms ({duration/1000:.2f}s)")


# ========== 示例用法 ==========

def example_with_subtitle():
    """示例 1: 获取字幕"""
    print("\n" + "=" * 80)
    print("示例 1: 调用 API 并获取字幕")
    print("=" * 80)
    
    result = call_tts_api_with_subtitle(
        text="你好，这是一段测试文本。今天天气很好。",
        spk_audio_path="examples/voice_02.wav",
        save_audio_path="outputs/api_test_with_subtitle.wav",
        save_subtitle_path="outputs/api_test_subtitle.srt",
        max_text_tokens_per_sentence=120
    )
    
    if result and result.get('subtitles'):
        print_subtitles(result['subtitles'])


def example_without_subtitle():
    """示例 2: 不获取字幕（兼容旧版本）"""
    print("\n" + "=" * 80)
    print("示例 2: 调用 API 不获取字幕")
    print("=" * 80)
    
    audio_bytes = call_tts_api_without_subtitle(
        text="这是另一段测试文本。",
        spk_audio_path="examples/voice_02.wav",
        save_audio_path="outputs/api_test_without_subtitle.wav"
    )
    
    if audio_bytes:
        print(f">> 完成！")


def example_long_text():
    """示例 3: 长文本生成字幕"""
    print("\n" + "=" * 80)
    print("示例 3: 长文本生成字幕")
    print("=" * 80)
    
    long_text = """
离开神城已经一个多月了，重新回来，叶凡感受到了一种亲切。
这是一个充满激情的巨城。平静归来，他以为没有人注意。
可是刚进城不久，就有不少人眼神火热地迎了上来。
    """.strip()
    
    result = call_tts_api_with_subtitle(
        text=long_text,
        spk_audio_path="examples/voice_02.wav",
        save_audio_path="outputs/api_test_long.wav",
        save_subtitle_path="outputs/api_test_long.json",
        max_text_tokens_per_sentence=50  # 使用较小的值强制分段
    )
    
    if result and result.get('subtitles'):
        print_subtitles(result['subtitles'])


def example_with_emotion():
    """示例 4: 带情感控制的字幕生成"""
    print("\n" + "=" * 80)
    print("示例 4: 带情感控制的字幕生成")
    print("=" * 80)
    
    result = call_tts_api_with_subtitle(
        text="太棒了！今天真是美好的一天！",
        spk_audio_path="examples/voice_02.wav",
        save_audio_path="outputs/api_test_emotion.wav",
        save_subtitle_path="outputs/api_test_emotion.srt",
        emo_control_method=2,  # 使用情感向量
        emo_vec=[0.5, 0, 0, 0, 0, 0, 0, 0.3]  # happy=0.5, calm=0.3
    )
    
    if result and result.get('subtitles'):
        print_subtitles(result['subtitles'])


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    # 运行示例
    try:
        example_with_subtitle()
        example_without_subtitle()
        example_long_text()
        example_with_emotion()
        
        print("\n" + "=" * 80)
        print("所有示例执行完成！")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n错误: 无法连接到 API 服务器")
        print("请确保服务器正在运行: python api_server.py")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

