#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说话人音频上传示例

展示如何使用 /upload_speaker API 上传音频文件，然后在 TTS 生成中使用
"""

import requests
import base64
import os


API_BASE_URL = "http://localhost:6006"


def upload_speaker_audio(audio_file_path: str):
    """
    上传说话人音频文件
    
    参数:
        audio_file_path: 本地音频文件路径
    
    返回:
        dict: 包含上传结果的字典，其中 file_path 可用于后续 TTS 调用
    """
    if not os.path.exists(audio_file_path):
        print(f"错误: 文件不存在 - {audio_file_path}")
        return None
    
    print(f">> 上传音频文件: {audio_file_path}")
    
    # 读取文件
    with open(audio_file_path, 'rb') as f:
        files = {'file': (os.path.basename(audio_file_path), f, 'audio/wav')}
        
        # 发送上传请求
        response = requests.post(f"{API_BASE_URL}/upload_speaker", files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result.get('status') == 'success':
            print(f"✓ 上传成功!")
            print(f"  文件路径: {result['file_path']}")
            print(f"  文件名: {result['filename']}")
            print(f"  文件大小: {result['file_size']} 字节")
            return result
        else:
            print(f"✗ 上传失败: {result.get('error', '未知错误')}")
            return None
    else:
        print(f"✗ 请求失败: HTTP {response.status_code}")
        try:
            error_info = response.json()
            print(f"  错误信息: {error_info.get('error', '未知错误')}")
        except:
            print(f"  响应内容: {response.text}")
        return None


def generate_tts_with_uploaded_audio(spk_audio_path: str, text: str, return_subtitle: bool = False):
    """
    使用上传的音频文件进行 TTS 生成
    
    参数:
        spk_audio_path: 上传后返回的文件路径
        text: 要合成的文本
        return_subtitle: 是否返回字幕
    
    返回:
        dict 或 bytes: TTS 生成结果
    """
    print(f"\n>> 生成语音...")
    print(f"  参考音频: {spk_audio_path}")
    print(f"  文本: {text}")
    
    data = {
        "text": text,
        "spk_audio_path": spk_audio_path,
        "return_subtitle": return_subtitle
    }
    
    response = requests.post(f"{API_BASE_URL}/tts_url", json=data)
    
    if response.status_code == 200:
        if return_subtitle:
            result = response.json()
            print(f"✓ 生成成功!")
            print(f"  字幕片段数: {result.get('subtitle_count', 0)}")
            return result
        else:
            audio_bytes = response.content
            print(f"✓ 生成成功!")
            print(f"  音频大小: {len(audio_bytes)} 字节")
            return audio_bytes
    else:
        print(f"✗ 生成失败: HTTP {response.status_code}")
        try:
            error_info = response.json()
            print(f"  错误信息: {error_info.get('error', '未知错误')}")
        except:
            print(f"  响应内容: {response.text}")
        return None


def save_audio_from_base64(audio_base64: str, output_path: str):
    """从 base64 保存音频"""
    audio_bytes = base64.b64decode(audio_base64)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(audio_bytes)
    print(f"  音频已保存到: {output_path}")


# ========== 示例用法 ==========

def example_1_basic_upload_and_tts():
    """示例 1: 基本的上传和 TTS 生成"""
    print("\n" + "=" * 80)
    print("示例 1: 上传音频并生成语音（不带字幕）")
    print("=" * 80)
    
    # 步骤 1: 上传音频文件
    upload_result = upload_speaker_audio("test/shan_tian_fang_voices_nano.mp3")
    
    if not upload_result:
        return
    
    # 步骤 2: 使用上传的文件路径进行 TTS
    uploaded_path = upload_result['file_path']
    audio_bytes = generate_tts_with_uploaded_audio(
        spk_audio_path=uploaded_path,
        text="你好，这是使用上传音频文件生成的语音。",
        return_subtitle=False
    )
    
    # 步骤 3: 保存生成的音频
    if audio_bytes:
        output_path = "outputs/example1_output.wav"
        os.makedirs("outputs", exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        print(f"  输出音频已保存到: {output_path}")


def example_2_upload_with_subtitle():
    """示例 2: 上传音频并生成语音（带字幕）"""
    print("\n" + "=" * 80)
    print("示例 2: 上传音频并生成语音（带字幕）")
    print("=" * 80)
    
    # 步骤 1: 上传音频文件
    upload_result = upload_speaker_audio("test/shan_tian_fang_voices_nano.mp3")
    
    if not upload_result:
        return
    
    # 步骤 2: 使用上传的文件路径进行 TTS（带字幕）
    uploaded_path = upload_result['file_path']
    result = generate_tts_with_uploaded_audio(
        spk_audio_path=uploaded_path,
        text="今天天气很好，我们一起去公园玩吧。",
        return_subtitle=True
    )
    
    if result:
        # 步骤 3: 保存音频
        save_audio_from_base64(result['audio'], "outputs/example2_output.wav")
        
        # 步骤 4: 打印字幕
        print(f"\n  字幕信息:")
        for i, subtitle in enumerate(result['subtitles'], 1):
            duration = subtitle['time_end'] - subtitle['time_begin']
            print(f"    [{i}] {subtitle['text']}")
            print(f"        时间: {subtitle['time_begin']:.0f}ms - {subtitle['time_end']:.0f}ms ({duration/1000:.2f}s)")


def example_3_batch_upload():
    """示例 3: 批量上传多个说话人音频"""
    print("\n" + "=" * 80)
    print("示例 3: 批量上传多个说话人音频")
    print("=" * 80)
    
    audio_files = [
        "examples/voice_02.wav",
        "examples/voice_03.wav",
        "examples/voice_04.wav"
    ]
    
    uploaded_speakers = []
    
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            print(f"\n>> 上传: {audio_file}")
            result = upload_speaker_audio(audio_file)
            if result:
                uploaded_speakers.append({
                    'original': audio_file,
                    'uploaded_path': result['file_path'],
                    'filename': result['filename']
                })
    
    print(f"\n>> 成功上传 {len(uploaded_speakers)} 个音频文件:")
    for speaker in uploaded_speakers:
        print(f"  - {speaker['original']} -> {speaker['uploaded_path']}")
    
    # 使用第一个上传的音频生成语音
    if uploaded_speakers:
        print(f"\n>> 使用第一个上传的音频生成语音...")
        audio_bytes = generate_tts_with_uploaded_audio(
            spk_audio_path=uploaded_speakers[0]['uploaded_path'],
            text="这是使用批量上传的第一个音频生成的语音。",
            return_subtitle=False
        )
        
        if audio_bytes:
            output_path = "outputs/example3_output.wav"
            os.makedirs("outputs", exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            print(f"  输出音频已保存到: {output_path}")


def example_4_error_handling():
    """示例 4: 错误处理"""
    print("\n" + "=" * 80)
    print("示例 4: 错误处理示例")
    print("=" * 80)
    
    # 测试 1: 上传不存在的文件
    print("\n>> 测试 1: 上传不存在的文件")
    upload_speaker_audio("non_existent_file.wav")
    
    # 测试 2: 上传不支持的格式（如果有的话）
    print("\n>> 测试 2: 尝试上传文本文件")
    # 创建一个临时文本文件
    temp_file = "temp_test.txt"
    with open(temp_file, 'w') as f:
        f.write("This is not an audio file")
    upload_speaker_audio(temp_file)
    os.remove(temp_file)  # 清理
    
    # 测试 3: 使用不存在的路径进行 TTS
    print("\n>> 测试 3: 使用不存在的音频路径进行 TTS")
    generate_tts_with_uploaded_audio(
        spk_audio_path="uploads/speakers/non_existent.wav",
        text="这应该会失败",
        return_subtitle=False
    )


def example_5_complete_workflow():
    """示例 5: 完整工作流程"""
    print("\n" + "=" * 80)
    print("示例 5: 完整工作流程（上传 -> 生成 -> 保存字幕）")
    print("=" * 80)
    
    # 步骤 1: 上传
    upload_result = upload_speaker_audio("examples/voice_02.wav")
    if not upload_result:
        return
    
    # 步骤 2: 生成（带字幕）
    result = generate_tts_with_uploaded_audio(
        spk_audio_path=upload_result['file_path'],
        text="离开神城已经一个多月了，重新回来，叶凡感受到了一种亲切。这是一个充满激情的巨城。",
        return_subtitle=True
    )
    
    if not result:
        return
    
    # 步骤 3: 保存音频
    save_audio_from_base64(result['audio'], "outputs/example5_output.wav")
    
    # 步骤 4: 保存字幕为 JSON
    import json
    subtitle_path = "outputs/example5_subtitle.json"
    with open(subtitle_path, 'w', encoding='utf-8') as f:
        json.dump(result['subtitles'], f, ensure_ascii=False, indent=2)
    print(f"  字幕已保存到: {subtitle_path}")
    
    # 步骤 5: 保存字幕为 SRT
    def format_srt_time(ms):
        s = int(ms / 1000)
        ms = int(ms % 1000)
        h, m = s // 3600, (s % 3600) // 60
        s = s % 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    srt_path = "outputs/example5_subtitle.srt"
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(result['subtitles'], 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(sub['time_begin'])} --> {format_srt_time(sub['time_end'])}\n")
            f.write(f"{sub['text']}\n\n")
    print(f"  SRT 字幕已保存到: {srt_path}")


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    try:
        example_1_basic_upload_and_tts()
        example_2_upload_with_subtitle()
        # example_3_batch_upload()
        # example_4_error_handling()
        # example_5_complete_workflow()
        
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

