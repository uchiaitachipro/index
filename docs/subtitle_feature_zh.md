# IndexTTS2 字幕功能说明

## 概述

`IndexTTS2SubTitle` 类在原有的 TTS 功能基础上，增加了字幕生成功能。它可以在生成语音的同时，返回每个语音片段对应的文本和时间戳信息。

## 功能特点

- **时间戳同步**: 为每个语音片段提供精确的开始和结束时间（毫秒级）
- **原始文本保留**: 同时保留原始输入文本和经过 tokenizer 处理后的发音文本
- **自动分段**: 根据标点符号和最大 token 数自动将长文本分段
- **静音间隔考虑**: 时间戳计算自动考虑片段之间的静音间隔

## 使用方法

### 基本用法

```python
from indextts.infer_v2_subtitle import IndexTTS2SubTitle

# 初始化模型
tts = IndexTTS2SubTitle(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True
)

# 生成语音和字幕
result = tts.infer(
    spk_audio_prompt="examples/voice_02.wav",
    text="你好，这是一段测试文本。",
    output_path="output.wav"
)

# 获取字幕信息
subtitles = result['subtitles']
audio_path = result['audio_path']
```

### 返回数据格式

```python
{
    "audio_path": "output.wav",  # 或 "audio_data": (sample_rate, audio_array)
    "subtitles": [
        {
            "text": "你好，这是一段测试文本。",
            "pronounce_text": "你好，这是一段测试文本。",
            "time_begin": 0.0,          # 开始时间（毫秒）
            "time_end": 2500.5,         # 结束时间（毫秒）
        }
    ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | str | 原始输入文本（该片段） |
| `pronounce_text` | str | 经过 tokenizer 处理后重新解码的文本 |
| `time_begin` | float | 片段开始时间（毫秒） |
| `time_end` | float | 片段结束时间（毫秒） |

## 高级用法

### 导出为 SRT 字幕格式

```python
def save_srt(subtitles, output_file):
    """保存为 SRT 字幕文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, subtitle in enumerate(subtitles, 1):
            start = format_srt_time(subtitle['time_begin'])
            end = format_srt_time(subtitle['time_end'])
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle['text']}\n\n")

def format_srt_time(milliseconds):
    """转换为 SRT 时间格式"""
    seconds = int(milliseconds / 1000)
    ms = int(milliseconds % 1000)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

# 使用示例
save_srt(result['subtitles'], "output.srt")
```

### 导出为 JSON 格式

```python
import json

# 保存字幕为 JSON
with open("subtitles.json", 'w', encoding='utf-8') as f:
    json.dump(result['subtitles'], f, ensure_ascii=False, indent=2)
```

### 控制文本分段

```python
result = tts.infer(
    spk_audio_prompt="examples/voice_02.wav",
    text="很长的文本...",
    output_path="output.wav",
    max_text_tokens_per_segment=120,  # 每个片段的最大 token 数
    interval_silence=200,              # 片段间静音间隔（毫秒）
)
```

## 参数说明

### `infer()` 方法参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `spk_audio_prompt` | str | 必需 | 说话人参考音频路径 |
| `text` | str | 必需 | 要合成的文本 |
| `output_path` | str | None | 输出音频文件路径 |
| `max_text_tokens_per_segment` | int | 120 | 每个片段的最大 token 数 |
| `interval_silence` | int | 200 | 片段间静音间隔（毫秒） |
| `verbose` | bool | False | 是否打印详细信息 |

## 注意事项

1. **时间精度**: 时间戳基于生成的音频波形长度计算，精确到毫秒级
2. **文本分段**: 系统会根据标点符号自动分段，较长的文本会被分成多个片段
3. **静音间隔**: 片段之间的静音间隔会被计入总时间，但不会影响各片段的时间范围
4. **原始文本 vs 发音文本**: 
   - `text`: 原始输入文本
   - `pronounce_text`: 经过规范化和 tokenizer 处理后的文本，可能与原始文本略有不同

## 完整示例

参见 `examples/subtitle_example.py` 文件获取完整的使用示例。

## 与原版本的区别

- **IndexTTS2** (原版): 只返回音频文件路径或音频数据
- **IndexTTS2SubTitle** (新版): 返回包含音频和字幕信息的字典

兼容性说明：如果你需要保持与原版本的兼容性，可以这样使用：

```python
result = tts.infer(...)
if isinstance(result, dict):
    # 新版本，支持字幕
    audio_path = result['audio_path']
    subtitles = result['subtitles']
else:
    # 原版本，只有音频
    audio_path = result
```

## 常见问题

### Q: 字幕时间戳不准确怎么办？
A: 时间戳是基于实际生成的音频长度计算的，应该是准确的。如果发现不准，请检查：
   - `interval_silence` 参数设置是否正确
   - 音频采样率是否为 22050 Hz

### Q: 如何减少字幕片段数量？
A: 增加 `max_text_tokens_per_segment` 参数的值，这样每个片段会包含更多文本。

### Q: 支持流式返回吗？
A: 目前字幕功能需要在全部音频生成完成后才能返回完整的字幕信息，不支持流式返回字幕。

