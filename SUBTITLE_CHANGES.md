# IndexTTS2 字幕功能更新说明

## 更新概述

为 `IndexTTS2` 添加了字幕生成功能，创建了新的 `IndexTTS2SubTitle` 类，可以在生成语音的同时返回每个片段的时间戳和文本信息。

## 修改的文件

### 1. `indextts/infer_v2_subtitle.py`
这是主要的修改文件，在 `infer_generator` 方法中添加了字幕生成逻辑：

#### 主要改动：
- **文本分段处理** (第 491-500 行)
  - 在文本分词后，保存每个片段的原始文本
  - 使用 `tokenizer.decode()` 方法将 tokens 还原为文本

- **字幕信息收集** (第 532-534 行)
  - 添加 `subtitles` 列表用于存储字幕信息
  - 添加 `current_time_ms` 变量追踪累计时间

- **时间戳计算** (第 669-693 行)
  - 根据生成的音频波形长度计算每个片段的时长
  - 考虑片段间的静音间隔 (`interval_silence`)
  - 记录每个片段的 `time_begin` 和 `time_end`

- **返回值修改** (第 714-743 行)
  - 修改返回格式为字典，包含音频和字幕信息
  - 有输出路径时返回: `{"audio_path": "...", "subtitles": [...]}`
  - 无输出路径时返回: `{"audio_data": (sr, data), "subtitles": [...]}`

## 新增的文件

### 1. `examples/subtitle_example.py`
完整的使用示例，展示如何：
- 初始化 `IndexTTS2SubTitle` 模型
- 生成语音和字幕
- 导出字幕为 JSON 格式
- 导出字幕为 SRT 格式

### 2. `docs/subtitle_feature_zh.md`
详细的功能说明文档，包括：
- 功能特点和使用方法
- 返回数据格式说明
- 高级用法（导出 SRT、JSON）
- 参数说明和注意事项
- 常见问题解答

### 3. `tests/test_subtitle.py`
自动化测试脚本，包含三个测试：
- 基本字幕生成测试
- 长文本分段测试
- 无输出文件测试

## 返回数据格式

```python
{
    "audio_path": "output.wav",  # 音频文件路径（或 "audio_data" 如果不保存文件）
    "subtitles": [
        {
            "text": "原始文本片段",
            "pronounce_text": "经过 tokenizer 处理的文本",
            "time_begin": 0.0,      # 开始时间（毫秒）
            "time_end": 2500.5,     # 结束时间（毫秒）
        },
        # ... 更多片段
    ]
}
```

## 使用示例

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
    text="你好，这是一段测试文本。今天天气很好。",
    output_path="output.wav"
)

# 访问结果
audio_path = result['audio_path']
subtitles = result['subtitles']

# 打印字幕信息
for subtitle in subtitles:
    print(f"{subtitle['time_begin']:.2f}ms - {subtitle['time_end']:.2f}ms: {subtitle['text']}")
```

## 技术细节

### 时间戳计算方法
1. 每个片段生成音频后，根据波形长度计算时长：
   ```python
   duration_ms = (wav.shape[-1] / sampling_rate) * 1000.0
   ```

2. 累加时间戳，考虑片段间的静音间隔：
   ```python
   time_begin = current_time_ms
   time_end = current_time_ms + segment_duration_ms
   current_time_ms = time_end + interval_silence
   ```

### 文本处理
- 原始文本通过 `tokenizer.tokenize()` 分词
- 根据标点符号和最大 token 数分段
- 每个片段的 tokens 通过 `tokenizer.decode()` 还原为文本

### 兼容性
- 不影响原有的 `IndexTTS2` 类（在 `infer_v2.py` 中）
- 新功能封装在独立的 `IndexTTS2SubTitle` 类中
- 可以根据需要选择使用原版或带字幕的版本

## 测试方法

1. 运行示例程序：
```bash
python examples/subtitle_example.py
```

2. 运行自动化测试：
```bash
python tests/test_subtitle.py
```

3. 直接运行修改后的文件：
```bash
python indextts/infer_v2_subtitle.py
```

## 注意事项

1. **流式返回**: 目前字幕功能在 `stream_return=True` 时不返回字幕信息
2. **时间精度**: 时间戳精确到毫秒级，基于实际生成的音频长度
3. **文本差异**: `pronounce_text` 可能与 `text` 略有不同，因为经过了规范化处理

## 未来改进方向

- [ ] 支持流式返回时的实时字幕生成
- [ ] 添加更多字幕格式导出（WebVTT、ASS 等）
- [ ] 支持字级别的时间戳（而不仅是片段级别）
- [ ] 支持自定义分段策略

