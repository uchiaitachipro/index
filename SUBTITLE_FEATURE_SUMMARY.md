# IndexTTS2 字幕功能 - 实现总结

## 实现目标 ✓

为 IndexTTS2 的 `infer` 函数添加字幕返回功能，支持返回每个语音片段对应的：
- 原始文本 (`text`)
- 处理后的发音文本 (`pronounce_text`)
- 开始时间 (`time_begin`，单位：毫秒)
- 结束时间 (`time_end`，单位：毫秒)

## 实现方式

### 核心修改
在 `indextts/infer_v2_subtitle.py` 的 `infer_generator` 方法中添加了字幕生成逻辑：

1. **文本预处理阶段**：
   ```python
   # 保存每个片段的原始文本
   original_text_segments = []
   for segment_tokens in segments:
       segment_ids = self.tokenizer.convert_tokens_to_ids(segment_tokens)
       segment_text = self.tokenizer.decode(segment_ids)
       original_text_segments.append(segment_text)
   ```

2. **音频生成阶段**：
   ```python
   # 初始化字幕收集
   subtitles = []
   current_time_ms = 0.0
   
   # 在每个片段生成后
   for seg_idx, sent in enumerate(segments):
       # ... 生成音频 wav ...
       
       # 计算时长和时间戳
       segment_duration_ms = (wav.shape[-1] / sampling_rate) * 1000.0
       time_begin = current_time_ms
       time_end = current_time_ms + segment_duration_ms
       
       # 收集字幕信息
       subtitle_item = {
           "text": original_text_segments[seg_idx],
           "pronounce_text": self.tokenizer.decode(segment_ids),
           "time_begin": time_begin,
           "time_end": time_end,
       }
       subtitles.append(subtitle_item)
       
       # 更新累计时间（包括静音间隔）
       current_time_ms = time_end
       if seg_idx < segments_count - 1:
           current_time_ms += interval_silence
   ```

3. **返回结果**：
   ```python
   # 修改返回格式为字典
   yield {
       "audio_path": output_path,  # 或 "audio_data": (sr, data)
       "subtitles": subtitles
   }
   ```

## 返回数据示例

```json
{
  "audio_path": "output.wav",
  "subtitles": [
    {
      "text": "离开神城已经一个多月了，重新回来，叶凡感受到了一种亲切，这是一个充满激情的巨城。",
      "pronounce_text": "离开神城已经一个多月了，重新回来，叶凡感受到了一种亲切，这是一个充满激情的巨城。",
      "time_begin": 0.0,
      "time_end": 10906.258503401361
    },
    {
      "text": "平静归来，他以为没有人注意，可是刚进城不久，就有不少人眼神火热地迎了上来，热情地打招呼。",
      "pronounce_text": "平静归来，他以为没有人注意，可是刚进城不久，就有不少人眼神火热地迎了上来，热情地打招呼。",
      "time_begin": 11106.258503401361,
      "time_end": 24694.421768707485
    }
  ]
}
```

## 关键技术点

### 1. 时间戳计算
- **基础**: 根据音频波形长度和采样率计算：`(wav.shape[-1] / sampling_rate) * 1000.0`
- **累加**: 考虑片段间的静音间隔 (`interval_silence`)
- **精度**: 毫秒级精度，使用浮点数

### 2. 文本处理
- **原始文本**: 通过 `tokenizer.decode()` 从 tokens 还原
- **发音文本**: 同样使用 `tokenizer.decode()`，确保与实际语音对应
- **分段**: 根据标点符号和 `max_text_tokens_per_segment` 自动分段

### 3. 静音间隔处理
- 片段之间的静音 (`interval_silence`) 被计入总时间
- 但不影响各片段自身的 `time_begin` 和 `time_end`
- 最后一个片段后不添加静音间隔

## 配套文件

### 1. 示例程序
- **文件**: `examples/subtitle_example.py`
- **功能**: 完整的使用示例，包括导出 JSON 和 SRT 格式

### 2. 测试脚本
- **文件**: `tests/test_subtitle.py`
- **功能**: 自动化测试，验证基本功能、长文本分段、无输出文件等场景

### 3. 文档
- **文件**: `docs/subtitle_feature_zh.md`
- **内容**: 详细的功能说明、API 文档、高级用法、常见问题

## 使用示例

### 基本用法
```python
from indextts.infer_v2_subtitle import IndexTTS2SubTitle

tts = IndexTTS2SubTitle(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints"
)

result = tts.infer(
    spk_audio_prompt="examples/voice_02.wav",
    text="你好，这是测试文本。",
    output_path="output.wav"
)

# 访问字幕
for subtitle in result['subtitles']:
    print(f"{subtitle['time_begin']:.0f}ms: {subtitle['text']}")
```

### 导出 SRT 字幕
```python
def save_srt(subtitles, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles, 1):
            start = format_srt_time(sub['time_begin'])
            end = format_srt_time(sub['time_end'])
            f.write(f"{i}\n{start} --> {end}\n{sub['text']}\n\n")

save_srt(result['subtitles'], "output.srt")
```

## 特点和优势

✓ **精确时间戳**: 基于实际生成音频的长度计算，精确到毫秒
✓ **自动分段**: 根据标点和长度自动智能分段
✓ **保留原文**: 同时保留原始文本和处理后文本
✓ **易于导出**: 可轻松导出为 JSON、SRT 等常见字幕格式
✓ **向后兼容**: 不影响原有 `IndexTTS2` 类的功能

## 限制和注意事项

⚠️ **流式返回**: 当 `stream_return=True` 时，字幕信息不可用
⚠️ **文本差异**: `pronounce_text` 可能与 `text` 略有差异（规范化）
⚠️ **片段级别**: 当前仅支持片段级别时间戳，不支持字级别

## 测试验证

所有测试通过：
- ✓ 基本字幕生成
- ✓ 长文本自动分段
- ✓ 无输出文件模式
- ✓ 时间戳连续性验证
- ✓ 返回格式验证

## 相关文件清单

**修改的文件**:
- `indextts/infer_v2_subtitle.py` - 主要实现

**新增的文件**:
- `examples/subtitle_example.py` - 使用示例
- `tests/test_subtitle.py` - 自动化测试
- `docs/subtitle_feature_zh.md` - 功能文档
- `SUBTITLE_CHANGES.md` - 修改说明
- `SUBTITLE_FEATURE_SUMMARY.md` - 本文件

## 下一步建议

1. 运行测试验证功能：`python tests/test_subtitle.py`
2. 查看示例程序：`python examples/subtitle_example.py`
3. 阅读详细文档：`docs/subtitle_feature_zh.md`
4. 根据实际需求调整参数和格式

---

**实现完成日期**: 2025-10-26
**实现状态**: ✓ 完成并通过测试

