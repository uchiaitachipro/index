# IndexTTS2 API 字幕功能文档

## 概述

IndexTTS2 API 服务器现在支持返回字幕信息。您可以选择以下两种模式之一：

1. **仅音频模式**（默认）：返回 WAV 音频字节流
2. **音频+字幕模式**：返回 JSON 格式，包含 base64 编码的音频和字幕信息

## API 端点

### POST /tts_url

生成语音，可选返回字幕信息。

#### 请求参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | ✓ | - | 要合成的文本 |
| `spk_audio_path` | string | ✓ | - | 说话人参考音频路径 |
| `return_subtitle` | boolean | ✗ | `false` | 是否返回字幕信息 |
| `emo_control_method` | integer | ✗ | `0` | 情感控制方法（0-3） |
| `emo_ref_path` | string | ✗ | `null` | 情感参考音频路径 |
| `emo_weight` | float | ✗ | `1.0` | 情感权重 |
| `emo_vec` | array | ✗ | `[0,0,0,0,0,0,0,0]` | 情感向量 |
| `emo_text` | string | ✗ | `null` | 情感文本 |
| `emo_random` | boolean | ✗ | `false` | 是否随机情感 |
| `max_text_tokens_per_sentence` | integer | ✗ | `120` | 每句最大 token 数 |

#### 情感控制方法

- `0`: 不使用情感控制
- `1`: 使用情感参考音频
- `2`: 使用情感向量
- `3`: 使用情感文本自动检测

#### 情感向量格式

8 维向量，对应 8 种情感：
```
[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
```

所有值的总和不应超过 1.5。

## 响应格式

### 模式 1: 仅音频（`return_subtitle=false`）

**Content-Type**: `application/octet-stream`

返回 WAV 格式的音频字节流。

### 模式 2: 音频+字幕（`return_subtitle=true`）

**Content-Type**: `application/json`

```json
{
  "status": "success",
  "audio": "base64_encoded_audio_data",
  "audio_format": "wav",
  "sample_rate": 22050,
  "subtitles": [
    {
      "text": "原始文本片段",
      "pronounce_text": "处理后的发音文本",
      "time_begin": 0.0,
      "time_end": 2500.5
    }
  ],
  "subtitle_count": 1
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 状态：`"success"` 或 `"error"` |
| `audio` | string | Base64 编码的音频数据 |
| `audio_format` | string | 音频格式（目前为 `"wav"`） |
| `sample_rate` | integer | 采样率（Hz） |
| `subtitles` | array | 字幕信息数组 |
| `subtitle_count` | integer | 字幕片段数量 |

#### 字幕对象字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 原始输入文本片段 |
| `pronounce_text` | string | 经过 tokenizer 处理的发音文本 |
| `time_begin` | float | 开始时间（毫秒） |
| `time_end` | float | 结束时间（毫秒） |

## 使用示例

### Python 示例

#### 示例 1: 获取字幕

```python
import requests
import base64

# 请求数据
data = {
    "text": "你好，这是测试。",
    "spk_audio_path": "examples/voice_02.wav",
    "return_subtitle": True
}

# 发送请求
response = requests.post("http://localhost:6006/tts_url", json=data)
result = response.json()

# 保存音频
audio_bytes = base64.b64decode(result['audio'])
with open("output.wav", "wb") as f:
    f.write(audio_bytes)

# 打印字幕
for subtitle in result['subtitles']:
    print(f"{subtitle['time_begin']}ms - {subtitle['time_end']}ms: {subtitle['text']}")
```

#### 示例 2: 仅获取音频（不要字幕）

```python
import requests

# 请求数据
data = {
    "text": "你好，这是测试。",
    "spk_audio_path": "examples/voice_02.wav",
    "return_subtitle": False  # 或省略此参数
}

# 发送请求
response = requests.post("http://localhost:6006/tts_url", json=data)

# 保存音频
with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### 示例 3: 带情感控制和字幕

```python
import requests
import base64

data = {
    "text": "太棒了！真是美好的一天！",
    "spk_audio_path": "examples/voice_02.wav",
    "return_subtitle": True,
    "emo_control_method": 2,
    "emo_vec": [0.5, 0, 0, 0, 0, 0, 0, 0.3]  # happy + calm
}

response = requests.post("http://localhost:6006/tts_url", json=data)
result = response.json()

print(f"生成了 {result['subtitle_count']} 个字幕片段")
```

### cURL 示例

#### 获取字幕

```bash
curl -X POST http://localhost:6006/tts_url \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是测试。",
    "spk_audio_path": "examples/voice_02.wav",
    "return_subtitle": true
  }'
```

#### 仅获取音频

```bash
curl -X POST http://localhost:6006/tts_url \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是测试。",
    "spk_audio_path": "examples/voice_02.wav"
  }' \
  --output output.wav
```

### JavaScript 示例

```javascript
// 获取字幕
async function getTTSWithSubtitle(text, spkAudioPath) {
  const response = await fetch('http://localhost:6006/tts_url', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: text,
      spk_audio_path: spkAudioPath,
      return_subtitle: true
    })
  });
  
  const result = await response.json();
  
  // 解码音频
  const audioBytes = atob(result.audio);
  const audioArray = new Uint8Array(audioBytes.length);
  for (let i = 0; i < audioBytes.length; i++) {
    audioArray[i] = audioBytes.charCodeAt(i);
  }
  
  // 创建 Blob 和音频 URL
  const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
  const audioUrl = URL.createObjectURL(audioBlob);
  
  return {
    audioUrl: audioUrl,
    subtitles: result.subtitles
  };
}

// 使用示例
getTTSWithSubtitle("你好，这是测试。", "examples/voice_02.wav")
  .then(result => {
    console.log('音频 URL:', result.audioUrl);
    console.log('字幕:', result.subtitles);
  });
```

## 字幕格式转换

### 转换为 SRT 格式

```python
def save_srt(subtitles, filename):
    """保存为 SRT 字幕文件"""
    with open(filename, 'w', encoding='utf-8') as f:
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
```

### 转换为 VTT 格式

```python
def save_vtt(subtitles, filename):
    """保存为 WebVTT 字幕文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for i, subtitle in enumerate(subtitles, 1):
            start = format_vtt_time(subtitle['time_begin'])
            end = format_vtt_time(subtitle['time_end'])
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle['text']}\n\n")

def format_vtt_time(milliseconds):
    """转换为 VTT 时间格式"""
    seconds = int(milliseconds / 1000)
    ms = int(milliseconds % 1000)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"
```

## 错误处理

### 错误响应格式

```json
{
  "status": "error",
  "error": "错误信息"
}
```

### 常见错误

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| 情感向量之和不能超过1.5 | `emo_vec` 总和 > 1.5 | 降低情感向量的值 |
| TTS model not initialized | 模型未加载 | 等待服务器启动完成 |
| 文件路径不存在 | 音频文件路径错误 | 检查路径是否正确 |

## 性能优化建议

1. **长文本分段**: 使用 `max_text_tokens_per_sentence` 参数控制分段，较小的值会生成更多字幕片段但更精确
2. **缓存音频**: 如果只需要字幕而不需要多次下载音频，可以本地缓存
3. **批量处理**: 对于多个文本，依次调用 API 而不是并行，以避免服务器过载

## 启动服务器

```bash
# 基本启动
python api_server.py

# 使用 FP16 加速
python api_server.py --is_fp16

# 使用 CUDA 内核加速（需要 CUDA）
python api_server.py --is_fp16 --use_cuda_kernel

# 自定义端口
python api_server.py --port 8080

# 启用详细日志
python api_server.py --verbose
```

## 健康检查

```bash
curl http://localhost:6006/health
```

响应：
```json
{
  "status": "healthy",
  "message": "Service is running",
  "timestamp": 1234567890.123
}
```

## 更多示例

完整的客户端示例代码请参见：`examples/api_subtitle_client.py`

运行示例：
```bash
python examples/api_subtitle_client.py
```

## 技术支持

如有问题，请查看：
- [字幕功能文档](subtitle_feature_zh.md)
- [项目 README](../README.md)

