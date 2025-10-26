# API 服务器字幕功能更新说明

## 更新概述

为 `api_server.py` 的 `/tts_url` 端点添加了字幕返回功能，支持在生成语音的同时返回时间戳和文本信息。

## 主要改动

### 1. 修改的文件：`api_server.py`

#### 新增导入
```python
import base64  # 用于编码音频数据
```

#### 修改的端点：`POST /tts_url`

**新增请求参数**：
- `return_subtitle` (boolean, 可选, 默认 `false`): 是否返回字幕信息

**返回格式变化**：

1. **当 `return_subtitle=false` 时**（默认，向后兼容）：
   - Content-Type: `application/octet-stream`
   - 返回：WAV 音频字节流

2. **当 `return_subtitle=true` 时**（新功能）：
   - Content-Type: `application/json`
   - 返回：JSON 对象，包含 base64 编码的音频和字幕信息

### 2. 新增的文件

#### `examples/api_subtitle_client.py`
完整的客户端示例代码，包含：
- `call_tts_api_with_subtitle()` - 获取字幕
- `call_tts_api_without_subtitle()` - 不获取字幕
- 字幕格式转换（SRT）
- 多个使用示例

#### `docs/API_SUBTITLE.md`
详细的 API 文档，包含：
- 端点说明
- 请求/响应格式
- 多语言示例（Python, cURL, JavaScript）
- 错误处理
- 字幕格式转换

#### `tests/test_api_subtitle.py`
自动化测试脚本，测试：
- 健康检查
- 不返回字幕的请求
- 返回字幕的请求
- 长文本处理
- 情感控制 + 字幕

## 返回格式详解

### 字幕模式返回格式

```json
{
  "status": "success",
  "audio": "UklGRiQAAABXQVZFZm10...",
  "audio_format": "wav",
  "sample_rate": 22050,
  "subtitles": [
    {
      "text": "你好，今天天气很好。",
      "pronounce_text": "你好，今天天气很好。",
      "time_begin": 0.0,
      "time_end": 2856.8
    },
    {
      "text": "我们去公园吧。",
      "pronounce_text": "我们去公园吧。",
      "time_begin": 3056.8,
      "time_end": 5123.4
    }
  ],
  "subtitle_count": 2
}
```

## 使用示例

### Python 客户端

```python
import requests
import base64

# 获取字幕
data = {
    "text": "你好，这是测试。",
    "spk_audio_path": "examples/voice_02.wav",
    "return_subtitle": True
}

response = requests.post("http://localhost:6006/tts_url", json=data)
result = response.json()

# 解码音频
audio_bytes = base64.b64decode(result['audio'])
with open("output.wav", "wb") as f:
    f.write(audio_bytes)

# 使用字幕
for subtitle in result['subtitles']:
    print(f"{subtitle['time_begin']:.0f}ms: {subtitle['text']}")
```

### 不获取字幕（向后兼容）

```python
import requests

data = {
    "text": "你好，这是测试。",
    "spk_audio_path": "examples/voice_02.wav"
    # return_subtitle 默认为 False
}

response = requests.post("http://localhost:6006/tts_url", json=data)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## 向后兼容性

✓ **完全向后兼容**：
- 默认行为保持不变（返回音频字节流）
- 旧客户端无需修改即可继续使用
- 只有显式设置 `return_subtitle=true` 才会改变返回格式

## 性能影响

- **字幕生成**: 几乎无额外开销（已在 TTS 过程中计算）
- **Base64 编码**: 约增加 33% 数据大小
- **网络传输**: JSON 格式略大于纯二进制，但可通过 gzip 压缩

## 测试方法

### 1. 启动服务器
```bash
python api_server.py --is_fp16
```

### 2. 运行自动化测试
```bash
python tests/test_api_subtitle.py
```

### 3. 运行示例程序
```bash
python examples/api_subtitle_client.py
```

### 4. 手动测试（cURL）
```bash
# 获取字幕
curl -X POST http://localhost:6006/tts_url \
  -H "Content-Type: application/json" \
  -d '{"text":"你好","spk_audio_path":"examples/voice_02.wav","return_subtitle":true}' \
  | jq .

# 不获取字幕
curl -X POST http://localhost:6006/tts_url \
  -H "Content-Type: application/json" \
  -d '{"text":"你好","spk_audio_path":"examples/voice_02.wav"}' \
  --output test.wav
```

## API 端点完整参数列表

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | ✓ | - | 要合成的文本 |
| `spk_audio_path` | string | ✓ | - | 说话人参考音频路径 |
| `return_subtitle` | boolean | ✗ | `false` | **新增：是否返回字幕** |
| `emo_control_method` | integer | ✗ | `0` | 情感控制方法（0-3） |
| `emo_ref_path` | string | ✗ | `null` | 情感参考音频路径 |
| `emo_weight` | float | ✗ | `1.0` | 情感权重 |
| `emo_vec` | array | ✗ | `[0]*8` | 情感向量 |
| `emo_text` | string | ✗ | `null` | 情感文本 |
| `emo_random` | boolean | ✗ | `false` | 是否随机情感 |
| `max_text_tokens_per_sentence` | integer | ✗ | `120` | 每句最大 token 数 |

## 技术细节

### 时间戳计算
- 基于实际生成的音频波形长度
- 精确到毫秒级
- 考虑片段间的静音间隔

### Base64 编码原因
- JSON 不支持直接传输二进制数据
- Base64 是标准的二进制编码方式
- 前端可直接解码使用

### 错误处理
所有错误仍然返回 JSON 格式：
```json
{
  "status": "error",
  "error": "错误信息详情"
}
```

## 常见问题

### Q1: 为什么使用 Base64 而不是直接返回二进制？
A: 当 `return_subtitle=true` 时，需要同时返回音频和字幕，JSON 格式最为合适。而 JSON 不支持二进制数据，因此使用 Base64 编码。

### Q2: Base64 会增加多少数据量？
A: 约 33%。例如 1MB 音频会变成约 1.33MB 的 base64 字符串。

### Q3: 如何减少传输数据量？
A: 可以使用 gzip 压缩（大多数 HTTP 客户端自动支持）。

### Q4: 旧客户端是否会受影响？
A: 不会。只要不设置 `return_subtitle=true`，行为完全不变。

### Q5: 字幕功能会增加多少计算时间？
A: 几乎没有额外开销，字幕信息在 TTS 过程中自然产生。

## 迁移指南

### 从旧 API 迁移到新 API（带字幕）

**旧代码**：
```python
response = requests.post(API_URL, json={
    "text": text,
    "spk_audio_path": audio_path
})
audio_bytes = response.content
```

**新代码**：
```python
response = requests.post(API_URL, json={
    "text": text,
    "spk_audio_path": audio_path,
    "return_subtitle": True  # 添加此行
})
result = response.json()
audio_bytes = base64.b64decode(result['audio'])  # 修改此行
subtitles = result['subtitles']  # 新增：获取字幕
```

## 相关文档

- [API 详细文档](docs/API_SUBTITLE.md)
- [字幕功能说明](docs/subtitle_feature_zh.md)
- [客户端示例](examples/api_subtitle_client.py)
- [自动化测试](tests/test_api_subtitle.py)

## 更新历史

- **2025-10-26**: 添加字幕返回功能
- 向后兼容旧版本 API
- 添加完整文档和示例

---

**实现状态**: ✓ 完成并测试通过

