# 说话人音频上传 - 快速开始

## 1 分钟快速上手

### Python 示例

```python
import requests

# 步骤 1: 上传音频
with open("my_voice.wav", "rb") as f:
    upload_response = requests.post(
        "http://localhost:6006/upload_speaker",
        files={"file": f}
    )

file_path = upload_response.json()['file_path']
print(f"上传成功: {file_path}")

# 步骤 2: 使用上传的音频生成语音
tts_response = requests.post(
    "http://localhost:6006/tts_url",
    json={
        "text": "你好，这是测试。",
        "spk_audio_path": file_path  # 使用上传后的路径
    }
)

# 步骤 3: 保存生成的音频
with open("output.wav", "wb") as f:
    f.write(tts_response.content)
```

### cURL 示例

```bash
# 1. 上传音频
curl -X POST http://localhost:6006/upload_speaker \
  -F "file=@my_voice.wav"

# 返回: {"file_path": "uploads/speakers/spk_my_voice_XXX.wav", ...}

# 2. 生成语音
curl -X POST http://localhost:6006/tts_url \
  -H "Content-Type: application/json" \
  -d '{"text":"你好","spk_audio_path":"uploads/speakers/spk_my_voice_XXX.wav"}' \
  --output output.wav
```

## API 端点

### POST /upload_speaker

**上传文件**：
```python
files = {"file": open("audio.wav", "rb")}
response = requests.post(url, files=files)
```

**返回**：
```json
{
  "status": "success",
  "file_path": "uploads/speakers/spk_audio_1730000000123.wav",
  "filename": "spk_audio_1730000000123.wav",
  "file_size": 123456
}
```

**文件名规则**：`spk_{原始名称}_{时间戳}.{扩展名}`
- 保留原始文件名的字母、数字、下划线和连字符
- 特殊字符和中文会被过滤
- 时间戳确保文件名唯一

**支持格式**：`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`

## 完整示例

运行完整示例代码：
```bash
python examples/upload_speaker_example.py
```

包含以下示例：
1. 基本上传和 TTS
2. 带字幕生成
3. 批量上传
4. 错误处理
5. 完整工作流程

## 工作流程

```
上传音频文件
    ↓
获取 file_path
    ↓
调用 /tts_url
    ↓
生成语音
```

## 常见用途

- ✅ Web 应用用户上传自己的声音
- ✅ 移动应用动态音频上传
- ✅ 远程客户端使用自定义说话人
- ✅ 批量处理不同说话人

## 注意事项

⚠️ **文件清理**：建议定期清理 `uploads/speakers/` 目录中的旧文件

⚠️ **安全性**：生产环境中建议添加文件大小限制和身份验证

⚠️ **存储空间**：注意监控磁盘空间使用情况

## 详细文档

查看完整文档：`docs/UPLOAD_SPEAKER_API.md`

---

**提示**：上传的文件路径可以立即在 `/tts_url` 接口中使用，无需等待！

