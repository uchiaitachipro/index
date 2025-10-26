# 说话人音频上传 API 文档

## 概述

`/upload_speaker` 端点允许你上传说话人参考音频文件到服务器，返回的文件路径可以在 `/tts_url` 接口中使用。

这在以下场景中特别有用：
- 远程客户端需要使用自定义说话人音频
- Web 应用中用户上传自己的声音
- 移动应用需要动态上传音频文件

## API 端点

### POST /upload_speaker

上传说话人参考音频文件

#### 请求格式

使用 `multipart/form-data` 格式上传文件：

**参数**：
- `file` (File, 必需): 音频文件

**支持的音频格式**：
- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- OGG (.ogg)
- M4A (.m4a)
- AAC (.aac)

#### 响应格式

**成功响应** (200):
```json
{
  "status": "success",
  "file_path": "uploads/speakers/spk_my_voice_1730000000123.wav",
  "filename": "spk_my_voice_1730000000123.wav",
  "file_size": 123456,
  "timestamp": 1730000000.123,
  "message": "文件已成功上传到 uploads/speakers/spk_my_voice_1730000000123.wav"
}
```

**字段说明**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 状态：`"success"` 或 `"error"` |
| `file_path` | string | 保存的文件路径（可用于 tts_url 的 spk_audio_path） |
| `filename` | string | 生成的唯一文件名 |
| `file_size` | integer | 文件大小（字节） |
| `timestamp` | float | 上传时间戳 |
| `message` | string | 成功消息 |

**错误响应** (400):
```json
{
  "status": "error",
  "error": "不支持的文件格式: .txt，支持的格式: .wav, .mp3, .flac, .ogg, .m4a, .aac"
}
```

**错误响应** (500):
```json
{
  "status": "error",
  "error": "错误详情..."
}
```

## 使用示例

### Python 示例

#### 基本用法

```python
import requests

# 上传文件
with open("my_voice.wav", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:6006/upload_speaker", files=files)

result = response.json()
print(f"上传成功！文件路径: {result['file_path']}")

# 使用上传的文件进行 TTS
tts_response = requests.post("http://localhost:6006/tts_url", json={
    "text": "你好，这是测试。",
    "spk_audio_path": result['file_path'],  # 使用上传后的路径
    "return_subtitle": False
})

audio_bytes = tts_response.content
```

#### 完整示例（带错误处理）

```python
import requests
import os

def upload_and_generate_tts(audio_file, text):
    """上传音频并生成 TTS"""
    
    # 步骤 1: 上传音频
    if not os.path.exists(audio_file):
        print(f"文件不存在: {audio_file}")
        return None
    
    with open(audio_file, 'rb') as f:
        files = {'file': (os.path.basename(audio_file), f)}
        upload_response = requests.post(
            "http://localhost:6006/upload_speaker",
            files=files
        )
    
    if upload_response.status_code != 200:
        print(f"上传失败: {upload_response.json()}")
        return None
    
    upload_result = upload_response.json()
    if upload_result['status'] != 'success':
        print(f"上传失败: {upload_result.get('error')}")
        return None
    
    print(f"✓ 上传成功: {upload_result['file_path']}")
    
    # 步骤 2: 生成 TTS
    tts_response = requests.post(
        "http://localhost:6006/tts_url",
        json={
            "text": text,
            "spk_audio_path": upload_result['file_path'],
            "return_subtitle": False
        }
    )
    
    if tts_response.status_code != 200:
        print(f"TTS 生成失败: {tts_response.json()}")
        return None
    
    print(f"✓ TTS 生成成功")
    return tts_response.content

# 使用示例
audio_bytes = upload_and_generate_tts("my_voice.wav", "你好世界")
if audio_bytes:
    with open("output.wav", "wb") as f:
        f.write(audio_bytes)
    print("音频已保存到 output.wav")
```

### cURL 示例

```bash
# 上传文件
curl -X POST http://localhost:6006/upload_speaker \
  -F "file=@my_voice.wav"

# 返回示例：
# {
#   "status": "success",
#   "file_path": "uploads/speakers/spk_1730000000123.wav",
#   ...
# }

# 使用上传的文件进行 TTS
curl -X POST http://localhost:6006/tts_url \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是测试。",
    "spk_audio_path": "uploads/speakers/spk_1730000000123.wav"
  }' \
  --output output.wav
```

### JavaScript 示例

```javascript
// 上传文件
async function uploadSpeakerAudio(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:6006/upload_speaker', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    
    if (result.status === 'success') {
        console.log('上传成功:', result.file_path);
        return result.file_path;
    } else {
        console.error('上传失败:', result.error);
        return null;
    }
}

// 使用上传的文件生成 TTS
async function generateTTS(filePath, text) {
    const response = await fetch('http://localhost:6006/tts_url', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            text: text,
            spk_audio_path: filePath,
            return_subtitle: false
        })
    });
    
    const audioBlob = await response.blob();
    return URL.createObjectURL(audioBlob);
}

// HTML 表单示例
document.getElementById('uploadForm').onsubmit = async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];
    
    // 上传
    const filePath = await uploadSpeakerAudio(file);
    
    if (filePath) {
        // 生成 TTS
        const audioUrl = await generateTTS(filePath, "你好，这是测试。");
        
        // 播放音频
        const audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = audioUrl;
        audioPlayer.play();
    }
};
```

### HTML 表单示例

```html
<!DOCTYPE html>
<html>
<head>
    <title>说话人音频上传</title>
</head>
<body>
    <h1>上传说话人音频并生成语音</h1>
    
    <form id="uploadForm">
        <label>选择音频文件:</label>
        <input type="file" id="audioFile" accept=".wav,.mp3,.flac" required>
        <br><br>
        
        <label>输入文本:</label>
        <textarea id="textInput" rows="4" cols="50" required>你好，这是测试文本。</textarea>
        <br><br>
        
        <button type="submit">上传并生成</button>
    </form>
    
    <div id="status"></div>
    <audio id="audioPlayer" controls></audio>
    
    <script>
    document.getElementById('uploadForm').onsubmit = async (e) => {
        e.preventDefault();
        
        const statusDiv = document.getElementById('status');
        const fileInput = document.getElementById('audioFile');
        const textInput = document.getElementById('textInput');
        
        statusDiv.textContent = '正在上传...';
        
        // 上传文件
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        const uploadResponse = await fetch('http://localhost:6006/upload_speaker', {
            method: 'POST',
            body: formData
        });
        
        const uploadResult = await uploadResponse.json();
        
        if (uploadResult.status !== 'success') {
            statusDiv.textContent = '上传失败: ' + uploadResult.error;
            return;
        }
        
        statusDiv.textContent = '上传成功！正在生成语音...';
        
        // 生成 TTS
        const ttsResponse = await fetch('http://localhost:6006/tts_url', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                text: textInput.value,
                spk_audio_path: uploadResult.file_path,
                return_subtitle: false
            })
        });
        
        const audioBlob = await ttsResponse.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // 播放音频
        const audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = audioUrl;
        
        statusDiv.textContent = '生成成功！';
    };
    </script>
</body>
</html>
```

## 文件存储

### 存储位置

上传的文件保存在：
```
uploads/speakers/
```

### 文件命名规则

文件名格式：`spk_{原始名称}_{timestamp}{extension}`

示例：
- 上传 `my_voice.wav` → `spk_my_voice_1730000000123.wav`
- 上传 `user-audio.mp3` → `spk_user-audio_1730000001234.mp3`
- 上传 `语音样本.wav` → `spk_audio_1730000002345.wav` (中文字符被过滤)

其中：
- `原始名称`：保留原文件名的字母、数字、下划线和连字符
- `timestamp`：毫秒级时间戳，确保文件名唯一
- 特殊字符和中文会被过滤掉
- 如果过滤后为空，使用默认名称 `audio`

### 文件清理

建议定期清理旧的上传文件：

```python
import os
import time

def clean_old_uploads(directory="uploads/speakers", max_age_hours=24):
    """删除超过指定时间的上传文件"""
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(directory):
        if filename.startswith("spk_"):
            filepath = os.path.join(directory, filename)
            file_age = now - os.path.getmtime(filepath)
            
            if file_age > max_age_seconds:
                os.remove(filepath)
                print(f"删除旧文件: {filename}")

# 定期执行
clean_old_uploads(max_age_hours=24)
```

或使用 cron 任务：
```bash
# 每天凌晨 3 点删除 24 小时前的上传文件
0 3 * * * find /path/to/uploads/speakers -name "spk_*.wav" -mtime +1 -delete
```

## 安全考虑

### 文件大小限制

建议在生产环境中设置文件大小限制：

```python
from fastapi import FastAPI, UploadFile, HTTPException

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@app.post("/upload_speaker")
async def upload_speaker_audio(file: UploadFile = File(...)):
    # 读取文件内容
    file_content = await file.read()
    
    # 检查文件大小
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件太大，最大允许 {MAX_FILE_SIZE / (1024*1024):.1f} MB"
        )
    
    # 继续处理...
```

### 文件类型验证

API 已经内置了文件类型检查，只允许音频格式：
- `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`

### 恶意文件防护

建议添加额外的验证：

```python
import magic  # python-magic 库

def is_valid_audio(file_path):
    """验证文件是否为真实的音频文件"""
    mime = magic.from_file(file_path, mime=True)
    return mime.startswith('audio/')
```

## 常见问题

### Q1: 上传的文件会被永久保存吗？
A: 文件会保存在服务器上，建议定期清理旧文件（参见"文件清理"章节）。

### Q2: 可以上传多大的文件？
A: 默认无限制，建议在生产环境中设置合理的大小限制（如 10-50 MB）。

### Q3: 文件名会保留原始名称吗？
A: 会部分保留。系统会保留原始文件名中的字母、数字、下划线和连字符，并添加时间戳确保唯一性。格式为 `spk_{原始名称}_{时间戳}.{扩展名}`。特殊字符和中文会被过滤掉。

### Q4: 上传失败怎么办？
A: 检查以下几点：
- 文件格式是否支持
- 文件是否损坏
- 服务器磁盘空间是否充足
- 查看错误消息获取详细信息

### Q5: 可以上传多个说话人音频吗？
A: 可以。每次上传都会生成一个新的文件路径，可以保存多个说话人音频。

### Q6: 上传的文件路径在哪里可以看到？
A: 在上传成功的响应中查看 `file_path` 字段。

## 完整工作流程

1. **上传音频** → POST `/upload_speaker`
2. **获取路径** → 从响应中提取 `file_path`
3. **生成语音** → POST `/tts_url` 使用 `file_path` 作为 `spk_audio_path`
4. **获取结果** → 接收生成的音频和/或字幕

## 相关文档

- [API 字幕功能文档](API_SUBTITLE.md)
- [完整示例代码](../examples/upload_speaker_example.py)

## 示例代码

完整的使用示例请参见：
```bash
python examples/upload_speaker_example.py
```

---

**更新日期**: 2025-10-26  
**版本**: 1.0

