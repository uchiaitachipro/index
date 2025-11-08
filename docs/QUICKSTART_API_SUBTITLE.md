# API 字幕功能 - 快速开始

## 5 分钟快速入门

### 1. 启动服务器 (30 秒)

```bash
# 基本启动
python api_server.py

# 或使用 FP16 加速（推荐）
python api_server.py --is_fp16
```

等待看到 "Application startup complete" 消息。

### 2. 测试服务器 (10 秒)

```bash
curl http://localhost:6006/health
```

应该返回：
```json
{"status": "healthy", "message": "Service is running", ...}
```

### 3. 第一个带字幕的请求 (2 分钟)

创建 `test_subtitle.py`：

```python
import requests
import base64
import json

# 发送请求
response = requests.post("http://localhost:6006/tts_url", json={
    "text": "你好，这是测试。今天天气很好。",
    "spk_audio_path": "examples/voice_02.wav",
    "return_subtitle": True
})

result = response.json()

# 保存音频
audio_bytes = base64.b64decode(result['audio'])
with open("output.wav", "wb") as f:
    f.write(audio_bytes)

# 打印字幕
print(f"生成了 {result['subtitle_count']} 个字幕片段：\n")
for i, sub in enumerate(result['subtitles'], 1):
    print(f"{i}. [{sub['time_begin']:.0f}ms - {sub['time_end']:.0f}ms]")
    print(f"   {sub['text']}\n")

# 保存字幕
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(result['subtitles'], f, ensure_ascii=False, indent=2)

print("✓ 音频已保存到: output.wav")
print("✓ 字幕已保存到: output.json")
```

运行：
```bash
python test_subtitle.py
```

### 4. 运行完整示例 (2 分钟)

```bash
python examples/api_subtitle_client.py
```

这将运行 4 个示例：
- ✓ 基本字幕生成
- ✓ 不返回字幕
- ✓ 长文本处理
- ✓ 带情感控制

输出将保存在 `outputs/` 目录。

## 核心概念

### 两种模式

**模式 1: 仅音频**（默认）
```python
response = requests.post(url, json={
    "text": "你好",
    "spk_audio_path": "voice.wav"
    # return_subtitle 默认为 False
})
audio = response.content  # 直接得到 WAV 字节流
```

**模式 2: 音频 + 字幕**
```python
response = requests.post(url, json={
    "text": "你好",
    "spk_audio_path": "voice.wav",
    "return_subtitle": True  # 关键！
})
result = response.json()
audio = base64.b64decode(result['audio'])
subtitles = result['subtitles']
```

### 字幕数据结构

```python
{
  "text": "你好，今天天气很好。",      # 原始文本
  "pronounce_text": "你好，今天天气很好。",  # 处理后文本
  "time_begin": 0.0,                  # 开始时间（毫秒）
  "time_end": 2856.8                  # 结束时间（毫秒）
}
```

## 常用代码片段

### 保存为 SRT 字幕

```python
def save_srt(subtitles, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles, 1):
            ms_start = sub['time_begin']
            ms_end = sub['time_end']
            
            # 转换为 SRT 时间格式
            start = format_time(ms_start)
            end = format_time(ms_end)
            
            f.write(f"{i}\n{start} --> {end}\n{sub['text']}\n\n")

def format_time(ms):
    s = int(ms / 1000)
    ms = int(ms % 1000)
    h, m = s // 3600, (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
```

### 播放音频并显示字幕

```python
import soundfile as sf
import time

# 加载音频
audio, sr = sf.read("output.wav")

# 播放并显示字幕
start_time = time.time()
for subtitle in subtitles:
    # 等到字幕开始时间
    while (time.time() - start_time) * 1000 < subtitle['time_begin']:
        time.sleep(0.01)
    
    # 显示字幕
    print(f"\r{subtitle['text']}", end='', flush=True)
    
    # 等到字幕结束
    time.sleep((subtitle['time_end'] - subtitle['time_begin']) / 1000)
```

### 合并短字幕

```python
def merge_short_subtitles(subtitles, min_duration_ms=1000):
    """合并时长小于 min_duration_ms 的字幕"""
    merged = []
    current = None
    
    for sub in subtitles:
        duration = sub['time_end'] - sub['time_begin']
        
        if current is None:
            current = sub.copy()
        elif duration < min_duration_ms:
            # 合并到当前字幕
            current['text'] += sub['text']
            current['pronounce_text'] += sub['pronounce_text']
            current['time_end'] = sub['time_end']
        else:
            merged.append(current)
            current = sub.copy()
    
    if current:
        merged.append(current)
    
    return merged
```

## 实用技巧

### 1. 控制分段长度

```python
# 生成更多短片段（更精确的字幕）
data = {
    "text": long_text,
    "spk_audio_path": "voice.wav",
    "return_subtitle": True,
    "max_text_tokens_per_sentence": 50  # 较小的值
}

# 生成更少长片段（更少的字幕）
data["max_text_tokens_per_sentence"] = 200  # 较大的值
```

### 2. 添加情感

```python
data = {
    "text": "太棒了！",
    "spk_audio_path": "voice.wav",
    "return_subtitle": True,
    "emo_control_method": 2,
    "emo_vec": [0.5, 0, 0, 0, 0, 0, 0, 0.3]  # 快乐 + 平静
}
```

情感向量顺序：`[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`

### 3. 批量处理

```python
texts = ["第一段文本", "第二段文本", "第三段文本"]

for i, text in enumerate(texts):
    response = requests.post(url, json={
        "text": text,
        "spk_audio_path": "voice.wav",
        "return_subtitle": True
    })
    
    result = response.json()
    
    # 保存
    audio = base64.b64decode(result['audio'])
    with open(f"output_{i}.wav", "wb") as f:
        f.write(audio)
    
    with open(f"subtitle_{i}.json", "w") as f:
        json.dump(result['subtitles'], f, indent=2)
```

## 故障排除

### 问题 1: 连接被拒绝
```
requests.exceptions.ConnectionError: Connection refused
```

**解决**：确保服务器正在运行
```bash
python api_server.py --is_fp16
```

### 问题 2: 返回的是二进制而不是 JSON
```
JSONDecodeError: Expecting value
```

**解决**：检查是否设置了 `return_subtitle=True`
```python
data["return_subtitle"] = True  # 必须显式设置
```

### 问题 3: 字幕为空
```python
result['subtitle_count'] == 0
```

**解决**：检查是否使用了 `IndexTTS2SubTitle` 类
```python
# api_server.py 中应该是：
from indextts.infer_v2_subtitle import IndexTTS2SubTitle
tts = IndexTTS2SubTitle(...)  # 而不是 IndexTTS2
```

### 问题 4: 音频解码失败
```
binascii.Error: Invalid base64-encoded string
```

**解决**：确保正确解码
```python
import base64
audio_bytes = base64.b64decode(result['audio'])
```

## 下一步

- 📖 阅读完整 API 文档：`docs/API_SUBTITLE.md`
- 🧪 运行自动化测试：`python tests/test_api_subtitle.py`
- 💡 查看更多示例：`examples/api_subtitle_client.py`
- 🎯 了解字幕功能细节：`docs/subtitle_feature_zh.md`

## 获取帮助

遇到问题？检查：
1. 服务器是否正常运行：`curl http://localhost:6006/health`
2. 音频文件路径是否正确
3. 查看服务器日志输出
4. 运行测试：`python tests/test_api_subtitle.py`

---

**祝您使用愉快！** 🎉

