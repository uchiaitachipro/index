# 故事音频批量生成功能

> 本节内容可添加到主 README.md 中

---

## 🎙️ 故事音频批量生成

新增功能：一键将完整故事文本转换为语音和字幕！

### ✨ 功能特点

- 📚 批量处理整个故事章节
- 🎭 智能匹配角色音色
- 😊 支持情感控制（开心、悲伤、愤怒等）
- 🎵 自动合并所有语音片段
- 📝 自动生成时间同步的字幕
- 📄 输出 JSON 和 SRT 两种字幕格式

### 🚀 快速开始

#### 1. 准备故事 JSON 文件

```json
[
  {
    "type": "c",
    "text": "你好，今天天气真不错。",
    "sex": "man",
    "name": "张三",
    "emotion": "happy"
  },
  {
    "type": "c",
    "text": "是啊，我们出去走走吧。",
    "sex": "wo",
    "name": "李四",
    "emotion": "happy"
  }
]
```

#### 2. 调用 API 生成

```bash
# 使用客户端脚本（推荐）
python api_story_client.py \
  --json-file examples/test_story_short.json \
  --output-dir outputs

# 或使用 curl
curl -X POST "http://localhost:6006/generate_story_audio" \
  -F "file=@story.json" \
  -o response.json
```

#### 3. 获取结果

生成三个文件：
- `story.wav` - 完整音频
- `story.json` - JSON 字幕
- `story.srt` - SRT 字幕

### 📖 详细文档

- [完整 API 文档](STORY_AUDIO_API.md)
- [快速入门指南](QUICKSTART_STORY_API.md)
- [功能概述](STORY_AUDIO_FEATURE.md)

### 🎯 使用场景

- 📚 有声书制作
- 🎬 视频配音
- 🎮 游戏对话生成
- 📻 播客内容制作
- 🎓 教育音频材料

### ⚙️ API 端点

#### POST /generate_story_audio

批量生成故事音频和字幕。

**请求**：上传 JSON 文件

**响应**：
```json
{
  "status": "success",
  "audio": "base64编码的音频",
  "subtitles": [...],
  "total_duration": 360.5,
  "processed_items": 145
}
```

### 📊 性能

| 文本条目 | 处理时间 | 输出时长 |
|---------|----------|----------|
| 10 条   | ~30秒    | ~30秒    |
| 100 条  | ~3分钟   | ~5分钟   |
| 1000 条 | ~30分钟  | ~50分钟  |

*基于 GPU 加速的预估时间*

### 💡 示例

```python
import requests
import base64

# 上传 JSON 文件
with open('story.json', 'rb') as f:
    response = requests.post(
        'http://localhost:6006/generate_story_audio',
        files={'file': f}
    )

result = response.json()

# 保存音频
audio_data = base64.b64decode(result['audio'])
with open('output.wav', 'wb') as f:
    f.write(audio_data)

print(f"✓ 生成完成！时长：{result['total_duration']:.2f}秒")
```

---

**相关链接**：
- [上传说话人音频](UPLOAD_SPEAKER_QUICKSTART.md)
- [字幕功能说明](API_SUBTITLE_UPDATE.md)

