# 故事音频批量生成功能

## 功能概述

新增的故事音频批量生成功能允许用户上传一个包含完整故事文本的 JSON 文件，系统会自动：

1. 🎤 根据人物性别和名称选择合适的说话人音频
2. 🎭 根据情感状态调整语音表现
3. 🔊 将每段文本转换为语音
4. 📝 记录每段语音的字幕时间戳
5. 🎵 将所有语音片段合并成一个完整的音频文件
6. ⏱️ 自动累加字幕时间戳
7. 📄 生成 JSON 和 SRT 两种格式的字幕文件

## 使用场景

- 📚 **有声书制作**：将小说章节转换为有声读物
- 🎬 **视频配音**：为视频内容生成配音和字幕
- 🎮 **游戏对话**：批量生成游戏角色对话
- 📻 **播客内容**：生成播客剧本音频
- 🎓 **教育内容**：制作教学音频材料

## 新增文件

### 服务端
- `api_server.py` - 更新，添加了 `/generate_story_audio` API 端点
  - 新增 `find_speaker_audio()` 函数：智能查找说话人音频
  - 新增 `generate_story_audio()` API：批量生成故事音频

### 客户端
- `api_story_client.py` - Python 客户端脚本
  - 上传 JSON 文件
  - 自动保存音频和字幕
  - 生成 SRT 格式字幕

### 文档
- `STORY_AUDIO_API.md` - 完整 API 文档
- `QUICKSTART_STORY_API.md` - 快速入门指南
- `STORY_AUDIO_FEATURE.md` - 本文件，功能概述

### 测试文件
- `examples/test_story_short.json` - 短故事测试文件（6个条目）

## API 端点

### POST /generate_story_audio

批量生成故事音频和字幕。

**请求**：
- 上传 JSON 文件，包含故事文本数组

**响应**：
- `audio`: base64 编码的 WAV 音频
- `subtitles`: 字幕数据（包含累加的时间戳）
- `total_duration`: 总时长
- `processed_items`: 处理的条目数

## JSON 文件格式

```json
[
  {
    "type": "c",           // 文本类型：c=对话, s=旁白, m=心理, t=标题
    "text": "你好世界",     // 要转换的文本
    "sex": "man",          // 性别：man/wo/unknown
    "name": "张三",        // 人物名称
    "emotion": "happy"     // 情感（可选）
  }
]
```

## 核心特性

### 1. 智能说话人匹配

系统会按以下优先级查找说话人音频：

```
精确匹配 (性别+名字)
    ↓
性别默认匹配
    ↓
文件名模糊搜索
    ↓
通用默认音频
```

### 2. 情感控制

支持多种情感状态：
- happy（开心）
- sad（悲伤）
- angry（愤怒）
- surprised（惊讶）
- afraid/fearful（害怕）
- calm（平静）
- disgusted（厌恶）
- melancholic（忧郁）

### 3. 音频合并

- 自动合并所有音频片段
- 处理不同采样率（自动重采样）
- 无缝拼接，无停顿

### 4. 字幕累加

- 自动计算每段音频时长
- 累加时间戳（start/end）
- 支持 JSON 和 SRT 两种格式

### 5. 错误处理

- 跳过空文本条目
- 处理失败时继续下一条
- 详细的错误日志
- 返回处理统计信息

## 配置说明

### 说话人音频目录

```python
SPEAKER_AUDIO_DIRS = [
    "uploads/speakers",  # 用户上传的音频
    "examples",          # 示例音频
    "assets"             # 资源音频
]
```

### 默认音频映射

```python
DEFAULT_SPEAKER_MAP = {
    ("man", "叶凡"): "voice_01.wav",
    ("wo", "安妙依"): "voice_02.wav",
    ("man", "default"): "voice_03.wav",
    ("wo", "default"): "voice_04.wav",
    ("unknown", "default"): "voice_05.wav",
}
```

## 使用示例

### 快速测试

```bash
# 1. 启动服务器
python api_server.py --verbose

# 2. 生成音频
python api_story_client.py \
  --json-file examples/test_story_short.json \
  --output-dir outputs

# 3. 播放结果
afplay outputs/test_story_short_story.wav
```

### 处理长故事

```bash
# 处理完整章节（可能需要较长时间）
python api_story_client.py \
  --json-file examples/role_778_True.json \
  --output-dir outputs
```

### Python 代码集成

```python
import requests
import base64

with open('story.json', 'rb') as f:
    response = requests.post(
        'http://localhost:6006/generate_story_audio',
        files={'file': f},
        timeout=600
    )

result = response.json()
if result['status'] == 'success':
    # 保存音频
    audio_data = base64.b64decode(result['audio'])
    with open('story.wav', 'wb') as f:
        f.write(audio_data)
    
    print(f"生成成功！时长：{result['total_duration']:.2f}秒")
```

## 输出文件

运行后生成三个文件：

```
outputs/
├── story_name.wav   # 合成的完整音频
├── story_name.json  # JSON 格式字幕（带元数据）
└── story_name.srt   # SRT 格式字幕（标准格式）
```

### JSON 字幕格式

```json
[
  {
    "text": "你好世界",
    "start": 0.0,
    "end": 2.5,
    "speaker": "张三",
    "type": "c",
    "emotion": "happy"
  }
]
```

### SRT 字幕格式

```srt
1
00:00:00,000 --> 00:00:02,500
[张三] 你好世界
```

## 性能指标

基于实际测试（使用 GPU）：

| 条目数 | 预估时间 | 输出时长 |
|--------|----------|----------|
| 10     | ~30秒    | ~30秒    |
| 100    | ~3分钟   | ~5分钟   |
| 500    | ~15分钟  | ~25分钟  |
| 1000   | ~30分钟  | ~50分钟  |

*实际时间取决于硬件性能和文本长度*

## 技术实现

### 异步处理

```python
# 使用 asyncio 避免阻塞
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(
    None,
    lambda: tts.infer(...)
)
```

### 音频合并

```python
# 使用 numpy 高效合并
final_audio = np.concatenate([
    audio_segment for _, audio_segment in all_audio_segments
])
```

### 时间累加

```python
# 累加字幕时间戳
for subtitle in subtitles:
    adjusted_subtitle = {
        "text": subtitle["text"],
        "start": subtitle["start"] + current_time,
        "end": subtitle["end"] + current_time,
        ...
    }
current_time += segment_duration
```

## 扩展性

### 添加新角色

1. 准备说话人音频样本
2. 上传到服务器
3. 更新 `DEFAULT_SPEAKER_MAP`
4. 重启服务器

### 自定义文本类型

可以在 JSON 中使用任意 `type` 值，系统会保留该信息在字幕中。

### 批量处理

可以编写脚本批量处理多个 JSON 文件：

```bash
for file in stories/*.json; do
    python api_story_client.py --json-file "$file" --output-dir outputs
done
```

## 注意事项

⚠️ **长文本处理**
- 超过 500 条目的文件建议分段处理
- 设置足够的客户端超时时间（推荐 10 分钟+）

⚠️ **内存使用**
- 长音频会占用较多内存
- 建议服务器至少有 8GB 内存

⚠️ **音频质量**
- 说话人音频样本质量直接影响生成效果
- 推荐使用清晰、无背景噪音的音频
- 样本时长建议 3-10 秒

⚠️ **并发限制**
- 避免同时提交多个大文件请求
- API 是同步处理的，会按顺序执行

## 相关文档

- [完整 API 文档](STORY_AUDIO_API.md)
- [快速入门指南](QUICKSTART_STORY_API.md)
- [上传说话人音频](UPLOAD_SPEAKER_QUICKSTART.md)
- [字幕功能说明](API_SUBTITLE_UPDATE.md)

## 更新日志

### v1.0.0 (2024-11-01)
- ✨ 新增 `/generate_story_audio` API 端点
- ✨ 智能说话人音频匹配
- ✨ 自动音频合并
- ✨ 字幕时间累加
- ✨ 支持 JSON 和 SRT 两种字幕格式
- 📝 完整文档和示例

## 反馈与改进

如有建议或问题，欢迎反馈！

## 许可证

本功能遵循项目主许可证。

