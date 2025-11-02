# 故事音频生成 API 使用文档

## 概述

`/generate_story_audio` API 端点用于批量生成故事音频和字幕。它可以将包含故事文本描述的 JSON 文件转换成一个完整的语音文件，并生成对应的字幕数据。

## 功能特性

- ✅ 批量处理故事文本（对话、旁白、心理活动等）
- ✅ 自动根据人物名称和性别选择合适的说话人音频
- ✅ 支持情感控制（开心、悲伤、愤怒等）
- ✅ 自动合并所有音频片段
- ✅ 生成累加时间戳的字幕数据
- ✅ 支持多种文本类型（对话、旁白、心理活动、标题）

## API 接口

### 端点地址

```
POST /generate_story_audio
```

### 请求参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| file | UploadFile | 是 | 故事 JSON 文件 |

### JSON 文件格式

JSON 文件应包含一个数组，每个元素代表一段故事文本：

```json
[
  {
    "type": "c",           // 文本类型
    "text": "我要再去西漠。",  // 要转成语音的文本
    "sex": "wo",           // 性别
    "name": "安妙依",       // 人物名
    "emotion": null        // 情绪状态（可选）
  },
  {
    "type": "s",
    "text": "夜月幽冷，大地空旷...",
    "sex": "unknown",
    "name": "",
    "emotion": null
  }
]
```

#### 字段说明

**type** - 文本类型：
- `c` - 人物对话 (character dialogue)
- `s` - 故事情节描写 (story narrative)
- `m` - 心理活动描写 (mental activity)
- `t` - 标题 (title)
- `unknown` - 未知类型

**sex** - 人物性别：
- `man` - 男性
- `wo` - 女性
- `unknown` - 未知/旁白

**name** - 人物名称
- 具体的人物名称（如 "叶凡"、"安妙依"）
- 空字符串表示旁白或无特定人物

**emotion** - 情绪状态（可选）：
- `happy` - 开心
- `sad` - 悲伤
- `angry` - 愤怒
- `afraid`/`fearful` - 害怕
- `surprised` - 惊讶
- `disgusted` - 厌恶
- `melancholic` - 忧郁
- `calm` - 平静
- `null` - 无特定情绪

### 返回结果

#### 成功响应 (200)

```json
{
  "status": "success",
  "audio": "base64编码的音频数据",
  "audio_format": "wav",
  "sample_rate": 24000,
  "subtitles": [
    {
      "text": "我要再去西漠。",
      "start": 0.0,
      "end": 2.5,
      "speaker": "安妙依",
      "type": "c",
      "emotion": null
    },
    ...
  ],
  "subtitle_count": 150,
  "processed_items": 145,
  "skipped_items": 5,
  "total_duration": 360.5,
  "message": "成功生成 145 个音频片段，总时长 360.50 秒"
}
```

#### 字段说明

- `audio`: base64 编码的 WAV 格式音频数据
- `audio_format`: 音频格式（固定为 "wav"）
- `sample_rate`: 音频采样率
- `subtitles`: 字幕数据数组
  - `text`: 字幕文本
  - `start`: 开始时间（秒）
  - `end`: 结束时间（秒）
  - `speaker`: 说话人名称
  - `type`: 文本类型
  - `emotion`: 情绪状态
- `subtitle_count`: 字幕条目总数
- `processed_items`: 成功处理的文本条目数
- `skipped_items`: 跳过的条目数（空文本等）
- `total_duration`: 总音频时长（秒）

#### 错误响应

```json
{
  "status": "error",
  "error": "错误信息"
}
```

## 说话人音频配置

### 音频查找规则

系统会按以下优先级查找说话人音频：

1. **精确匹配**: 根据 `(sex, name)` 精确匹配
2. **性别默认**: 根据 `(sex, "default")` 匹配
3. **文件名搜索**: 在指定目录中搜索包含名字或性别的音频文件
4. **通用默认**: 使用 `("unknown", "default")` 的默认音频

### 音频搜索目录

```python
SPEAKER_AUDIO_DIRS = [
    "uploads/speakers",  # 用户上传的音频
    "examples",          # 示例音频
    "assets"             # 资源音频
]
```

### 默认音频映射

在 `api_server.py` 中可以配置默认映射：

```python
DEFAULT_SPEAKER_MAP = {
    ("man", "叶凡"): "voice_01.wav",
    ("wo", "安妙依"): "voice_02.wav",
    ("man", "default"): "voice_03.wav",
    ("wo", "default"): "voice_04.wav",
    ("unknown", "default"): "voice_05.wav",
}
```

### 自定义音频映射

要添加新的人物音频：

1. **方法一：添加映射**
   ```python
   DEFAULT_SPEAKER_MAP[("man", "张三")] = "zhang_san.wav"
   ```

2. **方法二：文件命名**
   将音频文件命名为包含人物名称的格式，如：
   - `zhang_san_voice.wav`
   - `voice_zhang_san.wav`
   - `man_zhang_san.wav`

3. **方法三：上传音频**
   使用 `/upload_speaker` API 上传音频后，手动添加映射

## 使用示例

### 1. 使用 Python 客户端

```bash
python api_story_client.py \
  --json-file examples/role_778_True.json \
  --api-url http://localhost:6006 \
  --output-dir outputs
```

### 2. 使用 curl

```bash
curl -X POST "http://localhost:6006/generate_story_audio" \
  -F "file=@examples/role_778_True.json" \
  -o response.json
```

### 3. 使用 Python requests

```python
import requests
import base64

# 读取 JSON 文件
with open('examples/role_778_True.json', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:6006/generate_story_audio',
        files=files
    )

# 解析响应
result = response.json()
if result['status'] == 'success':
    # 保存音频
    audio_data = base64.b64decode(result['audio'])
    with open('output.wav', 'wb') as f:
        f.write(audio_data)
    
    # 保存字幕
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(result['subtitles'], f, ensure_ascii=False, indent=2)
```

### 4. 使用 JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:6006/generate_story_audio', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(result => {
  if (result.status === 'success') {
    // 解码 base64 音频
    const audioData = atob(result.audio);
    const audioBlob = new Blob([audioData], { type: 'audio/wav' });
    
    // 创建下载链接
    const url = URL.createObjectURL(audioBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'story.wav';
    a.click();
  }
});
```

## 客户端脚本说明

提供的 `api_story_client.py` 脚本包含以下功能：

- 上传 JSON 文件到 API
- 自动保存返回的音频文件
- 自动生成 JSON 格式字幕
- 自动生成 SRT 格式字幕（可用于视频编辑软件）

### 输出文件

运行客户端后会生成三个文件：

```
outputs/
├── role_778_True_story.wav      # 合成的音频文件
├── role_778_True_story.json     # JSON 格式字幕
└── role_778_True_story.srt      # SRT 格式字幕
```

### SRT 字幕格式

生成的 SRT 字幕可以直接用于视频编辑软件（如 Premiere、Final Cut Pro 等）：

```srt
1
00:00:00,000 --> 00:00:02,500
[安妙依] 我要再去西漠。

2
00:00:02,500 --> 00:00:08,300
夜月幽冷，大地空旷，唯有风吹过时才有沙沙的声响...
```

## 性能和限制

### 处理时间

- 单个文本条目：约 1-3 秒
- 100 条目：约 2-5 分钟
- 1000 条目：约 20-50 分钟

### 建议

1. **分段处理**: 对于超长故事（>500 条目），建议分段处理
2. **并发限制**: 避免同时提交多个大文件请求
3. **超时设置**: 客户端请求超时建议设置为 10 分钟以上
4. **内存管理**: 服务器需要足够内存来处理长音频

## 故障排查

### 常见问题

**1. "无法找到任何说话人音频文件"**
- 确保 `examples` 目录下有 `voice_01.wav` ~ `voice_05.wav` 等文件
- 检查 `DEFAULT_SPEAKER_MAP` 配置是否正确
- 检查音频文件路径是否存在

**2. "JSON 解析失败"**
- 确认 JSON 文件格式正确
- 检查文件编码（应为 UTF-8）
- 验证 JSON 语法（可使用 jsonlint.com）

**3. "没有成功生成任何音频片段"**
- 检查 JSON 文件中是否所有条目都是空文本
- 查看服务器日志了解具体错误
- 确认 TTS 模型已正确加载

**4. 请求超时**
- 增加客户端超时时间
- 减少 JSON 文件的条目数量
- 检查服务器性能和 GPU 状态

### 调试模式

启动服务器时添加 `--verbose` 参数查看详细日志：

```bash
python api_server.py --verbose
```

## 相关 API

- `/upload_speaker` - 上传说话人音频
- `/tts_url` - 单次 TTS 生成
- `/health` - 健康检查

## 更新日志

### v1.0.0 (2024-11-01)
- ✨ 初始版本
- ✨ 支持批量故事文本转语音
- ✨ 自动音频合并和字幕累加
- ✨ 支持多种文本类型和情感控制

## 许可证

本项目遵循与主项目相同的许可证。

