# 故事音频批量生成功能 - 更改说明

## 概述

本次更新为 API 服务器添加了故事音频批量生成功能，允许用户上传包含完整故事文本的 JSON 文件，自动生成合成的语音和字幕。

## 更改日期

2024-11-01

## 新增文件

### 1. 文档文件

| 文件名 | 说明 | 用途 |
|--------|------|------|
| `STORY_AUDIO_API.md` | 完整 API 文档 | 详细的 API 接口说明、参数、返回值等 |
| `QUICKSTART_STORY_API.md` | 快速入门指南 | 帮助用户快速上手使用 |
| `STORY_AUDIO_FEATURE.md` | 功能概述文档 | 功能介绍、使用场景、技术实现等 |
| `STORY_AUDIO_CHANGES.md` | 本文件 | 更改说明和文件清单 |

### 2. 客户端脚本

| 文件名 | 说明 | 功能 |
|--------|------|------|
| `api_story_client.py` | Python 客户端脚本 | 调用 API、保存音频和字幕 |

### 3. 测试文件

| 文件名 | 说明 | 用途 |
|--------|------|------|
| `examples/test_story_short.json` | 短故事测试文件 | 用于快速测试（6个条目） |

## 修改文件

### api_server.py

**修改内容**：

1. **导入新模块**（第 17-18 行）
   ```python
   import numpy as np
   from glob import glob
   ```

2. **新增配置变量**（第 25-39 行）
   ```python
   SPEAKER_AUDIO_DIRS = [...]
   DEFAULT_SPEAKER_MAP = {...}
   ```

3. **新增辅助函数**（第 41-92 行）
   ```python
   def find_speaker_audio(sex: str, name: str) -> Optional[str]:
       """根据性别和名字查找说话人音频文件"""
   ```

4. **新增 API 端点**（第 362-607 行）
   ```python
   @app.post("/generate_story_audio")
   async def generate_story_audio(file: UploadFile = File(...)):
       """批量生成故事音频和字幕"""
   ```

5. **修复参数名**（第 99 行）
   - 从 `args.is_fp16` 改为 `args.fp16`

**新增代码行数**：约 550 行

## 功能详情

### 1. 智能说话人音频查找 (`find_speaker_audio`)

**功能**：
- 根据性别和名字智能查找说话人音频
- 支持精确匹配、性别默认、文件名搜索等多种策略

**查找优先级**：
1. 精确匹配 `(sex, name)`
2. 性别默认 `(sex, "default")`
3. 文件名模糊搜索
4. 通用默认音频

### 2. 故事音频生成 API (`/generate_story_audio`)

**功能**：
- 接收 JSON 文件上传
- 批量处理故事文本
- 自动选择说话人音频
- 支持情感控制
- 合并所有音频片段
- 生成累加时间戳的字幕

**处理流程**：
```
上传 JSON 文件
    ↓
解析 JSON 数据
    ↓
遍历每个文本条目
    ↓
查找说话人音频
    ↓
调用 TTS 生成音频
    ↓
记录字幕数据
    ↓
累加时间戳
    ↓
合并所有音频
    ↓
返回结果
```

### 3. 客户端脚本 (`api_story_client.py`)

**功能**：
- 上传 JSON 文件到 API
- 接收并解析响应
- 保存音频文件（WAV）
- 保存字幕文件（JSON）
- 生成 SRT 格式字幕
- 显示处理进度和统计信息

**使用示例**：
```bash
python api_story_client.py \
  --json-file examples/test_story_short.json \
  --api-url http://localhost:6006 \
  --output-dir outputs
```

## API 接口变更

### 新增端点

#### POST /generate_story_audio

**请求**：
- Content-Type: `multipart/form-data`
- Body: `file` (JSON 文件)

**响应**：
```json
{
  "status": "success",
  "audio": "base64编码的音频",
  "audio_format": "wav",
  "sample_rate": 24000,
  "subtitles": [...],
  "subtitle_count": 150,
  "processed_items": 145,
  "skipped_items": 5,
  "total_duration": 360.5,
  "message": "成功生成 145 个音频片段，总时长 360.50 秒"
}
```

## JSON 文件格式

### 输入格式

```json
[
  {
    "type": "c",           // 文本类型
    "text": "你好世界",     // 文本内容
    "sex": "man",          // 性别
    "name": "张三",        // 人物名
    "emotion": "happy"     // 情感（可选）
  }
]
```

### 支持的字段值

**type**（文本类型）:
- `c` - 人物对话
- `s` - 故事旁白
- `m` - 心理活动
- `t` - 标题
- `unknown` - 未知

**sex**（性别）:
- `man` - 男性
- `wo` - 女性
- `unknown` - 未知/旁白

**emotion**（情感）:
- `happy`, `sad`, `angry`, `surprised`, `afraid`, `fearful`, `calm`, `disgusted`, `melancholic`
- `null` - 默认情感

## 配置说明

### 说话人音频搜索目录

```python
SPEAKER_AUDIO_DIRS = [
    "uploads/speakers",  # 用户上传
    "examples",          # 示例音频
    "assets"             # 资源音频
]
```

### 默认说话人映射

```python
DEFAULT_SPEAKER_MAP = {
    ("man", "叶凡"): "voice_01.wav",
    ("wo", "安妙依"): "voice_02.wav",
    ("man", "default"): "voice_03.wav",
    ("wo", "default"): "voice_04.wav",
    ("unknown", "default"): "voice_05.wav",
}
```

**自定义方法**：
1. 编辑 `api_server.py` 中的 `DEFAULT_SPEAKER_MAP`
2. 添加新的 `(sex, name): "audio_file.wav"` 映射
3. 确保音频文件存在于搜索目录中
4. 重启服务器

## 依赖项

### 新增 Python 依赖

- `numpy` - 音频数据处理和合并
- `glob` - 文件搜索（Python 标准库）

**无需额外安装**，这些依赖已包含在项目中。

## 使用示例

### 1. 启动服务器

```bash
python api_server.py --verbose
```

### 2. 快速测试

```bash
# 使用客户端脚本
python api_story_client.py \
  --json-file examples/test_story_short.json \
  --output-dir outputs

# 使用 curl
curl -X POST "http://localhost:6006/generate_story_audio" \
  -F "file=@examples/test_story_short.json" \
  -o response.json
```

### 3. 处理完整故事

```bash
python api_story_client.py \
  --json-file examples/role_778_True.json \
  --output-dir outputs
```

## 输出文件

每次成功运行后会生成三个文件：

```
outputs/
├── [名称]_story.wav   # 合成的音频文件
├── [名称]_story.json  # JSON 格式字幕（带元数据）
└── [名称]_story.srt   # SRT 格式字幕（标准格式）
```

## 兼容性

### 向后兼容

✅ **完全兼容**
- 不影响现有 API 端点
- 不修改现有功能
- 仅新增功能

### 服务器要求

- Python 3.8+
- 已有的 TTS 模型和依赖
- 建议 8GB+ 内存
- GPU 加速（可选，但推荐）

## 性能

### 处理速度（基于 GPU）

| 条目数 | 预计时间 | 输出时长 |
|--------|----------|----------|
| 10     | ~30秒    | ~30秒    |
| 100    | ~3分钟   | ~5分钟   |
| 500    | ~15分钟  | ~25分钟  |
| 1000   | ~30分钟  | ~50分钟  |

### 优化建议

1. 使用 `--fp16` 启动服务器
2. 启用 CUDA kernel（如支持）
3. 分段处理超长故事
4. 使用 GPU 加速

## 已知限制

1. **单个请求处理时间**
   - 长故事可能需要较长时间
   - 建议客户端设置 10 分钟以上超时

2. **内存使用**
   - 长音频会占用较多内存
   - 建议服务器至少 8GB 内存

3. **并发处理**
   - API 按顺序处理请求
   - 建议避免同时提交多个大文件

4. **音频质量**
   - 依赖说话人音频样本质量
   - 建议使用清晰、无噪音的音频

## 测试

### 测试文件

提供了两个测试文件：

1. **test_story_short.json**（6个条目）
   - 用于快速功能测试
   - 处理时间：约 30 秒

2. **role_778_True.json**（1874个条目）
   - 完整故事章节
   - 处理时间：约 30-60 分钟

### 测试步骤

```bash
# 1. 启动服务器
python api_server.py --verbose

# 2. 运行快速测试
python api_story_client.py \
  --json-file examples/test_story_short.json \
  --output-dir outputs

# 3. 检查输出文件
ls -lh outputs/test_story_short_story.*

# 4. 播放音频
afplay outputs/test_story_short_story.wav  # macOS
# 或
aplay outputs/test_story_short_story.wav   # Linux

# 5. 查看字幕
cat outputs/test_story_short_story.srt
```

## 文档

### 完整文档列表

1. **STORY_AUDIO_API.md** - 完整 API 文档
   - API 接口详细说明
   - 参数和返回值
   - 配置说明
   - 故障排查

2. **QUICKSTART_STORY_API.md** - 快速入门
   - 快速开始步骤
   - 常用命令
   - 示例代码

3. **STORY_AUDIO_FEATURE.md** - 功能概述
   - 功能介绍
   - 使用场景
   - 技术实现

4. **STORY_AUDIO_CHANGES.md** - 本文件
   - 更改清单
   - 文件说明
   - 技术细节

## 未来改进

可能的改进方向：

1. ✨ 支持多种输出格式（MP3、OGG 等）
2. ✨ 支持流式返回（边生成边返回）
3. ✨ 添加进度查询接口
4. ✨ 支持音频效果（混响、均衡器等）
5. ✨ 支持背景音乐混合
6. ✨ 支持更多字幕格式（ASS、VTT 等）
7. ✨ 添加音频质量选项
8. ✨ 支持批量处理队列

## 技术支持

### 遇到问题？

1. 查看服务器日志（使用 `--verbose`）
2. 检查 JSON 文件格式
3. 确认说话人音频配置
4. 查看完整文档

### 反馈和建议

欢迎提供反馈和改进建议！

## 总结

本次更新为系统添加了强大的批量故事音频生成功能，通过智能音频匹配、自动合并和字幕累加，极大地提升了批量内容制作的效率。

### 主要优势

- ✅ 批量处理，提高效率
- ✅ 智能匹配说话人音频
- ✅ 自动合并和字幕生成
- ✅ 支持情感控制
- ✅ 完整的文档和示例
- ✅ 向后兼容现有功能

---

**更新时间**: 2024-11-01  
**版本**: v1.0.0

