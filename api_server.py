import os
import asyncio
import io
import base64
from tabnanny import verbose
import traceback
from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import time
import soundfile as sf
from typing import List, Optional, Union
import numpy as np
from glob import glob

from indextts.infer_v2_subtitle import IndexTTS2SubTitle

tts = None
api_verbose = False

# 说话人音频查找配置
SPEAKER_AUDIO_DIRS = [
    "uploads/speakers",
    "examples",
]

# 默认说话人音频映射（可根据实际情况修改）
DEFAULT_SPEAKER_MAP = {
    ("man", "叶凡"): "fanqie_man_main_role_12x.mp3",
    ("wo", "安妙依"): "fanqie_woman_yaorao_12x.mp3",
    ("woman", "安妙依"): "fanqie_woman_yaorao_12x.mp3",
    ("wo", "姬紫月"): "minmax_young_girl_12x.mp3",
     ("woman", "姬紫月"): "minmax_young_girl_12x.mp3",
    ("man", "default"): "minmax_yongth_qingche_12x.mp3",
    ("wo", "default"): "minmax_yongth_qingche_12x.mp3",
    ("woman", "default"): "minmax_yongth_qingche_12x.mp3",
    ("unknown", "default"): "fanqie_speaker_default_12x.mp3"
}

def find_speaker_audio(sex: str, name: str) -> Optional[str]:
    """
    根据性别和名字查找说话人音频文件
    
    Args:
        sex: 性别 (man, wo, unknown)
        name: 人物名
        
    Returns:
        音频文件路径，如果未找到则返回 None
    """
    # 尝试精确匹配
    key = (sex, name)
    if key in DEFAULT_SPEAKER_MAP:
        filename = DEFAULT_SPEAKER_MAP[key]
        for dir_path in SPEAKER_AUDIO_DIRS:
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                return file_path
    
    # 尝试按性别默认匹配
    key = (sex, "default")
    if key in DEFAULT_SPEAKER_MAP:
        filename = DEFAULT_SPEAKER_MAP[key]
        for dir_path in SPEAKER_AUDIO_DIRS:
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                return file_path
    
    # 搜索目录中包含名字或性别的文件
    search_terms = [name.lower(), sex.lower()] if name else [sex.lower()]
    for dir_path in SPEAKER_AUDIO_DIRS:
        if not os.path.exists(dir_path):
            continue
        for ext in [".wav", ".mp3", ".flac"]:
            files = glob(os.path.join(dir_path, f"*{ext}"))
            for file_path in files:
                filename = os.path.basename(file_path).lower()
                for term in search_terms:
                    if term and term in filename:
                        return file_path
    
    # 返回通用默认音频
    key = ("unknown", "default")
    if key in DEFAULT_SPEAKER_MAP:
        filename = DEFAULT_SPEAKER_MAP[key]
        for dir_path in SPEAKER_AUDIO_DIRS:
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                return file_path
    
    return None


def find_speaker_audio_with_emotion(
    sex: str, 
    name: str, 
    type: str,
    emotion: Optional[str] = None,
    custom_voice_map: Optional[list] = None
) -> Optional[str]:
    """
    根据性别、名字和情感查找说话人音频文件（支持自定义音色映射）
    
    查找优先级：
    1. 自定义 voice_map 中的 (name, sex, emotion) 精确匹配
    2. 自定义 voice_map 中的 (name, sex, emotion=null) 默认匹配
    3. DEFAULT_SPEAKER_MAP 中的匹配
    4. 文件名搜索
    5. 通用默认
    
    Args:
        sex: 性别 (man, wo, unknown)
        name: 人物名
        emotion: 情感状态 (happy, sad, angry 等)
        custom_voice_map: 自定义音色映射列表
        
    Returns:
        音频文件路径，如果未找到则返回 None
    """
    if type == "unknown" or type == "s" or type == "story":
        return find_speaker_audio("unknown", "default")
    
    # 1. 优先使用自定义音色映射
    if custom_voice_map:
        # 1.1 尝试精确匹配 (name, sex, emotion)
        if emotion:
            for mapping in custom_voice_map:
                if (mapping.get("name") == name and 
                    mapping.get("sex") == sex and 
                    mapping.get("emotion") == emotion):
                    voice_path = mapping.get("voice_id")
                    if voice_path and os.path.exists(voice_path):
                        return voice_path
        
        # 1.2 尝试匹配 (name, sex, emotion=null) - 人物默认音色
        for mapping in custom_voice_map:
            if (mapping.get("name") == name and 
                mapping.get("sex") == sex and 
                mapping.get("emotion") is None):
                voice_path = mapping.get("voice_id")
                if voice_path and os.path.exists(voice_path):
                    return voice_path
    
    # 2. 使用原有的 find_speaker_audio 逻辑（DEFAULT_SPEAKER_MAP + 文件搜索）
    return find_speaker_audio(sex, name)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2SubTitle(
        model_dir=args.model_dir,
        # use_fp16=args.fp16,
        use_fp16=True,

        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
    )
    global api_verbose
    api_verbose = args.verbose
    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time()
        }
    )


@app.post("/upload_speaker", responses={
    200: {"content": {"application/json": {}}},
    400: {"content": {"application/json": {}}},
    500: {"content": {"application/json": {}}}
})
async def upload_speaker_audio(file: UploadFile = File(...)):
    """
    上传说话人参考音频文件
    
    用于上传 spk_audio 文件到服务器，返回的文件路径可以在 /tts_url 接口的 spk_audio_path 参数中使用。
    
    请求参数：
    - file (UploadFile, 必需): 音频文件（支持 wav, mp3, flac 等格式）
    
    返回：
    - status (string): 状态 "success" 或 "error"
    - file_path (string): 保存的文件路径（可用于 tts_api_url）
    - filename (string): 文件名
    - file_size (int): 文件大小（字节）
    - timestamp (float): 上传时间戳
    """
    try:
        # 检查文件是否存在
        if not file or not file.filename:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": "未提供文件或文件名为空"
                }
            )
        
        # 检查文件扩展名
        allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": f"不支持的文件格式: {file_ext}，支持的格式: {', '.join(allowed_extensions)}"
                }
            )
        
        # 创建上传目录
        upload_dir = "uploads/speakers"
        os.makedirs(upload_dir, exist_ok=True)
        
        # 生成唯一文件名（保留原始名称 + 时间戳）
        timestamp = int(time.time() * 1000)
        # 获取原始文件名（不含扩展名）
        original_name = os.path.splitext(file.filename)[0]
        # 清理文件名，只保留字母、数字、下划线和连字符
        safe_original_name = "".join(c for c in original_name if c.isalnum() or c in ('_', '-'))
        # 如果清理后为空，使用默认名称
        if not safe_original_name:
            safe_original_name = "audio"
        safe_filename = f"spk_{safe_original_name}_{timestamp}{file_ext}"
        file_path = os.path.join(upload_dir, safe_filename)
        
        # 保存文件
        file_content = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        file_size = len(file_content)
        
        print(f">> Uploaded speaker audio: {file_path} ({file_size} bytes)")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "file_path": file_path,
                "filename": safe_filename,
                "file_size": file_size,
                "timestamp": time.time(),
                "message": f"文件已成功上传到 {file_path}"
            }
        )
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/tts_url", responses={
    200: {"content": {
        "application/octet-stream": {},
        "application/json": {}
    }},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    """
    TTS API 端点（带字幕支持）
    
    使用 IndexTTS2SubTitle 模型生成语音，支持两种返回格式：
    1. return_subtitle=False（默认）：返回音频字节流（WAV 格式）
    2. return_subtitle=True：返回 JSON，包含 base64 编码的音频和字幕信息
    
    请求参数：
    - text (string, 必需): 要合成的文本
    - spk_audio_path (string, 必需): 说话人参考音频路径
    - return_subtitle (boolean, 可选): 是否返回字幕信息，默认 False
    - emo_control_method (int, 可选): 情感控制方法，0-3，默认 0
    - emo_ref_path (string, 可选): 情感参考音频路径
    - emo_weight (float, 可选): 情感权重，默认 1.0
    - emo_vec (array, 可选): 情感向量 [happy,angry,sad,afraid,disgusted,melancholic,surprised,calm]
    - emo_text (string, 可选): 情感文本（用于自动检测情感）
    - emo_random (boolean, 可选): 是否随机情感，默认 False
    - max_text_tokens_per_sentence (int, 可选): 每句最大 token 数，默认 120
    
    返回：
    - return_subtitle=False: audio/wav 字节流
    - return_subtitle=True: JSON 对象，包含 audio(base64), subtitles 等字段
    """
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)
        return_subtitle = data.get("return_subtitle", False)  # 新增：是否返回字幕

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        else:
            vec = None

        print(f"Emo control mode:{emo_control_method},vec:{vec},return_subtitle:{return_subtitle}")
        global api_verbose
        
        # 调用 TTS 推理（在线程池中运行，避免阻塞事件循环）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # 使用默认线程池
            lambda: tts.infer(
                spk_audio_prompt=spk_audio_path, 
                text=text,
                output_path=None,
                emo_audio_prompt=emo_ref_path, 
                emo_alpha=emo_weight,
                emo_vector=vec,
                use_emo_text=(emo_control_method==3), 
                emo_text=emo_text,
                use_random=emo_random,
                max_text_tokens_per_segment=int(max_text_tokens_per_sentence),
                verbose=api_verbose
            )
        )
        
        # 解析返回结果（IndexTTS2SubTitle 总是返回字典格式）
        audio_data = result['audio_data']
        sr, wav = audio_data
        subtitles = result.get('subtitles', [])
        
        # 生成音频字节流
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()
        
        if return_subtitle:
            # 返回 JSON 格式（包含 base64 编码的音频和字幕）
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "audio": audio_base64,
                    "audio_format": "wav",
                    "sample_rate": int(sr),
                    "subtitles": subtitles,
                    "subtitle_count": len(subtitles)
                }
            )
        else:
            # 返回音频字节流
            return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


async def _process_story_audio_generation(story_data: list, custom_voice_map: Optional[list] = None) -> dict:
    """
    核心音频生成逻辑（内部函数）
    
    参数：
    - story_data: 故事数据列表
    - custom_voice_map: 自定义音色映射列表
    
    返回：
    - dict: 包含生成结果的字典
    
    注意：
    - 音频片段之间会添加 80ms 静音间隔
    - 字幕时间戳已自动调整以匹配静音间隔
    - 每个音频片段会添加 10ms 淡入淡出效果
    - 使用 PyTorch 进行高效合成（降级方案为 NumPy）
    """
    if not isinstance(story_data, list):
        raise ValueError("story_data 必须是数组格式")
    
    print(f">> 开始处理故事音频生成，共 {len(story_data)} 个条目")
    
    # 用于存储所有音频片段和字幕
    all_audio_segments = []
    all_subtitles = []
    current_time = 0.0  # 单位：毫秒
    processed_count = 0
    skipped_count = 0
    
    # 静音间隔配置（毫秒）
    silence_interval_ms = 80.0  # 80ms 静音间隔（从50ms增加以减少杂音）
    
    global tts, api_verbose
    loop = asyncio.get_event_loop()
    
    # 定义中文标点符号集合
    chinese_punctuation = set('。，、""''：；？⁇！…—·《》〈〉【】「」『』（）(),.;:!?\'"…“”\n\r\t')
    
    for idx, item in enumerate(story_data):
        text = item.get("text", "").strip()
        item_type = item.get("type", "unknown")
        sex = item.get("sex", "unknown")
        name = item.get("name", "")
        emotion = item.get("emotion", None)
        
        # 去除文本首尾的标点符号
        while text and text[0] in chinese_punctuation:
            text = text[1:]
        while text and text[-1] in chinese_punctuation:
            text = text[:-1]
        text = text.strip()
        
        # 跳过空文本或 unknown 类型且文本为空的条目
        if not text or (item_type == "unknown" and not text):
            skipped_count += 1
            if api_verbose:
                print(f"  [{idx+1}/{len(story_data)}] 跳过空条目")
            continue
        
        # 跳过只包含标点符号的文本
        text_without_punctuation = ''.join(c for c in text if c not in chinese_punctuation)
        if not text_without_punctuation:
            skipped_count += 1
            if api_verbose:
                print(f"  [{idx+1}/{len(story_data)}] 跳过纯标点符号条目: {text}")
            continue
        
        # 查找说话人音频（优先使用自定义音色映射）
        spk_audio_path = find_speaker_audio_with_emotion(
            sex=sex,
            name=name,
            type=item_type,
            emotion=emotion,
            custom_voice_map=custom_voice_map
        )
        
        if not spk_audio_path:
            print(f"  警告: 未找到 sex={sex}, name={name}, emotion={emotion} 的音频，使用默认音频")
            spk_audio_path = find_speaker_audio("unknown", "default")
            if not spk_audio_path:
                raise FileNotFoundError("无法找到任何说话人音频文件")
        
        # 如果使用了自定义音色映射，在 verbose 模式下显示匹配信息
        if api_verbose and custom_voice_map:
            print(f"    使用音色文件: {spk_audio_path}")
        
        # 准备情感参数
        emo_vec = [0] * 8
        emo_weight = 0.15
        emo_control_method = 0
        
        if emotion and emotion.lower() != "unknown":
            # 情感映射到向量 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emotion_map = {
                "happy": 0,       # 高兴
                "angry": 1,       # 愤怒
                "sad": 2,         # 悲伤
                "afraid": 3,      # 害怕
                "disgusted": 4,   # 厌恶
                "melancholic": 5, # 忧郁
                "surprised": 6,   # 惊讶
                "calm": 7,        # 冷静/平静
                "fearful": 3,     # 害怕（映射到 afraid）
            }
            
            emotion_lower = emotion.lower()
            if emotion_lower in emotion_map:
                emo_idx = emotion_map[emotion_lower]
                emo_vec[emo_idx] = 1.0
                emo_control_method = 2
            else:
                # 未识别的情感，记录警告但继续处理
                if api_verbose:
                    print(f"    警告: 未识别的情感 '{emotion}'，将使用默认情感")
        
        print(f"  [{idx+1}/{len(story_data)}] 处理: type={item_type}, name={name}, text={text[:30]}...")
        
        # 调用 TTS 生成音频（在线程池中运行，失败时最多重试 3 次）
        max_retries = 3
        success = False
        last_error = None
        
        for retry in range(max_retries):
            try:
                if retry > 0:
                    print(f"    重试 {retry}/{max_retries-1}...")
                    # 重试前短暂等待
                    await asyncio.sleep(1)
                
                result = await loop.run_in_executor(
                    None,
                    lambda: tts.infer(
                        spk_audio_prompt=spk_audio_path,
                        text=text,
                        output_path=None,
                        emo_alpha=emo_weight,
                        emo_vector=emo_vec if emo_control_method == 2 else None,
                        use_emo_text=False,
                        use_random=False,
                        max_text_tokens_per_segment=120,
                        verbose=api_verbose
                    )
                )
                
                # 解析结果
                audio_data = result['audio_data']
                sr, wav = audio_data
                subtitles = result.get('subtitles', [])
                
                # 计算当前音频片段时长（毫秒）
                segment_duration = len(wav) / sr * 1000.0
                
                # 存储音频片段
                all_audio_segments.append((sr, wav))
                
                # 调整字幕时间并添加到总列表
                for subtitle in subtitles:
                    adjusted_subtitle = {
                        "text": subtitle["text"],
                        "start": subtitle["time_begin"] + current_time,
                        "end": subtitle["time_end"] + current_time,
                        "pronounce_text": subtitle["pronounce_text"],
                        "speaker": name if name else "旁白",
                        "type": item_type,
                        "emotion": emotion
                    }
                    all_subtitles.append(adjusted_subtitle)
                
                # 更新当前时间（毫秒）
                current_time += segment_duration
                
                # 如果不是最后一个片段，添加静音间隔时间
                # 注意：这里 idx 是 enumerate(story_data) 的索引，但只有成功的才会添加
                # 所以需要检查是否是最后一个成功处理的片段
                # 简化处理：总是添加静音间隔，最后在合成时最后一个不加
                current_time += silence_interval_ms
                
                processed_count += 1
                success = True
                
                if api_verbose:
                    retry_info = f" (重试 {retry} 次后成功)" if retry > 0 else ""
                    print(f"    生成音频: {segment_duration:.0f}毫秒, 字幕: {len(subtitles)}条, 加静音: {silence_interval_ms:.0f}毫秒{retry_info}")
                
                break  # 成功，跳出重试循环
                
            except Exception as ex:
                last_error = ex
                if retry < max_retries - 1:
                    print(f"    警告: 生成失败 ({str(ex)}), 准备重试...")
                continue
        
        # 如果所有重试都失败
        if not success:
            print(f"  错误: 处理条目 {idx+1} 时失败（已重试 {max_retries} 次）: {str(last_error)}")
            skipped_count += 1
            continue
    
    if not all_audio_segments:
        raise ValueError("没有成功生成任何音频片段")
    
    # 调整总时长：移除最后一个片段后多加的静音间隔
    # （因为最后一个片段后面不需要静音）
    if processed_count > 0:
        current_time -= silence_interval_ms
        if api_verbose:
            print(f"   调整总时长: 移除最后 {silence_interval_ms:.0f}ms 静音间隔")
    
    # 使用 pydub 合并所有音频片段
    print(f">> 合并 {len(all_audio_segments)} 个音频片段（含 {silence_interval_ms:.0f}ms 静音间隔）...")
    final_sr = all_audio_segments[0][0]
    
    try:
        from pydub import AudioSegment
        
        # 创建静音片段（用于间隔）
        silence = AudioSegment.silent(duration=int(silence_interval_ms), frame_rate=int(final_sr))
        
        # 合并所有音频片段
        combined = None
        
        for idx, (sr, wav) in enumerate(all_audio_segments):
            # 使用 soundfile 写入 BytesIO，然后用 AudioSegment 读取
            with io.BytesIO() as buffer:
                sf.write(buffer, wav, sr, format='WAV')
                buffer.seek(0)
                audio_segment = AudioSegment.from_wav(buffer)
            
            # 拼接音频片段
            if combined is None:
                combined = audio_segment
            else:
                combined = combined + audio_segment
            
            # 在片段之间添加静音（最后一个片段除外）
            if idx < len(all_audio_segments) - 1:
                combined = combined + silence
        
        # 导出为 WAV 字节流
        with io.BytesIO() as wav_buffer:
            combined.export(wav_buffer, format="wav")
            wav_bytes = wav_buffer.getvalue()
        
        if api_verbose:
            print(f"   使用 pydub 合成: {len(all_audio_segments)} 个片段，总时长 {len(combined)}ms")
            
    except ImportError:
        # 如果没有 pydub，使用简化的 numpy 方式作为降级方案
        if api_verbose:
            print(f"   pydub 不可用，使用 NumPy 合成")
        
        audio_arrays = []
        silence_duration = silence_interval_ms / 1000.0
        silence_samples = int(final_sr * silence_duration)
        silence_np = np.zeros(silence_samples, dtype=np.float32)
        
        fade_samples = int(final_sr * 0.01)  # 10ms 淡入淡出
        
        for idx, (sr, wav) in enumerate(all_audio_segments):
            # 展平
            if wav.ndim > 1:
                wav = wav.flatten()
            
            # 重采样
            if sr != final_sr:
                ratio = final_sr / sr
                new_length = int(len(wav) * ratio)
                wav = np.interp(
                    np.linspace(0, len(wav), new_length),
                    np.arange(len(wav)),
                    wav
                )
            
            wav = wav.astype(np.float32)
            
            # 简单淡入淡出
            if len(wav) > fade_samples * 2:
                fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
                wav[:fade_samples] *= fade_in
                fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
                wav[-fade_samples:] *= fade_out
            
            audio_arrays.append(wav)
            
            # 添加静音
            if idx < len(all_audio_segments) - 1:
                audio_arrays.append(silence_np)
        
        # 合并
        final_audio = np.concatenate(audio_arrays)
        
        # 归一化
        max_val = np.abs(final_audio).max()
        if max_val > 0:
            final_audio = final_audio * (0.95 / max_val)
        
        # 生成音频字节流
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, final_audio, final_sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()
    
    # 编码为 base64
    audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
    
    print(f">> 完成! 处理: {processed_count}, 跳过: {skipped_count}, 总时长: {current_time:.0f}毫秒, 字幕: {len(all_subtitles)}条")
    
    return {
        "status": "success",
        "audio": audio_base64,
        "audio_format": "wav",
        "sample_rate": int(final_sr),
        "subtitles": all_subtitles,
        "subtitle_count": len(all_subtitles),
        "processed_items": processed_count,
        "skipped_items": skipped_count,
        "total_duration": current_time,
        "message": f"成功生成 {processed_count} 个音频片段，总时长 {current_time/1000:.2f} 秒"
    }


@app.post("/generate_story_audio", responses={
    200: {"content": {"application/json": {}}},
    400: {"content": {"application/json": {}}},
    500: {"content": {"application/json": {}}}
})
async def generate_story_audio(
    file: UploadFile = File(...),
    voice_map: Optional[str] = Form(None)
):
    """
    批量生成故事音频和字幕（文件上传方式）
    
    上传一个 JSON 文件，包含故事文本的描述信息，生成合成的语音和字幕。
    
    JSON 格式示例：
    [
        {
            "type": "c",  // 文本类型：s=故事情节, m=心理活动, c=人物对话, t=标题, unknown=未知
            "text": "我要再去西漠。",  // 要转成语音的文本
            "sex": "wo",  // 性别：man=男, wo=女, unknown=未知
            "name": "安妙依",  // 人物名
            "emotion": null  // 情绪状态：happy, sad, angry, afraid, disgusted, surprised, calm, fearful 等
        },
        ...
    ]
    
    请求参数：
    - file (UploadFile, 必需): JSON 文件
    - voice_map (string, 可选): 自定义音色映射 JSON 字符串
      格式示例：
      [
          {
              "name": "叶凡",
              "sex": "man",
              "emotion": "happy",
              "voice_id": "/uploads/speakers/ye_fan_happy.mp3"
          },
          {
              "name": "叶凡",
              "sex": "man",
              "emotion": null,
              "voice_id": "/uploads/speakers/ye_fan_default.mp3"
          }
      ]
    
    返回：
    - status (string): 状态 "success" 或 "error"
    - audio (string): base64 编码的合成音频（WAV 格式）
    - audio_format (string): 音频格式
    - sample_rate (int): 采样率
    - subtitles (array): 字幕数据列表
    - subtitle_count (int): 字幕条目数量
    - processed_items (int): 处理的文本条目数
    - skipped_items (int): 跳过的条目数
    - total_duration (float): 总时长（毫秒）
    """
    try:
        # 读取并解析 JSON 文件
        content = await file.read()
        try:
            story_data = json.loads(content)
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": f"JSON 解析失败: {str(e)}"
                }
            )
        
        # 解析自定义音色映射（如果提供）
        custom_voice_map = None
        if voice_map:
            try:
                custom_voice_map = json.loads(voice_map)
                if not isinstance(custom_voice_map, list):
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "error": "voice_map 必须是数组格式"
                        }
                    )
                print(f">> 使用自定义音色映射，共 {len(custom_voice_map)} 条规则")
            except json.JSONDecodeError as e:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "error": f"voice_map JSON 解析失败: {str(e)}"
                    }
                )
        
        # 调用核心生成逻辑
        result = await _process_story_audio_generation(story_data, custom_voice_map)
        return JSONResponse(status_code=200, content=result)
    
    except ValueError as ve:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": str(ve)
            }
        )
    except FileNotFoundError as fe:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(fe)
            }
        )
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(f">> 错误: {tb_str}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/generate_story_audio_json", responses={
    200: {"content": {"application/json": {}}},
    400: {"content": {"application/json": {}}},
    500: {"content": {"application/json": {}}}
})
async def generate_story_audio_json(request: Request):
    """
    批量生成故事音频和字幕（直接 JSON 方式）
    
    直接发送 JSON 数据，包含故事文本的描述信息，生成合成的语音和字幕。
    
    请求体格式：
    {
        "story_data": [
            {
                "type": "c",  // 文本类型：s=故事情节, m=心理活动, c=人物对话, t=标题, unknown=未知
                "text": "我要再去西漠。",  // 要转成语音的文本
                "sex": "wo",  // 性别：man=男, wo=女, unknown=未知
                "name": "安妙依",  // 人物名
                "emotion": null  // 情绪状态：happy, sad, angry, afraid, disgusted, surprised, calm, fearful 等
            },
            ...
        ],
        "voice_map": [  // 可选：自定义音色映射
            {
                "name": "叶凡",
                "sex": "man",
                "emotion": "happy",
                "voice_id": "/uploads/speakers/ye_fan_happy.mp3"
            },
            {
                "name": "叶凡",
                "sex": "man",
                "emotion": null,
                "voice_id": "/uploads/speakers/ye_fan_default.mp3"
            }
        ]
    }
    
    返回：
    - status (string): 状态 "success" 或 "error"
    - audio (string): base64 编码的合成音频（WAV 格式）
    - audio_format (string): 音频格式
    - sample_rate (int): 采样率
    - subtitles (array): 字幕数据列表
    - subtitle_count (int): 字幕条目数量
    - processed_items (int): 处理的文本条目数
    - skipped_items (int): 跳过的条目数
    - total_duration (float): 总时长（毫秒）
    """
    try:
        # 解析请求体
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": f"JSON 解析失败: {str(e)}"
                }
            )
        
        # 获取故事数据
        story_data = body.get("story_data")
        if not story_data:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": "缺少 story_data 字段"
                }
            )
        
        if not isinstance(story_data, list):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": "story_data 必须是数组格式"
                }
            )
        
        # 获取自定义音色映射（可选）
        custom_voice_map = body.get("voice_map")
        if custom_voice_map:
            if not isinstance(custom_voice_map, list):
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "error": "voice_map 必须是数组格式"
                    }
                )
            print(f">> 使用自定义音色映射，共 {len(custom_voice_map)} 条规则")
        
        # 调用核心生成逻辑
        result = await _process_story_audio_generation(story_data, custom_voice_map)
        return JSONResponse(status_code=200, content=result)
    
    except ValueError as ve:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": str(ve)
            }
        )
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(f">> 错误: {tb_str}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
    parser.add_argument("--fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--use_cuda_kernel", action="store_true", default=False, help="Use CUDA kernel")
    parser.add_argument("--use_deepspeed", action="store_true", default=False, help="Use DeepSpeed")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    args = parser.parse_args()
    
    # 创建必要的目录
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    if not os.path.exists("uploads/speakers"):
        os.makedirs("uploads/speakers")

    uvicorn.run(app=app, host=args.host, port=args.port)