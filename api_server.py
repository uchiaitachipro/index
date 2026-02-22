import os
import asyncio
import io
import base64
import zipfile
import shutil
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
from audio_detect_noise_v2 import detect_chi_noise
from text_spilter import split_text_by_characters

tts = None
api_verbose = False
api_diagnose_mode = False

# 说话人音频查找配置
SPEAKER_AUDIO_DIRS = [
    "uploads/speakers",
    "examples",
]

# 默认说话人音频映射（可根据实际情况修改）
DEFAULT_SPEAKER_MAP = {
    ("man", "叶凡"): "minmax_shulang_man_2dot6_1x.mp3",
    ("wo", "安妙依"): "minmax_soft_girl_2dot6_1x.mp3",
    ("woman", "安妙依"): "minmax_soft_girl_2dot6_1x.mp3",
    ("wo", "姬紫月"): "minmax_warm_girl_2dot6_1x.mp3",
    ("woman", "姬紫月"): "minmax_warm_girl_2dot6_1x.mp3",
    ("man", "default"): "minmax_unrestrained_young_man_2dot6_1x.mp3",
    ("wo", "default"): "minmax_arrogant_girl_2dot6_1x.mp3",
    ("woman", "default"): "minmax_arrogant_girl_2dot6_1x.mp3",
    ("unknown", "default"): "fanqie_speaker_1x.mp3"
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


def save_diagnose_data_to_zip(
    diagnose_output_dir: str,
    wav_bytes: bytes,
    noise_result: dict,
    text: str,
    speaker: str,
    emotion: str,
    retry: int
) -> Optional[str]:
    """
    将诊断数据和音频保存到 zip 文件
    
    Args:
        diagnose_output_dir: 诊断输出目录路径
        wav_bytes: 生成的音频字节数据
        noise_result: 杂音检测结果
        text: 文本内容
        speaker: 说话人名称
        emotion: 情感
        retry: 重试次数
        
    Returns:
        zip 文件路径，失败则返回 None
    """
    if not diagnose_output_dir or not os.path.isdir(diagnose_output_dir):
        return None
    
    diag_timestamp = int(time.time() * 1000)
    zip_dir = "uploads/diagnose"
    os.makedirs(zip_dir, exist_ok=True)
    zip_path = f"{zip_dir}/{diag_timestamp}.zip"
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 添加诊断目录中的所有文件
            for root, dirs, files in os.walk(diagnose_output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, diagnose_output_dir)
                    zf.write(file_path, arcname)
            
            # 添加当前生成的音频文件
            zf.writestr("generated_audio.wav", wav_bytes)
            
            # 添加杂音检测结果
            noise_info = {
                "noise_result": noise_result,
                "text": text,
                "speaker": speaker,
                "emotion": emotion,
                "retry": retry
            }
            zf.writestr("noise_detection.json", json.dumps(noise_info, ensure_ascii=False, indent=2))
        
        print(f"    诊断数据已保存到 {zip_path}")
        
        # 清理诊断输出目录
        shutil.rmtree(diagnose_output_dir, ignore_errors=True)
        
        return zip_path
    except Exception as zip_error:
        print(f"    警告: 保存诊断数据失败: {str(zip_error)}")
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
    global api_diagnose_mode
    api_diagnose_mode = args.diagnose_mode
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
    segment_noise_results = []  # 记录每个片段的杂音检测结果
    
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
        chapter_num = item.get("chapter_number", 0)  # 获取章节号，默认为 0
        
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
        if custom_voice_map:
            print(f"    使用自定义音色文件: {spk_audio_path}")
        
        # 准备情感参数
        emo_vec = [0] * 8
        emo_weight = 0.35
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
        
        # 调用 TTS 生成音频（在线程池中运行，失败或检测到杂音时最多重试 3 次）
        max_retries = 3
        success = False
        last_error = None
        
        for retry in range(max_retries):
            try:
                if retry > 0:
                    print(f"    重试 {retry}/{max_retries-1}...")
                    # 重试前短暂等待
                    await asyncio.sleep(1)
                
                def infer_tts():
                    return tts.infer(
                        spk_audio_prompt=spk_audio_path,
                        text=text,
                        output_path=None,
                        emo_alpha=emo_weight,
                        emo_vector=emo_vec if emo_control_method == 2 else None,
                        use_emo_text=False,
                        use_random=False,
                        max_text_tokens_per_segment=120,
                        verbose=api_verbose,
                        return_diagnose=api_diagnose_mode
                    )
                
                result = await loop.run_in_executor(None, infer_tts)
                
                # 解析结果
                audio_data = result['audio_data']
                sr, wav = audio_data
                subtitles = result.get('subtitles', [])
                
                # 检测杂音
                segment_has_noise = False
                segment_noise_score = 0.0
                try:
                    # 使用 AudioSegment 导出字节数组，然后调用 detect_chi_noise 检测
                    # 这样可以避免类型转换问题，detect_chi_noise 会自动处理字节数组
                    from pydub import AudioSegment
                    with io.BytesIO() as buffer:
                        sf.write(buffer, wav, sr, format='WAV')
                        buffer.seek(0)
                        audio_segment = AudioSegment.from_wav(buffer)
                    
                    # 导出为 WAV 字节数组
                    with io.BytesIO() as wav_buffer:
                        audio_segment.export(wav_buffer, format="wav")
                        wav_bytes = wav_buffer.getvalue()
                    
                    # 调用 detect_chi_noise 检测（支持 bytes 输入）
                    noise_result = detect_chi_noise(wav_bytes)
                    has_noise = noise_result.get('has_chi', False)
                    segment_has_noise = has_noise
                    segment_noise_score = noise_result.get('score', 0.0)
                    
                    if has_noise:
                        if api_verbose:
                            timestamp = int(time.time() * 1000)
                            save_audio_path = f"uploads/noise/{chapter_num}_noise_{timestamp}.wav"
                            os.makedirs(os.path.dirname(save_audio_path), exist_ok=True)
                            with open(save_audio_path, "wb") as f:
                                f.write(wav_bytes)
                            print(f"    警告: 检测到杂音，已保存到 {save_audio_path}")
                        
                        # 如果启用了诊断模式，保存诊断数据和音频到 zip 文件
                        if api_diagnose_mode:
                            diagnose_data = result.get('diagnose', {})
                            diagnose_output_dir = diagnose_data.get('output_dir', '')
                            save_diagnose_data_to_zip(
                                diagnose_output_dir=diagnose_output_dir,
                                wav_bytes=wav_bytes,
                                noise_result=noise_result,
                                text=text,
                                speaker=name if name else "旁白",
                                emotion=emotion if emotion else "unknown",
                                retry=retry
                            )
                        # 检测到杂音
                        if retry < max_retries - 1:
                            # 还有重试机会，抛出异常触发重试
                            raise ValueError(f"检测到杂音 (score={noise_result.get('score', 0):.2f}, "
                                           f"ratio={noise_result.get('ratio_db_peak', 0):.2f}dB)")
                        else:
                            # 最后一次重试仍有问题，输出日志并使用最后一次生成的音频（当前 audio_data）
                            print(f"    错误: 条目 {idx+1} 经过 {max_retries} 次重试后仍检测到杂音，使用最后一次生成的音频")
                            print(f"      杂音检测详情: score={noise_result.get('score', 0):.2f}, "
                                  f"ratio={noise_result.get('ratio_db_peak', 0):.2f}dB, "
                                  f"flux={noise_result.get('flux_db_peak', 0):.2f}dB")
                            print(f"      文本: {text[:50]}...")
                            # 使用当前生成的音频（已经是最后一次的）
                except ValueError as noise_error:
                    # 杂音检测失败，重新抛出以触发重试
                    raise
                except Exception as detect_error:
                    # 检测过程出错，记录但继续使用当前音频
                    print(f"    警告: 杂音检测失败: {str(detect_error)}，使用当前生成的音频")
                
                # 记录当前片段的杂音检测结果
                segment_noise_results.append({
                    'has_noise': segment_has_noise,
                    'score': segment_noise_score
                })
                
                # 音频生成成功且无杂音（或检测失败但继续使用），处理音频
                # 计算当前音频片段时长（毫秒）
                segment_duration = len(wav) / sr * 1000.0
                
                # 存储音频片段
                all_audio_segments.append((sr, wav))
                
                # 调整字幕时间并添加到总列表
                for subtitle in subtitles:
                    adjusted_subtitle = {
                        "text": subtitle["text"],
                        "time_begin": subtitle["time_begin"] + current_time,
                        "time_end": subtitle["time_end"] + current_time,
                        "pronounce_text": subtitle["pronounce_text"],
                        "speaker": name if name else "旁白",
                        "type": item_type,
                        "emotion": emotion if emotion else "unknown"
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
                    error_msg = str(ex)
                    if "检测到杂音" in error_msg:
                        print(f"    警告: {error_msg}，准备重试...")
                    else:
                        print(f"    警告: 生成失败 ({error_msg}), 准备重试...")
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
        # 如果没有 pydub，抛出错误
        raise ImportError("pydub 不可用，无法合并音频片段")
    
    # 检测最终合并音频的杂音（直接使用片段检测结果）
    # 如果有一个片段有杂音，就认为最终音频也有杂音
    segments_with_noise = [r for r in segment_noise_results if r.get('has_noise', False)]
    has_noise = len(segments_with_noise) > 0
    
    if has_noise:
        noise_count = len(segments_with_noise)
        print(f">> 警告: 检测到 {noise_count} 个片段包含杂音，最终音频标记为有杂音")
    
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
        "has_noise": has_noise,
        "message": f"成功生成 {processed_count} 个音频片段，总时长 {current_time/1000:.2f} 秒"
    }


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
                "emotion": null  // 情绪状态：happy, sad, angry, afraid, disgusted, surprised, calm, fearful 等,
                "chapter_number":  -1 // 章节号：-1 表示不分章节
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


@app.post("/generate_story_audio_single", responses={
    200: {"content": {"application/json": {}}},
    400: {"content": {"application/json": {}}},
    500: {"content": {"application/json": {}}}
})
async def generate_story_audio_single(request: Request):
    """
    生成单个故事音频和字幕（直接 JSON 方式）
    
    直接发送单个故事数据对象，生成合成的语音和字幕。
    
    请求体格式：
    {
        "story_data": {
            "type": "c",  // 文本类型：s=故事情节, m=心理活动, c=人物对话, t=标题, unknown=未知
            "text": "我要再去西漠。",  // 要转成语音的文本
            "sex": "wo",  // 性别：man=男, wo=女, unknown=未知
            "name": "安妙依",  // 人物名
            "emotion": null  // 情绪状态：happy, sad, angry, afraid, disgusted, surprised, calm, fearful 等
            "chapter_number":  -1 // 章节号：-1 表示不分章节
        },
        "voice_map": [  // 可选：自定义音色映射
            {
                "name": "叶凡",
                "sex": "man",
                "emotion": "happy",
                "voice_id": "/uploads/speakers/ye_fan_happy.mp3"
            },
        ]
        }
    }
    
    返回：
    {
        "data": {
            "audio_base64": "base64_encoded_audio_data",
            "status": 2
        },
        "extra_info": {
            "audio_length": 10000,
            "audio_sample_rate": 22050,
            "audio_format": "wav",
            "audio_channel": 1,
            "has_noise": False
        },
        "base_resp": {
            "status_code": 0,
            "status_msg": "success"
        },
        "subtitles": [...字幕数据...]
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
        
        # 验证 story_data 是否为字典格式（单个对象）
        if not isinstance(story_data, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": "story_data 必须是对象格式"
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
        
        # 将单个对象转换为数组格式
        story_data_list = [story_data]
        
        # 调用核心生成逻辑
        result = await _process_story_audio_generation(story_data_list, custom_voice_map)
        
        # 构建返回结果（新格式）
        # 状态映射：success -> 2
        status_code = 2 if result.get("status", "success") == "success" else 1
        
        return JSONResponse(
            status_code=200,
            content={
                "data": {
                    "audio_base64": result.get("audio", ""),
                    "status": 2 #1 表示合成中，2 表示合成结束
                },
                "extra_info": {
                    "audio_length": int(result.get("total_duration", 0.0)),
                    "audio_sample_rate": result.get("sample_rate", 32000),
                    "audio_format": result.get("audio_format", "wav"),
                    "audio_channel": 1,  # TTS 生成单声道音频
                    "has_noise": result.get("has_noise", False)
                },
                "base_resp": {
                    "status_code": 0,
                    "status_msg": "success"
                },
                "subtitles": result.get("subtitles", [])
            }
        )
    
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

@app.post("/split_text", responses={
    200: {"content": {"application/json": {}}},
    400: {"content": {"application/json": {}}},
    500: {"content": {"application/json": {}}}
})
async def split_text(request: Request):
    """
    分割文本
    
    请求体格式：
    {
        "text": "我要再去西漠。",
        "max_chars_per_segment": 100
    }
    
    返回：
    {
        "status": "success",
        "segments": [
            "我要再去西漠。",
            "我要再去西漠。",   
            "我要再去西漠。"
    }
    """
    try:
        # 解析请求体
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            return JSONResponse(status_code=400, content={"status": "error", "error": f"JSON 解析失败: {str(e)}"})

        # 获取文本
        text = body.get("text")
        if not text:
            return JSONResponse(status_code=400, content={"status": "error", "error": "缺少 text 字段"})

        # 获取最大字符数
        max_chars_per_segment = body.get("max_chars_per_segment", 100)
        if not isinstance(max_chars_per_segment, int):
            return JSONResponse(status_code=400, content={"status": "error", "error": "max_chars_per_segment 必须是整数"})

        # 分割文本
        segments = split_text_by_characters(text, max_chars_per_segment)
        
        return JSONResponse(status_code=200, content={"status": "success", "segments": segments})

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"status": "error", "error": str(ve)})
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(f">> 错误: {tb_str}")
        return JSONResponse(status_code=500, content={"status": "error", "error": str(tb_str)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
    parser.add_argument("--fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--use_cuda_kernel", action="store_true", default=False, help="Use CUDA kernel")
    parser.add_argument("--use_deepspeed", action="store_true", default=False, help="Use DeepSpeed")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    parser.add_argument("--diagnose_mode", action="store_true", default=False, help="Enable diagnose mode for noise detection")
    args = parser.parse_args()
    
    # 创建必要的目录
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    if not os.path.exists("uploads/speakers"):
        os.makedirs("uploads/speakers")

    uvicorn.run(app=app, host=args.host, port=args.port)