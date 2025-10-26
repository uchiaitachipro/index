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

from indextts.infer_v2_subtitle import IndexTTS2SubTitle

tts = None
api_verbose = False
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2SubTitle(
        model_dir=args.model_dir,
        # use_fp16=args.is_fp16,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
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