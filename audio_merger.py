from pydub import AudioSegment
import io
import soundfile as sf
import os

def merge_audio_segments(audio_segments):
    """
    Merge a list of audio segments into a single audio file.
    """
    combined = None
    for idx, item in enumerate(audio_segments):
        audio_segment = item["audio"]  # audio_segments 是字典列表，每个元素包含 "name" 和 "audio"
        if combined is None:
            combined = audio_segment
        else:
            combined = combined + audio_segment
    return combined

def read_audio_from_path_batch(dir):
    audio_segments = []
    for file in os.listdir(dir):
        if file.endswith('.wav'):
            audio_segment = read_audio_from_path(os.path.join(dir, file))
            audio_segments.append({"name": file, "audio": audio_segment})

    # 按数字排序：提取文件名中的数字部分进行排序
    def get_sort_key(item):
        name = item["name"]
        # 去掉扩展名，提取数字部分
        base_name = os.path.splitext(name)[0]  # 例如 "1.wav" -> "1"
        try:
            # 尝试转换为整数进行排序
            return int(base_name)
        except ValueError:
            # 如果无法转换为整数，使用字符串排序
            return base_name
    
    sorted_audio_segments = sorted(audio_segments, key=get_sort_key)
    for x in sorted_audio_segments:
        print(f"read audio name: {x['name']}")
    return sorted_audio_segments

def read_audio_from_path(file_path):
    audio_segment = AudioSegment.from_wav(file_path)
    return audio_segment

def write_audio_to_path(audio_segment, file_path):
    audio_segment.export(file_path, format='wav')
    return audio_segment

def write_audio_to_path_batch(audio_segments, dir):
    for audio_segment in audio_segments:
        write_audio_to_path(audio_segment, os.path.join(dir, audio_segment.name))
    return audio_segments

if __name__ == "__main__":
    audio_segments = read_audio_from_path_batch("./outputs/")
    merged_audio = merge_audio_segments(audio_segments)
    write_audio_to_path(merged_audio, "./outputs/763_merge_2.wav")