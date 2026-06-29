import asyncio
import tempfile
import os

# 微软神经网络语音 — 真人般自然
# 男声: zh-CN-YunjianNeural (成熟稳重), zh-CN-YunxiNeural (明亮), zh-CN-YunyangNeural (新闻风格)
# 女声: zh-CN-XiaoxiaoNeural (温暖), zh-CN-XiaoyiNeural (清晰)
EDGE_VOICE = "zh-CN-YunxiNeural"


async def _generate_edge_tts(text: str, path: str) -> bool:
    """使用 edge_tts 生成 MP3 音频，成功返回 True"""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, EDGE_VOICE)
        await communicate.save(path)
        return True
    except Exception:
        return False


def _generate_pyttsx3(text: str, path: str):
    """fallback: 离线 pyttsx3 生成 WAV"""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    # 尝试选择中文语音
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.save_to_file(text, path)
    engine.runAndWait()


def generate_tts_audio(text):
    """生成 TTS 音频，返回临时文件路径。
    优先使用 edge_tts (MP3/真人语音)，失败则 fallback 到 pyttsx3 (WAV)。
    """
    # 先尝试 edge_tts (MP3)
    fd, path = tempfile.mkstemp(suffix='.mp3')
    os.close(fd)
    try:
        success = asyncio.run(_generate_edge_tts(text, path))
        if success and os.path.getsize(path) > 0:
            return path
    except Exception:
        pass

    # fallback: pyttsx3 (WAV)
    try:
        os.unlink(path)
    except OSError:
        pass
    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    _generate_pyttsx3(text, path)
    return path
