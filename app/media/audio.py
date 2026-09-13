"""งานเสียงเล็ก ๆ ที่ต้องทำก่อนเข้า pipeline

มีไฟล์นี้เพราะ provider แต่ละเจ้าคืนฟอร์แมตไม่เหมือนกัน (OpenRouter TTS คืน mp3,
JaiTTS คืน wav) แต่ทุกขั้นหลังจากนี้ — forced alignment, ffmpeg concat,
การวัด LUFS — สมมติว่า voice.wav เป็น WAV จริง ๆ จึงแปลงให้จบตรงจุดรับไฟล์
ที่เดียว ดีกว่าให้แต่ละขั้นไปเดาเอาเองว่าไฟล์เป็นอะไร
"""
from __future__ import annotations

import subprocess
from pathlib import Path

SAMPLE_RATE = 48000


def write_voice_wav(data: bytes, out: Path, *, mime: str = "") -> Path:
    """เขียนเสียงพากย์ลง out เป็น WAV 48 kHz mono เสมอ

    ถ้าได้ WAV มาอยู่แล้วก็ยังแปลงซ้ำ เพราะ sample rate ของแต่ละเจ้าไม่เท่ากัน
    (F5-TTS 24 kHz, Qwen TTS 24 kHz) และ ffmpeg จะ resample ตอน concat อยู่ดี
    ทำตรงนี้ทีเดียวจะได้ไม่ต้องไปเดาทีหลัง
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    src = out.with_suffix(".src")
    src.write_bytes(data)
    try:
        p = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(out)],
            capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(
                f"แปลงเสียงเป็น WAV ไม่ได้ (mime={mime or 'ไม่ระบุ'}): {p.stderr[:200]}")
    finally:
        src.unlink(missing_ok=True)
    return out
