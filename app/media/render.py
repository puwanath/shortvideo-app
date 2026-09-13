"""ตัดต่อขั้นสุดท้ายจาก clip ที่ generate มาจริง

ต่างจาก animatic ตรงที่ input เป็นวิดีโอไม่ใช่ภาพนิ่ง และตั้งแต่ 2026-09-13
คลิปมี *เสียงพูดจากโมเดลวิดีโอ* อยู่ข้างใน — ไม่มีเสียงพากย์แยก ไม่มีซับ
ซึ่งเปลี่ยนปัญหาที่ต้องจัดการไปสามข้อ:

1. ตัดคลิปให้พอดี timeline ไม่ได้อีกแล้ว
   เดิมสั่ง 5 วินาทีแล้วตัดท้ายทิ้งให้เหลือ 4.62 ได้ เพราะเสียงพากย์อยู่แยก
   ตอนนี้บทพูดอยู่ในคลิป ตัดท้าย = ตัดคำ จึงใช้ความยาวจริงของคลิปเป็น timeline
   (runner เรียก retime_shots หลัง video_gen) ถ้าคลิปสั้นกว่าที่วางไว้มาก
   ก็แค่เตือน ไม่ยืด ไม่ตัด

2. พารามิเตอร์ไม่ตรงกันระหว่าง clip — ทั้งภาพและเสียง
   คนละ shot อาจมาจากคนละโมเดล (fallback) ได้คนละ fps คนละ pix_fmt และ
   คนละ sample rate / จำนวน channel concat demuxer แบบ -c copy จะพังเงียบ ๆ
   ต้อง normalize ทั้งสอง stream ก่อนเสมอ

3. คลิปที่ไม่มีเสียงต้องได้ track เงียบ
   โมเดลบางตัว (หรือ fallback ที่ไม่ทำเสียง) คืนคลิปไม่มี audio stream
   concat ระหว่างไฟล์ที่มีกับไม่มี audio จะได้ผลลัพธ์ที่เสียงหายทั้งเรื่อง
   จึงยัด anullsrc ให้ทุกคลิปที่ไม่มีเสียง เพื่อให้ทุกไฟล์มี stream ครบเหมือนกัน

ความดังทำรวมทีเดียวตอนท้ายด้วย loudnorm สองรอบ เพราะเสียงจากแต่ละ shot
ดังไม่เท่ากันอยู่แล้ว (คนละ job) และ platform วัดที่ -14 LUFS ทั้งไฟล์
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .animatic import FPS, H, RenderError, W, _run

log = logging.getLogger(__name__)

AUDIO_RATE = 48000


@dataclass
class VideoShot:
    clip: Path
    duration_s: float          # ความยาวที่ timeline วางไว้ — ใช้เทียบเตือนเท่านั้น
    fade_in: float = 0.0


def probe_stream(path: Path) -> dict:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True)
    if p.returncode != 0:
        raise RenderError(f"ffprobe อ่าน {path.name} ไม่ได้: {p.stderr[:200]}")
    data = json.loads(p.stdout)
    vs = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not vs:
        raise RenderError(f"{path.name} ไม่มี video stream")
    st = vs[0]
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "duration": float(data["format"].get("duration") or 0),
        "pix_fmt": st.get("pix_fmt"),
        "fps": st.get("r_frame_rate", "0/1"),
        "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
    }


async def normalize_clip(shot: VideoShot, out: Path) -> Path:
    """ทำให้ clip หนึ่งตัวมีขนาด fps pix_fmt และรูปแบบเสียงตรงสเปกเป๊ะ

    ทุก clip ต้องผ่านตรงนี้ ไม่มีข้อยกเว้น แม้จะดูเหมือนถูกอยู่แล้ว
    เพราะ concat จะพังแบบเงียบ ๆ ถ้ามีตัวใดตัวหนึ่งต่าง
    ความยาวไม่แตะ — เสียงพูดอยู่ข้างใน
    """
    info = probe_stream(shot.clip)
    have = info["duration"]
    if abs(have - shot.duration_s) > 0.5:
        log.warning("clip %s ยาว %.2fs แต่วางไว้ %.2fs — ใช้ความยาวจริง ไม่ตัด",
                    shot.clip.name, have, shot.duration_s)

    # ครอบให้เป็น 9:16 เต็มจอ: ขยายให้คลุมแล้วครอบกลาง
    vf = [
        f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={W}:{H}",
        f"fps={FPS}",
        "setsar=1",
        "format=yuv420p",
    ]
    if shot.fade_in > 0:
        vf.insert(0, f"fade=t=in:st=0:d={shot.fade_in:.2f}")

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(shot.clip)]
    if info["has_audio"]:
        amap = ["-map", "0:v:0", "-map", "0:a:0"]
    else:
        # คลิปใบ้ — เติมเสียงเงียบให้ยาวเท่าภาพ (-shortest ตัดให้พอดี)
        log.warning("clip %s ไม่มี audio stream — เติมเสียงเงียบ", shot.clip.name)
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
        amap = ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]

    cmd += [
        *amap,
        "-vf", ",".join(vf),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-video_track_timescale", str(FPS * 1000),  # กัน timestamp เพี้ยนตอน concat
        "-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_RATE), "-ac", "2",
        str(out),
    ]
    await asyncio.to_thread(_run, cmd, what=f"normalize {shot.clip.name}")

    got = probe_stream(out)
    if not got["has_audio"]:
        raise RenderError(f"{shot.clip.name} หลัง normalize ไม่มีเสียง")
    if abs(got["duration"] - have) > 0.15:
        raise RenderError(
            f"{shot.clip.name} หลัง normalize ได้ {got['duration']:.3f}s จากเดิม {have:.3f}s")
    return out


async def render_final(
    shots: list[VideoShot],
    *,
    workdir: Path,
    out: Path,
    target_lufs: float = -14.0,
    nvenc: bool = False,
    concurrency: int = 2,
) -> Path:
    """clip จริง → วิดีโอสุดท้ายพร้อมโพสต์"""
    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(shots):
        if not s.clip.exists():
            raise RenderError(f"ไม่พบ clip ของ shot {i}: {s.clip}")

    # 1) normalize ขนานกัน แต่จำกัด concurrency เพราะ x264 กิน CPU เต็มแกน
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int, s: VideoShot) -> Path:
        async with sem:
            return await normalize_clip(s, workdir / f"norm_{i:03d}.mp4")

    parts = await asyncio.gather(*(one(i, s) for i, s in enumerate(shots)))

    # 2) concat — ปลอดภัยที่จะ -c copy เพราะ normalize ทั้งภาพและเสียงแล้ว
    listfile = workdir / "concat.txt"
    listfile.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    joined = workdir / "joined.mp4"
    await asyncio.to_thread(_run, [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listfile),
        "-c", "copy", "-fflags", "+genpts", str(joined),
    ], what="concat clips")

    # 3) ความดัง — loudnorm สองรอบบนไฟล์รวม ภาพ copy ไม่ต้อง encode ซ้ำ
    #    (nvenc ไม่จำเป็นในสายนี้แล้ว: ไม่มีซับให้เบิร์น จึงไม่ต้อง re-encode ภาพ)
    measured = await asyncio.to_thread(_measure_loudness, joined, target_lufs)
    if measured:
        # รอบสอง: ป้อนค่าที่วัดได้กลับเข้าไป ได้ผลแม่นกว่า single-pass มาก
        af = (f"loudnorm=I={target_lufs}:TP=-1.0:LRA=11:"
              f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
              f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
              f"offset={measured.get('target_offset', '0.0')}:linear=true")
    else:
        af = f"loudnorm=I={target_lufs}:TP=-1.0:LRA=11"

    await asyncio.to_thread(_run, [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(joined),
        "-map", "0:v", "-map", "0:a",
        "-af", f"{af},aresample={AUDIO_RATE}",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_RATE), "-ac", "2",
        "-movflags", "+faststart",
        str(out),
    ], what="loudnorm final")
    return out


def _measure_loudness(path: Path, target: float) -> dict | None:
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"loudnorm=I={target}:TP=-1.0:LRA=11:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True)
    i = p.stderr.rfind("{")
    if i < 0:
        return None
    try:
        return json.loads(p.stderr[i:p.stderr.rfind("}") + 1])
    except json.JSONDecodeError:
        return None
