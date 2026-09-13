"""ต่อภาพนิ่งเป็นวิดีโอที่มีจังหวะจริง — ตัวที่ทำให้ Gate 1 มีความหมาย

ราคา: $0 ใช้เวลาไม่กี่สิบวินาที ทำให้เห็นปัญหา 80% ของ short-form
(hook ไม่ติด จังหวะช้า ยาวเกิน ซับอ่านไม่ทัน) ก่อนจ่ายค่า video generation

Ken Burns ที่ไม่กระตุก
----------------------
zoompan คำนวณตำแหน่ง crop เป็นจำนวนเต็มพิกเซล ที่อัตราซูมต่ำ ๆ ค่ามันจะ
กระโดดเป็นขั้น ทำให้ภาพสั่นเป็นจังหวะ วิธีแก้มาตรฐานคือ upscale ก่อน zoompan
แล้วค่อย scale ลงมา — ที่ความละเอียดสูงขึ้น 4 เท่า ขั้นของการปัดเศษเล็กลง 4 เท่า

ข้อควรระวัง: พารามิเตอร์ d ของ zoompan ต้องเท่ากับ duration*fps เป๊ะ
ไม่งั้นจะได้เฟรมค้างท้ายคลิปหรือคลิปสั้นกว่าที่สั่ง
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

W, H, FPS = 1080, 1920, 30
UPSCALE = 3  # 3x พอสำหรับ 1080p; 4x กินแรมเยอะขึ้นโดยแทบไม่ต่าง


class RenderError(RuntimeError):
    pass


@dataclass
class ShotClip:
    image: Path
    duration_s: float
    zoom_start: float = 1.0
    zoom_end: float = 1.12
    pan_x: float = 0.0
    pan_y: float = 0.0


def _run(cmd: list[str], *, what: str) -> None:
    """รัน ffmpeg แล้วโยน error พร้อม stderr ท้าย ๆ (ส่วนที่บอกสาเหตุจริง)"""
    log.debug("ffmpeg: %s", " ".join(cmd[:12]))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-15:])
        raise RenderError(f"{what} ล้มเหลว (exit {p.returncode}):\n{tail}")


def _ffprobe(path: Path) -> dict:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True)
    if p.returncode != 0:
        raise RenderError(f"ffprobe อ่าน {path.name} ไม่ได้: {p.stderr[:200]}")
    return json.loads(p.stdout)


def check_tools() -> None:
    for t in ("ffmpeg", "ffprobe"):
        if not shutil.which(t):
            raise RenderError(f"ไม่พบ {t} ใน PATH")


def ken_burns_filter(c: ShotClip) -> str:
    """สร้าง filter chain สำหรับหนึ่ง shot"""
    frames = max(1, int(round(c.duration_s * FPS)))
    z0, z1 = c.zoom_start, c.zoom_end
    # เดินค่า zoom เชิงเส้นตามหมายเลขเฟรม (on) แทนการสะสม zoom+step
    # ซึ่งจะเพี้ยนสะสมเมื่อคลิปยาว
    zexpr = f"{z0}+({z1}-{z0})*on/{max(1, frames - 1)}"
    # แพนโดยเลื่อนจุดกึ่งกลาง crop เป็นสัดส่วนของภาพ
    xexpr = f"iw/2-(iw/zoom/2)+({c.pan_x})*iw*on/{max(1, frames - 1)}"
    yexpr = f"ih/2-(ih/zoom/2)+({c.pan_y})*ih*on/{max(1, frames - 1)}"

    return (
        f"scale={W * UPSCALE}:{H * UPSCALE}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={W * UPSCALE}:{H * UPSCALE},"
        f"zoompan=z='{zexpr}':x='{xexpr}':y='{yexpr}'"
        f":d={frames}:s={W}x{H}:fps={FPS},"
        f"setsar=1,format=yuv420p"
    )


async def render_shot(clip: ShotClip, out: Path) -> Path:
    """หนึ่งภาพนิ่ง → หนึ่งคลิป"""
    vf = ken_burns_filter(clip)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(FPS), "-i", str(clip.image),
        "-t", f"{clip.duration_s:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        str(out),
    ]
    await asyncio.to_thread(_run, cmd, what=f"render shot {clip.image.name}")
    return out


async def render_animatic(
    clips: list[ShotClip],
    *,
    workdir: Path,
    out: Path,
    voice: Path | None = None,
    music: Path | None = None,
    subtitles: Path | None = None,
    fonts_dir: Path | None = None,
    music_gain: float = 0.16,
    target_lufs: float = -14.0,
) -> Path:
    """pipeline เต็ม: ภาพนิ่ง → คลิป → ต่อกัน → เบิร์นซับ → ผสมเสียง"""
    check_tools()
    workdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1) แต่ละ shot
    parts: list[Path] = []
    for i, c in enumerate(clips):
        if not c.image.exists():
            raise RenderError(f"ไม่พบภาพของ shot {i}: {c.image}")
        p = workdir / f"shot_{i:03d}.mp4"
        await render_shot(c, p)
        parts.append(p)

    # 2) ต่อกันด้วย concat demuxer — cut ล้วน ไม่ใส่ transition
    #    short-form ที่มี transition เยอะจะดูเป็นสไลด์โชว์ ซึ่งเป็นสิ่งที่ต้องเลี่ยง
    listfile = workdir / "concat.txt"
    listfile.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    silent = workdir / "silent.mp4"
    await asyncio.to_thread(_run, [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listfile),
        "-c", "copy", str(silent),
    ], what="concat")

    # 3) เบิร์นซับ
    burned = silent
    if subtitles and subtitles.exists():
        burned = workdir / "burned.mp4"
        # escape path สำหรับ filtergraph: : และ ' มีความหมายพิเศษ
        sub_arg = str(subtitles).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        vf = f"ass='{sub_arg}'"
        if fonts_dir:
            vf = f"ass='{sub_arg}':fontsdir='{str(fonts_dir)}'"
        await asyncio.to_thread(_run, [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(silent),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
            "-pix_fmt", "yuv420p", "-an", str(burned),
        ], what="burn subtitles")

    # 4) เสียง
    if not voice and not music:
        shutil_copy(burned, out)
        return out

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(burned)]
    idx = 1
    filters, labels = [], []
    if voice:
        cmd += ["-i", str(voice)]
        labels.append(f"[{idx}:a]")
        idx += 1
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music)]
        filters.append(f"[{idx}:a]volume={music_gain},afade=t=out:st=%(fade)s:d=2.5[mus]")
        labels.append("[mus]")
        idx += 1

    dur = float(_ffprobe(burned)["format"]["duration"])
    filters = [f % {"fade": f"{max(0.0, dur - 2.5):.2f}"} for f in filters]

    if len(labels) == 1:
        mix = f"{labels[0]}aresample=48000[mixed]"
    else:
        mix = ("".join(labels) +
               f"amix=inputs={len(labels)}:duration=first:dropout_transition=0,"
               "aresample=48000[mixed]")
    filters.append(mix)
    filters.append(f"[mixed]loudnorm=I={target_lufs}:TP=-1.0:LRA=11[aout]")

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", "-movflags", "+faststart",
        str(out),
    ]
    await asyncio.to_thread(_run, cmd, what="mix audio")
    return out


def shutil_copy(src: Path, dst: Path) -> None:
    import shutil as _s
    _s.copyfile(src, dst)
