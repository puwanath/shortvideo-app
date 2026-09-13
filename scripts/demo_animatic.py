"""ทดสอบชั้น media ทั้งหมดโดยไม่ต้องมี API key

สร้างภาพนิ่งปลอม + เสียงปลอม + word timing ปลอม แล้วเรนเดอร์ animatic จริง
ใช้เช็กว่า ffmpeg / libass / ฟอนต์ไทย / การตัดคำ ทำงานถูกก่อนต่อ provider

    python scripts/demo_animatic.py
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.media.animatic import ShotClip, render_animatic, W, H          # noqa: E402
from app.media.ass import CaptionStyle, karaoke_ass                      # noqa: E402
from app.media.thai_text import (                                        # noqa: E402
    display_width, group_words_into_cues, insert_break_opportunities, tokenize, wrap,
)

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "var" / "demo"
FONTS = ROOT / "assets" / "fonts"

SCRIPT = [
    ("อย่าเพิ่งซื้อการ์ดจอใหม่", "push_in", (0.20, 0.35)),
    ("ถ้าโมเดลที่คุณรันใหญ่กว่าแรมการ์ดใบเดียว", "pan_left", (0.20, 0.30)),
    ("การแบ่งโมเดลข้ามการ์ดจะช้าลงทันที", "static", (0.25, 0.30)),
    ("เพราะข้อมูลต้องวิ่งผ่านแรมเครื่องแทน", "pull_out", (0.30, 0.25)),
    ("ลองวัดก่อนว่าคอขวดอยู่ตรงไหนจริง", "push_in", (0.15, 0.40)),
    ("แล้วค่อยตัดสินใจว่าจะจ่ายเงินหรือจะจูน", "static", (0.10, 0.45)),
]

PALETTE = [
    "0x1B3A5C", "0x2E5F4F", "0x5C3B1B", "0x3D2E5C", "0x5C1B2E", "0x1B5C58",
]


def make_placeholder(i: int, color: str, path: Path) -> None:
    """ภาพนิ่งปลอมพร้อมเลข shot — แทนที่ด้วย keyframe จริงตอนต่อ provider"""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c={color}:s={W}x{H}",
        "-vf", (f"drawtext=text='SHOT {i + 1}':fontcolor=white@0.35:fontsize=180:"
                f"x=(w-text_w)/2:y=(h-text_h)/2,"
                f"noise=alls=8:allf=t+u"),
        "-frames:v", "1", str(path),
    ], check=True, capture_output=True)


def fake_words(text: str, start: float, dur: float) -> list[tuple[str, float, float]]:
    """จำลองผลจาก Whisper — แบ่งเวลาตามความกว้างของแต่ละคำ"""
    toks = [t for t in tokenize(text) if t.strip()]
    widths = [max(display_width(t), 0.5) for t in toks]
    total = sum(widths)
    out, t = [], start
    for tok, w in zip(toks, widths):
        d = dur * (w / total)
        out.append((tok, round(t, 3), round(t + d * 0.94, 3)))
        t += d
    return out


def make_tone(path: Path, seconds: float) -> None:
    """เสียงปลอมแทน TTS"""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency=180:duration={seconds}",
        "-af", "tremolo=f=3.5:d=0.7,volume=0.35",
        "-ar", "48000", "-ac", "1", str(path),
    ], check=True, capture_output=True)


async def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("1) ตัดคำและตัดบรรทัดภาษาไทย")
    print("=" * 62)
    sample = "ถ้าโมเดลที่คุณรันใหญ่กว่าแรมการ์ดใบเดียวการแบ่งโมเดลข้ามการ์ดจะช้าลง"
    print("ตัดคำ   :", " | ".join(tokenize(sample)[:12]), "...")
    print("ZWSP    :", repr(insert_break_opportunities(sample)[:52]))
    for ln in wrap(sample, 15.0, 3):
        print(f"  บรรทัด [{display_width(ln):5.1f}] {ln}")

    print()
    print("=" * 62)
    print("2) สร้าง shot / timing / ซับ")
    print("=" * 62)

    clips: list[ShotClip] = []
    all_words: list[tuple[str, float, float]] = []
    t = 0.0
    from app.schemas import KenBurns

    for i, (text, move, _) in enumerate(SCRIPT):
        img = WORK / f"kf_{i:02d}.png"
        make_placeholder(i, PALETTE[i % len(PALETTE)], img)
        dur = round(1.4 + display_width(text) * 0.115, 2)
        kb = KenBurns.from_camera(move)
        clips.append(ShotClip(image=img, duration_s=dur,
                              zoom_start=kb.zoom_start, zoom_end=kb.zoom_end,
                              pan_x=kb.pan_x, pan_y=kb.pan_y))
        all_words += fake_words(text, t, dur)
        print(f"  shot {i + 1}  {dur:5.2f}s  {move:<10} {text}")
        t += dur

    total = round(t, 2)
    print(f"  รวม {total}s")

    bounds, acc = [], 0.0
    for c in clips[:-1]:
        acc += c.duration_s
        bounds.append(round(acc, 3))
    cues = group_words_into_cues(all_words, max_width=13.0, boundaries=bounds)
    print(f"  จัดเป็น {len(cues)} cue จาก {len(all_words)} คำ")

    sub = WORK / "captions.ass"
    sub.write_text(
        karaoke_ass(cues, CaptionStyle(font="Noto Sans Thai"),
                    on_screen=[(SCRIPT[0][0], 0.15, 2.6)]),
        encoding="utf-8")
    print(f"  เขียน {sub.name} ({sub.stat().st_size} bytes)")

    voice = WORK / "voice.wav"
    make_tone(voice, total)

    print()
    print("=" * 62)
    print("3) เรนเดอร์ animatic")
    print("=" * 62)
    out = WORK / "animatic.mp4"
    await render_animatic(
        clips, workdir=WORK / "tmp", out=out,
        voice=voice, subtitles=sub, fonts_dir=FONTS,
    )
    size = out.stat().st_size / 1024
    print(f"  {out}  ({size:.0f} KB)")

    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration", "-of", "default=nw=1", str(out),
    ], capture_output=True, text=True)
    print("  " + probe.stdout.strip().replace("\n", "  "))


if __name__ == "__main__":
    asyncio.run(main())
