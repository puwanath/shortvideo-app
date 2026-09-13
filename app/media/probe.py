"""ตรวจงานด้วยเครื่องมือที่ให้คำตอบแน่นอน ก่อนจะจ่ายเงินให้ VLM ดู

ลำดับสำคัญ: ตรวจ deterministic ก่อนเสมอ ฟรีและจับปัญหาได้เยอะกว่าที่คิด
(ความยาวเพี้ยน เสียงดังผิด เฟรมดำค้าง อัตราส่วนผิด) VLM ควรถูกเรียก
เฉพาะเรื่องที่ต้องใช้ตาคนดูจริง ๆ เท่านั้น
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..schemas import QCIssue, QCReport

TARGET_W, TARGET_H = 1080, 1920
TARGET_LUFS = -14.0
LUFS_TOLERANCE = 1.5


def probe(path: Path) -> dict:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe ล้มเหลว: {p.stderr[:200]}")
    return json.loads(p.stdout)


def measure_loudness(path: Path) -> dict | None:
    """รัน loudnorm แบบ analysis อย่างเดียว คืนค่าที่วัดได้"""
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.0:LRA=11:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True)
    txt = p.stderr
    start = txt.rfind("{")
    if start < 0:
        return None
    try:
        return json.loads(txt[start:txt.rfind("}") + 1])
    except json.JSONDecodeError:
        return None


def detect_freeze(path: Path, min_s: float = 0.6) -> list[float]:
    """หาช่วงภาพนิ่งค้าง — สัญญาณว่า clip generation คืนเฟรมซ้ำ"""
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-vf", f"freezedetect=n=-60dB:d={min_s}", "-map", "0:v", "-f", "null", "-"],
        capture_output=True, text=True)
    out = []
    for line in p.stderr.splitlines():
        if "freeze_start" in line:
            try:
                out.append(float(line.rsplit(":", 1)[1].strip()))
            except ValueError:
                pass
    return out


def detect_black(path: Path, min_s: float = 0.4) -> list[float]:
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-vf", f"blackdetect=d={min_s}:pic_th=0.98", "-an", "-f", "null", "-"],
        capture_output=True, text=True)
    out = []
    for line in p.stderr.splitlines():
        if "black_start" in line:
            for tok in line.split():
                if tok.startswith("black_start:"):
                    try:
                        out.append(float(tok.split(":", 1)[1]))
                    except ValueError:
                        pass
    return out


def deterministic_qc(path: Path, *, expected_duration_s: float | None = None,
                     tolerance_s: float = 0.6, mode: str = "video") -> QCReport:
    """mode="animatic" ปิดการตรวจภาพค้างและไม่บังคับว่าต้องมีเสียง

    เจอตอนทดสอบจริง: animatic คือภาพนิ่งที่ขยับช้ามาก freezedetect จะฟ้อง
    ทุก shot ที่ camera_move=static ซึ่งเป็นพฤติกรรมที่ตั้งใจ ไม่ใช่บั๊ก
    การตรวจภาพค้างมีความหมายเฉพาะกับวิดีโอที่ generate มาจริง ๆ ซึ่งเฟรมค้าง
    แปลว่าโมเดลคืนเฟรมซ้ำ

    animatic ไม่มีเสียงโดยตั้งใจ (เสียงพูดมาจากโมเดลวิดีโอตอน video_gen)
    ส่วนวิดีโอจริงต้องมีเสมอ — ไม่มีแปลว่าโมเดลไม่ได้สร้างเสียงให้ทั้งที่สั่ง"""
    issues: list[QCIssue] = []
    info = probe(path)

    vstreams = [s for s in info["streams"] if s["codec_type"] == "video"]
    astreams = [s for s in info["streams"] if s["codec_type"] == "audio"]

    if not vstreams:
        return QCReport(verdict="fail", issues=[
            QCIssue(severity="block", code="no_video", detail="ไม่มี video stream")])

    v = vstreams[0]
    w, h = int(v["width"]), int(v["height"])
    if (w, h) != (TARGET_W, TARGET_H):
        issues.append(QCIssue(
            severity="block", code="bad_resolution",
            detail=f"ได้ {w}x{h} ต้องการ {TARGET_W}x{TARGET_H}",
            suggested_fix="ตรวจ scale/crop filter ในขั้น render"))

    if v.get("pix_fmt") != "yuv420p":
        issues.append(QCIssue(
            severity="block", code="bad_pixfmt",
            detail=f"pix_fmt={v.get('pix_fmt')} แพลตฟอร์มบางเจ้าจะแสดงผลเพี้ยน",
            suggested_fix="เพิ่ม -pix_fmt yuv420p"))

    dur = float(info["format"].get("duration", 0))
    if expected_duration_s and abs(dur - expected_duration_s) > tolerance_s:
        issues.append(QCIssue(
            severity="block", code="duration_drift",
            detail=f"ยาว {dur:.2f}s แต่ timeline บอก {expected_duration_s:.2f}s",
            suggested_fix="animatic: เช็ก d ของ zoompan = duration*fps; "
                          "วิดีโอจริง: เช็กว่า retime_shots ถูกเรียกหลัง video_gen"))
    if dur > 60.5:
        issues.append(QCIssue(
            severity="warn", code="too_long",
            detail=f"ยาว {dur:.1f}s เกิน 60 วินาที YouTube จะไม่จัดเป็น Shorts"))

    if not astreams:
        if mode == "video":
            issues.append(QCIssue(
                severity="block", code="no_audio",
                detail="ไม่มี audio stream — โมเดลวิดีโอไม่ได้สร้างเสียงพูดมาให้",
                suggested_fix="เช็กว่าโมเดลใน routing profile รองรับ generate_audio"))
    else:
        ln = measure_loudness(path)
        if ln:
            try:
                measured = float(ln["input_i"])
                tp = float(ln["input_tp"])
                if abs(measured - TARGET_LUFS) > LUFS_TOLERANCE:
                    issues.append(QCIssue(
                        severity="warn", code="loudness",
                        detail=f"วัดได้ {measured:.1f} LUFS ต้องการ {TARGET_LUFS}",
                        suggested_fix="รัน loudnorm สองรอบแทนรอบเดียว"))
                if tp > -0.5:
                    issues.append(QCIssue(
                        severity="warn", code="true_peak",
                        detail=f"true peak {tp:.1f} dBTP เสี่ยงแตกบนลำโพงมือถือ"))
            except (KeyError, ValueError):
                pass

    for t in detect_black(path):
        issues.append(QCIssue(severity="block", code="black_frames",
                              detail=f"ภาพดำที่วินาที {t:.1f}"))
    freezes = detect_freeze(path) if mode == "video" else []
    if len(freezes) > 1:
        issues.append(QCIssue(
            severity="warn", code="freeze",
            detail=f"ภาพค้าง {len(freezes)} จุด แรกสุดที่ {freezes[0]:.1f}s",
            suggested_fix="โมเดล i2v อาจคืนเฟรมซ้ำ ลองเปลี่ยน motion_intent ให้ชัดขึ้น"))

    blocking = [i for i in issues if i.severity == "block"]
    verdict = "fail" if blocking else ("fix" if issues else "pass")
    return QCReport(verdict=verdict, issues=issues)


def extract_frames(path: Path, times: list[float], outdir: Path) -> list[Path]:
    """ดึงเฟรมกลาง shot ไปให้ VLM ดู — ส่งทั้งวิดีโอไปแพงโดยไม่จำเป็น

    คืนไฟล์เท่าจำนวน times เสมอและเรียงลำดับเดียวกัน เพราะ run_qc_vlm บอกโมเดล
    ว่า "เฟรมที่ i คือ shot ที่ i" ถ้าข้ามไปหนึ่งใบ ปัญหาจะถูกโยนไปผิด shot

    หนีบเวลาให้อยู่ในไฟล์ก่อนเสมอ: เคยเจอ shot ยาว 0 วินาทีที่ท้ายคลิป ทำให้
    -ss ตกพอดีที่ EOF แล้ว ffmpeg จบด้วย exit 234 โดยไม่เขียนไฟล์ ซึ่งลาก QC
    ทั้งขั้นล้มไปด้วย
    """
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        dur = float(probe(path)["format"].get("duration", 0)) or None
    except (RuntimeError, KeyError, ValueError):
        dur = None
    last = max(dur - 0.1, 0.0) if dur else None

    out = []
    for i, t in enumerate(times):
        ts = max(0.0, t if last is None else min(t, last))
        p = outdir / f"qc_{i:03d}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ts:.3f}",
             "-i", str(path), "-frames:v", "1", "-vf", "scale=540:-2",
             "-q:v", "4", str(p)],
            check=True, capture_output=True)
        out.append(p)
    return out
