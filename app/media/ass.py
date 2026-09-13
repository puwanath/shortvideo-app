"""สร้างซับ .ass แบบคาราโอเกะ (คำสว่างตามเสียง) — รูปแบบที่ short-form ใช้กันจริง

Safe zone ของ 1080x1920: UI ของแพลตฟอร์มทับพื้นที่ล่างและขวาเยอะ
  บน   ~130px  (เวลา/ปุ่มปิด)
  ล่าง ~320px  (แคปชัน ชื่อผู้ใช้ เพลง)
  ขวา  ~120px  (ปุ่มไลก์/แชร์)
ซับจึงต้องอยู่กลางค่อนล่าง แต่ไม่ต่ำกว่าเส้น 320px
"""
from __future__ import annotations

from dataclasses import dataclass

from .thai_text import wrap

W, H = 1080, 1920
SAFE_TOP, SAFE_BOTTOM, SAFE_RIGHT = 130, 320, 120


def _ts(t: float) -> str:
    """ASS ใช้ h:mm:ss.cc (เศษส่วนสองหลัก)"""
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


@dataclass
class CaptionStyle:
    font: str = "Noto Sans Thai"
    size: int = 104
    primary: str = "&H00FFFFFF"      # &HAABBGGRR — ขาว
    highlight: str = "&H0033E1FF"    # เหลืองส้ม ใช้ตอนคำถูกอ่าน
    outline_col: str = "&H00000000"
    back_col: str = "&HA0000000"
    outline: int = 7
    shadow: int = 0
    bold: int = 1
    margin_v: int = SAFE_BOTTOM + 60
    max_width_units: float = 13.0
    max_lines: int = 2


def _header(st: CaptionStyle) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st.font},{st.size},{st.highlight},{st.primary},{st.outline_col},{st.back_col},{st.bold},0,0,0,100,100,0,0,1,{st.outline},{st.shadow},2,{SAFE_RIGHT},{SAFE_RIGHT},{st.margin_v},1
Style: Title,{st.font},{int(st.size * 1.15)},&H00FFFFFF,&H00FFFFFF,{st.outline_col},{st.back_col},1,0,0,0,100,100,0,0,1,{st.outline + 1},0,5,{SAFE_RIGHT},{SAFE_RIGHT},0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def karaoke_ass(
    cues: list[list[tuple[str, float, float]]],
    style: CaptionStyle | None = None,
    *,
    on_screen: list[tuple[str, float, float]] | None = None,
) -> str:
    """cues = ผลจาก group_words_into_cues
    on_screen = ข้อความใหญ่กลางจอ (hook / ตัวเลข) เป็น (text, start, end)
    """
    st = style or CaptionStyle()
    lines = [_header(st)]

    for cue in cues:
        if not cue:
            continue
        start, end = cue[0][1], cue[-1][2]
        # \k ใช้หน่วยเซนติวินาที
        parts = []
        for word, ws, we in cue:
            dur_cs = max(1, int(round((we - ws) * 100)))
            parts.append(f"{{\\k{dur_cs}}}{word}")
        text = "".join(parts)

        # ตัดบรรทัดตามขอบเขตคำ โดยไม่ทำลาย \k tag
        plain = "".join(w for w, _, _ in cue)
        wrapped = wrap(plain, st.max_width_units, st.max_lines)
        if len(wrapped) > 1:
            text = _apply_wrap(cue, wrapped)

        lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{text}")

    for text, s, e in on_screen or []:
        wrapped = wrap(text, 12.0, 3)
        body = "\\N".join(wrapped)
        lines.append(
            f"Dialogue: 1,{_ts(s)},{_ts(e)},Title,,0,0,0,,"
            f"{{\\pos({W // 2},{int(H * 0.34)})\\fad(180,180)}}{body}"
        )

    return "\n".join(lines) + "\n"


def _apply_wrap(cue: list[tuple[str, float, float]], wrapped: list[str]) -> str:
    """ใส่ \\N ลงใน karaoke string ตรงจุดที่ wrap() คำนวณไว้
    ทำโดยเดินตามความยาวสะสมของแต่ละบรรทัด"""
    targets = [len(l) for l in wrapped]
    out: list[str] = []
    consumed = 0
    line_i = 0
    for word, ws, we in cue:
        dur_cs = max(1, int(round((we - ws) * 100)))
        if line_i < len(targets) - 1 and consumed >= targets[line_i]:
            out.append("\\N")
            line_i += 1
            consumed = 0
        out.append(f"{{\\k{dur_cs}}}{word}")
        consumed += len(word)
    return "".join(out)


def safe_zone_overlay_ass() -> str:
    """เส้นบอกเขตปลอดภัยสำหรับดูตอนรีวิว ไม่ใช้ใน render จริง"""
    st = CaptionStyle()
    body = _header(st)
    box = (f"{{\\p1\\bord0\\shad0\\1a&H80&\\1c&HFF0000&\\pos(0,0)}}"
           f"m 0 0 l {W} 0 l {W} {SAFE_TOP} l 0 {SAFE_TOP}{{\\p0}}")
    box2 = (f"{{\\p1\\bord0\\shad0\\1a&H80&\\1c&HFF0000&\\pos(0,0)}}"
            f"m 0 {H - SAFE_BOTTOM} l {W} {H - SAFE_BOTTOM} l {W} {H} l 0 {H}{{\\p0}}")
    body += f"Dialogue: 9,0:00:00.00,9:59:59.99,Title,,0,0,0,,{box}\n"
    body += f"Dialogue: 9,0:00:00.00,9:59:59.99,Title,,0,0,0,,{box2}\n"
    return body
