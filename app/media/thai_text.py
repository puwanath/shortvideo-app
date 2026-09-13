"""ภาษาไทยกับ libass

ปัญหา: ไทยไม่มีช่องว่างระหว่างคำ libass ตัดบรรทัดที่ช่องว่างเท่านั้น
ผลคือประโยคยาวจะล้นจอ หรือถ้าใส่ \\N เองแบบมั่ว ๆ ก็จะตัดกลางคำ

วิธีแก้: ตัดคำด้วย pythainlp แล้วแทรก U+200B (zero-width space) ระหว่างคำ
libass ถือว่า ZWSP เป็นจุดตัดบรรทัดได้ แต่ไม่กินความกว้าง
สำหรับซับคาราโอเกะเราคุมความยาวบรรทัดเองอยู่แล้ว จึงใช้ \\N ตรงจุดที่คำนวณไว้

หน่วยความกว้าง: ใช้จำนวน "คลัสเตอร์" ไม่ใช่ len() เพราะสระบน/ล่างและวรรณยุกต์
ไม่กินความกว้างแนวนอน ถ้านับด้วย len() บรรทัดจะสั้นกว่าที่ควรเยอะ
"""
from __future__ import annotations

import bisect
import re
import unicodedata

ZWSP = "\u200b"

# สระบน สระล่าง วรรณยุกต์ และเครื่องหมายที่ซ้อนบนพยัญชนะ ไม่กินความกว้าง
_COMBINING = set(
    "\u0e31\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39\u0e3a"
    "\u0e47\u0e48\u0e49\u0e4a\u0e4b\u0e4c\u0e4d\u0e4e"
)

try:
    from pythainlp.tokenize import word_tokenize as _pythai_tokenize
    _HAS_PYTHAINLP = True
except ImportError:  # pragma: no cover
    _HAS_PYTHAINLP = False


def has_thai(s: str) -> bool:
    return any("\u0e00" <= ch <= "\u0e7f" for ch in s)


def display_width(s: str) -> float:
    """ความกว้างโดยประมาณเป็นหน่วย 'ตัวอักษรไทย 1 ตัว'
    ละตินแคบกว่าไทยราว 0.6 เท่าที่ขนาดฟอนต์เดียวกัน"""
    w = 0.0
    for ch in s:
        if ch in _COMBINING or unicodedata.combining(ch):
            continue
        if ch == ZWSP:
            continue
        if "\u0e00" <= ch <= "\u0e7f":
            w += 1.0
        elif ch.isspace():
            w += 0.5
        else:
            w += 0.62
    return w


def tokenize(text: str) -> list[str]:
    """คืนรายการคำ ถ้าไม่มี pythainlp จะ fallback เป็นการตัดหยาบ ๆ
    (fallback ใช้ได้ในเทสต์เท่านั้น production ต้องติดตั้ง pythainlp)"""
    if not has_thai(text):
        return text.split()
    if _HAS_PYTHAINLP:
        return [t for t in _pythai_tokenize(text, engine="newmm", keep_whitespace=True) if t]
    return re.findall(r"[\u0e00-\u0e7f]+|[^\u0e00-\u0e7f]+", text)


def insert_break_opportunities(text: str) -> str:
    """แทรก ZWSP ระหว่างคำ ให้ libass มีจุดตัดบรรทัดที่ถูกต้อง
    ใช้กับข้อความที่ปล่อยให้ libass wrap เอง เช่น ข้อความบนจอที่ยาวไม่แน่นอน"""
    toks = tokenize(text)
    out = []
    for i, t in enumerate(toks):
        out.append(t)
        if i < len(toks) - 1 and not t.isspace() and not toks[i + 1].isspace():
            out.append(ZWSP)
    return "".join(out)


def wrap(text: str, max_width: float = 16.0, max_lines: int | None = None) -> list[str]:
    """ตัดบรรทัดตามขอบเขตคำ คืนรายการบรรทัด

    max_width นับเป็นหน่วยตัวอักษรไทย ค่า 15-16 เหมาะกับซับตัวใหญ่บนจอกว้าง 1080

    ตัดที่ max_width เสมอ ไม่เคยปล่อยให้บรรทัดล้น — บรรทัดที่ล้นคือบรรทัดที่
    คนอ่านไม่ทันและตัวหนังสือจะโดนขอบจอกิน ถ้าผลลัพธ์เกิน max_lines แปลว่า
    ข้อความยาวเกินไปสำหรับ cue เดียว ซึ่งเป็นปัญหาของชั้นที่จัดกลุ่มคำ
    ไม่ใช่ปัญหาของการตัดบรรทัด — ผู้เรียกต้องตรวจ len() เอง

    คำเดี่ยวที่กว้างเกิน max_width จะได้บรรทัดของตัวเอง (ไม่ตัดกลางคำ
    เพราะภาษาไทยตัดกลางคำแล้วอ่านไม่ออก)
    """
    toks = tokenize(text)
    lines: list[str] = []
    cur = ""
    for t in toks:
        if t.isspace():
            if cur:
                cur += t
            continue
        cand = cur + t
        if cur and display_width(cand) > max_width:
            lines.append(cur.strip())
            cur = t
        else:
            cur = cand
    if cur.strip():
        lines.append(cur.strip())
    if max_lines is not None and len(lines) > max_lines:
        log_overflow(text, len(lines), max_lines)
    return lines or [""]


def log_overflow(text: str, got: int, allowed: int) -> None:
    import logging
    logging.getLogger(__name__).warning(
        "ข้อความยาวเกิน cue เดียว: ได้ %d บรรทัด เกิน %d — %.30s…", got, allowed, text)


def spread_over_span(toks: list[str], start: float, end: float) -> list[tuple[str, float, float]]:
    """กระจายคำลงบนช่วงเวลาเดียวตามความกว้างที่มองเห็น

    ใช้ display_width ไม่ใช่ len() เพราะสระบน/ล่างและวรรณยุกต์ไม่กินเวลาพูดเพิ่ม
    """
    if not toks:
        return []
    span = max(end - start, 1e-3)
    widths = [max(display_width(t), 0.5) for t in toks]
    total = sum(widths)
    out: list[tuple[str, float, float]] = []
    t = start
    for tok, w in zip(toks, widths):
        d = span * (w / total)
        out.append((tok, round(t, 3), round(min(t + d, end), 3)))
        t += d
    return out


def align_script_to_spans(script: str,
                          spans: list[tuple[float, float]]) -> list[tuple[str, float, float]]:
    """ปูคำจาก *บทที่รู้อยู่แล้ว* ลงบนช่วงเวลาของวลีที่ ASR หามาให้

    นี่คือหัวใจของการทำ forced alignment ด้วย ASR ธรรมดา และมีเหตุผลสองชั้น:

    1. **ห้ามใช้ข้อความที่ ASR ถอดได้เป็นซับ** — วัดกับของจริงแล้ว ASR ฟังเสียง
       TTS ผิดเกือบทั้งประโยค ("เครื่องแรงที่สุด ไม่ได้แปลว่าลูกจะเก่งที่สุดนะลูก"
       ถูกถอดเป็น "ไม่ได้แปลว่าลูกจะเก่งที่สุดแล้วลูก") ถ้าเอาไปทำคาราโอเกะ
       คนดูจะเห็นคำที่ไม่ตรงกับเสียง ทั้งที่เรามีบทต้นฉบับอยู่ในมือ

    2. **ห้ามใช้ word timestamp ของ ASR กับภาษาไทย** — มันตัดคำด้วยช่องว่าง
       ภาษาไทยไม่มี ผลคือได้ก้อนละทั้งวลี และหลายก้อน start == end

    จึงใช้ ASR แค่ "เวลาของวลี" ซึ่งเชื่อได้เพราะมาจากเสียงจริง แล้วปูบทลงไป
    โดยแบ่งให้แต่ละวลีตามสัดส่วนความยาวเวลา ความคลาดเคลื่อนถูกขังอยู่ในวลีเดียว
    ไม่สะสมข้ามคลิปแบบการหารเฉลี่ยทั้งไฟล์
    """
    toks = [t for t in tokenize(script) if t.strip()]
    spans = [(a, b) for a, b in spans if b > a]
    if not toks or not spans:
        return []

    widths = [max(display_width(t), 0.5) for t in toks]
    total_w = sum(widths)
    total_t = sum(b - a for a, b in spans)
    if total_t <= 0:
        return []

    # จุดตัดของ "คำ" ที่ความกว้างสะสมตรงกับเวลาสะสมของวลี
    cuts: list[int] = []
    acc_t = 0.0
    for a, b in spans[:-1]:
        acc_t += b - a
        target = total_w * (acc_t / total_t)
        run, k = 0.0, 0
        while k < len(toks) and run + widths[k] / 2 < target:
            run += widths[k]
            k += 1
        cuts.append(k)
    # ไม่ถอยหลัง และต้องเหลือคำให้วลีที่ยังไม่ได้แบ่ง
    for i in range(len(cuts)):
        lo = cuts[i - 1] if i else 0
        hi = len(toks) - (len(cuts) - i)
        cuts[i] = max(lo, min(cuts[i], hi))

    out: list[tuple[str, float, float]] = []
    start_i = 0
    for i, (a, b) in enumerate(spans):
        end_i = cuts[i] if i < len(cuts) else len(toks)
        out.extend(spread_over_span(toks[start_i:end_i], a, b))
        start_i = end_i
    return out


def group_words_into_cues(
    words: list[tuple[str, float, float]],
    *,
    max_width: float = 15.0,
    max_gap_s: float = 0.55,
    max_dur_s: float = 3.2,
    boundaries: list[float] | None = None,
) -> list[list[tuple[str, float, float]]]:
    """จัดคำเป็นกลุ่มละ 1 cue สำหรับซับคาราโอเกะ

    ตัดกลุ่มเมื่อ:
      * ข้ามรอยต่อ shot  — สำคัญที่สุด ซับที่คร่อมรอยตัดภาพดูเหมือนงานพัง
      * เงียบนานเกิน max_gap_s — ซับค้างข้ามช่วงเงียบทำให้จังหวะเสีย
      * กว้างเกิน max_width
      * ยาวเกิน max_dur_s

    boundaries = เวลาที่ shot เปลี่ยน (วินาที) ส่งมาจาก shot list
    """
    cues: list[list[tuple[str, float, float]]] = []
    cur: list[tuple[str, float, float]] = []
    bounds = sorted(boundaries or [])

    def shot_of(t: float) -> int:
        """คำนี้อยู่ shot ไหน — bisect_right ทำให้คำที่เริ่มตรงรอยต่อพอดี
        ถูกนับเป็นของ shot ถัดไป ซึ่งเป็นสิ่งที่ต้องการ

        เดิมเช็กว่ามี boundary ตกอยู่ใน "ช่องว่าง" ระหว่างคำสองคำ
        (t0 < b <= t1) ซึ่งใช้ไม่ได้จริง เพราะ runner ส่ง boundary เป็น end_s
        ของคำสุดท้ายใน beat พอดี ทำให้ t0 < b เป็นเท็จเสมอ และถ้า word
        timestamp ติดกัน (t0 == t1 แบบที่ forced alignment มักให้) ก็ไม่มีค่า b
        ใดผ่านเงื่อนไขได้เลย ซับจึงคร่อมรอยตัดภาพทุกครั้ง
        """
        return bisect.bisect_right(bounds, t)

    for w in words:
        if not cur:
            cur = [w]
            continue
        gap = w[1] - cur[-1][2]
        width = display_width("".join(x[0] for x in cur) + w[0])
        dur = w[2] - cur[0][1]
        if (shot_of(w[1]) != shot_of(cur[0][1])
                or gap > max_gap_s
                or width > max_width
                or dur > max_dur_s):
            cues.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        cues.append(cur)
    return cues
