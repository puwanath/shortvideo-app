"""agent แต่ละขั้น — แต่ละตัวรับ state เข้า คืน object ที่ validate แล้วออก

เจตนา: ไม่มี agent ตัวไหน "คุย" กับตัวอื่น ทุกตัวอ่านจาก run แล้วเขียนกลับลง run
ทำให้ resume ได้ ทดสอบทีละตัวได้ และรู้เสมอว่าอะไรพังตรงไหน

ลำดับ (ตั้งแต่ 2026-09-13): บท → shot plan → ภาพตัวละคร → keyframe → animatic
→ วิดีโอ (โมเดลสร้างเสียงพูดเอง) — ไม่มี TTS ไม่มี alignment ไม่มีซับ
เวลาของแต่ละ shot จึงมาจาก Beat.duration_s ที่ LLM กำหนด ไม่ใช่จากเสียงจริง
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from ..providers.base import ImageRequest, JobHandle, VideoRequest
from ..providers.registry import Registry, Route
from ..schemas import (
    ConceptOut, KenBurns, PostMeta, QCReport, Shot, ShotPlanOut, Story,
)
from .formats import (
    AVATAR_STORY_ADDENDUM, DESIGN_SHEET_PROMPT, PORTRAIT_PROMPT, RENDER_MODES, fmt as _fmt,
)
from .llm import structured

log = logging.getLogger(__name__)


@dataclass
class GenRecord:
    """หนึ่งครั้งที่เรียก provider — cost คือค่าจริงที่ provider รายงาน ไม่ใช่ประมาณ
    runner เอาไปเขียนเป็นแถว Generation เพื่อดูย้อนหลังได้ว่า shot ไหน job ไหน เท่าไหร่"""
    stage: str
    route: Route
    cost_usd: float
    shot_id: str | None = None
    external_id: str | None = None


TH = "ตอบเป็นภาษาไทยทั้งหมด ยกเว้นฟิลด์ที่ลงท้ายด้วย _en และ image_prompt ที่ต้องเป็นภาษาอังกฤษ"


# ---------------------------------------------------------------- 1 concept

CONCEPT_SYS = f"""คุณคือคนคิดคอนเทนต์วิดีโอสั้นที่ทำงานกับตลาดไทยมานาน
งานของคุณคือหามุมเล่าที่ทำให้คนหยุดนิ้วภายใน 3 วินาทีแรก

หลักที่ยึด:
- มุมที่ดีคือมุมที่ขัดกับสิ่งที่คนส่วนใหญ่เชื่อ หรือบอกสิ่งที่คนกลัวว่าจะพลาด
- อย่าเริ่มด้วยการแนะนำตัวหรือเกริ่น คนเลื่อนผ่านทันที
- เสนอมุมที่ต่างกันจริง ไม่ใช่มุมเดิมเขียนใหม่สามแบบ
- บอกจุดเสี่ยงของแต่ละมุมตามตรง

{TH}"""


async def run_concept(reg: Registry, brief: str, brand: dict, budget=None):
    user = (
        f"บรีฟ: {brief}\n\n"
        f"โทนแบรนด์: {brand.get('tone_of_voice') or 'ไม่ระบุ'}\n"
        f"ข้อห้าม: {brand.get('do_donts') or 'ไม่ระบุ'}\n\n"
        "เสนอมุมเล่า 3 มุมแล้วเลือกมาหนึ่งมุมพร้อมเหตุผล"
    )
    return await structured(reg, "concept", ConceptOut,
                            system=CONCEPT_SYS, user=user, budget=budget)


# ---------------------------------------------------------------- ความยาวที่โมเดลวิดีโอรับ

async def video_durations(reg: Registry) -> list[int]:
    """ชุดความยาว (วินาที) ที่โมเดลวิดีโอตัวหลักรับจริง — ถามก่อนเขียนบท

    ต้องรู้ตั้งแต่ตอนวางบท เพราะคลิปที่ได้มีเสียงพูดอยู่ข้างใน จะตัดท้ายทิ้ง
    ให้พอดี timeline แบบเดิมไม่ได้แล้ว (บทพูดจะขาด) ความยาวที่ LLM เลือก
    จึงต้องเป็นค่าที่โมเดลรับได้เป๊ะ

    ลำดับที่เชื่อ: params.durations ใน routing profile (คนตั้งเอง) → catalog
    ของ provider → ว่าง (ปล่อยอิสระ แล้วไปปัดตอนสั่งวิดีโอ)
    """
    try:
        routes = reg.routes_for("video")
    except Exception as e:  # noqa: BLE001 — ไม่มี route video = ยังเขียนบทได้
        log.warning("อ่าน route ของ stage video ไม่ได้ (%s) — ปล่อยความยาวอิสระ", e)
        return []
    if not routes:
        return []
    primary = routes[0]
    declared = primary.params.get("durations")
    if declared:
        return sorted({int(d) for d in declared})
    try:
        caps = await reg.provider(primary.provider).capabilities(primary.model)
        return sorted({int(d) for d in caps.durations if float(d).is_integer()})
    except Exception as e:  # noqa: BLE001
        log.warning("ถามความยาวที่ %s รองรับไม่ได้ (%s) — ปล่อยความยาวอิสระ",
                    primary.model, e)
        return []


SNAP_SLACK_S = 0.3


def snap_duration(want: float, allowed: list[int] | list[float]) -> float:
    """ปัด *ขึ้น* ไปหาค่าที่โมเดลรับ — ปัดลงจะทำให้บทพูดขาด

    ยกเว้นเมื่อ want ห่างจากค่าที่รับได้ไม่ถึง SNAP_SLACK_S: ใช้ค่านั้นเลย
    เจอจริง: หลัง retime_shots ความยาว shot กลายเป็น 6.05 (คลิปจริงยาวเกิน
    6 นิดเดียว) พอ QC สั่งสร้าง shot นั้นใหม่ ปัดขึ้นแบบเถรตรงได้ 7 วินาที
    คลิปใหม่ยาวเกินเพื่อนไปหนึ่งวินาทีเต็มโดยไม่มีเหตุผล
    """
    if not allowed:
        return want
    near = min(allowed, key=lambda d: abs(d - want))
    if abs(near - want) <= SNAP_SLACK_S:
        return float(near)
    up = [d for d in allowed if d >= want]
    return float(min(up)) if up else float(max(allowed))


def _durations_hint(allowed: list[int]) -> str:
    if not allowed:
        return "ความยาวแต่ละ beat เป็นจำนวนเต็มวินาที"
    return (f"duration_s ของแต่ละ beat ต้องเป็นค่าใดค่าหนึ่งใน {allowed} เท่านั้น "
            "(เป็นชุดที่โมเดลวิดีโอรองรับ ค่าอื่นจะถูกปัดขึ้น)")


# ---------------------------------------------------------------- 2 story

def story_sys(format_name: str | None, render: str = "scene") -> str:
    sys_ = _fmt(format_name)["story_sys"].replace("{TH}", TH)
    if render == "avatar":
        sys_ += "\n" + AVATAR_STORY_ADDENDUM
    return sys_


async def run_story(reg: Registry, concept: ConceptOut, brief: str,
                    target_s: int, brand: dict, allowed_durations: list[int],
                    budget=None, format_name: str | None = None, render: str = "scene",
                    product_images: list[bytes] | None = None):
    f = _fmt(format_name)
    angle = concept.angles[concept.chosen_index]
    user = (
        f"บรีฟเดิม: {brief}\n\n"
        f"มุมที่เลือก: {angle.title}\n{angle.premise}\n"
        f"เหตุผล: {concept.rationale}\n"
        f"จุดเสี่ยงที่ต้องเลี่ยง: {angle.risk}\n\n"
        f"ความยาวเป้าหมาย {target_s} วินาที\n"
        f"{_durations_hint(allowed_durations)}\n"
        f"โทน: {brand.get('tone_of_voice') or f['default_tone']}\n"
        f"สไตล์ภาพของคลิป (ออกแบบตัวละครให้เข้ากับสไตล์นี้): "
        f"{brand.get('style_suffix') or 'live-action, natural light'}\n"
        + (f"\nแนบภาพสินค้า {len(product_images)} ภาพมาด้วย — ดูภาพแล้วอธิบายสินค้า "
           "(รูปทรง สี ของที่อยู่ในภาพ) ให้ตรงของจริงในบทและใน visual_intent\n"
           if product_images else "")
        + "\nเขียนบทเต็มออกมา"
    )
    return await structured(reg, "story", Story, system=story_sys(format_name, render),
                            user=user, images=product_images or None, budget=budget)


# ---------------------------------------------------------------- 3 shots

SHOT_SYS = """You plan the visual shots for a vertical 9:16 short-form video in which
the characters speak on camera. The video model will animate each still and
generate the speech itself, so every shot with dialogue must show the speaker
clearly enough to be lip-synced.

Rules:
- One shot per beat, same order, same count. Never merge or split beats.
- characters: names (exactly as given) of the characters visible in the frame.
  The speaker of a beat must be in the frame unless the beat is a cutaway.
- image_prompt must be ENGLISH, describe a still frame, and include:
  subject, setting, lighting, lens feel, and mood. Refer to characters by
  their given description, not by name. No text or words in the image
  (AI models render text badly and it looks broken).
- Vary composition across shots. Six close-ups in a row is a dead video.
- motion_intent is the CAMERA only. One clear movement. Complex camera motion
  breaks i2v models.
- camera_move must be consistent with motion_intent.
{RULES}

Return JSON only."""


def shot_sys(format_name: str | None) -> str:
    return SHOT_SYS.replace("{RULES}", _fmt(format_name)["shot_rules"])


async def run_shot_plan(reg: Registry, story: Story, brand: dict, budget=None,
                        format_name: str | None = None):
    chars_txt = "\n".join(
        f"- {c.name} ({c.role}): {c.appearance_en}" for c in story.characters)
    beats_txt = "\n".join(
        f"{b.idx}. [{b.role}] {b.duration_s}s "
        + (f"{b.speaker} says: \"{b.dialogue}\"" if b.dialogue else "(no dialogue)")
        + f"  — visual: {b.visual_intent}" + (f"  — sfx: {b.sfx}" if b.sfx else "")
        for b in story.beats
    )
    user = (
        f"Video title: {story.title}\nTone: {story.tone}\n"
        f"Visual style to follow: {brand.get('style_suffix') or 'clean, natural light, documentary feel'}\n\n"
        f"Characters:\n{chars_txt}\n\n"
        f"Beats ({len(story.beats)} total, produce exactly {len(story.beats)} shots):\n{beats_txt}"
    )
    out, route, cost = await structured(reg, "shots", ShotPlanOut,
                                        system=shot_sys(format_name), user=user, budget=budget)
    if len(out.shots) != len(story.beats):
        raise ValueError(
            f"shot planner คืน {len(out.shots)} shot แต่มี {len(story.beats)} beat")
    known = {c.name for c in story.characters}
    for sp in out.shots:
        unknown = [n for n in sp.characters if n not in known]
        if unknown:
            log.warning("shot beat=%s อ้างตัวละครที่ไม่มี %s — ตัดทิ้ง", sp.beat_idx, unknown)
            sp.characters = [n for n in sp.characters if n in known]
    return out, route, cost


def build_shots(plan: ShotPlanOut, story: Story,
                allowed_durations: list[int] | None = None) -> list[Shot]:
    """รวม plan เข้ากับ timeline จากบท — จุดที่ timeline กลายเป็นของจริง

    ความยาวแต่ละ shot = Beat.duration_s ปัดขึ้นให้ตรงชุดที่โมเดลวิดีโอรับ
    เรียงต่อกันจาก 0 ไม่มีช่องว่าง
    """
    by_idx = {b.idx: b for b in story.beats}
    shots: list[Shot] = []
    t = 0.0
    for i, sp in enumerate(plan.shots):
        beat = by_idx.get(sp.beat_idx)
        if beat is None and i < len(story.beats):
            # สัญญาคือ "หนึ่ง shot ต่อหนึ่ง beat เรียงลำดับเดียวกัน" และ
            # run_shot_plan บังคับจำนวนให้เท่ากันแล้ว ตำแหน่งจึงเชื่อถือได้กว่า
            # beat_idx ที่โมเดลกรอกมา — ใช้เป็นตาข่ายรับเมื่อโมเดลนับเลขคนละฐาน
            beat = story.beats[i]
            log.warning("shot %d อ้าง beat_idx %s ที่ไม่มีในบท — ใช้ beat ตำแหน่งที่ %d แทน",
                        i, sp.beat_idx, i)
        if beat is None:
            raise ValueError(f"shot อ้าง beat_idx {sp.beat_idx} ที่ไม่มีในบท")
        dur = snap_duration(beat.duration_s, allowed_durations or [])
        if dur != beat.duration_s:
            log.info("beat %d ขอ %ds — ปัดเป็น %.0fs ตามที่โมเดลวิดีโอรับ",
                     beat.idx, beat.duration_s, dur)
        shots.append(Shot(
            id=f"s{i:03d}",
            idx=i,
            beat_idx=beat.idx,
            start_s=round(t, 3),
            end_s=round(t + dur, 3),
            plan=sp,
            ken_burns=KenBurns.from_camera(sp.camera_move),
        ))
        t += dur
    return shots


def retime_shots(shots: list[Shot], actual: dict[str, float]) -> None:
    """เขียน timeline ทับด้วยความยาวจริงของคลิปที่ได้มา — แก้ในที่

    โมเดลวิดีโออาจคืนคลิปยาวไม่เท่าที่สั่งพอดี (เช่น 5.04s) และเราตัดไม่ได้
    เพราะมีเสียงพูดอยู่ข้างใน timeline ที่ QC / metadata / เฟรมกลาง shot ใช้
    จึงต้องตามของจริง ไม่ใช่ตามที่วางไว้
    """
    t = 0.0
    for s in shots:
        dur = actual.get(s.id, s.duration_s)
        s.start_s = round(t, 3)
        s.end_s = round(t + dur, 3)
        t += dur


# ---------------------------------------------------------------- 4 ภาพตัวละคร

DEFAULT_STYLE = "live-action short film, photorealistic, natural light"


async def run_character_sheets(reg: Registry, story: Story, brand: dict,
                               outdir: Path, budget=None):
    """สองใบต่อตัวละคร — portrait (ส่งให้โมเดลวิดีโอ/avatar และเป็น reference หลัก)
    กับ design sheet (หน้าตา ชุด สีหน้า อิริยาบถ ของประจำตัว — "ตัวละครคือใคร")

    แทนที่ style anchor เดิมที่เป็นภาพฉากรวม: โมเดลภาพรับ input_references ได้หลายใบ
    ส่งภาพตัวละครที่ปรากฏใน shot นั้น ๆ ตรง ๆ แม่นกว่าให้มันเดาจากภาพฉากเดียว
    design sheet ทำหลัง portrait โดยใช้ portrait เป็น reference เพื่อให้เป็นคนเดียวกัน
    """
    outdir.mkdir(parents=True, exist_ok=True)
    style = brand.get("style_suffix") or DEFAULT_STYLE
    sheets: list[dict] = []
    records: list[GenRecord] = []

    async def gen(prompt: str, what: str, refs: list[bytes], negative: str):
        async def call(prov, route: Route):
            return await prov.generate_image(ImageRequest(
                prompt=prompt, model=route.model, n=1, negative_prompt=negative,
                reference_images=refs,
                aspect_ratio=route.params.get("aspect", "9:16"),
                size=route.params.get("size", "1080x1920"),
            ))
        resp, route = await reg.run("character", call, budget=budget)
        if budget and resp.cost_usd:
            await budget.charge(resp.cost_usd, what=what)
        records.append(GenRecord("character", route, resp.cost_usd, shot_id=what))
        return resp.images[0]

    for i, ch in enumerate(story.characters):
        portrait = await gen(
            PORTRAIT_PROMPT.format(appearance=ch.appearance_en, style=style).strip(),
            f"char:{ch.name}", [], "text, watermark, logo, multiple people, collage, full body")
        p1 = outdir / f"char_{i}.png"
        p1.write_bytes(portrait)
        sheet = await gen(
            DESIGN_SHEET_PROMPT.format(appearance=ch.appearance_en, style=style).strip(),
            f"sheet:{ch.name}", [portrait], "text, letters, watermark, logo, photo collage")
        p2 = outdir / f"char_{i}_sheet.png"
        p2.write_bytes(sheet)
        sheets.append({"name": ch.name, "role": ch.role, "ref_path": str(p1),
                       "sheet_path": str(p2),
                       "cost_usd": sum(r.cost_usd for r in records[-2:])})
    return sheets, records


# ---------------------------------------------------------------- 5 ภาพนิ่ง

async def run_keyframes(reg: Registry, shots: list[Shot], story: Story,
                        character_refs: dict[str, Path], brand: dict, outdir: Path,
                        only: list[str] | None = None, concurrency: int = 3,
                        budget=None, character_sheets: dict[str, Path] | None = None,
                        product_refs: list[Path] | None = None) -> list[GenRecord]:
    """สร้างภาพนิ่งราย shot — only=[shot_id] สำหรับ regenerate เฉพาะตัว
    คืนรายการ GenRecord หนึ่งตัวต่อหนึ่ง call พร้อม cost จริงจาก provider

    ภาพอ้างอิง = ภาพตัวละครที่ shot นั้นระบุว่าปรากฏ (ถ้าไม่ระบุ ส่งทุกตัว)
    และเติมคำอธิบายหน้าตาลง prompt ด้วย — ภาพอ้างอิงคุมหน้า คำอธิบายคุมเสื้อผ้า
    """
    outdir.mkdir(parents=True, exist_ok=True)
    ref_bytes = {name: p.read_bytes() for name, p in character_refs.items() if p.exists()}
    sheet_bytes = {name: p.read_bytes() for name, p in (character_sheets or {}).items() if p.exists()}
    subject_refs = [Path(p).read_bytes() for p in (brand.get("subject_ref_paths") or [])
                    if Path(p).exists()]
    # ภาพสินค้าที่แนบมากับ task — ส่งเป็น reference ทุก shot ให้ของในภาพเป็นของจริง
    product_bytes = [p.read_bytes() for p in (product_refs or []) if p.exists()]
    sem = asyncio.Semaphore(concurrency)
    records: list[GenRecord] = []
    suffix = brand.get("style_suffix") or ""

    async def one(shot: Shot):
        async with sem:
            names = shot.plan.characters or list(ref_bytes)
            refs = ([ref_bytes[n] for n in names if n in ref_bytes]
                    + [sheet_bytes[n] for n in names if n in sheet_bytes]
                    + subject_refs + product_bytes)
            desc = "; ".join(
                f"{c.appearance_en}" for n in names if (c := story.character(n)))
            prompt = f"{shot.plan.image_prompt}. {shot.plan.composition} shot."
            if desc:
                prompt += f" The people in frame look exactly like the reference images: {desc}."
            if product_bytes:
                prompt += (" The product shown must match the attached product photos exactly "
                           "(shape, colors, label) — do not redesign it.")
            prompt = f"{prompt} {suffix}".strip()

            async def call(prov, route: Route):
                return await prov.generate_image(ImageRequest(
                    prompt=prompt,
                    model=route.model,
                    negative_prompt=shot.plan.negative_prompt or "text, watermark, logo, caption",
                    reference_images=refs,
                    aspect_ratio=route.params.get("aspect", "9:16"),
                    size=route.params.get("size", "1080x1920"),
                ))

            resp, route = await reg.run("keyframe", call, budget=budget)
            if budget and resp.cost_usd:
                await budget.charge(resp.cost_usd, what=f"keyframe:{shot.id}")
            records.append(GenRecord("keyframe", route, resp.cost_usd, shot_id=shot.id))
            p = outdir / f"{shot.id}.png"
            p.write_bytes(resp.images[0])
            shot.keyframe_path = str(p)
            shot.keyframe_cost_usd += resp.cost_usd
            shot.regen_count += 1 if only else 0

    targets = [s for s in shots if not only or s.id in only]
    if not only:
        # resume: ภาพที่มีอยู่แล้วไม่สร้างซ้ำ (regenerate ระบุ only จึงไม่เข้าทางนี้)
        targets = [s for s in targets
                   if not (s.keyframe_path and Path(s.keyframe_path).exists())]
    await asyncio.gather(*(one(s) for s in targets))
    return records


# ---------------------------------------------------------------- 6 วิดีโอ

HEYGEN_VOICE_FEMALE = "80441555167a467e967ab9487d844a30"   # Kore, Multilingual
HEYGEN_VOICE_MALE = "3097f9a8fd3b4340b6bbe913177b378f"     # Orus, Multilingual
_FEMALE_WORDS = ("female", "girl", "woman", "she ", "her ", "lady", "mother", "mom", "aunt", "grandma")


def avatar_voice_id(params: dict, voice_en: str) -> str:
    if params.get("voice_id"):
        return params["voice_id"]
    v = (voice_en or "").lower()
    female = any(w in v for w in _FEMALE_WORDS)
    return params.get("voice_id_female", HEYGEN_VOICE_FEMALE) if female \
        else params.get("voice_id_male", HEYGEN_VOICE_MALE)


def video_prompt(shot: Shot, story: Story, brand: dict | None = None) -> str:
    """เรียง: บทพูด → การกระทำ → ภาพ → กล้อง → สไตล์

    บทพูดต้องมาก่อน เพราะโมเดลเป็นคนสร้างเสียงเอง และ i2v ให้น้ำหนักคำต้น ๆ
    มากที่สุด ถ้าไปอยู่ท้ายจะได้คลิปที่ปากขยับแต่ไม่ตรงประโยค หรือไม่พูดเลย
    action ก่อน camera ด้วยเหตุผลเดิม: ส่งแต่ camera move จะได้ Ken Burns
    """
    beat = next((b for b in story.beats if b.idx == shot.beat_idx), None)
    parts: list[str] = []
    if beat and beat.dialogue and beat.speaker:
        ch = story.character(beat.speaker)
        voice = f" ({ch.voice_en})" if ch and ch.voice_en else ""
        parts.append(f"The {ch.role if ch else beat.speaker}{voice} speaks in Thai, "
                     f"lip-synced, saying exactly: \"{beat.dialogue}\"")
    else:
        parts.append("No one speaks")
    if beat and beat.sfx:
        parts.append(f"Sound: {beat.sfx}")
    elif not (beat and beat.dialogue):
        parts.append("Natural ambient sound only")
    if shot.plan.subject_action:
        parts.append(shot.plan.subject_action)
    parts.append(shot.plan.image_prompt)
    parts.append(f"Camera: {shot.plan.motion_intent}")
    style = (brand or {}).get("style_suffix")
    if style:
        # i2v ตามภาพตั้งต้นเป็นหลัก แต่บอกสไตล์ซ้ำกันโมเดลค่อย ๆ ดริฟต์ไปทางสมจริง
        parts.append(f"Style: {style}")
    return ". ".join(p.rstrip(".") for p in parts) + "."


async def run_video(reg: Registry, shots: list[Shot], story: Story, outdir: Path,
                    brand: dict | None = None, only: list[str] | None = None,
                    poll_every: float = 10.0, max_wait_s: float = 900.0,
                    budget=None, render: str = "scene",
                    character_refs: dict[str, Path] | None = None,
                    ) -> tuple[list[GenRecord], list[tuple[str, Exception]]]:
    """i2v ราย shot พร้อมเสียงจากโมเดล — submit ทั้งหมดก่อน แล้วค่อย poll พร้อมกัน
    ไม่ทำแบบ submit-รอ-submit-รอ เพราะจะช้าเป็นผลรวมแทนที่จะเป็นค่ามากสุด

    cost ของแต่ละคลิปคือ usage.cost ที่ provider รายงานตอน job เสร็จ (ไม่ใช่ประมาณ)
    บันทึกลง Shot.clip_cost_usd และคืนเป็น GenRecord ต่อ job"""
    outdir.mkdir(parents=True, exist_ok=True)
    stage = RENDER_MODES.get(render, RENDER_MODES["scene"])["video_stage"]
    targets = [s for s in shots if (not only or s.id in only) and s.keyframe_path]
    handles: list[tuple[Shot, JobHandle, Route]] = []
    records: list[GenRecord] = []
    errors: list[tuple[str, Exception]] = []
    warned_audio: set[str] = set()
    beats = {b.idx: b for b in story.beats}

    if not only:
        # resume: คลิปที่เจนเสร็จก่อน run ล้ม (เช่นเครดิตหมดกลางทาง) ยังอยู่บนดิสก์
        # ไม่จ่ายซ้ำ — regenerate (only=...) ตั้งใจสร้างใหม่จึงไม่ข้าม
        done = [s for s in targets if (outdir / f"{s.id}.mp4").exists()]
        for s in done:
            s.clip_path = str(outdir / f"{s.id}.mp4")
        if done:
            log.info("ข้าม %d shot ที่มีคลิปอยู่แล้ว: %s", len(done), [s.id for s in done])
        targets = [s for s in targets if s not in done]

    for shot in targets:
        first = Path(shot.keyframe_path).read_bytes()
        beat = beats.get(shot.beat_idx)

        async def call(prov, route: Route, _s=shot, _f=first, _b=beat):
            caps = await prov.capabilities(route.model)
            if render == "avatar":
                # avatar: prompt คือ "คำพูด" ล้วน ๆ ภาพตัวละครไปทาง input_references
                # ความยาวตามเสียงที่โมเดลสร้าง — ไม่มี duration ให้สั่ง
                if not (_b and _b.dialogue):
                    raise ValueError(f"shot {_s.id} ไม่มีบทพูด — โหมด avatar ต้องมีทุก shot")
                ref = _f
                if character_refs and _b.speaker in character_refs:
                    ref = character_refs[_b.speaker].read_bytes()
                extra = {"motion_prompt": _s.plan.subject_action or _b.visual_intent}
                for k in ("voice_settings", "expressiveness", "fit", "remove_background", "background"):
                    if k in route.params:
                        extra[k] = route.params[k]
                # เสียง: HeyGen ไม่มีเสียงไทย แต่เสียง "Multilingual" พูดไทยได้ (วัดแล้ว: ASR
                # ถอดตรงบททุกคำ) เลือกหญิง/ชายจาก voice_en ของตัวละคร override ได้จาก
                # params.voice_id_female / voice_id_male / voice_id — และต้องส่งใต้
                # provider.options.heygen ไม่ใช่ระดับบน
                ch = story.character(_b.speaker)
                extra["provider"] = {"options": {"heygen": {
                    "voice_id": avatar_voice_id(route.params, ch.voice_en if ch else "")}}}
                return await prov.submit(VideoRequest(
                    prompt=_b.dialogue,
                    model=route.model,
                    duration_s=_s.duration_s,
                    aspect_ratio=route.params.get("aspect", "9:16"),
                    resolution=route.params.get("resolution", "720p"),
                    reference_images=[ref],
                    with_audio=True,
                    extra=extra,
                ))
            # timeline ปัดให้ตรงชุดของโมเดลหลักไว้แล้ว แต่ fallback อาจรับคนละชุด
            dur = snap_duration(_s.duration_s, caps.durations)
            if not caps.supports_audio and route.model not in warned_audio:
                # หลักเดิม: ความสามารถต้องเช็ก ไม่ใช่เดา — โมเดลที่ไม่ทำเสียง
                # จะรับ generate_audio ไปเงียบ ๆ แล้วคืนคลิปใบ้ ซึ่งในสายนี้
                # แปลว่าบทพูดหายทั้ง shot
                warned_audio.add(route.model)
                log.warning("%s ไม่ประกาศว่าสร้างเสียงได้ — shot %s อาจไม่มีเสียงพูด",
                            route.model, _s.id)
            return await prov.submit(VideoRequest(
                prompt=video_prompt(_s, story, brand),
                model=route.model,
                duration_s=dur,
                aspect_ratio=route.params.get("aspect", "9:16"),
                resolution=route.params.get("resolution", "1080p"),
                first_frame=_f,
                with_audio=True,
            ))

        try:
            handle, route = await reg.run(stage, call, budget=budget)
        except Exception as e:  # noqa: BLE001 — submit ล้มหนึ่งตัว ไม่ทิ้งตัวที่ส่งไปแล้ว
            log.warning("submit video shot=%s ล้ม: %s", shot.id, e)
            errors.append((shot.id, e))
            if getattr(e, "code", "") == "http_402" or "402" in str(e):
                break   # เครดิตหมด — shot ที่เหลือก็จะล้มเหมือนกัน ไม่ต้องยิงต่อ
            continue
        handles.append((shot, handle, route))
        log.info("ส่งงาน video shot=%s job=%s", shot.id, handle.external_id)

    async def wait(shot: Shot, handle: JobHandle, route: Route):
        prov = reg.provider(route.provider)
        waited = 0.0
        while waited < max_wait_s:
            st = await prov.poll(handle)
            if st.state == "succeeded":
                data = await prov.fetch(handle, st.url)
                p = outdir / f"{shot.id}.mp4"
                tmp = outdir / f"{shot.id}.part"
                tmp.write_bytes(data)
                tmp.replace(p)   # ไฟล์ .mp4 โผล่ทีเดียวตอนครบ — resume จะไม่เจอไฟล์ครึ่งเดียว
                shot.clip_path = str(p)
                shot.clip_cost_usd += st.cost_usd
                shot.clip_job_id = handle.external_id
                shot.clip_model = route.model
                if not st.cost_usd:
                    log.warning("video shot=%s job=%s ไม่รายงาน usage.cost — "
                                "งบจะไม่เห็นค่าใช้จ่ายคลิปนี้", shot.id, handle.external_id)
                records.append(GenRecord("video", route, st.cost_usd, shot_id=shot.id,
                                         external_id=handle.external_id))
                if budget and st.cost_usd:
                    await budget.charge(st.cost_usd, what=f"video:{shot.id}")
                return
            if st.state == "failed":
                raise RuntimeError(f"video shot={shot.id} ล้มเหลว: {st.error}")
            await asyncio.sleep(poll_every)
            waited += poll_every
        raise TimeoutError(f"video shot={shot.id} ไม่เสร็จใน {max_wait_s}s "
                           f"(job {handle.external_id} ยังค้างอยู่ที่ provider)")

    results = await asyncio.gather(*(wait(s, h, r) for s, h, r in handles),
                                   return_exceptions=True)
    for (shot, _h, _r), res in zip(handles, results):
        if isinstance(res, BaseException):
            errors.append((shot.id, res))
    # คืนทั้งคู่ — ผู้เรียกต้องบันทึก records/clip ที่สำเร็จ *ก่อน* โยน error
    # ไม่งั้นเงินที่จ่ายไปแล้วหายจากบัญชีและ resume จะเจนซ้ำ
    return records, errors


# ---------------------------------------------------------------- 7 QC

QC_SYS = """You review frames from a short vertical video before it is published.

Report only real, visible problems. Do not invent issues to seem thorough —
a clean video should return verdict "pass" with an empty issues list.

severity "block" = must not publish (deformed hands or faces, garbled text
rendered in the image, the subject visibly changing appearance between shots,
anything unsafe or off-brand).
severity "warn" = worth a human look but not fatal.

You only see one still frame per shot, so you cannot judge speech or
lip-sync — do not report on audio.

Return JSON only."""


async def run_qc_vlm(reg: Registry, frames: list[bytes], shot_ids: list[str],
                     story_title: str, budget=None):
    user = (
        f"Video: {story_title}\n"
        f"Frames in order, one per shot: {', '.join(shot_ids)}\n\n"
        "Review them and report problems."
    )
    return await structured(reg, "qc", QCReport, system=QC_SYS,
                            user=user, images=frames, budget=budget)


# ---------------------------------------------------------------- 8 metadata

META_SYS = f"""คุณเขียนแคปชันและแฮชแท็กสำหรับวิดีโอสั้นภาษาไทย

- แคปชันเปิดด้วยประโยคที่ทำให้อยากดู ไม่ใช่สรุปว่าคลิปนี้เกี่ยวกับอะไร
- แฮชแท็ก 5-8 อัน ผสมกว้างกับเฉพาะทาง ภาษาไทยและอังกฤษได้
- title_yt ต้องลงท้ายด้วย #Shorts
- cover_time_s เลือกวินาทีที่ภาพน่าจะหยุดคนได้ที่สุด

{TH}"""


async def run_metadata(reg: Registry, story: Story, duration_s: float, budget=None):
    user = (
        f"หัวข้อ: {story.title}\nมุมเล่า: {story.angle}\n"
        f"กลุ่มเป้าหมาย: {story.audience}\nประโยคเปิด: {story.hook_line}\n"
        f"CTA: {story.cta}\nความยาวจริง {duration_s:.1f} วินาที"
    )
    return await structured(reg, "meta", PostMeta, system=META_SYS,
                            user=user, budget=budget)
