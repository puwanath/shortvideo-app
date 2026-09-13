"""Concept agent — จุดแรกที่คุ้มกับการเป็น agent จริง

เหตุผลที่ควรวน ไม่ใช่ยิงครั้งเดียว: hook สามวินาทีแรกตัดสินว่าคลิปรอดหรือตาย
และการรู้ว่า "เราเคยใช้มุมนี้ไปแล้วสามคลิปที่แล้ว" เป็นข้อมูลที่ต้องไปดึงมา
ไม่ใช่สิ่งที่เดาจาก prompt ได้ จำนวนรอบจึงไม่รู้ล่วงหน้า

tool ที่ให้:
  past_angles   — มุมที่เคยใช้กับแบรนด์นี้ (จาก DB ของเราเอง สัญญาณจริง)
  score_hook    — ให้โมเดลอีกตัวให้คะแนน hook แยกจากคนเขียน
  search_web    — ดู trend (ต้องตั้งค่า search provider ก่อน ไม่งั้นบอกว่าไม่มี)
  submit        — จบ loop

ตัว score_hook ทำให้เกิดสิ่งที่ one-shot ทำไม่ได้: agent เสนอ hook ได้คะแนนต่ำ
แล้วเขียนใหม่เองก่อนส่ง โดยที่คนยังไม่ต้องเห็น
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ..providers.base import TextRequest
from ..providers.registry import Registry, Route
from ..schemas import ConceptOut
from .loop import Toolbox, run_loop, tool_from_model

log = logging.getLogger(__name__)

# system prompt ของ loop นี้อยู่ใน agents/formats.py (แยกตาม format: content / cartoon)


class HookScore(BaseModel):
    score: int = Field(ge=0, le=10, description="0-10 หยุดนิ้วคนได้แค่ไหน")
    why: str
    stronger_version: str | None = Field(
        default=None, description="เขียนใหม่ให้แรงขึ้น ถ้าคะแนนต่ำกว่า 8")


JUDGE_SYSTEM = """คุณให้คะแนนประโยคเปิดของวิดีโอสั้นภาษาไทย ตัดสินอย่างเดียวว่า
คนที่กำลังเลื่อนฟีดเร็ว ๆ จะหยุดดูไหม

ให้คะแนนต่ำกับ: การเกริ่น การแนะนำตัว คำถามกว้าง ๆ คำโปรยที่ไม่บอกอะไร
ให้คะแนนสูงกับ: ข้ออ้างที่ขัดความเชื่อทั่วไป ตัวเลขที่เฉพาะเจาะจง
ความเสี่ยงที่ผู้ฟังกำลังเจออยู่โดยไม่รู้ตัว

เข้มงวด คะแนน 8 ขึ้นไปต้องดีจริง ไม่ใช่แค่ผ่าน ตอบเป็นภาษาไทย"""


async def run_concept_agent(
    reg: Registry,
    brief: str,
    brand: dict,
    *,
    past_angle_titles: list[str] | None = None,
    search_fn=None,
    budget=None,
    max_steps: int = 8,
    format_name: str | None = None,
):
    """คืน (ConceptOut, cost, transcript) — ใช้แทน run_concept เดิมได้ตรง ๆ
    format_name เลือก system prompt: คอนเทนต์หา "มุม" การ์ตูนหา "พล็อต+มุก" """
    from .formats import fmt as _fmt
    system = _fmt(format_name)["concept_sys"]
    tb = Toolbox()
    seen = list(past_angle_titles or [])

    async def past_angles(limit: int = 10):
        if not seen:
            return {"angles": [], "note": "ยังไม่มีงานเก่าของแบรนด์นี้ เสนอได้อิสระ"}
        return {"angles": seen[:limit],
                "note": "อย่าเสนอมุมที่ซ้ำหรือใกล้เคียงกับรายการนี้"}

    async def score_hook(hook: str):
        async def call(prov, route: Route):
            return await prov.generate(TextRequest(
                system=JUDGE_SYSTEM,
                user=f"ให้คะแนนประโยคเปิดนี้:\n\n{hook}",
                model=route.model.removeprefix("local/") if route.provider == "vllm_local" else route.model,
                temperature=0.2,
                max_tokens=700,
                json_schema=HookScore.model_json_schema(),
            ))

        resp, _ = await reg.run("concept", call, budget=budget)
        if budget and resp.cost_usd:
            await budget.charge(resp.cost_usd, what="concept:judge")
        try:
            return HookScore.model_validate_json(
                resp.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        except Exception:  # noqa: BLE001
            return {"error": "ผู้ให้คะแนนตอบผิดรูปแบบ ใช้วิจารณญาณตัวเองแทน"}

    async def search_web(query: str):
        if search_fn is None:
            return {"error": "ยังไม่ได้ตั้งค่า search provider — "
                             "ทำงานต่อโดยไม่ใช้ข้อมูล trend"}
        return await search_fn(query)

    async def submit(**kwargs):
        return ConceptOut.model_validate(kwargs)

    tb.add("past_angles", "ดูมุมเล่าที่เคยใช้กับแบรนด์นี้ เพื่อไม่เสนอซ้ำ",
           {"type": "object", "properties": {
               "limit": {"type": "integer", "description": "จำนวนสูงสุด"}}},
           past_angles)
    tb.add("score_hook", "ให้คนนอกให้คะแนนประโยคเปิด 0-10 พร้อมเหตุผล",
           {"type": "object", "required": ["hook"], "properties": {
               "hook": {"type": "string", "description": "ประโยคเปิดที่จะวัด"}}},
           score_hook)
    tb.add("search_web", "ค้นหา trend หรือข้อมูลประกอบ",
           {"type": "object", "required": ["query"], "properties": {
               "query": {"type": "string"}}},
           search_web)
    tool_from_model(tb, "submit", "ส่งมุมเล่าสุดท้าย 3 มุมพร้อมตัวที่เลือก",
                    ConceptOut, submit, terminal=True)

    user = (
        f"บรีฟ: {brief}\n\n"
        f"โทนแบรนด์: {brand.get('tone_of_voice') or 'ไม่ระบุ'}\n"
        f"ข้อห้าม: {brand.get('do_donts') or 'ไม่ระบุ'}\n\n"
        "เริ่มจากดูมุมที่เคยใช้ก่อน"
    )

    res = await run_loop(reg, "concept", system=system, user=user,
                         toolbox=tb, max_steps=max_steps, budget=budget)

    if res.output is None:
        raise ValueError(
            f"concept agent ไม่ได้ผลลัพธ์: {res.stopped_reason} "
            f"(เดินไป {res.steps} รอบ ใช้ ${res.cost_usd:.3f})")

    log.info("concept agent จบใน %d รอบ ($%.4f)", res.steps, res.cost_usd)
    return res.output, res.cost_usd, res.transcript
