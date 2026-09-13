"""เรียก LLM แล้วบังคับให้ได้ Pydantic object ที่ถูกต้อง

repair loop: ถ้า validate ไม่ผ่าน ส่ง error กลับเข้าไปใน prompt แล้วให้แก้
วิธีนี้ทำให้โมเดลขนาดกลางที่รันเองใช้งานได้จริง โดยไม่ต้องขยับไปโมเดลใหญ่
"""
from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..providers.base import TextRequest
from ..providers.registry import Registry, Route

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _extract_json(text: str) -> str:
    """โมเดลชอบใส่ ```json ครอบ หรือพูดนำหน้า — ดึงก้อน JSON ออกมา"""
    t = _FENCE.sub("", text).strip()
    if t.startswith("{") or t.startswith("["):
        return t
    start = min((i for i in (t.find("{"), t.find("[")) if i >= 0), default=-1)
    if start < 0:
        return t
    end = max(t.rfind("}"), t.rfind("]"))
    return t[start:end + 1] if end > start else t[start:]


async def structured(
    registry: Registry,
    stage: str,
    model_cls: type[T],
    *,
    system: str,
    user: str,
    images: list[bytes] | None = None,
    max_repairs: int = 2,
    budget=None,
) -> tuple[T, Route, float]:
    """คืน (object, route, cost) — ราคาถูกรวมทุกครั้งที่ retry แล้ว"""
    schema = model_cls.model_json_schema()
    total_cost = 0.0
    convo_user = user

    async def call(prov, route: Route):
        return await prov.generate(TextRequest(
            system=system,
            user=convo_user,
            model=route.model.removeprefix("local/") if route.provider == "vllm_local" else route.model,
            temperature=float(route.params.get("temperature", 0.7)),
            max_tokens=int(route.params.get("max_tokens", 4096)),
            json_schema=schema,
            images=images or [],
        ))

    last_err = ""
    for attempt in range(max_repairs + 1):
        resp, route = await registry.run(stage, call, budget=budget)
        total_cost += resp.cost_usd
        if budget and resp.cost_usd:
            await budget.charge(resp.cost_usd, what=stage)

        raw = _extract_json(resp.text)
        try:
            obj = model_cls.model_validate_json(raw)
            if attempt:
                log.info("stage=%s ซ่อม JSON สำเร็จที่รอบ %d", stage, attempt + 1)
            return obj, route, total_cost
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = str(e)[:1200]
            log.warning("stage=%s JSON ไม่ผ่าน validation รอบ %d", stage, attempt + 1)
            convo_user = (
                f"{user}\n\n"
                f"--- คำตอบก่อนหน้าของคุณผิดรูปแบบ ---\n{raw[:1500]}\n\n"
                f"--- error ที่เกิด ---\n{last_err}\n\n"
                "แก้ให้ถูกต้องแล้วตอบกลับมาเป็น JSON ล้วนอย่างเดียว "
                "ห้ามมีคำอธิบาย ห้ามมี markdown fence"
            )

    raise ValueError(f"stage '{stage}' ให้ JSON ที่ผ่าน validation ไม่ได้หลัง "
                     f"{max_repairs + 1} ครั้ง — error สุดท้าย: {last_err}")
