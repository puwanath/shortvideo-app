"""เครื่องยนต์ agent loop

ใช้เฉพาะสองที่: คิดมุมเล่า และวนแก้งานหลัง QC — สองจุดที่จำนวนรอบไม่รู้ล่วงหน้าจริง
ส่วนที่เหลือของ pipeline ยังเป็น DAG เหมือนเดิม loop นี้อยู่ "ข้างใน" node ตัวเดียว
ไม่ได้คุยข้าม node

ขอบเขตที่บังคับไว้ ไม่ใช่ตัวเลือก:
  * max_steps — กันวนไม่จบ
  * budget — เกินเพดานแล้วหยุด ไม่ใช่เตือน
  * tool ทุกตัวต้องประกาศล่วงหน้า เรียกชื่อที่ไม่มีจะได้ error กลับเข้า transcript
    แทนที่จะพัง (โมเดลแก้ตัวเองได้ในรอบถัดไป)
  * ต้องจบด้วยการเรียก finish tool เท่านั้น ไม่รับคำตอบเป็นข้อความเปล่า
    เพราะข้อความเปล่าแปลว่าเราต้องมานั่ง parse เอง ซึ่งคือจุดที่ระบบพัง
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from ..providers.base import TextRequest, ToolSpec
from ..providers.registry import BudgetExceeded, Registry, Route

log = logging.getLogger(__name__)

ToolFn = Callable[..., Awaitable[Any]]


@dataclass
class Tool:
    spec: ToolSpec
    fn: ToolFn
    # tool ที่จบ loop — ผลลัพธ์ของมันคือคำตอบสุดท้าย
    terminal: bool = False
    # ถ้าใส่ loop จะ validate อาร์กิวเมนต์ให้ก่อนเรียก fn
    args_model: type[BaseModel] | None = None


@dataclass
class LoopResult:
    output: Any
    steps: int
    cost_usd: float
    transcript: list[dict] = field(default_factory=list)
    stopped_reason: str = "finished"


class Toolbox:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def add(self, name: str, description: str, parameters: dict,
            fn: ToolFn, *, terminal: bool = False,
            args_model: type[BaseModel] | None = None) -> None:
        """parameters คือ JSON schema ที่ส่งให้โมเดล
        args_model คือ Pydantic model ที่ใช้ validate ตอนโมเดลเรียกกลับมา
        ใส่ทั้งคู่เมื่อเป็นไปได้ — ไม่งั้น tool ที่รับ **kwargs จะกลืน
        อาร์กิวเมนต์ผิด ๆ ไปเงียบ ๆ แล้วไปพังลึกกว่านั้น"""
        self._tools[name] = Tool(
            ToolSpec(name, description, parameters), fn, terminal, args_model)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)


async def run_loop(
    registry: Registry,
    stage: str,
    *,
    system: str,
    user: str,
    toolbox: Toolbox,
    max_steps: int = 8,
    budget=None,
    temperature: float | None = None,
) -> LoopResult:
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    total_cost = 0.0
    specs = toolbox.specs()

    for step in range(max_steps):
        # เช็กก่อนยิง — ไม่ใช่รอให้ charge โยน exception หลังจ่ายไปแล้ว
        if budget and budget.spent >= budget.cap:
            return LoopResult(None, step, total_cost, messages,
                              f"งบหมด (ใช้ไป ${budget.spent:.3f})")

        async def call(prov, route: Route):
            return await prov.generate(TextRequest(
                system=system, user=user,
                model=route.model.removeprefix("local/") if route.provider == "vllm_local" else route.model,
                temperature=temperature if temperature is not None
                else float(route.params.get("temperature", 0.7)),
                max_tokens=int(route.params.get("max_tokens", 3000)),
                tools=specs,
                messages=messages,
            ))

        try:
            resp, _route = await registry.run(stage, call, budget=budget)
            total_cost += resp.cost_usd
            if budget and resp.cost_usd:
                await budget.charge(resp.cost_usd, what=f"{stage}:step{step}")
        except BudgetExceeded as e:
            # ไม่โยนต่อ — ให้ผู้เรียกตัดสินใจเอง
            # concept agent: ไม่มีผลลัพธ์ = run ล้ม (ถูกต้อง ยังไม่ได้จ่ายอะไรแพง)
            # qc agent: ไม่มีผลลัพธ์ = ส่งให้คนดู (ถูกต้อง วิดีโอเรนเดอร์เสร็จแล้ว
            #   จะทิ้งทั้ง run เพราะงบตรวจหมดไม่สมเหตุสมผล)
            return LoopResult(None, step, total_cost, messages, f"งบหมด: {e}")

        # เก็บข้อความของผู้ช่วยลง transcript ในรูปแบบที่ API รับกลับไปได้
        assistant_msg = (resp.raw or {}).get("message") or {
            "role": "assistant", "content": resp.text}
        messages.append(assistant_msg)

        if not resp.tool_calls:
            # ไม่เรียก tool = ยังไม่จบ เตือนแล้วให้ลองใหม่
            messages.append({
                "role": "user",
                "content": "คุณต้องจบด้วยการเรียก tool ที่กำหนดไว้ ไม่ใช่ตอบเป็นข้อความ "
                           "เลือก tool ที่เหมาะแล้วเรียกมา",
            })
            continue

        terminal_out = None
        for tc in resp.tool_calls:
            tool = toolbox.get(tc.name)
            if tool is None:
                result = {"error": f"ไม่มี tool ชื่อ '{tc.name}' — "
                                   f"ที่ใช้ได้: {[s.name for s in specs]}"}
            else:
                try:
                    args = tc.arguments
                    if tool.args_model is not None:
                        # validate ที่ loop ไม่ใช่ปล่อยให้ tool แต่ละตัวทำเอง
                        args = tool.args_model.model_validate(args).model_dump()
                    result = await tool.fn(**args)
                    if tool.terminal:
                        terminal_out = result
                except ValidationError as e:
                    result = {"error": f"อาร์กิวเมนต์ผิดรูปแบบ: {str(e)[:600]}"}
                except Exception as e:  # noqa: BLE001
                    log.exception("tool %s ล้มเหลว", tc.name)
                    result = {"error": f"{type(e).__name__}: {e}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": json.dumps(_jsonable(result), ensure_ascii=False)[:4000],
            })

        if terminal_out is not None:
            return LoopResult(terminal_out, step + 1, total_cost, messages, "finished")

    return LoopResult(None, max_steps, total_cost, messages, "ครบ max_steps แล้วยังไม่เรียก finish")


def _jsonable(v: Any) -> Any:
    if isinstance(v, BaseModel):
        return v.model_dump()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def tool_from_model(tb: "Toolbox", name: str, description: str,
                    model_cls: type[BaseModel], fn: ToolFn, *,
                    terminal: bool = False) -> None:
    """ลงทะเบียน tool จาก Pydantic model ตัวเดียว — ได้ทั้ง schema และ validation
    ไม่ต้องเขียน JSON schema สองที่แล้วลืมอัปเดตตัวใดตัวหนึ่ง"""
    tb.add(name, description, schema_of(model_cls), fn,
           terminal=terminal, args_model=model_cls)


def schema_of(model_cls: type[BaseModel]) -> dict:
    """ใช้ Pydantic schema เป็น tool parameters โดยตรง
    ได้ validation ฟรีและไม่ต้องเขียน JSON schema สองที่"""
    s = model_cls.model_json_schema()
    s.pop("title", None)
    return s
