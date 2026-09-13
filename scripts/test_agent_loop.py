"""ทดสอบ agent loop โดยไม่ต้องต่อเน็ต

ใช้ provider ปลอมที่เขียนสคริปต์ tool call ไว้ล่วงหน้า เพื่อพิสูจน์ว่า
**ขอบเขต** ทำงาน ไม่ใช่แค่พิสูจน์ว่า loop เดินได้ — เพราะ loop ที่เดินได้
แต่หยุดไม่เป็นคือสิ่งที่ทำให้ระบบ agent เผาเงิน

    python scripts/test_agent_loop.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel                          # noqa: E402
from app.agents.loop import Toolbox, run_loop, tool_from_model  # noqa: E402
from app.providers.base import TextResponse, ToolCall   # noqa: E402
from app.providers.registry import BudgetGuard, Route   # noqa: E402


class FakeProvider:
    """เล่นตามสคริปต์ [(tool_name, args), ...] — None = ตอบข้อความเปล่า"""
    name = "fake"

    def __init__(self, script, cost=0.01):
        self.script = list(script)
        self.calls = 0
        self.cost = cost

    async def generate(self, req):
        self.calls += 1
        if not self.script:
            return TextResponse(text="ไม่รู้จะทำอะไรต่อ", cost_usd=self.cost)
        item = self.script.pop(0)
        if item is None:
            return TextResponse(text="ขอตอบเป็นข้อความเฉย ๆ", cost_usd=self.cost)
        name, args = item
        return TextResponse(
            text="", cost_usd=self.cost,
            tool_calls=[ToolCall(id=f"c{self.calls}", name=name, arguments=args)],
            raw={"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{self.calls}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}]}},
        )


class FakeRegistry:
    def __init__(self, provider):
        self.p = provider

    async def run(self, stage, fn, *, budget=None, attempts_per_route=3):
        return await fn(self.p, Route("fake", "fake/model", {})), Route("fake", "fake/model", {})


def box(calls: list):
    tb = Toolbox()

    async def ping(**kw):
        calls.append(("ping", kw))
        return {"pong": True}

    async def finish(**kw):
        calls.append(("finish", kw))
        return {"done": kw}

    class PingArgs(BaseModel):
        n: int

    tool_from_model(tb, "ping", "ทดสอบ", PingArgs, ping)
    tb.add("finish", "จบ", {"type": "object", "properties": {
        "answer": {"type": "string"}}}, finish, terminal=True)
    return tb


async def main():
    ok = 0
    fail = 0

    def check(label, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  ✓ {label}")
        else:
            fail += 1
            print(f"  ✗ {label}  {detail}")

    print("=" * 64)
    print("1) เส้นทางปกติ — เรียก tool แล้วจบด้วย terminal tool")
    print("=" * 64)
    calls = []
    reg = FakeRegistry(FakeProvider([
        ("ping", {"n": 1}),
        ("ping", {"n": 2}),
        ("finish", {"answer": "เสร็จ"}),
    ]))
    r = await run_loop(reg, "test", system="s", user="u", toolbox=box(calls), max_steps=8)
    check("จบด้วย finished", r.stopped_reason == "finished", r.stopped_reason)
    check("ได้ output", r.output == {"done": {"answer": "เสร็จ"}}, str(r.output))
    check("เดิน 3 รอบ", r.steps == 3, f"ได้ {r.steps}")
    check("เรียก tool ครบ 3 ครั้ง", len(calls) == 3, str(calls))

    print()
    print("=" * 64)
    print("2) เบรก: max_steps — agent ที่ไม่ยอมจบต้องถูกตัด")
    print("=" * 64)
    calls = []
    reg = FakeRegistry(FakeProvider([("ping", {"n": i}) for i in range(50)]))
    r = await run_loop(reg, "test", system="s", user="u", toolbox=box(calls), max_steps=4)
    check("หยุดที่ max_steps", r.steps == 4, f"ได้ {r.steps}")
    check("output เป็น None", r.output is None)
    check("บอกเหตุผลชัด", "max_steps" in r.stopped_reason, r.stopped_reason)

    print()
    print("=" * 64)
    print("3) เบรก: งบหมดกลางคัน")
    print("=" * 64)
    calls = []
    guard = BudgetGuard(cap_usd=0.025)
    prov = FakeProvider([("ping", {"n": i}) for i in range(20)], cost=0.01)
    r = await run_loop(FakeRegistry(prov), "test", system="s", user="u",
                       toolbox=box(calls), max_steps=20, budget=guard)
    check("หยุดเพราะงบหมด", "งบหมด" in r.stopped_reason, r.stopped_reason)
    check("ไม่เดินเกิน 4 รอบ", r.steps <= 4, f"เดินไป {r.steps} รอบ")
    check(f"ใช้เงิน ${guard.spent:.3f} ไม่เกินเพดานมาก", guard.spent <= 0.045)

    print()
    print("=" * 64)
    print("4) ทนทาน: เรียก tool ที่ไม่มี / อาร์กิวเมนต์ผิด / ตอบข้อความเปล่า")
    print("=" * 64)
    calls = []
    reg = FakeRegistry(FakeProvider([
        ("no_such_tool", {}),                 # ชื่อ tool ผิด
        None,                                  # ตอบข้อความเปล่า ไม่เรียก tool
        ("ping", {"wrong_arg": "x"}),          # อาร์กิวเมนต์ผิด
        ("finish", {"answer": "รอดมาได้"}),
    ]))
    r = await run_loop(reg, "test", system="s", user="u", toolbox=box(calls), max_steps=8)
    check("ไม่ crash และจบได้", r.output == {"done": {"answer": "รอดมาได้"}}, str(r.output))
    tool_msgs = [m for m in r.transcript if m.get("role") == "tool"]
    errs = [m for m in tool_msgs if "error" in m["content"]]
    check("error ถูกส่งกลับเข้า transcript ให้แก้ตัว", len(errs) == 2, f"เจอ {len(errs)}")
    nudge = [m for m in r.transcript
             if m.get("role") == "user" and "ต้องจบด้วยการเรียก tool" in (m.get("content") or "")]
    check("เตือนเมื่อไม่เรียก tool", len(nudge) == 1, f"เจอ {len(nudge)}")

    print()
    print("=" * 64)
    print("5) QC repair: เบรก 'แก้แล้วไม่ดีขึ้นต้องหยุด'")
    print("=" * 64)
    from app.agents.qc_agent import MAX_REGEN_PER_SHOT
    check(f"เพดาน regen ต่อ shot = {MAX_REGEN_PER_SHOT}", MAX_REGEN_PER_SHOT == 2)

    import inspect as _i
    from app.agents import qc_agent
    src = _i.getsource(qc_agent.run_qc_repair)
    check("มีเบรกเมื่อ blocking ไม่ลด", "stalled" in src)
    check("เช็กเพดาน regen ก่อนสร้างใหม่", "MAX_REGEN_PER_SHOT" in src)
    check("เช็ก max_rounds ก่อนแก้", "state[\"round\"] >= max_rounds" in src)
    check("agent ไม่รู้จัก provider โดยตรง",
          "OpenRouter" not in src and "httpx" not in src)

    print()
    print("=" * 64)
    print(f"ผ่าน {ok} ข้อ  ล้มเหลว {fail} ข้อ")
    print("=" * 64)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
