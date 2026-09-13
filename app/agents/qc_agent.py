"""QC repair loop — จุดที่สองที่คุ้มกับการเป็น agent

เดิม: ตรวจแล้วรายงาน คนต้องมากดแก้เอง
ตอนนี้: ตรวจ → เลือก shot ที่พัง → เขียน prompt ใหม่ → สร้างใหม่ → ตรวจซ้ำ

ทำไมต้องเป็น loop ไม่ใช่ขั้นตอนเดียว: การแก้ครั้งแรกอาจไม่ผ่าน และ "ผ่านหรือยัง"
ตอบได้ก็ต่อเมื่อสร้างใหม่แล้วตรวจอีกรอบ จำนวนรอบจึงไม่รู้ล่วงหน้า

ขอบเขตที่บังคับ — สำคัญกว่าตัว loop เอง:
  * max_rounds — ดีฟอลต์ 2 ไม่ใช่ไม่จำกัด
  * เพดาน regen ต่อ shot — shot เดียวกันแก้ได้ไม่เกิน 2 ครั้งตลอด run
    ถ้าแก้สองรอบแล้วยังพัง แปลว่า prompt ไม่ใช่ปัญหา ให้คนดู
  * ต้องดีขึ้นจริง — ถ้าจำนวน blocking issue ไม่ลดลงหลังแก้ ให้หยุดทันที
    ไม่วนเผาเงินต่อ นี่คือเบรกที่ agent loop ส่วนใหญ่ลืมใส่
  * budget guard เดียวกับทั้ง run
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from ..media import probe as qcprobe
from ..providers.registry import Registry
from ..schemas import QCIssue, QCReport, Shot
from .loop import Toolbox, run_loop, tool_from_model

log = logging.getLogger(__name__)

MAX_REGEN_PER_SHOT = 2

SYSTEM = """คุณตรวจงานวิดีโอสั้นแนวตั้งก่อนส่งให้คนอนุมัติ

วิธีทำงาน:
1. เรียก inspect เพื่อดูรายงานปัญหาและภาพจากแต่ละ shot
2. ถ้าไม่มีปัญหาร้ายแรง เรียก accept ทันที อย่าหาเรื่องแก้เพื่อให้ดูขยัน
3. ถ้ามี ให้เลือกเฉพาะ shot ที่พังจริง เขียน image_prompt ใหม่ที่แก้ที่ต้นเหตุ
   แล้วเรียก repair — ระบบจะสร้างใหม่เฉพาะ shot นั้นแล้วส่งผลกลับมาให้ตรวจซ้ำ
4. ถ้าแก้แล้วยังไม่ดีขึ้น เรียก escalate เพื่อส่งให้คนดู อย่าวนแก้ไปเรื่อย

การเขียน prompt แก้:
- แก้ที่ต้นเหตุ ไม่ใช่เติมคำว่า "high quality" ต่อท้าย
- มือหรือใบหน้าผิดรูป → เปลี่ยนมุมกล้องหรือระยะภาพให้เลี่ยงส่วนนั้น
- ตัวอักษรในภาพมั่ว → เอาสิ่งที่มีตัวหนังสือออกจากฉากไปเลย
- ตัวละครหน้าเปลี่ยน → อธิบายลักษณะให้ตรงกับ shot อื่นอย่างเจาะจง

ตอบภาษาไทย prompt เป็นภาษาอังกฤษ"""


class RepairItem(BaseModel):
    shot_id: str
    reason: str = Field(description="พังตรงไหน")
    new_image_prompt: str = Field(description="prompt ใหม่ ภาษาอังกฤษ")
    new_motion_intent: str | None = None


class RepairPlan(BaseModel):
    items: list[RepairItem] = Field(min_length=1, max_length=6)


class Accept(BaseModel):
    note: str = Field(description="สรุปสั้น ๆ ว่าทำไมถึงผ่าน")


class Escalate(BaseModel):
    reason: str
    shot_ids: list[str] = []


@dataclass
class RepairOutcome:
    report: QCReport
    rounds: int
    repaired: list[str]
    cost_usd: float
    escalated: bool = False
    note: str = ""


async def run_qc_repair(
    reg: Registry,
    *,
    video: Path,
    shots: list[Shot],
    story_title: str,
    workdir: Path,
    regenerate,           # async fn(shot_ids) -> None — สร้างใหม่แล้ว re-render
    mode: str = "video",
    expected_duration_s: float | Callable[[], float] | None = None,
    max_rounds: int = 2,
    budget=None,
) -> RepairOutcome:
    """regenerate คือ callback ที่ runner ส่งเข้ามา ทำให้ loop นี้ไม่ต้องรู้จัก
    provider หรือ ffmpeg เลย ทดสอบแยกได้

    expected_duration_s รับ callable ได้ เพราะ regenerate เปลี่ยน timeline ได้
    (คลิปใหม่ยาวไม่เท่าเดิม แล้ว step_video เรียก retime_shots) ถ้าจำค่าตอนเริ่ม
    จะฟ้อง duration_drift เท็จหลังแก้ — เจอจริงใน run แรกของสายใหม่
    """
    state = {
        "round": 0,
        "report": None,
        "prev_blocking": None,
        "repaired": [],
        "escalated": False,
        "note": "",
    }
    regen_count = {s.id: s.regen_count for s in shots}
    by_id = {s.id: s for s in shots}
    cost = 0.0

    async def _scan() -> QCReport:
        nonlocal cost
        expected = (expected_duration_s() if callable(expected_duration_s)
                    else expected_duration_s)
        rep = qcprobe.deterministic_qc(video, expected_duration_s=expected, mode=mode)
        if not rep.blocking:
            if not reg.stage_can_see("qc"):
                # ไม่มีเส้นทางที่ดูภาพได้จริง — บอกตรง ๆ ว่าไม่ได้ตรวจ
                # ดีกว่าส่งภาพไปให้โมเดลที่ทิ้งภาพแล้วได้ verdict ที่ไม่มีใครดู
                # ไม่ทำให้ทั้ง run พัง เพราะยังไงก็มี Gate 2 ให้คนดูอยู่แล้ว
                rep.issues.append(QCIssue(
                    severity="warn", code="no_vlm",
                    detail="ไม่มี route ที่ตรวจภาพได้ — ตรวจแค่ระดับไฟล์เท่านั้น",
                    suggested_fix="ตั้ง params.vision=true ใน routing profile "
                                  "ถ้าโมเดลที่ serve อยู่ดูภาพเป็น"))
                log.warning("stage qc ไม่มี route ที่ดูภาพได้ — ข้ามการตรวจด้วย VLM")
            else:
                mids = [(s.start_s + s.end_s) / 2 for s in shots]
                frames = qcprobe.extract_frames(video, mids, workdir / "qc")
                from .stages import run_qc_vlm
                vlm, _route, c = await run_qc_vlm(
                    reg, [f.read_bytes() for f in frames],
                    [s.id for s in shots], story_title, budget=budget)
                cost += c
                rep.issues.extend(vlm.issues)
                if vlm.verdict != "pass" and rep.verdict == "pass":
                    rep.verdict = vlm.verdict
        state["report"] = rep
        return rep

    async def inspect():
        rep = await _scan()
        return {
            "round": state["round"],
            "verdict": rep.verdict,
            "issues": [i.model_dump() for i in rep.issues],
            "shots": [
                {"id": s.id, "idx": s.idx, "prompt": s.plan.image_prompt,
                 "composition": s.plan.composition,
                 "regen_left": MAX_REGEN_PER_SHOT - regen_count.get(s.id, 0)}
                for s in shots
            ],
        }

    async def repair(**kwargs):
        plan = RepairPlan.model_validate(kwargs)
        if state["round"] >= max_rounds:
            return {"error": f"แก้ครบ {max_rounds} รอบแล้ว เรียก escalate แทน"}

        targets, skipped = [], []
        for item in plan.items:
            shot = by_id.get(item.shot_id)
            if shot is None:
                skipped.append({"shot_id": item.shot_id, "why": "ไม่มี shot นี้"})
                continue
            if regen_count.get(item.shot_id, 0) >= MAX_REGEN_PER_SHOT:
                skipped.append({"shot_id": item.shot_id,
                                "why": f"แก้ครบ {MAX_REGEN_PER_SHOT} ครั้งแล้ว "
                                       "ถ้ายังพังต้องให้คนดู"})
                continue
            shot.plan.image_prompt = item.new_image_prompt
            if item.new_motion_intent:
                shot.plan.motion_intent = item.new_motion_intent
            regen_count[item.shot_id] = regen_count.get(item.shot_id, 0) + 1
            shot.regen_count = regen_count[item.shot_id]
            targets.append(item.shot_id)

        if not targets:
            return {"error": "ไม่มี shot ที่แก้ได้", "skipped": skipped}

        prev_blocking = len(state["report"].blocking) if state["report"] else None
        state["prev_blocking"] = prev_blocking

        await regenerate(targets)
        state["repaired"].extend(targets)
        state["round"] += 1

        rep = await _scan()
        now_blocking = len(rep.blocking)

        # เบรกสำคัญ: แก้แล้วไม่ดีขึ้น = หยุด ไม่วนเผาเงินต่อ
        stalled = prev_blocking is not None and now_blocking >= prev_blocking
        return {
            "regenerated": targets,
            "skipped": skipped,
            "verdict": rep.verdict,
            "blocking_before": prev_blocking,
            "blocking_after": now_blocking,
            "issues": [i.model_dump() for i in rep.issues],
            "advice": ("แก้แล้วไม่ดีขึ้น ให้เรียก escalate" if stalled
                       else "ถ้าผ่านแล้วเรียก accept"),
        }

    async def accept(**kwargs):
        a = Accept.model_validate(kwargs)
        state["note"] = a.note
        return a

    async def escalate(**kwargs):
        e = Escalate.model_validate(kwargs)
        state["escalated"] = True
        state["note"] = e.reason
        return e

    tb = Toolbox()
    tb.add("inspect", "ตรวจวิดีโอรอบปัจจุบัน คืนรายการปัญหาและข้อมูล shot",
           {"type": "object", "properties": {}}, inspect)
    tool_from_model(tb, "repair",
                    "แก้ prompt แล้วสร้าง shot ที่ระบุใหม่ แล้วตรวจซ้ำอัตโนมัติ",
                    RepairPlan, repair)
    tool_from_model(tb, "accept", "ยืนยันว่างานผ่าน ส่งต่อให้คนอนุมัติ",
                    Accept, accept, terminal=True)
    tool_from_model(tb, "escalate",
                    "ส่งให้คนดู เมื่อแก้แล้วไม่ดีขึ้นหรือเกินขอบเขตที่แก้เองได้",
                    Escalate, escalate, terminal=True)

    res = await run_loop(
        reg, "qc", system=SYSTEM,
        user=f"ตรวจวิดีโอ '{story_title}' ({len(shots)} shot) เริ่มด้วย inspect",
        toolbox=tb, max_steps=max_rounds * 3 + 2, budget=budget, temperature=0.3)

    cost += res.cost_usd
    report = state["report"] or QCReport(verdict="fail", issues=[])

    if res.output is None:
        # loop ไม่จบเอง — ไม่ใช่เหตุให้ fail ทั้ง run เพราะยังไงก็มี Gate 2 รออยู่
        log.warning("qc loop ไม่จบด้วยตัวเอง: %s", res.stopped_reason)
        state["escalated"] = True
        state["note"] = res.stopped_reason

    return RepairOutcome(
        report=report,
        rounds=state["round"],
        repaired=sorted(set(state["repaired"])),
        cost_usd=cost,
        escalated=state["escalated"],
        note=state["note"],
    )
