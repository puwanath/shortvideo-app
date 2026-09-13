"""HTTP API — สิ่งที่ UI เรียก

ทุก endpoint ที่ทำให้ pipeline เดินต่อจะคืน state ใหม่กลับไปเสมอ
UI จะได้ไม่ต้องเดาว่าตอนนี้อยู่ตรงไหน
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .config import get_settings
from .db import engine, get_session
from .models import (
    Base, Event, Generation, PublishAttempt, PublishTarget, Run, Review, Task,
)
from .publishers import REGISTRY as PUBLISHERS
from .agents.formats import FORMATS, RENDER_MODES, STYLE_PRESETS
from .orchestrator import runner
from .queue import enqueue
from .schemas import GATES, RunState

# เทียบด้วย string ไม่ใช่ RunState(...) เพราะ run เก่าที่ค้างใน state ที่ถูกถอดไป
# (tts / aligning / style_anchor) จะทำให้ทั้งรายการล้มด้วย ValueError
GATE_VALUES = {g.value for g in GATES}

log = logging.getLogger(__name__)
settings = get_settings()

runs_started = Counter("video_run_started_total", "จำนวน run ที่เริ่ม")
runs_state = Counter("video_run_state_total", "run ที่เข้าสู่แต่ละ state", ["state"])
stage_seconds = Histogram("video_stage_duration_seconds", "เวลาต่อ stage", ["stage"])
cost_total = Counter("video_generation_cost_usd_total", "ค่าใช้จ่ายสะสม", ["stage"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="Short video pipeline", lifespan=lifespan)


async def load_ctx(session: AsyncSession, run_id: str) -> runner.Ctx:
    """ตัวเดียวกับที่ worker ใช้ — เพิ่มแค่การแปลง 404 ให้ชั้น HTTP"""
    try:
        return await runner.load_ctx(session, run_id, settings=settings)
    except runner.RunNotFound:
        raise HTTPException(404, "ไม่พบ run นี้")


# ---------------------------------------------------------------- schemas

class CreateTask(BaseModel):
    brief: str
    brand_kit_id: str | None = None
    routing_profile_id: str | None = None
    target_duration_s: int = 30
    target_platforms: list[str] = ["youtube"]
    budget_cap_usd: float | None = None
    format: str = "content"            # content | cartoon
    style_preset: str | None = None    # ดู GET /formats
    render: str = "scene"              # scene | avatar


class ApproveIn(BaseModel):
    reviewer: str = "human"
    notes: str = ""
    animatic_only: bool = False


class RejectIn(BaseModel):
    reviewer: str = "human"
    reason_code: str = "quality"
    notes: str = ""
    back_to: str | None = None


class RegenIn(BaseModel):
    shot_ids: list[str]
    prompt_overrides: dict[str, str] = {}


class TargetIn(BaseModel):
    platform: str                       # youtube | facebook | tiktok
    account_handle: str
    access_token: str
    refresh_token: str | None = None
    audited: bool = False
    config: dict = {}                   # facebook: {"page_id": "..."}


class PublishIn(BaseModel):
    target_ids: list[str]
    privacy: str = "private"            # private | unlisted | public


# ---------------------------------------------------------------- endpoints

@app.get("/formats")
async def list_formats():
    """รูปแบบงานและ preset สไตล์ที่หน้า New Task ให้เลือก"""
    return {
        "formats": [{"id": k, "label": v["label"]} for k, v in FORMATS.items()],
        "styles": [{"id": k, "label": v["label"]} for k, v in STYLE_PRESETS.items()],
        "renders": [{"id": k, "label": v["label"]} for k, v in RENDER_MODES.items()],
    }


async def _create_task(session: AsyncSession, body: CreateTask,
                       product_files: list[UploadFile] | None = None) -> dict:
    if body.format not in FORMATS:
        raise HTTPException(400, f"format ต้องเป็นหนึ่งใน {sorted(FORMATS)}")
    if body.style_preset and body.style_preset not in STYLE_PRESETS:
        raise HTTPException(400, f"style_preset ต้องเป็นหนึ่งใน {sorted(STYLE_PRESETS)}")
    if body.render not in RENDER_MODES:
        raise HTTPException(400, f"render ต้องเป็นหนึ่งใน {sorted(RENDER_MODES)}")
    task = Task(
        brief=body.brief, brand_kit_id=body.brand_kit_id,
        routing_profile_id=body.routing_profile_id,
        target_duration_s=body.target_duration_s,
        target_platforms=body.target_platforms,
        budget_cap_usd=body.budget_cap_usd or settings.default_task_budget_usd,
        format=body.format, style_preset=body.style_preset, render=body.render,
    )
    session.add(task)
    await session.flush()
    # ภาพสินค้า: เซฟก่อน enqueue — worker เริ่มขั้น concept ทันที บทต้องเห็นภาพตั้งแต่แรก
    refs: list[str] = []
    for i, f in enumerate(product_files or []):
        if not (f.content_type or "").startswith("image/"):
            raise HTTPException(400, f"ไฟล์ {f.filename} ไม่ใช่รูปภาพ")
        data = await f.read()
        if len(data) > 15 * 1024 * 1024:
            raise HTTPException(400, f"ไฟล์ {f.filename} ใหญ่เกิน 15MB")
        ext = ".png" if "png" in (f.content_type or "") else ".jpg"
        d = Path(settings.storage_root) / "tasks" / task.id
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"product_{i}{ext}"
        p.write_bytes(data)
        refs.append(str(p))
    task.product_refs = refs
    run = Run(task_id=task.id, state=RunState.DRAFT.value)
    session.add(run)
    await session.commit()
    runs_started.inc()
    await enqueue("advance_run", run.id)
    return {"task_id": task.id, "run_id": run.id, "state": run.state, "product_refs": len(refs)}


@app.post("/tasks")
async def create_task(body: CreateTask, session: AsyncSession = Depends(get_session)):
    """JSON — สำหรับ CLI/curl"""
    return await _create_task(session, body)


@app.post("/tasks/upload")
async def create_task_upload(
    brief: str = Form(...), target_duration_s: int = Form(30),
    format: str = Form("content"), style_preset: str | None = Form(None),
    render: str = Form("scene"), budget_cap_usd: float | None = Form(None),
    brand_kit_id: str | None = Form(None),
    products: list[UploadFile] = File(default=[]),
    session: AsyncSession = Depends(get_session),
):
    """multipart — หน้า UI ใช้ตัวนี้ เพื่อแนบภาพสินค้าไปพร้อมกัน"""
    body = CreateTask(brief=brief, target_duration_s=target_duration_s, format=format,
                      style_preset=style_preset or None, render=render,
                      budget_cap_usd=budget_cap_usd, brand_kit_id=brand_kit_id or None)
    return await _create_task(session, body, products)


@app.get("/runs")
async def list_runs(limit: int = 50, session: AsyncSession = Depends(get_session)):
    """รายการ run ล่าสุด — หน้า UI ใช้ตัวนี้ทำ sidebar

    selectinload(Run.task) เพราะต้องแสดง brief และ async session
    อ่าน relationship ที่ยัง lazy ไม่ได้
    """
    rows = (await session.execute(
        select(Run).options(selectinload(Run.task))
        .order_by(Run.created_at.desc()).limit(min(limit, 200)))).scalars().all()
    return [{
        "id": r.id, "state": r.state, "error": r.error,
        "brief": (r.task.brief if r.task else ""),
        "format": (r.task.format if r.task else "content"),
        "render": (r.task.render if r.task else "scene"),
        "cost_usd": round(r.total_cost_usd or 0, 4),
        "at_gate": r.state in GATE_VALUES,
        "created_at": r.created_at,
    } for r in rows]


@app.get("/runs/{run_id}")
async def get_run(run_id: str, session: AsyncSession = Depends(get_session)):
    run = (await session.execute(
        select(Run).where(Run.id == run_id)
        .options(selectinload(Run.task)))).scalar_one_or_none()
    if not run:
        raise HTTPException(404, "ไม่พบ run นี้")
    from sqlalchemy import func
    # ค่าใช้จ่ายจริงต่อ stage จากตาราง Generation (usage.cost ที่ provider รายงาน)
    rows = (await session.execute(
        select(Generation.stage, func.sum(Generation.cost_usd), func.count())
        .where(Generation.run_id == run_id).group_by(Generation.stage))).all()
    cost_by_stage = {st: {"usd": round(c or 0, 4), "calls": n} for st, c, n in rows}
    shots = run.shots or []
    story = run.story or {}
    duration = (shots[-1].get("end_s") if shots
                else sum(b.get("duration_s", 0) for b in story.get("beats", [])) or None)
    return {
        "id": run.id, "state": run.state, "error": run.error,
        "failed_state": run.failed_state,
        "cost_usd": round(run.total_cost_usd or 0, 4),
        "cost_by_stage": cost_by_stage,
        "at_gate": run.state in GATE_VALUES,
        "story": run.story, "shots": run.shots,
        "characters": [{"name": c.get("name"), "role": c.get("role")}
                       for c in (run.characters or [])],
        "qc_report": run.qc_report,
        "post_meta": run.post_meta,
        "animatic": bool(run.animatic_asset_id),
        "final": bool(run.final_asset_id),
        # หน้า UI ต้องใช้พวกนี้ครบในจอเดียว ไม่งั้นต้องยิงหลายรอบ
        "brief": (run.task.brief if run.task else ""),
        "budget_cap_usd": (run.task.budget_cap_usd if run.task else None),
        "target_duration_s": (run.task.target_duration_s if run.task else None),
        "format": (run.task.format if run.task else "content"),
        "style_preset": (run.task.style_preset if run.task else None),
        "render": (run.task.render if run.task else "scene"),
        "n_products": len(run.task.product_refs or []) if run.task else 0,
        "duration_s": duration,
        "paid_stages": run.paid_stages or [],
    }


@app.get("/runs/{run_id}/events")
async def get_events(run_id: str, session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(
        select(Event).where(Event.run_id == run_id).order_by(Event.id))).scalars().all()
    return [{"ts": e.ts, "level": e.level, "stage": e.stage, "message": e.message}
            for e in rows]


@app.get("/runs/{run_id}/video")
async def get_video(run_id: str, which: str = "animatic", download: bool = False,
                    session: AsyncSession = Depends(get_session)):
    """download=1 ส่ง Content-Disposition: attachment ให้เบราว์เซอร์เซฟไฟล์
    ชื่อไฟล์เอาจาก title ในบท ถ้ามี — คนได้ไฟล์ที่รู้ว่าเป็นคลิปอะไรโดยไม่ต้องเปิด"""
    run = (await session.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if not run:
        raise HTTPException(404, "ไม่พบ run นี้")
    path = run.final_asset_id if which == "final" else run.animatic_asset_id
    if not path or not Path(path).exists():
        raise HTTPException(404, f"ยังไม่มีไฟล์ {which}")
    if not download:
        return FileResponse(path, media_type="video/mp4")
    title = ((run.story or {}).get("title") or run_id[:8]).strip()
    # สระบน/ล่างและวรรณยุกต์ไทยไม่ผ่าน isalnum() — ต้องปล่อยทั้งบล็อก U+0E00–0E7F
    safe = "".join(ch for ch in title
                   if ch.isalnum() or ch in " _-" or "\u0e00" <= ch <= "\u0e7f")[:60].strip() or run_id[:8]
    return FileResponse(path, media_type="video/mp4",
                        filename=f"{safe}-{which}.mp4",
                        content_disposition_type="attachment")


@app.get("/runs/{run_id}/clip")
async def get_clip(run_id: str, idx: int = 0, session: AsyncSession = Depends(get_session)):
    """คลิปราย shot ที่โมเดลวิดีโอส่งกลับมา (ก่อนต่อเป็น final) — ดูได้แม้ run ล้มกลางทาง"""
    run = (await session.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if not run:
        raise HTTPException(404, "ไม่พบ run นี้")
    shots = run.shots or []
    if not 0 <= idx < len(shots):
        raise HTTPException(404, f"ไม่มี shot ลำดับที่ {idx}")
    raw = shots[idx].get("clip_path")
    if not raw:
        raise HTTPException(404, f"shot ลำดับที่ {idx} ยังไม่มีคลิป")
    workdir = (Path(settings.storage_root) / run_id).resolve()
    path = Path(raw).resolve()
    if not path.is_relative_to(workdir):
        raise HTTPException(403, "path อยู่นอกโฟลเดอร์ของ run นี้")
    if not path.exists():
        raise HTTPException(404, "ไม่พบไฟล์คลิป")
    return FileResponse(path, media_type="video/mp4")


@app.get("/runs/{run_id}/image")
async def get_image(run_id: str, kind: str = "keyframe", idx: int = 0,
                    session: AsyncSession = Depends(get_session)):
    """เสิร์ฟ keyframe / ภาพตัวละคร ให้หน้า UI เอาไปโชว์

    path มาจาก run.shots ซึ่งเป็นค่าที่ pipeline เขียนเอง ไม่ใช่ input จากผู้ใช้
    แต่ยัง resolve แล้วเช็กว่าอยู่ใต้ workdir ของ run นี้จริง เพราะ endpoint นี้
    รับ run_id จาก URL — ถ้าวันหนึ่งมีทางให้เขียน path ลง DB ได้ ตรงนี้จะเป็น
    ด่านสุดท้ายที่กัน path traversal
    """
    run = (await session.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if not run:
        raise HTTPException(404, "ไม่พบ run นี้")

    workdir = (Path(settings.storage_root) / run_id).resolve()
    if kind in ("character", "sheet"):
        chars = run.characters or []
        if not 0 <= idx < len(chars):
            raise HTTPException(404, f"ไม่มีตัวละครลำดับที่ {idx}")
        raw = chars[idx].get("sheet_path") if kind == "sheet" else chars[idx].get("ref_path")
        if not raw:
            raise HTTPException(404, "ยังไม่มีภาพนี้")
        path = Path(raw)
    elif kind == "product":
        task = (await session.execute(select(Task).where(Task.id == run.task_id))).scalar_one_or_none()
        refs = (task.product_refs if task else None) or []
        if not 0 <= idx < len(refs):
            raise HTTPException(404, f"ไม่มีภาพสินค้าลำดับที่ {idx}")
        # ภาพสินค้าอยู่ใต้ tasks/<task_id>/ ไม่ใช่ workdir ของ run
        workdir = (Path(settings.storage_root) / "tasks" / run.task_id).resolve()
        path = Path(refs[idx])
    else:
        shots = run.shots or []
        if not 0 <= idx < len(shots):
            raise HTTPException(404, f"ไม่มี shot ลำดับที่ {idx}")
        raw = shots[idx].get("keyframe_path")
        if not raw:
            raise HTTPException(404, f"shot ลำดับที่ {idx} ยังไม่มีภาพนิ่ง")
        path = Path(raw)

    path = path.resolve()
    if not path.is_relative_to(workdir):
        raise HTTPException(403, "path อยู่นอกโฟลเดอร์ของ run นี้")
    if not path.exists():
        raise HTTPException(404, "ไม่พบไฟล์ภาพ")
    return FileResponse(path, media_type="image/jpeg" if path.suffix == ".jpg" else "image/png")


@app.post("/runs/{run_id}/approve")
async def approve(run_id: str, body: ApproveIn, session: AsyncSession = Depends(get_session)):
    ctx = await load_ctx(session, run_id)
    gate = RunState(ctx.run.state)
    if gate not in GATES:
        raise HTTPException(409, f"run อยู่ที่ {gate.value} ไม่ใช่ gate")

    session.add(Review(run_id=run_id, gate=gate.value, reviewer=body.reviewer,
                       verdict="approve_animatic" if body.animatic_only else "approve",
                       notes=body.notes))
    await session.commit()
    await enqueue("approve_run", run_id, gate.value, body.animatic_only)
    return {"state": "กำลังเดินต่อ", "from_gate": gate.value}


@app.post("/runs/{run_id}/reject")
async def reject(run_id: str, body: RejectIn,
                 session: AsyncSession = Depends(get_session)):
    """ตีกลับ — reason_code เก็บแบบมีโครงสร้างเพื่อเอาไปวิเคราะห์ว่าพังตรงไหนบ่อย"""
    ctx = await load_ctx(session, run_id)
    gate = RunState(ctx.run.state)
    if gate not in GATES:
        raise HTTPException(409, f"run อยู่ที่ {gate.value} ไม่ใช่ gate")

    session.add(Review(run_id=run_id, gate=gate.value, reviewer=body.reviewer,
                       verdict="reject", reason_code=body.reason_code, notes=body.notes))
    target = RunState(body.back_to) if body.back_to else (
        RunState.STORY_DRAFT if gate == RunState.STORY_REVIEW else RunState.KEYFRAMING)
    ctx.run.state = target.value
    await session.commit()
    await enqueue("advance_run", run_id)
    return {"state": target.value}


@app.post("/runs/{run_id}/resume")
async def resume(run_id: str, session: AsyncSession = Depends(get_session)):
    """เดินต่อจาก state ที่ตาย (เช่นเครดิตหมดกลางทาง) — ของที่เจนเสร็จแล้วไม่จ่ายซ้ำ"""
    ctx = await load_ctx(session, run_id)
    if ctx.run.state != RunState.FAILED.value:
        raise HTTPException(409, f"run อยู่ที่ {ctx.run.state} ไม่ใช่ failed")
    if not ctx.run.failed_state:
        raise HTTPException(409, "run นี้ล้มก่อนจะมีการจำ state — ใช้ reject พร้อม back_to แทน")
    await enqueue("resume_run", run_id)
    return {"state": "กำลังเดินต่อ", "from": ctx.run.failed_state}


@app.post("/runs/{run_id}/regenerate")
async def regenerate(run_id: str, body: RegenIn, session: AsyncSession = Depends(get_session)):
    """สร้างใหม่เฉพาะ shot ที่ระบุ — ไม่รื้อทั้ง run

    ถ้ายังไม่ผ่าน Gate 1 จะทำแค่ภาพนิ่ง (ถูก) ถ้าผ่านแล้วจะทำวิดีโอด้วย (แพง)
    """
    ctx = await load_ctx(session, run_id)
    if body.prompt_overrides:
        shots = ctx.run.shots or []
        for s in shots:
            if s["id"] in body.prompt_overrides:
                s["plan"]["image_prompt"] = body.prompt_overrides[s["id"]]
        ctx.run.shots = shots
        await session.commit()

    already_paid = "video" in (ctx.run.paid_stages or [])
    await enqueue("regenerate_shots", run_id, body.shot_ids, already_paid)
    return {"shots": body.shot_ids, "will_regenerate_video": already_paid}


@app.post("/runs/{run_id}/publish")
async def publish(run_id: str, body: PublishIn,
                  session: AsyncSession = Depends(get_session)):
    """โพสต์ได้เฉพาะหลังผ่าน Gate 2 แล้วเท่านั้น — ไม่มีทางลัด
    หลาย target = หลาย job แยกกัน แต่ละอันมี idempotency ของตัวเอง"""
    run = (await session.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if not run:
        raise HTTPException(404, "ไม่พบ run นี้")
    if run.state not in (RunState.APPROVED.value, RunState.PUBLISHED.value):
        raise HTTPException(409, f"run อยู่ที่ {run.state} ยังไม่ผ่านการอนุมัติ")
    if body.privacy not in ("private", "unlisted", "public"):
        raise HTTPException(400, "privacy ต้องเป็น private | unlisted | public")
    if not body.target_ids:
        raise HTTPException(400, "ต้องเลือกอย่างน้อยหนึ่ง target")
    rows = (await session.execute(
        select(PublishTarget).where(PublishTarget.id.in_(body.target_ids)))).scalars().all()
    missing = set(body.target_ids) - {t.id for t in rows}
    if missing:
        raise HTTPException(404, f"ไม่พบ target: {sorted(missing)}")
    for t in rows:
        await enqueue("publish_run", run_id, t.id, body.privacy)
    return {"queued": [t.id for t in rows], "privacy": body.privacy}


@app.get("/runs/{run_id}/publishes")
async def list_publishes(run_id: str, session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(
        select(PublishAttempt, PublishTarget)
        .join(PublishTarget, PublishAttempt.target_id == PublishTarget.id)
        .where(PublishAttempt.run_id == run_id)
        .order_by(PublishAttempt.created_at.desc()))).all()
    return [{
        "id": a.id, "target_id": t.id, "platform": t.platform,
        "account_handle": t.account_handle, "status": a.status,
        "external_post_id": a.external_post_id, "url": a.url,
        "error": a.error, "posted_at": a.posted_at, "created_at": a.created_at,
    } for a, t in rows]


# ---------------------------------------------------------------- publish targets

@app.get("/targets")
async def list_targets(session: AsyncSession = Depends(get_session)):
    """ไม่ส่ง token กลับไป — หน้า UI ต้องรู้แค่ว่ามีบัญชีไหนบ้าง"""
    rows = (await session.execute(
        select(PublishTarget).order_by(PublishTarget.created_at))).scalars().all()
    return [{
        "id": t.id, "platform": t.platform, "account_handle": t.account_handle,
        "audited": t.audited, "expires_at": t.expires_at,
        "config": {k: v for k, v in (t.config or {}).items()},
        "has_refresh": bool(t.refresh_token_enc),
    } for t in rows]


@app.post("/targets")
async def create_target(body: TargetIn, session: AsyncSession = Depends(get_session)):
    if body.platform not in PUBLISHERS:
        raise HTTPException(400, f"platform ต้องเป็นหนึ่งใน {sorted(PUBLISHERS)}")
    if body.platform == "facebook" and not body.config.get("page_id"):
        raise HTTPException(400, "facebook ต้องมี config.page_id")
    t = PublishTarget(
        platform=body.platform, account_handle=body.account_handle,
        access_token_enc=body.access_token, refresh_token_enc=body.refresh_token,
        audited=body.audited, config=body.config,
    )
    session.add(t)
    await session.commit()
    return {"id": t.id, "platform": t.platform, "account_handle": t.account_handle}


@app.delete("/targets/{target_id}")
async def delete_target(target_id: str, session: AsyncSession = Depends(get_session)):
    t = (await session.execute(
        select(PublishTarget).where(PublishTarget.id == target_id))).scalar_one_or_none()
    if not t:
        raise HTTPException(404, "ไม่พบ target นี้")
    used = (await session.execute(
        select(PublishAttempt.id).where(PublishAttempt.target_id == target_id).limit(1))).first()
    if used:
        raise HTTPException(409, "target นี้เคยโพสต์แล้ว ลบไม่ได้ (ประวัติจะขาด)")
    await session.delete(t)
    await session.commit()
    return {"deleted": target_id}


@app.get("/metrics")
async def metrics():
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
async def healthz():
    return {"ok": True, "external_providers": settings.external_providers_enabled}


# ---------------------------------------------------------------- UI
# mount ท้ายสุดเสมอ เพื่อไม่ให้ StaticFiles ไปกลืน path ของ API
UI_DIR = Path(__file__).parent / "static"
if UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def _root():
        return RedirectResponse("/ui/")
