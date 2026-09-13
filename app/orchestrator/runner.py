"""ตัวเดิน pipeline

ทำไมไม่ใช้ Temporal: state ต้องอยู่ใน Postgres อยู่แล้วเพราะ UI ต้องอ่าน
พอมี state ใน DB ครบ การ resume ก็แค่ "อ่าน state แล้วทำขั้นถัดไป"
ซึ่งเป็นโค้ดไม่กี่ร้อยบรรทัดและ debug ได้ด้วย SQL ธรรมดา

หลักการ: ทุก transition เขียนลง DB ก่อนทำงาน ไม่ใช่หลังทำงาน
ถ้า process ตายกลางทาง เรารู้ว่ามันตายตอนอยู่ state ไหน
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..agents import stages
from ..agents.concept_agent import run_concept_agent
from ..agents.qc_agent import run_qc_repair
from ..media.animatic import ShotClip, render_animatic
from ..media.render import VideoShot, probe_stream, render_final
from ..models import Event, Generation, Run
from ..providers.registry import BudgetExceeded, BudgetGuard, Registry
from ..schemas import ConceptOut, RunState, Shot, Story

log = logging.getLogger(__name__)


class RunNotFound(LookupError):
    """ไม่มี run id นี้ใน DB — ให้ชั้น HTTP แปลงเป็น 404 เอง"""

# ขั้นที่ทำเองอัตโนมัติ → ขั้นถัดไป
# gate ไม่อยู่ในนี้ เพราะ gate ต้องรอ input จากคน
NEXT = {
    RunState.DRAFT: RunState.CONCEPT,
    RunState.CONCEPT: RunState.STORY_DRAFT,
    RunState.STORY_DRAFT: RunState.STORY_REVIEW,
    RunState.SHOT_PLANNING: RunState.CHARACTER_DESIGN,
    RunState.CHARACTER_DESIGN: RunState.KEYFRAMING,
    RunState.KEYFRAMING: RunState.ANIMATIC,
    RunState.ANIMATIC: RunState.STORYBOARD_REVIEW,
    RunState.VIDEO_GEN: RunState.FINAL_RENDER,
    RunState.FINAL_RENDER: RunState.QC,
    RunState.QC: RunState.FINAL_REVIEW,
}

# gate ผ่านแล้วไปไหนต่อ
AFTER_GATE = {
    RunState.STORY_REVIEW: RunState.SHOT_PLANNING,
    RunState.STORYBOARD_REVIEW: RunState.VIDEO_GEN,
    RunState.FINAL_REVIEW: RunState.APPROVED,
}


@dataclass
class Ctx:
    run: Run
    session: AsyncSession
    registry: Registry
    workdir: Path
    fonts_dir: Path
    brand: dict
    budget: BudgetGuard
    nvenc: bool = False


async def _event(ctx: Ctx, msg: str, *, level="info", stage="", **payload):
    ctx.session.add(Event(run_id=ctx.run.id, level=level, stage=stage,
                          message=msg, payload=payload))
    await ctx.session.commit()   # ให้ LOG ในหน้า UI ขึ้นทันที ไม่ต้องรอจบ job
    log.log(logging.WARNING if level == "warn" else logging.INFO,
            "[run=%s stage=%s] %s", ctx.run.id[:8], stage, msg)


async def _gen(ctx: Ctx, stage: str, route, cost: float, **kw):
    ctx.session.add(Generation(
        run_id=ctx.run.id, stage=stage, provider=route.provider,
        model=route.model, params=route.params, cost_usd=cost, **kw))
    ctx.run.total_cost_usd = (ctx.run.total_cost_usd or 0) + cost


async def _gens(ctx: Ctx, records) -> float:
    """เขียน Generation หนึ่งแถวต่อหนึ่ง call ด้วย cost จริงจาก provider แล้วคืนผลรวม"""
    total = 0.0
    for r in records:
        ctx.session.add(Generation(
            run_id=ctx.run.id, stage=r.stage, provider=r.route.provider,
            model=r.route.model, params=r.route.params, cost_usd=r.cost_usd,
            shot_id=r.shot_id, external_id=r.external_id))
        total += r.cost_usd
    ctx.run.total_cost_usd = (ctx.run.total_cost_usd or 0) + total
    return total


def _cost_breakdown(records) -> str:
    return ", ".join(f"{r.shot_id or r.stage} ${r.cost_usd:.3f}" for r in records)


async def _set_state(ctx: Ctx, state: RunState):
    """commit ไม่ใช่แค่ flush — "เขียน transition ก่อนทำงาน" มีความหมายก็ต่อเมื่อ
    ถึง DB จริง เดิม flush ไว้แล้วรอ commit ตอนจบ job ทำให้ (1) worker ตายกลาง
    job แล้ว rollback กลับไป state ก่อนหน้า ซึ่ง sweep จะเดินซ้ำตั้งแต่ต้น job
    และ (2) หน้า UI เห็นแต่ state ตอนต้น job จนกว่าจะถึง gate (resume โชว์
    failed อยู่หลายนาทีทั้งที่กำลังเจนวิดีโอ) session ตั้ง expire_on_commit=False
    จึงใช้ object ต่อได้หลัง commit"""
    from sqlalchemy import func
    ctx.run.state = state.value
    ctx.run.state_entered_at = func.now()
    await ctx.session.commit()


# ---------------------------------------------------------------- ขั้นต่าง ๆ

async def _past_angles(ctx: Ctx, limit: int = 12) -> list[str]:
    """มุมที่เคยใช้กับแบรนด์นี้ — สัญญาณจริงจาก DB ของเราเอง
    ดีกว่าให้ agent เดาจาก prompt ว่าอะไรซ้ำ"""
    from ..models import Task as TaskModel
    if not ctx.run.task or not ctx.run.task.brand_kit_id:
        return []
    rows = (await ctx.session.execute(
        select(Run.concept).join(TaskModel, Run.task_id == TaskModel.id)
        .where(TaskModel.brand_kit_id == ctx.run.task.brand_kit_id)
        .where(Run.id != ctx.run.id)
        .where(Run.concept.is_not(None))
        .order_by(Run.created_at.desc()).limit(limit))).scalars().all()
    out = []
    for c in rows:
        try:
            out.append(c["angles"][c["chosen_index"]]["title"])
        except (KeyError, IndexError, TypeError):
            continue
    return out


async def step_concept(ctx: Ctx):
    """agentic loop — ดูงานเก่า ให้คะแนน hook ตัวเอง แก้ก่อนส่ง"""
    task = ctx.run.task
    past = await _past_angles(ctx)
    out, cost, transcript = await run_concept_agent(
        ctx.registry, task.brief, ctx.brand,
        past_angle_titles=past, budget=ctx.budget, format_name=task.format)
    ctx.run.concept = out.model_dump()
    ctx.run.total_cost_usd = (ctx.run.total_cost_usd or 0) + cost
    await _event(ctx, f"เลือกมุม: {out.angles[out.chosen_index].title}",
                 stage="concept", steps=len([m for m in transcript if m.get("role") == "tool"]),
                 avoided=len(past))


async def step_story(ctx: Ctx):
    concept = ConceptOut.model_validate(ctx.run.concept)
    task = ctx.run.task
    allowed = await stages.video_durations(ctx.registry) if task.render != "avatar" else []
    products = [Path(p).read_bytes() for p in (task.product_refs or []) if Path(p).exists()]
    out, route, cost = await stages.run_story(
        ctx.registry, concept, task.brief, task.target_duration_s, ctx.brand, allowed,
        budget=ctx.budget, format_name=task.format, render=task.render,
        product_images=products or None)
    if task.render == "avatar":
        # โมเดล avatar ทำได้อย่างเดียวคือ "ตัวละครพูด" — beat ที่ไม่มีบทพูดไปต่อไม่ได้
        # จับตั้งแต่ตรงนี้ (ยังไม่จ่ายค่าภาพ) ดีกว่าไปล้มที่ video_gen
        mute = [b.idx for b in out.beats if not (b.speaker and b.dialogue)]
        if mute:
            raise ValueError(f"โหมด avatar ต้องมีบทพูดทุก beat แต่ beat {mute} ไม่มี — "
                             "ลองสั่งใหม่หรือแก้บรีฟให้เป็นเรื่องที่ตัวละครพูด")
    ctx.run.story = out.model_dump()
    await _gen(ctx, "story", route, cost)
    drift = out.total_duration_s - ctx.run.task.target_duration_s
    await _event(
        ctx,
        f"เขียนบท {len(out.beats)} beat ตัวละคร {len(out.characters)} ตัว "
        f"รวม {out.total_duration_s}s ({'เกิน' if drift > 0 else 'สั้นกว่า'}เป้า {abs(drift)}s)",
        stage="story", level="warn" if abs(drift) > 8 else "info")


async def step_shot_plan(ctx: Ctx):
    story = Story.model_validate(ctx.run.story)
    plan, route, cost = await stages.run_shot_plan(
        ctx.registry, story, ctx.brand, budget=ctx.budget, format_name=ctx.run.task.format)
    allowed = await stages.video_durations(ctx.registry) if ctx.run.task.render != "avatar" else []
    shots = stages.build_shots(plan, story, allowed)
    ctx.run.shots = [s.model_dump() for s in shots]
    await _gen(ctx, "shots", route, cost)
    await _event(ctx, f"วาง {len(shots)} shot รวม {shots[-1].end_s:.0f}s "
                      f"(ความยาวที่โมเดลวิดีโอรับ: {allowed or 'ไม่จำกัด'})", stage="shots")


async def step_characters(ctx: Ctx):
    story = Story.model_validate(ctx.run.story)
    have = [c for c in (ctx.run.characters or [])
            if Path(c.get("ref_path", "")).exists() and Path(c.get("sheet_path", "")).exists()]
    if have and len(have) == len(story.characters):
        await _event(ctx, f"ภาพตัวละคร {len(have)} ตัวมีอยู่แล้ว — ข้าม (resume)", stage="character")
        return
    sheets, records = await stages.run_character_sheets(
        ctx.registry, story, ctx.brand, ctx.workdir / "characters", budget=ctx.budget)
    ctx.run.characters = sheets
    cost = await _gens(ctx, records)
    await _event(ctx, f"สร้างภาพอ้างอิงตัวละคร {len(sheets)} ตัว ${cost:.3f} "
                      f"({_cost_breakdown(records)})", stage="character")


def _character_refs(ctx: Ctx) -> dict[str, Path]:
    return {c["name"]: Path(c["ref_path"]) for c in (ctx.run.characters or [])}


def _character_sheets(ctx: Ctx) -> dict[str, Path]:
    return {c["name"]: Path(c["sheet_path"]) for c in (ctx.run.characters or []) if c.get("sheet_path")}


def _product_refs(ctx: Ctx) -> list[Path]:
    return [Path(p) for p in (ctx.run.task.product_refs or []) if ctx.run.task]


async def step_keyframes(ctx: Ctx, only: list[str] | None = None):
    shots = [Shot.model_validate(s) for s in ctx.run.shots]
    story = Story.model_validate(ctx.run.story)
    refs = _character_refs(ctx)
    if not refs:
        raise ValueError("ยังไม่มีภาพอ้างอิงตัวละคร — ขั้น character_design ต้องผ่านก่อน")
    if ctx.run.task.render == "avatar":
        # ไม่สร้างฉาก — keyframe ของ shot คือ portrait ของคนพูด (ฟรี) ใช้ทำ animatic
        # และเป็นภาพที่ส่งให้ avatar model
        first = next(iter(refs.values()))
        for sh in shots:
            beat = next((b for b in story.beats if b.idx == sh.beat_idx), None)
            sh.keyframe_path = str(refs.get(beat.speaker if beat else None, first))
        ctx.run.shots = [s.model_dump() for s in shots]
        await _event(ctx, f"โหมด avatar — ใช้ portrait ตัวละครเป็นภาพของ {len(shots)} shot "
                          "(ไม่สร้างฉาก $0)", stage="keyframe")
        return
    records = await stages.run_keyframes(
        ctx.registry, shots, story, refs, ctx.brand,
        ctx.workdir / "keyframes", only=only, budget=ctx.budget,
        character_sheets=_character_sheets(ctx), product_refs=_product_refs(ctx))
    ctx.run.shots = [s.model_dump() for s in shots]
    cost = await _gens(ctx, records)
    await _event(ctx, f"สร้างภาพนิ่ง {len(only or shots)} ใบ ${cost:.3f} "
                      f"({_cost_breakdown(records)})", stage="keyframe")


async def step_animatic(ctx: Ctx):
    """ภาพนิ่งขยับตามจังหวะที่วางไว้ — ไม่มีเสียง ไม่มีซับ

    ยังคุ้มที่จะทำแม้ไม่มีเสียง: ฟรี เห็นจังหวะและลำดับภาพก่อนจ่ายค่าวิดีโอ
    (ซึ่งตอนนี้แพงเป็นสองเท่าของเดิมเพราะเปิดเสียงจากโมเดล)
    """
    shots = [Shot.model_validate(s) for s in ctx.run.shots]
    clips = [ShotClip(image=Path(s.keyframe_path), duration_s=s.duration_s,
                      zoom_start=s.ken_burns.zoom_start, zoom_end=s.ken_burns.zoom_end,
                      pan_x=s.ken_burns.pan_x, pan_y=s.ken_burns.pan_y)
             for s in shots]

    out = ctx.workdir / "animatic.mp4"
    await render_animatic(clips, workdir=ctx.workdir / "tmp_animatic", out=out)
    ctx.run.animatic_asset_id = str(out)
    await _event(ctx, f"เรนเดอร์ animatic เสร็จ — ใช้ไปทั้งหมด "
                      f"${ctx.run.total_cost_usd:.3f}", stage="animatic")


async def step_video(ctx: Ctx, only: list[str] | None = None):
    shots = [Shot.model_validate(s) for s in ctx.run.shots]
    story = Story.model_validate(ctx.run.story)
    records, errors = await stages.run_video(
        ctx.registry, shots, story, ctx.workdir / "clips", brand=ctx.brand,
        only=only, budget=ctx.budget, render=ctx.run.task.render,
        character_refs=_character_refs(ctx))
    # บันทึกของที่สำเร็จก่อนเสมอ — ถ้ามีตัวล้ม เงินที่จ่ายไปแล้วและคลิปที่ได้ต้องอยู่ใน DB
    ctx.run.shots = [s.model_dump() for s in shots]
    cost = await _gens(ctx, records)
    if records:
        paid = set(ctx.run.paid_stages or [])
        paid.add("video")
        ctx.run.paid_stages = sorted(paid)
        model = records[0].route.model
        await _event(ctx, f"สร้างวิดีโอ {len(records)} shot ด้วย {model} ${cost:.3f} "
                          f"({_cost_breakdown(records)})", stage="video",
                     jobs={r.shot_id: r.external_id for r in records})
    if errors:
        await ctx.session.flush()
        ids = [sid for sid, _ in errors]
        raise RuntimeError(
            f"video ล้ม {len(errors)} shot {ids} (สำเร็จ {len(records)}) — "
            f"กด resume แล้วจะทำต่อเฉพาะที่ขาด: {errors[0][1]}")
    # timeline ต้องตามคลิปจริง — ตัดคลิปที่มีเสียงพูดให้พอดีแผนไม่ได้
    actual = {s.id: probe_stream(Path(s.clip_path))["duration"]
              for s in shots if s.clip_path}
    stages.retime_shots(shots, actual)
    ctx.run.shots = [s.model_dump() for s in shots]


async def step_final_render(ctx: Ctx):
    """ต่อ clip จริง (พร้อมเสียงที่โมเดลสร้างมา) — ขั้นที่ปิดวงจร pipeline"""
    shots = [Shot.model_validate(s) for s in ctx.run.shots]
    missing = [s.id for s in shots if not s.clip_path]
    if missing:
        raise ValueError(f"ยังไม่มี clip ของ shot: {missing}")

    vshots = [VideoShot(clip=Path(s.clip_path), duration_s=s.duration_s,
                        fade_in=0.15 if s.plan.transition_in == "fade" else 0.0)
              for s in shots]

    out = ctx.workdir / "final.mp4"
    await render_final(vshots, workdir=ctx.workdir / "tmp_final", out=out,
                       nvenc=ctx.nvenc)
    ctx.run.final_asset_id = str(out)
    await _event(ctx, f"ตัดต่อขั้นสุดท้ายเสร็จ ({out.stat().st_size // 1024} KB)",
                 stage="final_render")


async def step_qc(ctx: Ctx):
    """agentic loop — ตรวจ แก้ prompt สร้าง shot ที่พังใหม่ ตรวจซ้ำ

    ขอบเขตอยู่ใน qc_agent: max 2 รอบ, shot ละไม่เกิน 2 ครั้ง,
    และหยุดทันทีถ้าแก้แล้วจำนวนปัญหาไม่ลดลง
    """
    is_animatic = not ctx.run.final_asset_id or (
        ctx.run.final_asset_id == ctx.run.animatic_asset_id)
    target = Path(ctx.run.final_asset_id or ctx.run.animatic_asset_id)
    story = Story.model_validate(ctx.run.story)
    shots = [Shot.model_validate(s) for s in ctx.run.shots]

    async def regenerate(shot_ids: list[str]) -> None:
        """callback ที่ agent เรียก — สร้างภาพ/คลิปใหม่แล้ว re-render
        agent ไม่รู้จัก provider หรือ ffmpeg เลย รู้แค่ว่าเรียกแล้วของอัปเดต"""
        ctx.run.shots = [s.model_dump() for s in shots]
        await step_keyframes(ctx, only=shot_ids)
        if not is_animatic:
            await step_video(ctx, only=shot_ids)
            await step_final_render(ctx)
        else:
            await step_animatic(ctx)
        for s in shots:
            for fresh in ctx.run.shots:
                if fresh["id"] == s.id:
                    s.keyframe_path = fresh.get("keyframe_path")
                    s.clip_path = fresh.get("clip_path")
                    # step_video เขียน timeline ใหม่ตามคลิปจริง ต้องตามมาด้วย
                    s.start_s, s.end_s = fresh["start_s"], fresh["end_s"]

    outcome = await run_qc_repair(
        ctx.registry, video=target, shots=shots, story_title=story.title,
        workdir=ctx.workdir, regenerate=regenerate,
        mode="animatic" if is_animatic else "video",
        expected_duration_s=lambda: shots[-1].end_s, budget=ctx.budget)

    ctx.run.shots = [s.model_dump() for s in shots]
    ctx.run.qc_report = outcome.report.model_dump()
    ctx.run.total_cost_usd = (ctx.run.total_cost_usd or 0) + outcome.cost_usd

    msg = f"QC: {outcome.report.verdict} ({len(outcome.report.issues)} ข้อ)"
    if outcome.repaired:
        msg += f" — แก้ {len(outcome.repaired)} shot ใน {outcome.rounds} รอบ"
    if outcome.escalated:
        msg += f" — ส่งให้คนดู: {outcome.note[:80]}"
    await _event(ctx, msg, stage="qc",
                 level="warn" if (outcome.escalated or outcome.report.blocking) else "info")


async def step_metadata(ctx: Ctx):
    """แคปชัน แฮชแท็ก ชื่อคลิป — ต้องมีก่อนโพสต์

    ทำตอนคนกดผ่าน Gate 2 ไม่ใช่ตอน QC เพราะถ้า QC ต้องวนแก้หลายรอบ
    metadata ที่สร้างไว้ก่อนจะอ้างอิงวิดีโอเวอร์ชันที่ถูกทิ้งไปแล้ว
    """
    story = Story.model_validate(ctx.run.story)
    shots = [Shot.model_validate(s) for s in ctx.run.shots]
    out, route, cost = await stages.run_metadata(
        ctx.registry, story, shots[-1].end_s, budget=ctx.budget)
    ctx.run.post_meta = out.model_dump()
    await _gen(ctx, "meta", route, cost)
    await _event(ctx, f"เตรียมข้อมูลโพสต์: {out.title_yt}", stage="meta")


STEPS = {
    RunState.CONCEPT: step_concept,
    RunState.STORY_DRAFT: step_story,
    RunState.SHOT_PLANNING: step_shot_plan,
    RunState.CHARACTER_DESIGN: step_characters,
    RunState.KEYFRAMING: step_keyframes,
    RunState.ANIMATIC: step_animatic,
    RunState.VIDEO_GEN: step_video,
    RunState.FINAL_RENDER: step_final_render,
    RunState.QC: step_qc,
}


async def _run_step(ctx: Ctx, state: RunState, step) -> bool:
    """รัน step หนึ่งตัวพร้อมดัก error — คืน False ถ้า run ตายไปแล้ว

    แยกออกมาเพราะ approve() ก็ต้องรัน step (metadata) และต้องจัดการ error
    แบบเดียวกับ advance() เป๊ะ ไม่งั้นจะมีทางที่ run พังแล้วไม่ถูกตั้งเป็น FAILED
    """
    try:
        await step(ctx)
        return True
    except BudgetExceeded as e:
        ctx.run.error = str(e)
        ctx.run.failed_state = state.value
        await _set_state(ctx, RunState.FAILED)
        await _event(ctx, f"งบหมด: {e}", level="warn", stage=state.value)
        return False
    except Exception as e:  # noqa: BLE001
        ctx.run.error = f"{type(e).__name__}: {e}"
        ctx.run.failed_state = state.value
        await _set_state(ctx, RunState.FAILED)
        await _event(ctx, f"ล้มเหลวที่ {state.value}: {e}",
                     level="error", stage=state.value)
        log.exception("run %s ล้มที่ %s", ctx.run.id, state.value)
        return False


async def advance(ctx: Ctx, *, max_steps: int = 20) -> RunState:
    """เดินไปเรื่อย ๆ จนกว่าจะชน gate, จบ, หรือพัง

    ปลอดภัยที่จะเรียกซ้ำ — ถ้า process ตายกลางทาง เรียกใหม่แล้วมันเดินต่อ
    จาก state ที่ค้างอยู่ใน DB
    """
    for _ in range(max_steps):
        state = RunState(ctx.run.state)

        if state in AFTER_GATE:
            await _event(ctx, f"รอคนอนุมัติที่ {state.value}", stage=state.value)
            return state
        # PUBLISHING อยู่ในมือ publish_run ไม่ใช่ของ advance() — ปล่อยไว้เฉย ๆ
        if state in (RunState.APPROVED, RunState.PUBLISHING, RunState.PUBLISHED,
                     RunState.REJECTED, RunState.FAILED):
            return state

        step = STEPS.get(state)
        if step and not await _run_step(ctx, state, step):
            return RunState.FAILED

        nxt = NEXT.get(state)
        if nxt is None:
            return state
        await _set_state(ctx, nxt)

    await _event(ctx, "เดินครบ max_steps แล้วยังไม่จบ — น่าจะมี loop ใน NEXT",
                 level="warn")
    return RunState(ctx.run.state)


async def approve(ctx: Ctx, gate: RunState, *, animatic_only: bool = False) -> RunState:
    """คนกดผ่าน gate — จุดเดียวที่ pipeline เดินต่อได้

    `gate` มาจาก job ที่ถูก enqueue ไว้ตอนคนกดปุ่ม ไม่ใช่สถานะปัจจุบัน จึงต้อง
    เช็กก่อนว่า run ยังค้างอยู่ที่ gate นั้นจริง มิฉะนั้นการกดซ้ำจาก UI ที่ค้าง
    (หน้าเว็บ poll ทุก 2.5 วินาที ปุ่มจึงอยู่ต่ออีกครู่หลัง gate ผ่านไปแล้ว)
    จะพา run *ถอยหลัง* ไปเข้าขั้นที่แพงกว่าเดิม — เจอจริง: run ที่ผ่าน Gate 1
    แบบ animatic แล้วไปถึง final_review ถูกกดซ้ำจนเด้งกลับเข้า video_gen
    """
    if RunState(ctx.run.state) != gate:
        log.warning("ข้ามการอนุมัติ: run อยู่ที่ %s แล้ว ไม่ใช่ %s",
                    ctx.run.state, gate.value)
        await _event(ctx, f"ข้ามการกดอนุมัติซ้ำ — run เดินผ่าน {gate.value} ไปแล้ว",
                     stage="gate", level="warn")
        return RunState(ctx.run.state)

    if gate == RunState.STORYBOARD_REVIEW and animatic_only:
        # จบที่ animatic ไม่จ่ายค่า video generation
        ctx.run.final_asset_id = ctx.run.animatic_asset_id
        await _set_state(ctx, RunState.QC)
        await _event(ctx, "อนุมัติเป็น animatic — ข้ามการสร้างวิดีโอ",
                     stage="gate1")
        return await advance(ctx)

    nxt = AFTER_GATE.get(gate)
    if nxt is None:
        raise ValueError(f"{gate} ไม่ใช่ gate")

    # Gate 2 = งานผ่านแล้วจริง ๆ ถึงค่อยจ่ายค่าเขียนแคปชัน
    # ถ้าไม่มี post_meta ตรงนี้ publish_run จะ validate PostMeta ไม่ผ่านทุกครั้ง
    if gate == RunState.FINAL_REVIEW and not ctx.run.post_meta:
        if not await _run_step(ctx, RunState.FINAL_REVIEW, step_metadata):
            return RunState.FAILED

    await _set_state(ctx, nxt)
    return await advance(ctx)


async def resume(ctx: Ctx) -> RunState:
    """เดินต่อจาก state ที่ตาย — ใช้ได้เพราะทุก transition เขียนก่อนทำงาน
    และ step ที่แพง (character/keyframe/video) ข้ามของที่มีอยู่แล้วบนดิสก์"""
    if RunState(ctx.run.state) != RunState.FAILED:
        raise ValueError(f"run อยู่ที่ {ctx.run.state} ไม่ใช่ failed")
    back = ctx.run.failed_state
    if not back:
        raise ValueError("ไม่รู้ว่า run นี้ตายที่ state ไหน (failed ก่อนมี failed_state) — "
                         "ใช้ reject/back_to แทน")
    # ถ้าตายตอนทำ metadata ใน approve() failed_state จะเป็น final_review —
    # กลับไปรอที่ gate นั้น advance() หยุดให้เอง แล้วคนกด approve ใหม่
    target = RunState(back)
    await _event(ctx, f"resume จาก {target.value} (เดิม: {(ctx.run.error or '')[:120]})",
                 stage="resume")
    ctx.run.error = None
    ctx.run.failed_state = None
    await _set_state(ctx, target)
    return await advance(ctx)


async def load_ctx(session: AsyncSession, run_id: str, *, settings) -> Ctx:
    """โหลด run + brand kit + routing profile มาประกอบเป็น Ctx

    ต้อง selectinload(Run.task) เสมอ — async session อ่าน relationship ที่ยัง
    lazy อยู่ไม่ได้ จะโยน MissingGreenlet ทันทีที่แตะ ctx.run.task
    ซึ่ง step_concept / step_story / _past_angles / BudgetGuard แตะทั้งหมด
    """
    from ..models import BrandKit, RoutingProfile

    run = (await session.execute(
        select(Run).where(Run.id == run_id)
        .options(selectinload(Run.task)))).scalar_one_or_none()
    if run is None:
        raise RunNotFound(run_id)

    profile: dict = {}
    if run.task and run.task.routing_profile_id:
        rp = (await session.execute(select(RoutingProfile).where(
            RoutingProfile.id == run.task.routing_profile_id))).scalar_one_or_none()
        if rp:
            profile = {"stages": rp.stages, "animatic_only": rp.animatic_only}
    if not profile:
        rp = (await session.execute(select(RoutingProfile).where(
            RoutingProfile.is_default.is_(True)))).scalar_one_or_none()
        if rp:
            profile = {"stages": rp.stages, "animatic_only": rp.animatic_only}

    brand: dict = {}
    if run.task and run.task.brand_kit_id:
        bk = (await session.execute(select(BrandKit).where(
            BrandKit.id == run.task.brand_kit_id))).scalar_one_or_none()
        if bk:
            brand = {
                "tone_of_voice": bk.tone_of_voice, "style_suffix": bk.style_suffix,
                "do_donts": bk.do_donts, "font_family": bk.font_family,
                "pronunciation": bk.pronunciation,
            }

    # preset สไตล์จากหน้า New Task ชนะ brand kit — คนเลือกตอนสั่งงานตั้งใจกว่าค่าที่ตั้งไว้นานแล้ว
    from ..agents.formats import style_suffix
    if run.task and (suffix := style_suffix(run.task.style_preset)) is not None:
        brand["style_suffix"] = suffix

    workdir = Path(settings.storage_root) / run.id
    workdir.mkdir(parents=True, exist_ok=True)
    return Ctx(
        run=run, session=session, registry=Registry(profile, settings=settings),
        workdir=workdir, fonts_dir=Path(settings.fonts_dir), brand=brand,
        budget=BudgetGuard(run.task.budget_cap_usd, run.total_cost_usd or 0.0),
        nvenc=settings.use_nvenc,
    )
