"""arq worker — งานหนักและงานยาววิ่งที่นี่ ไม่ใช่ในโปรเซส API

แยกคิว render ออกจากคิว agent เพราะ render กิน CPU/GPU และต้องจำกัด
concurrency แยก ถ้าปนกันคิว agent จะถูก render บล็อก
"""
from __future__ import annotations

import logging

from arq import cron
from arq.connections import RedisSettings

from .config import get_settings
from .db import Session
from .orchestrator import runner
from .queue import enqueue  # noqa: F401  — re-export ให้โค้ดเดิมที่เรียก worker.enqueue

log = logging.getLogger(__name__)
settings = get_settings()


async def advance_run(ctx, run_id: str):
    async with Session() as session:
        c = await runner.load_ctx(session, run_id, settings=settings)
        state = await runner.advance(c)
        await session.commit()
        return state.value


async def approve_run(ctx, run_id: str, gate: str, animatic_only: bool):
    from .schemas import RunState
    async with Session() as session:
        c = await runner.load_ctx(session, run_id, settings=settings)
        state = await runner.approve(c, RunState(gate), animatic_only=animatic_only)
        await session.commit()
        return state.value


async def resume_run(ctx, run_id: str):
    async with Session() as session:
        c = await runner.load_ctx(session, run_id, settings=settings)
        state = await runner.resume(c)
        await session.commit()
        return state.value


async def regenerate_shots(ctx, run_id: str, shot_ids: list[str], with_video: bool):
    async with Session() as session:
        c = await runner.load_ctx(session, run_id, settings=settings)
        await runner.step_keyframes(c, only=shot_ids)
        await runner.step_animatic(c)
        if with_video:
            await runner.step_video(c, only=shot_ids)
            await runner.step_final_render(c)
        await session.commit()


async def sweep_stuck_runs(ctx):
    """run ที่ค้างใน state เดิมเกิน 30 นาทีแปลว่ามีอะไรตายไป
    ปลุกให้เดินต่อ — advance() ปลอดภัยที่จะเรียกซ้ำ"""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import select
    from .models import Run
    from .schemas import GATES, RunState

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
    async with Session() as session:
        rows = (await session.execute(
            select(Run).where(Run.state_entered_at < cutoff))).scalars().all()
        for run in rows:
            st = RunState(run.state)
            if st in GATES or st in (RunState.PUBLISHING, RunState.PUBLISHED,
                                     RunState.FAILED, RunState.REJECTED,
                                     RunState.APPROVED):
                continue
            log.warning("run %s ค้างที่ %s — ปลุกให้เดินต่อ", run.id, run.state)
            await ctx["redis"].enqueue_job("advance_run", run.id)


async def publish_run(ctx, run_id: str, target_id: str, privacy: str = "private"):
    """โพสต์ขึ้นแพลตฟอร์ม — idempotent ต่อ (run, target, เนื้อไฟล์)

    privacy มาจากคนกด ไม่ใช่เดาจาก target.audited: ดีฟอลต์ private เสมอ
    publisher แต่ละตัวลดระดับเองถ้าแพลตฟอร์มไม่อนุญาต (TikTok ยังไม่ audit → SELF_ONLY)
    """
    from datetime import datetime, timezone
    from pathlib import Path
    from sqlalchemy import select
    from .models import PublishAttempt, PublishTarget, Run
    from .publishers import (
        PublishError, PublishRequest, get_publisher, idempotency_key,
    )
    from .schemas import PostMeta, RunState

    async with Session() as session:
        run = (await session.execute(select(Run).where(Run.id == run_id))).scalar_one()
        target = (await session.execute(
            select(PublishTarget).where(PublishTarget.id == target_id))).scalar_one()

        video = Path(run.final_asset_id or run.animatic_asset_id)
        key = idempotency_key(run_id, target_id, video)

        existing = (await session.execute(select(PublishAttempt).where(
            PublishAttempt.idempotency_key == key))).scalar_one_or_none()
        if existing and existing.status == "published":
            log.info("run %s โพสต์ไปแล้วที่ %s — ข้าม", run_id, target.platform)
            return existing.external_post_id

        attempt = existing or PublishAttempt(
            run_id=run_id, target_id=target_id, idempotency_key=key, status="pending")
        attempt.error = None
        session.add(attempt)
        prev_state = run.state
        run.state = RunState.PUBLISHING.value   # เขียนก่อนยิง ไม่ใช่หลัง
        await session.flush()
        await session.commit()   # ให้ UI เห็น pending/publishing ระหว่างอัปโหลด

        meta = PostMeta.model_validate(run.post_meta or {})
        cfg = dict(target.config or {})
        pub = get_publisher(target.platform, target.access_token_enc,
                            client_id=settings.google_client_id,
                            client_secret=settings.google_client_secret,
                            client_key=settings.tiktok_client_key,
                            client_secret_tiktok=settings.tiktok_client_secret,
                            audited=target.audited, handle=target.account_handle,
                            **cfg)
        shots = run.shots or []
        duration = float(shots[-1]["end_s"]) if shots else 0.0
        req = PublishRequest(
            video=video, title=meta.title_yt, caption=meta.caption_th,
            hashtags=meta.hashtags, privacy=privacy, ai_generated=True,
        )
        try:
            pub.precheck(req, duration)
            res = await pub.publish(req)
        except PublishError as e:
            attempt.status = "failed" if not e.retryable else "retry"
            attempt.error = f"[{e.kind}] {e}"
            # โพสต์ไม่สำเร็จไม่ทำให้ run เสีย — กลับไป state เดิม (approved/published)
            run.state = prev_state if prev_state != RunState.PUBLISHING.value \
                else RunState.APPROVED.value
            await session.commit()
            log.error("โพสต์ล้มเหลว run=%s %s: %s", run_id, target.platform, e)
            raise

        attempt.status = "published"
        attempt.external_post_id = res.external_id
        attempt.url = res.url
        attempt.posted_at = datetime.now(timezone.utc)
        run.state = RunState.PUBLISHED.value
        await session.commit()
        return res.external_id


class WorkerSettings:
    functions = [advance_run, approve_run, resume_run, regenerate_shots, publish_run]
    # ทุก 10 นาที ปลุก run ที่ค้าง — เดิมประกาศ sweep ไว้แต่ลืมผูก cron
    cron_jobs = [cron(sweep_stuck_runs, minute={0, 10, 20, 30, 40, 50}, run_at_startup=True)]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 8
    job_timeout = 1800
