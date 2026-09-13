"""ตาราง — Postgres เป็น source of truth ตัวเดียวของระบบ

จุดสำคัญ: ตาราง generation เก็บ params + seed ครบพอที่จะสร้างซ้ำได้
ถ้าไม่เก็บ พอบิลมาจะไม่รู้ว่าเงินหายไปกับ run ไหน และ regen จะได้ผลไม่เหมือนเดิม
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class BrandKit(Base, TimestampMixin):
    __tablename__ = "brand_kit"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120))
    tone_of_voice: Mapped[str] = mapped_column(Text, default="")
    style_suffix: Mapped[str] = mapped_column(Text, default="")
    palette: Mapped[dict] = mapped_column(JSON, default=dict)
    font_family: Mapped[str] = mapped_column(String(120), default="Noto Sans Thai")
    do_donts: Mapped[str] = mapped_column(Text, default="")
    subject_ref_asset_ids: Mapped[list] = mapped_column(JSON, default=list)
    # override การอ่านคำเฉพาะของ TTS — ต้องมีตั้งแต่ schema แรก ไม่ใช่แปะทีหลัง
    pronunciation: Mapped[dict] = mapped_column(JSON, default=dict)


class RoutingProfile(Base, TimestampMixin):
    """ผูกกับหน้าตั้งค่าเส้นทางโมเดล — JSON ที่ export จาก UI ลงตรงนี้"""
    __tablename__ = "routing_profile"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    stages: Mapped[dict] = mapped_column(JSON, default=dict)
    animatic_only: Mapped[bool] = mapped_column(Boolean, default=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)


class Task(Base, TimestampMixin):
    __tablename__ = "task"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    brief: Mapped[str] = mapped_column(Text)
    brand_kit_id: Mapped[str | None] = mapped_column(ForeignKey("brand_kit.id"))
    routing_profile_id: Mapped[str | None] = mapped_column(ForeignKey("routing_profile.id"))
    target_platforms: Mapped[list] = mapped_column(JSON, default=list)
    target_duration_s: Mapped[int] = mapped_column(Integer, default=30)
    budget_cap_usd: Mapped[float] = mapped_column(Float, default=5.0)
    created_by: Mapped[str] = mapped_column(String(120), default="system")
    # รูปแบบงาน (content | cartoon) และ preset สไตล์ภาพ — ดู agents/formats.py
    # (DB เก่า: ALTER TABLE task ADD COLUMN format varchar(20) DEFAULT 'content',
    #  ADD COLUMN style_preset varchar(30))
    format: Mapped[str] = mapped_column(String(20), default="content")
    style_preset: Mapped[str | None] = mapped_column(String(30))
    # scene = keyframe ทุก shot + i2v (Veo) | avatar = ภาพตัวละคร + บทพูด (HeyGen Avatar IV)
    # (DB เก่า: ALTER TABLE task ADD COLUMN render varchar(20) DEFAULT 'scene',
    #  ADD COLUMN product_refs json)
    render: Mapped[str] = mapped_column(String(20), default="scene")
    # ภาพสินค้าที่แนบมาตอนสร้าง task — path ใต้ STORAGE_ROOT/tasks/<id>/
    product_refs: Mapped[list] = mapped_column(JSON, default=list)

    runs: Mapped[list["Run"]] = relationship(back_populates="task")


class Run(Base, TimestampMixin):
    __tablename__ = "run"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(ForeignKey("task.id"), index=True)
    attempt_no: Mapped[int] = mapped_column(Integer, default=1)
    state: Mapped[str] = mapped_column(String(40), default="draft", index=True)
    state_entered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    error: Mapped[str | None] = mapped_column(Text)
    # state ที่ตายตอน failed — ให้ /resume พากลับไปเดินต่อจากตรงนั้นได้
    # (DB เก่า: ALTER TABLE run ADD COLUMN failed_state varchar(40))
    failed_state: Mapped[str | None] = mapped_column(String(40))

    # payload ของแต่ละ stage เก็บเป็น JSON ตาม schema ใน schemas.py
    concept: Mapped[dict | None] = mapped_column(JSON)
    story: Mapped[dict | None] = mapped_column(JSON)
    shots: Mapped[list | None] = mapped_column(JSON)
    qc_report: Mapped[dict | None] = mapped_column(JSON)
    post_meta: Mapped[dict | None] = mapped_column(JSON)

    # ภาพอ้างอิงตัวละคร [{name, role, ref_path}] — ตัวล็อกหน้าตาระดับ run
    # ไม่ใช่ระดับ shot เพื่อให้ regen ทีหลังยังได้คนเดิม
    # (แทน style_anchor_asset_id / style_seed เดิม — คอลัมน์เก่ายังอยู่ใน DB
    # ที่สร้างก่อน 2026-09-13 แต่ไม่มีใครอ่านแล้ว; DB ใหม่ต้อง
    # ALTER TABLE run ADD COLUMN characters json)
    characters: Mapped[list | None] = mapped_column(JSON)

    # สองคอลัมน์นี้ชื่อ *_asset_id เพราะตั้งใจจะเก็บ Asset.id (uuid 36 ตัว)
    # แต่ตอนนี้ยังไม่ได้ใช้ตาราง Asset โค้ดจึงเก็บ path ของไฟล์ตรง ๆ ซึ่งยาวกว่า 36
    # (เช่น /data/runs/<uuid>/animatic.mp4 = 60 ตัว) ต้องกว้างพอจนกว่าจะย้ายไป Asset
    # Postgres ตัดไม่ได้แล้วโยน error ส่วน sqlite ไม่บังคับความยาว จึงไม่เจอตอนเทสต์
    animatic_asset_id: Mapped[str | None] = mapped_column(String(512))
    final_asset_id: Mapped[str | None] = mapped_column(String(512))

    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    paid_stages: Mapped[list] = mapped_column(JSON, default=list)

    task: Mapped[Task] = relationship(back_populates="runs")
    generations: Mapped[list["Generation"]] = relationship(back_populates="run")

    __table_args__ = (Index("ix_run_state_entered", "state", "state_entered_at"),)


class Asset(Base, TimestampMixin):
    """content-addressed — sha256 ซ้ำแปลว่าไฟล์เดียวกัน ไม่ต้องเก็บซ้ำ"""
    __tablename__ = "asset"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(20))  # image | audio | video | subtitle
    mime: Mapped[str] = mapped_column(String(60))
    bytes: Mapped[int] = mapped_column(BigInteger)
    storage_key: Mapped[str] = mapped_column(String(400))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    duration_s: Mapped[float | None] = mapped_column(Float)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (UniqueConstraint("sha256", "kind", name="uq_asset_sha_kind"),)


class Generation(Base, TimestampMixin):
    """หนึ่งแถวต่อหนึ่งครั้งที่เรียก provider — ใช้คิดต้นทุนย้อนหลังและทำซ้ำ"""
    __tablename__ = "generation"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("run.id"), index=True)
    shot_id: Mapped[str | None] = mapped_column(String(64))
    stage: Mapped[str] = mapped_column(String(40), index=True)
    provider: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(120))
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    seed: Mapped[int | None] = mapped_column(BigInteger)
    external_id: Mapped[str | None] = mapped_column(String(200))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="ok")
    error: Mapped[str | None] = mapped_column(Text)
    asset_id: Mapped[str | None] = mapped_column(String(36))

    run: Mapped[Run] = relationship(back_populates="generations")


class Review(Base, TimestampMixin):
    __tablename__ = "review"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("run.id"), index=True)
    gate: Mapped[str] = mapped_column(String(40))
    reviewer: Mapped[str] = mapped_column(String(120))
    verdict: Mapped[str] = mapped_column(String(20))  # approve | reject | approve_animatic
    reason_code: Mapped[str | None] = mapped_column(String(40))
    notes: Mapped[str] = mapped_column(Text, default="")


class PublishTarget(Base, TimestampMixin):
    __tablename__ = "publish_target"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    platform: Mapped[str] = mapped_column(String(20))  # youtube | facebook | tiktok
    account_handle: Mapped[str] = mapped_column(String(120))
    access_token_enc: Mapped[str] = mapped_column(Text)
    refresh_token_enc: Mapped[str | None] = mapped_column(Text)
    scopes: Mapped[list] = mapped_column(JSON, default=list)
    audited: Mapped[bool] = mapped_column(Boolean, default=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # ค่าเฉพาะแพลตฟอร์ม: facebook ต้องมี page_id — เพิ่ม 2026-09-13
    # (DB เก่า: ALTER TABLE publish_target ADD COLUMN config json)
    config: Mapped[dict] = mapped_column(JSON, default=dict)


class PublishAttempt(Base, TimestampMixin):
    __tablename__ = "publish_attempt"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("run.id"), index=True)
    target_id: Mapped[str] = mapped_column(ForeignKey("publish_target.id"))
    idempotency_key: Mapped[str] = mapped_column(String(120))
    external_post_id: Mapped[str | None] = mapped_column(String(200))
    # ลิงก์ที่คนกดดูได้ — เพิ่ม 2026-09-13 (DB เก่า: ALTER TABLE publish_attempt ADD COLUMN url varchar(400))
    url: Mapped[str | None] = mapped_column(String(400))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_publish_idem"),
    )


class Event(Base):
    """append-only — ใช้ debug และทำ timeline ใน UI"""
    __tablename__ = "event"
    # sqlite ไม่ autoincrement ให้ BigInteger PK — ทำให้เทสต์ offline บน sqlite
    # ล้มที่ event แรก ส่วน Postgres ยังได้ bigserial เหมือนเดิม
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"),
                                    primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    level: Mapped[str] = mapped_column(String(10), default="info")
    stage: Mapped[str] = mapped_column(String(40), default="")
    message: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
