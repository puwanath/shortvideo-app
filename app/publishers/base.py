"""ชั้นกลางของการโพสต์

สามอย่างที่ต้องมีเหมือนกันทุกแพลตฟอร์ม และเป็นสามอย่างที่คนมักลืม:

1. **Idempotency** — โพสต์ซ้ำกู้คืนไม่ได้ ถ้า network timeout ระหว่าง upload
   เราไม่รู้ว่าโพสต์ไปแล้วหรือยัง จึงต้องมี key ต่อ (run, target) และเช็ก
   ก่อนยิงเสมอ ไม่ใช่ retry ดื้อ ๆ

2. **Token refresh เชิงรุก** — refresh ตาม cron ก่อนหมดอายุ ไม่ใช่รอ 401
   แล้วค่อย refresh เพราะตอนได้ 401 คุณอาจอัปโหลดไฟล์ไปครึ่งทางแล้ว

3. **แยกชนิด error** — quota เต็ม (รอ reset), metadata ผิด (ห้าม retry),
   5xx (backoff) ถ้าเหมารวมเป็น retry หมด จะยิงซ้ำจนโดนแบน
"""
from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class PublishErrorKind(StrEnum):
    QUOTA = "quota"            # รอ reset แล้วลองใหม่พรุ่งนี้
    RATE_LIMIT = "rate_limit"  # backoff สั้น ๆ แล้วลองใหม่
    AUTH = "auth"              # ต้องให้คน re-authorize
    INVALID = "invalid"        # metadata/ไฟล์ผิด ห้าม retry
    NOT_AUDITED = "not_audited"  # แอปยังไม่ผ่าน audit ของแพลตฟอร์ม
    TRANSIENT = "transient"    # 5xx, network — retry ได้


class PublishError(RuntimeError):
    def __init__(self, msg: str, kind: PublishErrorKind, *, retry_after_s: int | None = None):
        super().__init__(msg)
        self.kind = kind
        self.retry_after_s = retry_after_s

    @property
    def retryable(self) -> bool:
        return self.kind in (PublishErrorKind.RATE_LIMIT,
                             PublishErrorKind.TRANSIENT,
                             PublishErrorKind.QUOTA)


@dataclass
class PublishRequest:
    video: Path
    title: str
    caption: str
    hashtags: list[str] = field(default_factory=list)
    cover: Path | None = None
    privacy: str = "private"        # ดีฟอลต์ private เสมอ — ต้องตั้งใจถึงจะ public
    scheduled_at: str | None = None
    # ทุกแพลตฟอร์มบังคับให้ประกาศว่าเป็นเนื้อหาที่สร้างด้วย AI
    ai_generated: bool = True


@dataclass
class PublishResult:
    external_id: str
    url: str | None = None
    status: str = "published"
    raw: dict = field(default_factory=dict)


def idempotency_key(run_id: str, target_id: str, video: Path) -> str:
    """ผูกกับเนื้อไฟล์ด้วย ถ้า render ใหม่แล้วไฟล์เปลี่ยน key จะเปลี่ยนตาม
    ทำให้โพสต์เวอร์ชันแก้แล้วได้ โดยยังกันการยิงซ้ำของไฟล์เดิม"""
    h = hashlib.sha256()
    h.update(run_id.encode())
    h.update(target_id.encode())
    with video.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:40]


class Publisher(abc.ABC):
    platform: str = "base"
    max_duration_s: float = 60.0
    requires_audit: bool = False

    def __init__(self, access_token: str, **cfg):
        self.access_token = access_token
        self.cfg = cfg

    def precheck(self, req: PublishRequest, duration_s: float) -> None:
        """ตรวจสิ่งที่รู้ได้ก่อนอัปโหลด — ประหยัดเวลาและ quota"""
        if not req.video.exists():
            raise PublishError(f"ไม่พบไฟล์ {req.video}", PublishErrorKind.INVALID)
        if duration_s > self.max_duration_s:
            raise PublishError(
                f"ยาว {duration_s:.1f}s เกิน {self.max_duration_s}s ของ {self.platform}",
                PublishErrorKind.INVALID)

    @abc.abstractmethod
    async def publish(self, req: PublishRequest) -> PublishResult: ...

    @abc.abstractmethod
    async def refresh_token(self, refresh_token: str) -> dict:
        """คืน {access_token, refresh_token?, expires_in}"""
