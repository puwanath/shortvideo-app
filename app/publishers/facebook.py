"""Facebook Reels บน Page — Graph API v25 (ตรวจกับเอกสาร Reels Publishing แล้ว)

สามขั้นเสมอ ห้ามข้าม:
  1. POST /{page_id}/video_reels  upload_phase=start        → video_id + upload_url
  2. POST rupload.facebook.com/video-upload/v25.0/{video_id} ส่งไฟล์ทั้งก้อน
     header: Authorization: OAuth {token}, offset: 0, file_size: N
     (resume ได้ด้วย offset = bytes_transfered ที่อ่านจาก status)
  3. POST /{page_id}/video_reels  upload_phase=finish  video_state=PUBLISHED|DRAFT
     + description (ใส่ hashtag ในนี้ได้เลย)

หลัง finish วิดีโอยัง "processing" อยู่ — ต้อง poll GET /{video_id}?fields=status
จนกว่า video_status=ready ไม่งั้นจะได้ post ที่ยังไม่ขึ้นจริง

สเปก Reels: 9:16, ≥540×960 (แนะนำ 1080×1920), 24–60fps, **3–90 วินาที**,
H.264 + AAC 48kHz stereo — final.mp4 ของเราตรงทุกข้อ

token ต้องเป็น *Page* access token ที่มี pages_manage_posts +
pages_read_engagement + pages_show_list และคนที่ authorize ต้องมีสิทธิ์
CREATE_CONTENT บน Page นั้น Page token แบบ long-lived ไม่หมดอายุ จึงไม่มี refresh
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from .base import (
    Publisher, PublishError, PublishErrorKind, PublishRequest, PublishResult,
)

log = logging.getLogger(__name__)

API = "https://graph.facebook.com/v25.0"
RUPLOAD = "https://rupload.facebook.com/video-upload/v25.0"


class FacebookReelsPublisher(Publisher):
    platform = "facebook"
    max_duration_s = 90.0
    min_duration_s = 3.0
    requires_audit = False

    def __init__(self, access_token: str, *, page_id: str = "", **cfg):
        super().__init__(access_token, **cfg)
        if not page_id:
            raise PublishError("target ของ facebook ต้องมี config.page_id",
                               PublishErrorKind.INVALID)
        self.page_id = page_id

    @staticmethod
    def _classify(status: int, body: str) -> PublishError:
        low = body.lower()
        if status in (401, 403) or "oauthexception" in low or "access token" in low:
            return PublishError(f"token ใช้ไม่ได้: {body[:200]}", PublishErrorKind.AUTH)
        if status == 429 or "rate limit" in low or '"code":4' in body or '"code":32' in body:
            return PublishError("โดน rate limit ของ Graph API", PublishErrorKind.RATE_LIMIT,
                                retry_after_s=300)
        if status == 400:
            return PublishError(f"คำขอผิด ห้าม retry: {body[:300]}", PublishErrorKind.INVALID)
        if status >= 500:
            return PublishError(f"Facebook {status}", PublishErrorKind.TRANSIENT,
                                retry_after_s=30)
        return PublishError(f"Facebook {status}: {body[:300]}", PublishErrorKind.INVALID)

    def precheck(self, req: PublishRequest, duration_s: float) -> None:
        super().precheck(req, duration_s)
        if duration_s < self.min_duration_s:
            raise PublishError(f"Reels ต้องยาวอย่างน้อย {self.min_duration_s:.0f}s",
                               PublishErrorKind.INVALID)

    async def publish(self, req: PublishRequest) -> PublishResult:
        size = req.video.stat().st_size
        description = (req.caption + "\n\n" + " ".join(req.hashtags)).strip()[:2200]
        # Reels ไม่มี unlisted — private ของเราแปลว่าเก็บเป็น DRAFT ให้คนกดโพสต์เองใน Studio
        state = "PUBLISHED" if req.privacy == "public" else "DRAFT"

        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{API}/{self.page_id}/video_reels",
                             data={"upload_phase": "start",
                                   "access_token": self.access_token})
            if r.status_code >= 400:
                raise self._classify(r.status_code, r.text)
            start = r.json()
            video_id = start.get("video_id")
            if not video_id:
                raise PublishError(f"start ไม่ได้ video_id: {r.text[:200]}",
                                   PublishErrorKind.TRANSIENT)

            # ขั้น 2: ไฟล์ทั้งก้อน — rupload รับ body ดิบ ไม่ใช่ multipart
            with req.video.open("rb") as f:
                up = await c.post(
                    f"{RUPLOAD}/{video_id}",
                    content=f,
                    headers={
                        "Authorization": f"OAuth {self.access_token}",
                        "offset": "0",
                        "file_size": str(size),
                        "Content-Type": "application/octet-stream",
                    },
                    timeout=600,
                )
            if up.status_code >= 400 or not up.json().get("success"):
                raise self._classify(up.status_code, up.text)

            fin = await c.post(f"{API}/{self.page_id}/video_reels", data={
                "upload_phase": "finish",
                "video_id": video_id,
                "video_state": state,
                "description": description,
                "title": req.title[:255],
                "access_token": self.access_token,
            })
            if fin.status_code >= 400:
                raise self._classify(fin.status_code, fin.text)
            post_id = fin.json().get("post_id")

            status = await self._wait_ready(c, video_id)

        log.info("โพสต์ Facebook Reels สำเร็จ video_id=%s state=%s", video_id, state)
        return PublishResult(
            external_id=str(video_id),
            url=f"https://www.facebook.com/reel/{video_id}",
            status="published" if state == "PUBLISHED" else "draft",
            raw={"post_id": post_id, "status": status},
        )

    async def _wait_ready(self, c: httpx.AsyncClient, video_id: str,
                          max_wait_s: float = 600) -> dict:
        """finish คืน success ทันทีทั้งที่ยัง encode อยู่ — ต้องรอ ready จริง"""
        waited = 0.0
        while waited < max_wait_s:
            r = await c.get(f"{API}/{video_id}",
                            params={"fields": "status", "access_token": self.access_token})
            if r.status_code >= 400:
                raise self._classify(r.status_code, r.text)
            st = r.json().get("status") or {}
            vs = st.get("video_status")
            if vs == "ready":
                return st
            if vs in ("error", "expired", "upload_failed"):
                err = (st.get("processing_phase") or {}).get("error") or st
                raise PublishError(f"Facebook ประมวลผลไม่ผ่าน: {err}",
                                   PublishErrorKind.INVALID)
            await asyncio.sleep(5)
            waited += 5
        raise PublishError(f"Facebook ยัง processing อยู่หลัง {max_wait_s:.0f}s "
                           f"(video_id {video_id})", PublishErrorKind.TRANSIENT)

    async def refresh_token(self, refresh_token: str) -> dict:
        # Page token แบบ long-lived ไม่หมดอายุ — ถ้าหมด (คนเปลี่ยนรหัส/ถอนสิทธิ์)
        # ต้อง authorize ใหม่ทั้งรอบ ไม่มีทาง refresh อัตโนมัติ
        raise PublishError("Page token ของ Facebook refresh ไม่ได้ ต้อง authorize ใหม่",
                           PublishErrorKind.AUTH)
