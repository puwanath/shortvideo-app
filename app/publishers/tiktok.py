"""TikTok — Content Posting API "Direct Post" (ตรวจกับเอกสารแล้ว)

ลำดับ:
  1. POST /v2/post/publish/video/init/  body {post_info, source_info}
     → publish_id + upload_url (ใช้ได้ 1 ชั่วโมง)
  2. PUT upload_url เป็นก้อน header Content-Range: bytes a-b/total
  3. POST /v2/post/publish/status/fetch/ {publish_id} จน PUBLISH_COMPLETE
     (สถานะ: PROCESSING_UPLOAD → PROCESSING_DOWNLOAD/… → PUBLISH_COMPLETE | FAILED)

สิ่งที่ทำให้พังบ่อย:
  * privacy_level **บังคับ** ไม่มีค่าดีฟอลต์ — ไม่ใส่ได้ error ทันที
  * แอปที่ยังไม่ผ่าน audit ของ TikTok โพสต์ได้แค่ SELF_ONLY (เห็นคนเดียว)
    ขอ PUBLIC_TO_EVERYONE ไปจะโดนปฏิเสธ — จึงบังคับ private จนกว่า target.audited
  * ก้อนอัปโหลด: ไฟล์ ≤ 64MB ส่งก้อนเดียว (chunk_size = ขนาดไฟล์)
    ใหญ่กว่านั้นแบ่งก้อน 5–64MB และก้อนสุดท้ายรับเศษ
  * is_aigc=true คือช่องประกาศเนื้อหา AI ของ TikTok — ใส่เสมอเพราะทั้งคลิปสร้างจาก AI
  * scope ที่ต้องมี: video.publish (ผ่าน user authorization) token อายุ 24 ชม.
    refresh ด้วย client_key/secret ที่ /v2/oauth/token/
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from .base import (
    Publisher, PublishError, PublishErrorKind, PublishRequest, PublishResult,
)

log = logging.getLogger(__name__)

API = "https://open.tiktokapis.com/v2"
SINGLE_CHUNK_MAX = 64 * 1024 * 1024
CHUNK = 20 * 1024 * 1024


class TikTokPublisher(Publisher):
    platform = "tiktok"
    max_duration_s = 600.0
    requires_audit = True

    def __init__(self, access_token: str, *, client_key: str = "",
                 client_secret: str = "", audited: bool = False, **cfg):
        super().__init__(access_token, **cfg)
        self.client_key = client_key
        self.client_secret = client_secret
        self.audited = audited

    @staticmethod
    def _classify(status: int, body: str) -> PublishError:
        low = body.lower()
        if status == 401 or "access_token_invalid" in low or "scope_not_authorized" in low:
            return PublishError(f"token ใช้ไม่ได้: {body[:200]}", PublishErrorKind.AUTH)
        if status == 429 or "rate_limit_exceeded" in low or "spam_risk" in low:
            return PublishError("โดน rate limit / spam guard ของ TikTok",
                                PublishErrorKind.RATE_LIMIT, retry_after_s=600)
        if "unaudited_client" in low or "privacy_level_option_mismatch" in low:
            return PublishError(f"แอปยังไม่ผ่าน audit — โพสต์ได้แค่ SELF_ONLY: {body[:200]}",
                                PublishErrorKind.NOT_AUDITED)
        if status == 400 or "invalid_param" in low:
            return PublishError(f"คำขอผิด ห้าม retry: {body[:300]}", PublishErrorKind.INVALID)
        if status >= 500 or "internal_error" in low:
            return PublishError(f"TikTok {status}", PublishErrorKind.TRANSIENT,
                                retry_after_s=30)
        return PublishError(f"TikTok {status}: {body[:300]}", PublishErrorKind.INVALID)

    def _privacy(self, req: PublishRequest) -> str:
        if req.privacy == "public" and self.audited:
            return "PUBLIC_TO_EVERYONE"
        if req.privacy == "public":
            log.warning("TikTok target ยังไม่ audited — ลดเป็น SELF_ONLY แทน public")
        return "SELF_ONLY"

    async def publish(self, req: PublishRequest) -> PublishResult:
        size = req.video.stat().st_size
        if size <= SINGLE_CHUNK_MAX:
            chunk_size, n_chunks = size, 1
        else:
            chunk_size, n_chunks = CHUNK, size // CHUNK   # ก้อนสุดท้ายรับเศษ

        title = (req.title + "\n" + " ".join(req.hashtags)).strip()[:2200]
        body = {
            "post_info": {
                "title": title,
                "privacy_level": self._privacy(req),
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
                "video_cover_timestamp_ms": 1000,
                "is_aigc": bool(req.ai_generated),
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": size,
                "chunk_size": chunk_size,
                "total_chunk_count": n_chunks,
            },
        }
        headers = {"Authorization": f"Bearer {self.access_token}",
                   "Content-Type": "application/json; charset=UTF-8"}

        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{API}/post/publish/video/init/", json=body, headers=headers)
            data = r.json() if r.content else {}
            err = (data.get("error") or {})
            if r.status_code >= 400 or err.get("code", "ok") != "ok":
                raise self._classify(r.status_code, r.text)
            publish_id = data["data"]["publish_id"]
            upload_url = data["data"]["upload_url"]

            await self._upload(c, upload_url, req, size, chunk_size, n_chunks)
            status = await self._wait(c, publish_id, headers)

        post_ids = status.get("publicaly_available_post_id") or []
        log.info("โพสต์ TikTok สำเร็จ publish_id=%s", publish_id)
        return PublishResult(
            external_id=publish_id,
            url=(f"https://www.tiktok.com/@{self.cfg.get('handle', '')}/video/{post_ids[0]}"
                 if post_ids and self.cfg.get("handle") else None),
            status="published" if body["post_info"]["privacy_level"] != "SELF_ONLY"
            else "private",
            raw=status,
        )

    async def _upload(self, c: httpx.AsyncClient, url: str, req: PublishRequest,
                      size: int, chunk_size: int, n_chunks: int) -> None:
        with req.video.open("rb") as f:
            sent = 0
            for i in range(n_chunks):
                last_chunk = i == n_chunks - 1
                want = size - sent if last_chunk else chunk_size
                chunk = f.read(want)
                end = sent + len(chunk) - 1
                r = await c.put(url, content=chunk, headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {sent}-{end}/{size}",
                }, timeout=600)
                # 206 = รับก้อนแล้วรอก้อนต่อไป, 201 = ครบ
                if r.status_code not in (200, 201, 206):
                    raise self._classify(r.status_code, r.text)
                sent += len(chunk)

    async def _wait(self, c: httpx.AsyncClient, publish_id: str, headers: dict,
                    max_wait_s: float = 600) -> dict:
        waited = 0.0
        while waited < max_wait_s:
            r = await c.post(f"{API}/post/publish/status/fetch/",
                             json={"publish_id": publish_id}, headers=headers)
            data = r.json() if r.content else {}
            if r.status_code >= 400:
                raise self._classify(r.status_code, r.text)
            st = data.get("data") or {}
            status = st.get("status")
            if status == "PUBLISH_COMPLETE":
                return st
            if status == "FAILED":
                raise PublishError(f"TikTok ปฏิเสธ: {st.get('fail_reason')}",
                                   PublishErrorKind.INVALID)
            await asyncio.sleep(5)
            waited += 5
        raise PublishError(f"TikTok ยังไม่ publish หลัง {max_wait_s:.0f}s ({publish_id})",
                           PublishErrorKind.TRANSIENT)

    async def refresh_token(self, refresh_token: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{API}/oauth/token/", data={
                "client_key": self.client_key,
                "client_secret": self.client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }, headers={"Content-Type": "application/x-www-form-urlencoded"})
        if r.status_code >= 400 or "access_token" not in r.text:
            raise PublishError(f"refresh token ล้มเหลว: {r.text[:200]}", PublishErrorKind.AUTH)
        return r.json()
