"""YouTube Shorts — แพลตฟอร์มที่ควรทำก่อน เพราะไม่ต้องรอ audit แบบ TikTok

สิ่งที่ต้องรู้:
  * อัปโหลดผ่าน videos.insert แบบ resumable — ไฟล์วิดีโอไม่ควรส่งแบบ single request
  * จะถูกจัดเป็น Shorts อัตโนมัติเมื่อ 9:16 และยาวไม่เกิน 60 วินาที
    ไม่ต้องใส่ #Shorts ก็ได้ แต่ใส่ไว้ก็ไม่เสียหาย
  * ตั้งแต่กลางปี 2026 upload อยู่ใน quota bucket แยก คิด 1 unit ต่อครั้ง
    เพดานราว 100 คลิป/วัน ไม่กิน quota 10,000 รวมอีกแล้ว
    (บทความเก่าที่บอก 1,600 units = ~6 คลิป/วัน ล้าสมัย)
  * OAuth client ต้องผ่าน verification ของ Google ถึงจะโพสต์ public ได้
    ระหว่างที่ยังเป็น testing mode: token หมดอายุใน 7 วัน และจำกัด 100 test users
  * ประกาศเนื้อหาที่สร้างด้วย AI ผ่าน status.containsSyntheticMedia (มีใน API
    ตั้งแต่ 2024-10-30) — ไปโผล่ในช่อง "How this content was made" ใต้คลิป
    ไม่ใช่ selfDeclaredMadeForKids ซึ่งคนละเรื่อง
  * Shorts รับได้ถึง 3 นาทีแล้ว (ตั้งแต่ ต.ค. 2024) ไม่ใช่ 60 วินาที
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from .base import (
    Publisher, PublishError, PublishErrorKind, PublishRequest, PublishResult,
)

log = logging.getLogger(__name__)

UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
TOKEN_URL = "https://oauth2.googleapis.com/token"
CHUNK = 8 * 1024 * 1024


class YouTubePublisher(Publisher):
    platform = "youtube"
    max_duration_s = 180.0
    requires_audit = False

    def __init__(self, access_token: str, *, client_id: str = "",
                 client_secret: str = "", **cfg):
        super().__init__(access_token, **cfg)
        self.client_id = client_id
        self.client_secret = client_secret

    # ------------------------------------------------------------------

    @staticmethod
    def _classify(status: int, body: str) -> PublishError:
        low = body.lower()
        if status in (401, 403) and ("credential" in low or "unauthorized" in low
                                     or "authError" in body):
            return PublishError(f"token ใช้ไม่ได้: {body[:200]}", PublishErrorKind.AUTH)
        if status == 403 and "quota" in low:
            return PublishError(f"quota เต็ม: {body[:200]}", PublishErrorKind.QUOTA,
                                retry_after_s=3600)
        if status == 429:
            return PublishError("โดน rate limit", PublishErrorKind.RATE_LIMIT,
                                retry_after_s=60)
        if status == 400:
            return PublishError(f"คำขอผิด ห้าม retry: {body[:300]}",
                                PublishErrorKind.INVALID)
        if status >= 500:
            return PublishError(f"YouTube {status}", PublishErrorKind.TRANSIENT,
                                retry_after_s=30)
        return PublishError(f"YouTube {status}: {body[:300]}", PublishErrorKind.INVALID)

    def _metadata(self, req: PublishRequest) -> dict:
        tags = [h.lstrip("#") for h in req.hashtags][:15]
        title = req.title if "#Shorts" in req.title else f"{req.title} #Shorts"
        body = {
            "snippet": {
                "title": title[:100],
                "description": (req.caption + "\n\n" + " ".join(req.hashtags))[:5000],
                "tags": tags,
                "categoryId": "22",
            },
            "status": {
                "privacyStatus": req.privacy,       # private | unlisted | public
                "selfDeclaredMadeForKids": False,
                "madeForKids": False,
                "containsSyntheticMedia": bool(req.ai_generated),
            },
        }
        if req.scheduled_at:
            # ตั้งเวลาโพสต์ได้เฉพาะเมื่อ privacyStatus=private
            body["status"]["privacyStatus"] = "private"
            body["status"]["publishAt"] = req.scheduled_at
        return body

    # ------------------------------------------------------------------

    async def publish(self, req: PublishRequest) -> PublishResult:
        size = req.video.stat().st_size
        meta = self._metadata(req)

        async with httpx.AsyncClient(timeout=120) as c:
            # ขั้น 1: เปิด resumable session — ได้ URL สำหรับส่งไฟล์
            init = await c.post(
                UPLOAD_URL,
                params={"uploadType": "resumable",
                        "part": "snippet,status,contentDetails"},
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json; charset=UTF-8",
                    "X-Upload-Content-Length": str(size),
                    "X-Upload-Content-Type": "video/mp4",
                },
                json=meta,
            )
            if init.status_code >= 400:
                raise self._classify(init.status_code, init.text)

            session_url = init.headers.get("location") or init.headers.get("Location")
            if not session_url:
                raise PublishError("ไม่ได้ resumable session URL",
                                   PublishErrorKind.TRANSIENT)

            # ขั้น 2: ส่งไฟล์เป็นก้อน — resume ได้ถ้าขาดกลางทาง
            result = await self._upload_chunks(c, session_url, req.video, size)

        vid = result.get("id")
        if not vid:
            raise PublishError(f"อัปโหลดสำเร็จแต่ไม่ได้ video id: {json.dumps(result)[:200]}",
                               PublishErrorKind.TRANSIENT)

        log.info("โพสต์ YouTube สำเร็จ id=%s privacy=%s", vid, req.privacy)
        return PublishResult(
            external_id=vid,
            url=f"https://youtube.com/shorts/{vid}",
            status=result.get("status", {}).get("privacyStatus", "unknown"),
            raw=result,
        )

    async def _upload_chunks(self, c: httpx.AsyncClient, session_url: str,
                             path: Path, size: int) -> dict:
        sent = 0
        with path.open("rb") as f:
            while sent < size:
                chunk = f.read(CHUNK)
                if not chunk:
                    break
                last = sent + len(chunk) - 1
                r = await c.put(
                    session_url,
                    content=chunk,
                    headers={
                        "Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {sent}-{last}/{size}",
                    },
                )
                if r.status_code in (200, 201):
                    return r.json()
                if r.status_code == 308:
                    # ยังไม่จบ — YouTube บอกมาว่ารับไปถึงไบต์ไหนแล้ว
                    rng = r.headers.get("range")
                    if rng and "-" in rng:
                        sent = int(rng.rsplit("-", 1)[1]) + 1
                        f.seek(sent)
                    else:
                        sent = last + 1
                    continue
                raise self._classify(r.status_code, r.text)
        raise PublishError("ส่งไฟล์ครบแล้วแต่ไม่ได้ response สุดท้าย",
                           PublishErrorKind.TRANSIENT)

    async def refresh_token(self, refresh_token: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(TOKEN_URL, data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            })
        if r.status_code >= 400:
            raise PublishError(f"refresh token ล้มเหลว: {r.text[:200]}",
                               PublishErrorKind.AUTH)
        return r.json()
