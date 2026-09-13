"""สร้างภาพด้วย GPU ในเครื่อง — ฟรี ไม่มีโควตา ไม่ส่งข้อมูลออก

คุยกับ service เล็ก ๆ ที่รันข้าง ๆ GPU (ดู services/imagegen/) ผ่านสัญญา HTTP
ที่ตั้งใจให้บางที่สุด เพื่อให้สลับโมเดลข้างหลังได้โดยไม่ต้องแตะ pipeline

เรื่องที่ต้องรู้ก่อนใช้
----------------------
`run_keyframes` ส่ง `reference_images` = [ภาพอ้างอิงตัวละครที่อยู่ใน shot] เข้ามา
นั่นคือกลไก**เดียว**ที่คุมความต่อเนื่องของหน้าตาทั้งเรื่อง (prompt อย่างเดียวเอาไม่อยู่
— เขียนไว้ใน docstring ของ run_character_sheets)

โมเดล text-to-image ล้วน ๆ อย่าง Z-Image-Turbo **ใช้ reference ไม่ได้**
adapter จึงส่งไปให้ service ตัดสินใจเอง แล้วอ่านธงจาก /health มาเตือน
ครั้งเดียวตอนเริ่ม ไม่ใช่ปล่อยให้ความต่อเนื่องเพี้ยนแบบเงียบ ๆ
จนกว่าจะเปลี่ยนไปใช้โมเดลที่รับ reference ได้ ให้พึ่ง seed + style_suffix แทน
"""
from __future__ import annotations

import base64
import logging

import httpx

from .base import BaseProvider, ImageRequest, ImageResponse, ProviderError

log = logging.getLogger(__name__)


def _wh(req: ImageRequest) -> tuple[int, int]:
    """แปลง size เป็น (w, h) แล้วปัดลงเป็นพหุคูณของ 64

    โมเดล diffusion ส่วนใหญ่ต้องการขนาดที่หารด้วย 8/16/64 ลงตัว
    ถ้าส่ง 1080x1920 ตรง ๆ บางตัวจะ error บางตัวจะปัดเองเงียบ ๆ
    ปัดที่นี่ทีเดียวจะได้รู้แน่ว่าได้ขนาดอะไร (ffmpeg ขยายเป็น 1080x1920 ให้อยู่แล้ว)
    """
    try:
        w, h = (int(x) for x in req.size.lower().split("x"))
    except (ValueError, AttributeError):
        w, h = 768, 1344
    snap = lambda v: max(256, (v // 64) * 64)  # noqa: E731
    return snap(w), snap(h)


class LocalImageProvider(BaseProvider):
    """service สร้างภาพในเครื่อง — สัญญา: POST /generate, GET /health"""

    name = "image_local"

    def __init__(self, base_url: str, timeout: float = 600.0, **cfg):
        super().__init__(**cfg)
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._warned_no_ref = False

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    async def generate_image(self, req: ImageRequest) -> ImageResponse:
        w, h = _wh(req)
        payload: dict = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": w,
            "height": h,
            "seed": req.seed,
            "n": max(1, req.n),
        }
        # ส่ง steps/guidance ต่อได้จาก routing profile params ผ่าน extra
        for k in ("steps", "guidance", "sampler"):
            if k in (req.extra or {}):
                payload[k] = req.extra[k]
        if req.reference_images:
            payload["reference_images"] = [
                base64.b64encode(b).decode() for b in req.reference_images
            ]

        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self.base_url}/generate", json=payload)

        if r.status_code >= 400:
            raise ProviderError(
                f"image service {r.status_code}: {r.text[:300]}",
                retryable=r.status_code >= 500 or r.status_code == 429,
            )

        body = r.json()
        imgs = body.get("images") or []
        if not imgs:
            raise ProviderError("image service ไม่คืนรูปกลับมา",
                                retryable=False, code="no_image")

        if req.reference_images and not body.get("used_reference") and not self._warned_no_ref:
            self._warned_no_ref = True
            log.warning(
                "โมเดล %s ไม่รองรับ reference image — ส่ง style anchor ไปแล้วแต่ถูกทิ้ง "
                "ความต่อเนื่องของสไตล์จะพึ่ง seed กับ style_suffix เท่านั้น",
                body.get("model", "?"))

        return ImageResponse(
            images=[base64.b64decode(x) for x in imgs],
            cost_usd=0.0,                      # GPU ตัวเอง ไม่คิดเงินเข้า budget
            seed=body.get("seed", req.seed),
            model=body.get("model", req.model),
        )
