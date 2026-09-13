"""ส่งงานเข้าคิว arq

แยกจาก worker.py เพื่อให้ API import ฟังก์ชันนี้ได้โดยไม่ต้องลากตัว worker
ทั้งก้อน (ซึ่ง import main กลับมา) เข้ามาด้วย
"""
from __future__ import annotations

from arq.connections import RedisSettings, create_pool

from .config import get_settings

settings = get_settings()


async def enqueue(name: str, *args):
    """งานไปอยู่ที่ worker ไม่ใช่ในโปรเซส API
    ถ้า API restart กลางทาง งานยังอยู่ในคิว"""
    pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    try:
        return await pool.enqueue_job(name, *args)
    finally:
        await pool.close()
