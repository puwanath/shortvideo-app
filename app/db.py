"""engine + session factory

แยกออกมาจาก main.py เพราะทั้ง API และ worker ต้องใช้ ถ้าปล่อยไว้ใน main
worker จะต้อง import main ส่วน main ก็ต้อง import worker เพื่อเอา enqueue
กลายเป็นวงกลมที่ import ไม่ผ่านทั้งคู่
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, pool_pre_ping=True)
Session = async_sessionmaker(engine, expire_on_commit=False)


async def get_session():
    async with Session() as s:
        yield s
