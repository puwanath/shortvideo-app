"""ตั้งค่าจาก environment — ไม่มีค่า secret ฝังในโค้ด"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "shortvideo"
    public_url: str = "http://localhost:8000"
    debug: bool = False

    database_url: str = "postgresql+asyncpg://sv:sv@localhost:5432/shortvideo"
    redis_url: str = "redis://localhost:6379/0"

    # providers
    openrouter_api_key: str = ""
    vllm_base_url: str = "http://localhost:8001/v1"
    whisper_base_url: str = "http://localhost:8002"
    # JaiTTS (F5-TTS) เสียงไทยที่รันเอง
    tts_base_url: str = "http://localhost:8100"
    # service สร้างภาพบน GPU ในเครื่อง — ดู services/imagegen/
    image_base_url: str = "http://localhost:8102"

    # storage
    storage_root: Path = Path("./var/runs")
    s3_endpoint: str = ""
    s3_bucket: str = "shortvideo"
    s3_access_key: str = ""
    s3_secret_key: str = ""

    fonts_dir: Path = Path("./assets/fonts")

    # publishing
    google_client_id: str = ""
    google_client_secret: str = ""
    # TikTok Content Posting API — ใช้ refresh token (token อายุ 24 ชม.)
    tiktok_client_key: str = ""
    tiktok_client_secret: str = ""

    # กันเงินไหล — เพดานระดับระบบ ไม่ใช่แค่ระดับ task
    daily_budget_usd: float = 20.0
    default_task_budget_usd: float = 5.0
    external_providers_enabled: bool = True   # kill switch ตัวเดียวปิดของนอกทั้งหมด

    render_concurrency: int = 2
    # NVENC ใช้ VRAM ~300MB ต่อ session — ถ้า vLLM กิน VRAM เต็มอยู่ ให้ปิดไว้
    use_nvenc: bool = False
    video_poll_interval_s: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
