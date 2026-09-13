"""ตัวเลือกเส้นทาง — อ่าน routing profile (JSON จากหน้าตั้งค่า) แล้วเรียก provider ให้

ทำสามอย่าง:
  1. เลือก provider ตาม stage
  2. ไล่ fallback เมื่อ error ที่ retry ได้
  3. ตัดจบเมื่อเกินเพดานเงิน — hard fail ไม่ใช่แค่เตือน
"""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

from .base import ProviderError

log = logging.getLogger(__name__)
T = TypeVar("T")


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Route:
    provider: str
    model: str
    params: dict[str, Any]


class BudgetGuard:
    """นับเงินระดับ run — เกินแล้ว fail ทันที ไม่ให้ retry เผาต่อ"""

    def __init__(self, cap_usd: float, spent: float = 0.0):
        self.cap = cap_usd
        self.spent = spent
        self._lock = asyncio.Lock()

    async def charge(self, amount: float, *, what: str = "") -> None:
        async with self._lock:
            self.spent += amount
            if self.spent > self.cap:
                raise BudgetExceeded(
                    f"ใช้ไป ${self.spent:.3f} เกินเพดาน ${self.cap:.2f} ที่ {what}")

    def would_exceed(self, estimate: float) -> bool:
        return self.spent + estimate > self.cap


class Registry:
    def __init__(self, profile: dict, *, settings):
        self.profile = profile or {}
        self._providers: dict[str, Any] = {}
        self._settings = settings

    # -------------------------------------------------- instantiate lazily

    def provider(self, name: str):
        if name in self._providers:
            return self._providers[name]
        # import ตอนใช้จริง ไม่ใช่ตอน import module — adapter ทุกตัวลาก httpx มาด้วย
        # แต่สคริปต์ offline (demo_animatic / test_agent_loop) ไม่ได้แตะ provider จริงเลย
        s = self._settings
        if name == "openrouter":
            from .openrouter import OpenRouterProvider
            p = OpenRouterProvider(s.openrouter_api_key, referer=s.public_url,
                                   title=s.app_name)
        elif name == "vllm_local":
            from .local import VLLMProvider
            p = VLLMProvider(s.vllm_base_url)
        elif name == "whisper_local":
            from .local import ThaiWhisperProvider
            p = ThaiWhisperProvider(s.whisper_base_url)
        elif name == "tts_local":
            from .local import LocalTTSProvider
            p = LocalTTSProvider(s.tts_base_url)
        elif name == "image_local":
            from .image_local import LocalImageProvider
            p = LocalImageProvider(s.image_base_url)
        else:
            raise ProviderError(f"ไม่รู้จัก provider '{name}' — เพิ่ม adapter ก่อน",
                                retryable=False, code="unknown_provider")
        self._providers[name] = p
        return p

    # -------------------------------------------------- routing

    def routes_for(self, stage: str) -> list[Route]:
        cfg = self.profile.get("stages", {}).get(stage)
        if not cfg:
            raise ProviderError(f"routing profile ไม่มี stage '{stage}'",
                                retryable=False, code="no_route")
        if not cfg.get("enabled", True):
            return []
        primary = cfg["primary"]
        routes = [Route(primary["provider"], primary["model"], primary.get("params") or {})]
        for f in cfg.get("fallbacks", []):
            routes.append(Route(f["provider"], f["model"], f.get("params") or primary.get("params") or {}))
        return routes

    def stage_cfg(self, stage: str) -> dict:
        return self.profile.get("stages", {}).get(stage, {})

    def stage_can_see(self, stage: str) -> bool:
        """stage นี้มีเส้นทางที่ "ดูภาพจริง" ไหม

        ต้องถามก่อนส่งภาพไปตรวจ เพราะ adapter ที่ไม่รองรับจะทิ้ง images เงียบ ๆ
        แล้วโมเดลก็ตอบกลับมาทั้งที่ไม่ได้เห็นอะไรเลย — ได้รายงาน QC ที่ดูเหมือน
        ผ่าน ซึ่งอันตรายกว่าการไม่ตรวจแล้วบอกตรง ๆ ว่าไม่ได้ตรวจ

        ประกาศด้วย params.vision ใน routing profile ถ้าจะ serve โมเดล VL เอง
        ไม่ประกาศก็ถามความสามารถของ adapter แทน
        """
        for route in self.routes_for(stage):
            if "vision" in route.params:
                if route.params["vision"]:
                    return True
                continue
            if getattr(self.provider(route.provider), "supports_vision", False):
                return True
        return False

    # -------------------------------------------------- execute with fallback

    async def run(
        self,
        stage: str,
        fn: Callable[[Any, Route], Awaitable[T]],
        *,
        budget: BudgetGuard | None = None,
        attempts_per_route: int = 3,
    ) -> tuple[T, Route]:
        """fn รับ (provider_instance, route) แล้วคืนผลลัพธ์
        registry จัดการ retry / fallback / timeout ให้"""
        routes = self.routes_for(stage)
        if not routes:
            raise ProviderError(f"stage '{stage}' ถูกปิดไว้", retryable=False, code="disabled")

        cfg = self.stage_cfg(stage)
        timeout = float(cfg.get("timeout_s") or 180)
        last: Exception | None = None

        for r_i, route in enumerate(routes):
            prov = self.provider(route.provider)
            for attempt in range(attempts_per_route):
                if budget and budget.spent > budget.cap:
                    raise BudgetExceeded(f"งบหมดก่อนถึง {stage}")
                try:
                    result = await asyncio.wait_for(fn(prov, route), timeout=timeout)
                    if r_i > 0:
                        log.warning("stage=%s ใช้ fallback ตัวที่ %d (%s)", stage, r_i, route.model)
                    return result, route
                except asyncio.TimeoutError as e:
                    last = ProviderError(f"{route.model} timeout ที่ {timeout}s")
                    log.warning("stage=%s %s timeout", stage, route.model)
                except ProviderError as e:
                    last = e
                    if not e.retryable:
                        log.warning("stage=%s %s error ที่ retry ไม่ได้: %s — ข้ามไป fallback",
                                    stage, route.model, e)
                        break
                    log.warning("stage=%s %s attempt %d ล้ม: %s", stage, route.model, attempt + 1, e)
                except Exception as e:  # noqa: BLE001
                    last = e
                    log.exception("stage=%s %s error ไม่คาดคิด", stage, route.model)
                    break

                # exponential backoff + jitter
                await asyncio.sleep(min(2 ** attempt + random.random(), 20))

        raise ProviderError(
            f"stage '{stage}' ล้มทุกเส้นทาง ({len(routes)} routes) — ตัวสุดท้าย: {last}",
            retryable=False, code="all_routes_failed") from last
