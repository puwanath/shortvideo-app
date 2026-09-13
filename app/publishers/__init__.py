from .base import (
    Publisher, PublishError, PublishErrorKind, PublishRequest, PublishResult,
    idempotency_key,
)
from .facebook import FacebookReelsPublisher
from .tiktok import TikTokPublisher
from .youtube import YouTubePublisher

REGISTRY = {
    "youtube": YouTubePublisher,
    "facebook": FacebookReelsPublisher,
    "tiktok": TikTokPublisher,
}


# แต่ละ publisher รับ cfg ไม่เหมือนกัน — กรองให้ตรง ไม่งั้น TypeError ที่ __init__
_CFG_KEYS = {
    "youtube": {"client_id", "client_secret"},
    "facebook": {"page_id"},
    "tiktok": {"client_key", "client_secret_tiktok", "audited", "handle"},
}


def get_publisher(platform: str, access_token: str, **cfg) -> Publisher:
    cls = REGISTRY.get(platform)
    if cls is None:
        raise PublishError(
            f"ยังไม่ได้ทำ publisher ของ {platform} (มี: {sorted(REGISTRY)})",
            PublishErrorKind.INVALID)
    keep = {k: v for k, v in cfg.items() if k in _CFG_KEYS.get(platform, set())}
    if "client_secret_tiktok" in keep:
        keep["client_secret"] = keep.pop("client_secret_tiktok")
    return cls(access_token, **keep)


__all__ = ["Publisher", "PublishError", "PublishErrorKind", "PublishRequest",
           "PublishResult", "YouTubePublisher", "FacebookReelsPublisher",
           "TikTokPublisher", "REGISTRY", "get_publisher", "idempotency_key"]
