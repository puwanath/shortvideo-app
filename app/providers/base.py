"""ชั้นกลางระหว่าง pipeline กับผู้ให้บริการโมเดล

กฎเดียวที่ห้ามแหก: โค้ดใน app/agents ห้าม import provider ตัวใดตัวหนึ่งตรง ๆ
ต้องผ่าน registry เสมอ ไม่งั้นวันที่ provider เปลี่ยน API จะต้องรื้อทั้งระบบ
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


class ProviderError(RuntimeError):
    """error ที่ retry ได้"""
    def __init__(self, msg: str, *, retryable: bool = True, code: str = ""):
        super().__init__(msg)
        self.retryable = retryable
        self.code = code


# ---------------------------------------------------------------- ข้อความ

@dataclass(slots=True)
class ToolSpec:
    """นิยาม tool ที่ agent เรียกได้ — schema เดียวกับ OpenAI function calling"""
    name: str
    description: str
    parameters: dict

    def to_openai(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(slots=True)
class TextRequest:
    system: str
    user: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    json_schema: dict | None = None      # ถ้าใส่ จะบังคับ structured output
    images: list[bytes] = field(default_factory=list)  # สำหรับ VLM
    tools: list[ToolSpec] = field(default_factory=list)
    # ส่ง transcript เต็มเมื่ออยู่ใน agent loop — LLM ไม่มีความจำระหว่าง call
    messages: list[dict] | None = None


@dataclass(slots=True)
class TextResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    raw: dict | None = None


# ---------------------------------------------------------------- ภาพนิ่ง

@dataclass(slots=True)
class ImageRequest:
    prompt: str
    model: str
    negative_prompt: str | None = None
    aspect_ratio: str = "9:16"
    size: str = "1080x1920"
    seed: int | None = None
    reference_images: list[bytes] = field(default_factory=list)
    n: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ImageResponse:
    images: list[bytes]
    cost_usd: float = 0.0
    seed: int | None = None
    model: str = ""


# ---------------------------------------------------------------- วิดีโอ

@dataclass(slots=True)
class VideoRequest:
    prompt: str
    model: str
    duration_s: float
    aspect_ratio: str = "9:16"
    resolution: str = "1080p"
    first_frame: bytes | None = None     # i2v — ทางที่เราใช้จริง
    last_frame: bytes | None = None
    reference_images: list[bytes] = field(default_factory=list)
    with_audio: bool = False
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobHandle:
    provider: str
    external_id: str
    model: str
    submitted_params: dict = field(default_factory=dict)


@dataclass(slots=True)
class JobStatus:
    state: Literal["queued", "running", "succeeded", "failed"]
    progress: float = 0.0
    url: str | None = None
    error: str | None = None
    cost_usd: float = 0.0


@dataclass(slots=True)
class VideoCaps:
    """ดึงจาก Models API แล้ว validate ก่อนยิงเสมอ ไม่งั้นได้ 400 กลับมา"""
    model: str
    durations: list[float]
    resolutions: list[str]
    aspect_ratios: list[str]
    supports_first_frame: bool = False
    supports_audio: bool = False
    # โมเดลประกาศฟิลด์ generate_audio ไว้ใน catalog ไหม (null = ไม่รู้จักฟิลด์นี้เลย
    # ส่งไปจะโดน "unlisted value is rejected") — คนละเรื่องกับ supports_audio
    declares_audio: bool = False
    price_per_second: float | None = None
    passthrough: list[str] = field(default_factory=list)
    pricing_skus: dict = field(default_factory=dict)

    def validate(self, req: VideoRequest) -> None:
        if self.durations and req.duration_s not in self.durations:
            raise ProviderError(
                f"{self.model} ไม่รองรับความยาว {req.duration_s}s "
                f"(รองรับ {self.durations})", retryable=False, code="bad_duration")
        if self.aspect_ratios and req.aspect_ratio not in self.aspect_ratios:
            raise ProviderError(
                f"{self.model} ไม่รองรับ {req.aspect_ratio} "
                f"(รองรับ {self.aspect_ratios})", retryable=False, code="bad_aspect")
        if self.resolutions and req.resolution not in self.resolutions:
            raise ProviderError(
                f"{self.model} ไม่รองรับ {req.resolution}", retryable=False, code="bad_res")
        if req.first_frame and not self.supports_first_frame and not req.reference_images:
            raise ProviderError(
                f"{self.model} ไม่รองรับ image-to-video", retryable=False, code="no_i2v")


# ---------------------------------------------------------------- เสียง

@dataclass(slots=True)
class TTSRequest:
    text: str
    model: str
    voice: str = "default"
    speed: float = 1.0
    language: str = "th"
    pronunciation: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class TTSResponse:
    audio: bytes
    mime: str = "audio/wav"
    duration_s: float = 0.0
    cost_usd: float = 0.0


@dataclass(slots=True)
class STTWord:
    word: str
    start_s: float
    end_s: float


@dataclass(slots=True)
class STTResponse:
    text: str
    words: list[STTWord]
    cost_usd: float = 0.0


# ---------------------------------------------------------------- protocols

class TextProvider(Protocol):
    name: str
    async def generate(self, req: TextRequest) -> TextResponse: ...


class ImageProvider(Protocol):
    name: str
    # ชื่อ generate_image ไม่ใช่ generate เพราะ provider ตัวเดียวกัน
    # (OpenRouter) ทำได้ทั้ง text และ image จึงชนกันถ้าใช้ชื่อเดียว
    async def generate_image(self, req: ImageRequest) -> ImageResponse: ...


class VideoProvider(Protocol):
    """async job — ห้ามออกแบบเป็น blocking call เพราะใช้เวลาเป็นนาที
    orchestrator ต้อง resume ได้หลัง restart จาก external_id ที่เก็บใน DB"""
    name: str
    async def capabilities(self, model: str) -> VideoCaps: ...
    async def submit(self, req: VideoRequest) -> JobHandle: ...
    async def poll(self, handle: JobHandle) -> JobStatus: ...
    async def fetch(self, handle: JobHandle, url: str | None = None) -> bytes: ...


class TTSProvider(Protocol):
    name: str
    async def synthesize(self, req: TTSRequest) -> TTSResponse: ...


class STTProvider(Protocol):
    name: str
    async def transcribe(self, audio: bytes, *, language: str = "th",
                         model: str = "", hint: str = "") -> STTResponse: ...


class BaseProvider(abc.ABC):
    name: str = "base"

    # adapter ตัวนี้ "ส่งภาพต่อให้โมเดลจริง" หรือเปล่า
    # ต่างจากคำถามว่าโมเดลดูภาพเป็นไหม — adapter ที่ไม่รองรับจะ *ทิ้ง* images
    # เงียบ ๆ แล้วโมเดลก็ตอบมาทั้งที่ไม่ได้เห็นอะไร ซึ่งแย่กว่า error
    # เพราะรายงาน QC จะดูเหมือนผ่าน ทั้งที่ไม่มีใครดูภาพเลย
    supports_vision: bool = False

    def __init__(self, **cfg):
        self.cfg = cfg
