"""OpenRouter — ตัวหลัก ครอบคลุม text / image / video / audio ด้วย key เดียว

ข้อควรระวังที่เจอจริง:
  * video เป็น async job (POST แล้วได้ id กลับมา ต้อง poll หรือรับ webhook)
  * ต้อง validate duration / resolution / aspect กับ Models API ก่อนยิง
  * ถ้าเปิด ZDR ไว้ OpenRouter จะไม่ route งาน video ให้ เพราะ async ต้องเก็บไฟล์ชั่วคราว
  * รูปแบบ response ต่างกันระหว่าง endpoint — อย่าเขียน parser รวมตัวเดียว
"""
from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

import httpx

from ..media.thai_text import align_script_to_spans
from .base import (
    BaseProvider, ToolCall, ImageRequest, ImageResponse, JobHandle, JobStatus,
    ProviderError, STTResponse, STTWord, TTSRequest, TTSResponse,
    TextRequest, TextResponse, VideoCaps, VideoRequest,
)

BASE = "https://openrouter.ai/api/v1"

log = logging.getLogger(__name__)


def _price_per_second(skus: dict, resolution: str = "1080p") -> float | None:
    """pricing_skus แยกราคาตามความละเอียด — ชื่อคีย์ไม่เหมือนกันทุกโมเดล

    เจอจริงสองแบบ: per-video-second-1080p และ duration_seconds_768p
    (minimax/hailuo-3-max ใช้แบบหลัง) ถ้าอ่านไม่เจอจะได้ None ซึ่งแปลว่า
    ประเมินราคาไม่ได้ แล้ว BudgetGuard จะไม่เห็นค่าใช้จ่ายขั้น video เลย
    """
    for key in (f"per-video-second-{resolution}", f"duration_seconds_{resolution}",
                "per-video-second", "duration_seconds"):
        if key in skus:
            try:
                return float(skus[key])
            except (TypeError, ValueError):
                pass
    return None


_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


def _header_cost(headers) -> float:
    """endpoint เสียง/ภาพ คืนไฟล์ดิบ ไม่มี body ให้อ่าน usage — ราคามาทาง header"""
    for k in ("x-openrouter-cost", "x-or-cost", "openrouter-cost"):
        v = headers.get(k)
        if v:
            try:
                return float(v)
            except ValueError:
                pass
    return 0.0


def _b64_image(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


class OpenRouterProvider(BaseProvider):
    name = "openrouter"
    # ส่งภาพต่อเสมอ — จะเห็นจริงไหมขึ้นกับโมเดลที่เลือกใน routing profile
    supports_vision = True

    def __init__(self, api_key: str, *, referer: str = "", title: str = "shortvideo",
                 timeout: float = 120.0, callback_url: str = "", **cfg):
        super().__init__(**cfg)
        self.api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": referer or "https://localhost",
            "X-Title": title,
        }
        self._timeout = timeout
        self.callback_url = callback_url
        self._caps_cache: dict[str, VideoCaps] = {}
        self._img_caps_cache: dict[str, dict] = {}

    # ------------------------------------------------------------ helpers

    async def _post(self, path: str, payload: dict, *, timeout: float | None = None) -> dict:
        async with httpx.AsyncClient(timeout=timeout or self._timeout) as c:
            r = await c.post(f"{BASE}{path}", headers=self._headers, json=payload)
        if r.status_code == 429:
            raise ProviderError("openrouter rate limit", retryable=True, code="rate_limit")
        if r.status_code in (401, 403):
            raise ProviderError(f"auth ล้มเหลว: {r.text[:200]}", retryable=False, code="auth")
        if r.status_code >= 400:
            retryable = r.status_code >= 500
            raise ProviderError(
                f"openrouter {r.status_code}: {r.text[:400]}",
                retryable=retryable, code=f"http_{r.status_code}")
        return r.json()

    async def _get(self, path: str) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{BASE}{path}", headers=self._headers)
        if r.status_code >= 400:
            raise ProviderError(f"openrouter {r.status_code}: {r.text[:300]}",
                                retryable=r.status_code >= 500)
        return r.json()

    @staticmethod
    def _usage_cost(body: dict) -> float:
        u = body.get("usage") or {}
        # OpenRouter คืน cost มาให้ตรง ๆ เมื่อเปิด usage accounting — ใช้ค่านี้ อย่าคำนวณเอง
        return float(u.get("cost") or 0.0)

    # ------------------------------------------------------------ text / vlm

    async def generate(self, req: TextRequest) -> TextResponse:
        content: list[dict[str, Any]] = [{"type": "text", "text": req.user}]
        for img in req.images:
            content.append({"type": "image_url", "image_url": {"url": _b64_image(img, "image/jpeg")}})

        messages = req.messages or [
            {"role": "system", "content": req.system},
            {"role": "user", "content": content if req.images else req.user},
        ]
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "usage": {"include": True},
        }
        if req.tools:
            payload["tools"] = [t.to_openai() for t in req.tools]
            payload["tool_choice"] = "auto"
        if req.json_schema and not req.tools:
            # ใช้ร่วมกันไม่ได้ — tool calling กับ structured output คนละกลไก
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "strict": True, "schema": req.json_schema},
            }

        t0 = time.monotonic()
        body = await self._post("/chat/completions", payload)
        try:
            msg = body["choices"][0]["message"]
            text = msg.get("content") or ""
        except (KeyError, IndexError) as e:
            raise ProviderError(f"response ผิดรูป: {json.dumps(body)[:300]}") from e

        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments")}
            calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))

        u = body.get("usage") or {}
        return TextResponse(
            text=text,
            tool_calls=calls,
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            cost_usd=self._usage_cost(body),
            model=body.get("model", req.model),
            raw={"message": msg, "latency_ms": int((time.monotonic() - t0) * 1000)},
        )

    # ------------------------------------------------------------ image

    async def image_caps(self, model: str) -> dict:
        """พารามิเตอร์ที่โมเดลภาพตัวนี้รับจริง — cache ไว้ ถามครั้งเดียวต่อ process

        จำเป็นเพราะ /images **ไม่ error** เมื่อได้ฟิลด์ที่ไม่รู้จัก มันทิ้งเงียบ ๆ
        (วัดแล้ว: ส่ง reference_images ผิดชื่อไป ได้ 200 กลับมาพร้อมภาพคนละคน
        คนละห้อง โดยไม่มีสัญญาณอะไรบอกว่าภาพอ้างอิงถูกเมิน)
        """
        if model in self._img_caps_cache:
            return self._img_caps_cache[model]
        try:
            body = await self._get("/images/models")
        except ProviderError:
            self._img_caps_cache[model] = {}
            return {}
        caps = {}
        for m in (body.get("data") or body or []):
            if isinstance(m, dict) and m.get("id"):
                self._img_caps_cache[m["id"]] = m.get("supported_parameters") or {}
        caps = self._img_caps_cache.setdefault(model, {})
        return caps

    async def generate_image(self, req: ImageRequest) -> ImageResponse:
        caps = await self.image_caps(req.model)
        payload: dict[str, Any] = {
            "model": req.model,
            "prompt": req.prompt,
            "n": req.n,
            "aspect_ratio": req.aspect_ratio,
        }
        # ส่งเฉพาะที่โมเดลรับจริง ถ้า caps ว่าง (ถามไม่ได้) ก็ส่งไปตามเดิม
        if req.seed is not None and (not caps or "seed" in caps):
            payload["seed"] = req.seed
        elif req.seed is not None:
            _warn_once(f"img-seed-{req.model}",
                       "%s ไม่รับ seed — การล็อก seed ให้สไตล์ต่อเนื่องใช้ไม่ได้ "
                       "ตัวยึดสไตล์เหลือแค่ภาพ anchor ที่ส่งเป็น input_references",
                       req.model)
        if req.negative_prompt and (not caps or "negative_prompt" in caps):
            payload["negative_prompt"] = req.negative_prompt

        if req.reference_images:
            # ชื่อฟิลด์คือ input_references ไม่ใช่ reference_images
            # และต้องเป็น [{type, image_url:{url}}] ไม่ใช่ list ของ data URL เปล่า ๆ
            if caps and "input_references" not in caps:
                _warn_once(f"img-ref-{req.model}",
                           "%s ไม่รับภาพอ้างอิง — สไตล์จะไม่ต่อเนื่องข้าม shot "
                           "เลือกโมเดลที่มี input_references ใน routing profile",
                           req.model)
            else:
                payload["input_references"] = [
                    {"type": "image_url", "image_url": {"url": _b64_image(b)}}
                    for b in req.reference_images
                ]
        payload.update(req.extra)

        body = await self._post("/images", payload)
        out: list[bytes] = []
        for item in body.get("data", []):
            if item.get("b64_json"):
                out.append(base64.b64decode(item["b64_json"]))
            elif item.get("url"):
                async with httpx.AsyncClient(timeout=60) as c:
                    out.append((await c.get(item["url"])).content)
        if not out:
            raise ProviderError(f"ไม่ได้ภาพกลับมา: {json.dumps(body)[:300]}")
        return ImageResponse(images=out, cost_usd=self._usage_cost(body),
                             seed=req.seed, model=req.model)

    # ------------------------------------------------------------ video
    #
    # ตรวจกับเอกสารจริงแล้ว — จุดที่ต่างจากที่คนมักเดา:
    #   * video models ไม่โผล่ใน GET /models ธรรมดา ต้องใช้ /videos/models
    #   * i2v ใช้ frame_images (มี frame_type) ไม่ใช่ reference images
    #   * status enum คือ pending|in_progress|completed|failed
    #   * ผลลัพธ์อยู่ใน unsigned_urls[] ไม่ใช่ data.url
    #   * generate_audio ดีฟอลต์เป็น true สำหรับโมเดลที่รองรับ — ส่งค่าชัด ๆ เสมอ
    #     ตาม req.with_audio (ตั้งแต่ 2026-09-13 pipeline เปิดเสียงเพราะให้โมเดล
    #     พูดบทเอง) เสียงทำให้ราคาเป็นสองเท่า: seedance-1-5-pro คิด video_tokens
    #     0.0000024 กับเสียง / 0.0000012 ไม่มีเสียง
    #   * catalog บอกว่าโมเดลทำเสียงได้ด้วยฟิลด์ generate_audio: true ระดับบน
    #     (grok-imagine เป็น null = ไม่ทำ) ไม่ใช่จาก pricing_skus
    #   * ราคาอยู่ใน pricing_skus แยกตามความละเอียด
    #   * ZDR ปิดกั้น video generation — ถ้าเปิด ZDR ไว้จะไม่ route ให้เลย

    async def capabilities(self, model: str) -> VideoCaps:
        if model in self._caps_cache:
            return self._caps_cache[model]
        body = await self._get("/videos/models")
        found = next((m for m in body.get("data", []) if m.get("id") == model), None)
        if not found:
            available = [m.get("id") for m in body.get("data", [])][:8]
            raise ProviderError(
                f"ไม่พบโมเดล {model} ใน /videos/models — ที่มีเช่น {available}",
                retryable=False, code="unknown_model")

        skus = found.get("pricing_skus") or {}
        caps = VideoCaps(
            model=model,
            durations=[float(d) for d in found.get("supported_durations") or []],
            resolutions=list(found.get("supported_resolutions") or []),
            aspect_ratios=list(found.get("supported_aspect_ratios") or []),
            # heygen/avatar-iv ไม่มี frame_images เลย (รับภาพทาง input_references แทน)
            supports_first_frame="first_frame" in (found.get("supported_frame_images") or []),
            supports_audio=bool(found.get("generate_audio")) or bool(found.get("supports_audio")),
            declares_audio=found.get("generate_audio") is not None,
            # ส่ง resolution ที่ต่ำสุดที่รองรับเป็นตัวตั้ง แล้วให้ผู้เรียกถามซ้ำ
            # ตามความละเอียดจริงที่จะใช้ — ดีกว่าฮาร์ดโค้ด 1080p ซึ่งหลายโมเดลไม่มี
            price_per_second=_price_per_second(
                skus, (found.get("supported_resolutions") or ["1080p"])[0]),
        )
        caps.passthrough = list(found.get("allowed_passthrough_parameters") or [])
        caps.pricing_skus = skus
        self._caps_cache[model] = caps
        return caps

    async def submit(self, req: VideoRequest) -> JobHandle:
        caps = await self.capabilities(req.model)
        caps.validate(req)

        payload: dict[str, Any] = {
            "model": req.model,
            "prompt": req.prompt,
            "aspect_ratio": req.aspect_ratio,
            "resolution": req.resolution,
        }
        # ส่งเฉพาะฟิลด์ที่โมเดลรู้จัก — llms.txt ของ OpenRouter บอกชัด: "an unlisted value
        # is rejected" avatar-iv ไม่มี duration (ยาวตามคำพูด) และไม่มี generate_audio
        if caps.durations:
            payload["duration"] = (int(req.duration_s) if req.duration_s == int(req.duration_s)
                                   else req.duration_s)
        if caps.declares_audio:
            # ส่งชัด ๆ ทั้งสองทาง — ดีฟอลต์ของ API คือ true และคิดเงินสองเท่า
            payload["generate_audio"] = bool(req.with_audio)
        if req.first_frame and caps.supports_first_frame:
            payload["frame_images"] = [{
                "type": "image_url",
                "image_url": {"url": _b64_image(req.first_frame)},
                "frame_type": "first_frame",
            }]
            if req.last_frame:
                payload["frame_images"].append({
                    "type": "image_url",
                    "image_url": {"url": _b64_image(req.last_frame)},
                    "frame_type": "last_frame",
                })
        elif req.reference_images or req.first_frame:
            # ใช้ได้เฉพาะเมื่อไม่มี frame_images — ถ้ามีทั้งคู่ frame_images ชนะ
            # โมเดลที่ไม่รับ first_frame (avatar-iv) ได้ภาพทางนี้แทน
            refs = list(req.reference_images) or [req.first_frame]
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": _b64_image(b)}}
                for b in refs
            ]
        if req.seed is not None:
            payload["seed"] = req.seed
        if self.callback_url:
            # webhook ดีกว่า poll: ไม่ต้องยิงทุก 30 วินาทีต่องาน และ resume ได้
            payload["callback_url"] = self.callback_url
        if req.extra:
            # passthrough ต่อโมเดล (voice_id, motion_prompt, …) — กรองด้วยรายการที่ catalog
            # ประกาศ ไม่งั้นโดน 400 ทั้งคำขอเพราะคีย์เดียวที่โมเดลไม่รู้จัก
            allowed = set(caps.passthrough)
            for k, v in req.extra.items():
                if k == "provider" or not allowed or k in allowed:
                    # "provider" คือ routing/options ของ OpenRouter เอง (เช่น
                    # provider.options.heygen.voice_id — วัดแล้ว: ส่ง voice_id ระดับบน
                    # ทั้งที่อยู่ใน passthrough กลับโดนปฏิเสธ ต้องอยู่ใต้ provider.options)
                    payload[k] = v
                else:
                    _warn_once(f"vid-extra-{req.model}-{k}",
                               "%s ไม่รับพารามิเตอร์ %s — ทิ้ง", req.model, k)

        body = await self._post("/videos", payload, timeout=90)
        job_id = body.get("id")
        if not job_id:
            raise ProviderError(f"ไม่ได้ job id: {json.dumps(body)[:300]}")

        safe = {k: v for k, v in payload.items() if k not in ("frame_images", "input_references")}
        return JobHandle(provider=self.name, external_id=job_id,
                         model=req.model, submitted_params=safe)

    async def poll(self, handle: JobHandle) -> JobStatus:
        body = await self._get(f"/videos/{handle.external_id}")
        return self.parse_job(body)

    @staticmethod
    def parse_job(body: dict) -> JobStatus:
        """ใช้ร่วมกันระหว่าง poll กับ webhook — payload หน้าตาเดียวกัน
        เอา data ออกมาก่อนถ้าเป็น envelope ของ webhook"""
        if "data" in body and isinstance(body["data"], dict) and "status" in body["data"]:
            body = body["data"]
        raw = (body.get("status") or "").lower()
        state = {
            "pending": "queued",
            "in_progress": "running",
            "completed": "succeeded",
            "failed": "failed",
            "cancelled": "failed",
            "expired": "failed",
        }.get(raw, "running")
        urls = body.get("unsigned_urls") or []
        return JobStatus(
            state=state,
            url=urls[0] if urls else None,
            error=body.get("error") if isinstance(body.get("error"), str) else None,
            cost_usd=float((body.get("usage") or {}).get("cost") or 0.0),
        )

    async def fetch(self, handle: JobHandle, url: str | None = None) -> bytes:
        """unsigned_urls ชี้กลับมาที่ OpenRouter จึงต้องแนบ auth header
        (ต่างจาก provider อื่นที่ให้ signed URL ตรงไปที่ CDN)"""
        target = url or f"{BASE}/videos/{handle.external_id}/content?index=0"
        headers = self._headers if target.startswith(BASE) else {}
        async with httpx.AsyncClient(timeout=300, follow_redirects=True) as c:
            r = await c.get(target, headers=headers)
        if r.status_code >= 400:
            raise ProviderError(f"ดาวน์โหลดวิดีโอไม่สำเร็จ: {r.status_code} {r.text[:200]}")
        if not r.content:
            raise ProviderError("ดาวน์โหลดได้ไฟล์ว่าง")
        return r.content

    # ------------------------------------------------------------ audio

    # ทดสอบกับ API จริงแล้ว (qwen/qwen-audio-3.0-tts-flash) — อย่าแก้จากความจำ:
    #   * /audio/speech คืน "ไฟล์เสียงดิบ" ไม่ใช่ JSON ที่มี base64
    #   * response_format รับแค่ "mp3" กับ "pcm" — ส่ง "wav" ไปจะโดน ZodError
    #   * ต้องมี voice เสมอ ไม่งั้น 400 "An explicit voice is required"
    #     และชื่อ voice เป็นของเฉพาะรุ่น (flash มีสองเสียง: loongjohn,
    #     longanhuan_v3.6) ดูได้จากหน้าโมเดล ไม่มี endpoint ให้ query
    async def synthesize(self, req: TTSRequest) -> TTSResponse:
        text = req.text
        # เรียงจากคำยาวไปสั้น กัน "AI" ไปแทนทับกลางคำของ "AIS"
        for src in sorted(req.pronunciation or {}, key=len, reverse=True):
            text = text.replace(src, req.pronunciation[src])

        if not req.voice or req.voice == "default":
            raise ProviderError(
                f"{req.model} ต้องระบุ voice ใน routing profile "
                "(params.voice) — ฝั่ง provider ไม่มีเสียงดีฟอลต์",
                retryable=False, code="no_voice")

        payload: dict[str, Any] = {
            "model": req.model, "input": text,
            "voice": req.voice, "response_format": "mp3",
        }
        if req.speed and req.speed != 1.0:
            payload["speed"] = req.speed

        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{BASE}/audio/speech", headers=self._headers, json=payload)
        if r.status_code >= 400:
            raise ProviderError(f"tts {r.status_code}: {r.text[:300]}",
                                retryable=r.status_code >= 500)
        audio = r.content
        if not audio:
            raise ProviderError("ไม่ได้เสียงกลับมา", retryable=True)
        if audio[:3] not in (b"ID3", b"\xff\xfb", b"\xff\xf3") and audio[:4] != b"RIFF":
            raise ProviderError(
                f"เสียงที่ได้ไม่ใช่ mp3/wav (ขึ้นต้นด้วย {audio[:8]!r})",
                retryable=False, code="bad_audio")
        return TTSResponse(audio=audio, mime="audio/mpeg",
                           cost_usd=_header_cost(r.headers))

    # /audio/transcriptions เป็น multipart ไม่ใช่ JSON — ต้องส่งไฟล์จริง
    async def transcribe(self, audio: bytes, *, language: str = "th",
                         model: str = "", hint: str = "") -> STTResponse:
        data = {"model": model, "response_format": "verbose_json",
                "timestamp_granularities[]": "segment"}
        if language and language != "auto":
            data["language"] = language

        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(
                f"{BASE}/audio/transcriptions",
                headers={k: v for k, v in self._headers.items() if k != "Content-Type"},
                files={"file": ("audio.wav", audio, "audio/wav")}, data=data)
        if r.status_code >= 400:
            raise ProviderError(f"stt {r.status_code}: {r.text[:300]}",
                                retryable=r.status_code >= 500)
        body = r.json()
        cost = float(((body.get("usage") or {}).get("cost")) or 0.0) or _header_cost(r.headers)

        spans = [(float(sg["start"]), float(sg["end"]))
                 for sg in body.get("segments") or []
                 if sg.get("start") is not None and sg.get("end") is not None]

        # ถ้ารู้บทอยู่แล้ว (ทุกกรณีใน pipeline นี้) ให้ปูบทลงบนเวลาของวลี
        # อย่าใช้ words ที่ ASR คืนมา: ภาษาไทยไม่มีช่องว่าง มันจึงคืนก้อนละทั้งวลี
        # และหลายก้อน start == end (วัดแล้ว 44 วินาทีได้ 4 ก้อน)
        if hint and spans:
            words = [STTWord(w, a, b)
                     for w, a, b in align_script_to_spans(hint, spans)]
        else:
            words = [STTWord(w.get("word", ""), float(w["start"]), float(w["end"]))
                     for w in body.get("words") or []
                     if w.get("start") is not None and w.get("end") is not None]

        if not words:
            raise ProviderError(
                f"{model} ไม่คืนทั้ง segment และ word timestamp — "
                "timeline ทั้งระบบมาจากตรงนี้ ไปต่อไม่ได้",
                retryable=False, code="no_words")
        return STTResponse(text=hint or body.get("text", ""), words=words, cost_usd=cost)
