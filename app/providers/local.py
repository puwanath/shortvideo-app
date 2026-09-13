"""ของที่รันบนเครื่องเรา — ฟรี ไม่มีโควตา ไม่ส่งข้อมูลออก

vLLM ของ fatlama-ai พูด OpenAI protocol อยู่แล้ว จึงใช้ adapter บาง ๆ ตัวเดียวพอ
งานที่ควรวิ่งมาที่นี่: concept, shot planning, metadata — เรียกบ่อย ไม่ต้องการโมเดลใหญ่

Thai Whisper fine-tune คือของที่ cloud ทำแทนไม่ได้ดีเท่า ใช้ทำ word timestamp
ซึ่งเป็นแกนของ timeline ทั้งระบบ
"""
from __future__ import annotations

import base64
import json

import httpx

from .base import (
    BaseProvider, ToolCall, ProviderError, STTResponse, STTWord, TextRequest,
    TextResponse, TTSRequest, TTSResponse,
)


class VLLMProvider(BaseProvider):
    """OpenAI-compatible endpoint ของ vLLM"""
    name = "vllm_local"

    # adapter ส่งภาพต่อได้ แต่ *โมเดลที่ serve อยู่* จะดูภาพเป็นหรือไม่ ตัวนี้ไม่รู้
    # จึงต้องประกาศใน routing profile ด้วย params.vision = true
    # (วัดแล้วบนเครื่องจริง: Qwen3.6-35B-A3B-FP8 ตอบสีของภาพได้ถูก = ดูภาพเป็น)
    supports_vision = False

    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout: float = 300.0, **cfg):
        super().__init__(**cfg)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._timeout = timeout

    async def generate(self, req: TextRequest) -> TextResponse:
        if req.images and not req.messages:
            content: list[dict] = [{"type": "text", "text": req.user}]
            for img in req.images:
                b64 = base64.b64encode(img).decode()
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            user_msg: dict = {"role": "user", "content": content}
        else:
            user_msg = {"role": "user", "content": req.user}

        payload = {
            "model": req.model.removeprefix("local/"),
            "messages": req.messages or [
                {"role": "system", "content": req.system},
                user_msg,
            ],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.tools:
            payload["tools"] = [t.to_openai() for t in req.tools]
            payload["tool_choice"] = "auto"
        if req.json_schema and not req.tools:
            # ใช้ response_format แบบ OpenAI ไม่ใช่ guided_json
            # vLLM 0.19 "เมิน" guided_json เงียบ ๆ — ไม่ error แต่คืน JSON ที่
            # คิด schema ขึ้นมาเอง ทำให้ validate ไม่ผ่านแล้วเข้า repair loop ฟรี ๆ
            # ทดสอบบน 0.19.0 แล้ว: response_format และ structured_outputs ใช้ได้จริง
            # เลือก response_format เพราะเป็นมาตรฐาน OpenAI ย้าย provider ได้
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": req.json_schema, "strict": True},
            }

        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self.base_url}/chat/completions",
                             headers={"Authorization": f"Bearer {self.api_key}"},
                             json=payload)
        if r.status_code >= 400:
            raise ProviderError(f"vllm {r.status_code}: {r.text[:300]}",
                                retryable=r.status_code >= 500)
        body = r.json()
        try:
            msg = body["choices"][0]["message"]
            text = msg.get("content") or ""
        except (KeyError, IndexError) as e:
            raise ProviderError(f"vllm response ผิดรูป: {json.dumps(body)[:300]}") from e

        # Qwen3 ใส่ <think> มาด้วยเมื่อเปิด reasoning — ตัดออกก่อนส่งต่อ
        if "</think>" in text:
            text = text.split("</think>", 1)[1].lstrip()

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
            cost_usd=0.0,
            model=req.model,
            raw={"message": msg},
        )


class ThaiWhisperProvider(BaseProvider):
    """service ถอดเสียงไทยของเราเอง คาดหวัง endpoint POST /transcribe (multipart)
    ที่คืน {text, words:[{word,start,end}]}"""
    name = "whisper_local"

    def __init__(self, base_url: str, timeout: float = 600.0, **cfg):
        super().__init__(**cfg)
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def transcribe(self, audio: bytes, *, language: str = "th",
                         model: str = "", hint: str = "") -> STTResponse:
        files: dict = {"file": ("audio.wav", audio, "audio/wav")}
        # language / task เป็น "query param" ไม่ใช่ form field
        # ถ้าส่งเป็น form service จะเมินเงียบ ๆ แล้วตกไปใช้ language=auto
        # (เคยทำให้เสียงอังกฤษถูกถอดเป็นไทยมั่ว)
        params: dict[str, str] = {"language": language, "task": "transcribe"}
        if hint:
            # บทเต็มที่เรารู้อยู่แล้ว — service เอาไปปูลงบนเวลาของวลีที่ Whisper
            # หามาให้ ไม่ใช่เอาไปเป็น prompt ให้โมเดลเดา (Whisper จะเขียนต่อ)
            # ส่งเป็น form ไม่ใช่ query เพราะบทเต็มยาวเกินความยาว URL ที่ปลอดภัย
            # และต้องส่งทั้งก้อน ไม่ตัด ไม่งั้นท้ายคลิปจะไม่มีคำให้ปู
            files["align_text"] = (None, hint)

        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self.base_url}/transcribe", files=files, params=params)
        if r.status_code >= 400:
            raise ProviderError(f"whisper {r.status_code}: {r.text[:300]}",
                                retryable=r.status_code >= 500)
        body = r.json()
        # Thonburian Whisper คืนคีย์ชื่อ word_timestamps ไม่ใช่ words
        # เผื่อ words ไว้ด้วยสำหรับ service อื่นที่ใช้ชื่อมาตรฐานกว่า
        raw = body.get("word_timestamps") or body.get("words") or []
        words = [
            STTWord(w.get("word") or w.get("text", ""), float(w["start"]), float(w["end"]))
            for w in raw
            if w.get("start") is not None and w.get("end") is not None
        ]
        if not words:
            raise ProviderError(
                f"service ถอดข้อความได้ ({len(body.get('text') or '')} ตัวอักษร) "
                "แต่ไม่คืน word timestamp เลย — timeline ทั้งระบบมาจากตรงนี้ "
                "ต้องเปิด word-level timestamp ที่ตัว service ก่อน",
                retryable=False, code="no_words")
        return STTResponse(text=body.get("text", ""), words=words, cost_usd=0.0)


class LocalTTSProvider(BaseProvider):
    """JaiTTS (F5-TTS) ที่รันเอง — เสียงไทย ฟรี ไม่ส่งข้อความออกนอกเครื่อง

    สัญญา: POST /tts รับ JSON คืนไฟล์ WAV ดิบ (24 kHz mono)
    เลือกเสียงด้วย ref_audio_id — ดูรายชื่อจาก GET /voices
    """

    name = "tts_local"

    def __init__(self, base_url: str, timeout: float = 600.0, **cfg):
        super().__init__(**cfg)
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def voices(self) -> list[str]:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(f"{self.base_url}/voices")
        r.raise_for_status()
        return r.json().get("voices", [])

    async def synthesize(self, req: TTSRequest) -> TTSResponse:
        text = req.text
        # BrandKit.pronunciation คือ override การอ่านคำเฉพาะ (ชื่อแบรนด์ ศัพท์เทคนิค)
        # F5-TTS ไม่มีช่องรับ lexicon จึงแทนคำในข้อความก่อนส่ง
        # เรียงจากคำยาวไปสั้น กัน "AI" ไปแทนทับกลางคำของ "AIS"
        for src in sorted(req.pronunciation or {}, key=len, reverse=True):
            text = text.replace(src, req.pronunciation[src])

        payload: dict = {"text": text, "format": "wav"}
        if req.voice and req.voice != "default":
            payload["ref_audio_id"] = req.voice
        if req.speed and req.speed != 1.0:
            payload["speed"] = req.speed

        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self.base_url}/tts", json=payload)
        if r.status_code >= 400:
            raise ProviderError(f"tts {r.status_code}: {r.text[:300]}",
                                retryable=r.status_code >= 500)

        audio = r.content
        if audio[:4] != b"RIFF":
            raise ProviderError(
                f"tts ไม่ได้คืน WAV (ขึ้นต้นด้วย {audio[:8]!r}) — เช็ก format ที่ service",
                retryable=False, code="bad_audio")

        return TTSResponse(audio=audio, mime="audio/wav",
                           duration_s=_wav_duration(audio), cost_usd=0.0)


def _wav_duration(data: bytes) -> float:
    """อ่านความยาวจาก header ไม่ต้องเรียก ffprobe

    ใช้แค่ตอน log — เวลาจริงของ timeline มาจาก forced alignment เสมอ
    """
    try:
        import io
        import wave
        with wave.open(io.BytesIO(data)) as w:
            return round(w.getnframes() / float(w.getframerate()), 3)
    except Exception:  # noqa: BLE001
        return 0.0
