"""สัญญาข้อมูลระหว่าง stage — ทุก output ของ agent ต้อง validate ผ่านตรงนี้

หลักการ: ถ้า LLM ตอบผิดรูปแบบ ให้ retry พร้อมส่ง validation error กลับเข้าไปใน prompt
วิธีนี้แก้ปัญหา output พังได้เกือบทั้งหมดโดยไม่ต้องใช้โมเดลใหญ่ขึ้น
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# 1. Concept
# --------------------------------------------------------------------------

class Angle(BaseModel):
    title: str = Field(description="ชื่อมุมเล่าแบบสั้น")
    premise: str = Field(description="เล่าให้ฟังใน 1-2 ประโยคว่าคลิปนี้จะเล่าอะไร")
    why_it_works: str = Field(description="ทำไมมุมนี้จะหยุดนิ้วคนดูได้")
    risk: str = Field(description="จุดที่อาจพลาด")


class ConceptOut(BaseModel):
    angles: list[Angle] = Field(min_length=2, max_length=4)
    chosen_index: int = Field(ge=0)
    rationale: str

    @field_validator("chosen_index")
    @classmethod
    def _in_range(cls, v: int, info):
        angles = info.data.get("angles") or []
        if angles and v >= len(angles):
            raise ValueError(f"chosen_index {v} เกินจำนวน angle ({len(angles)})")
        return v


# --------------------------------------------------------------------------
# 2. Story  (Gate 0 อนุมัติที่นี่)
# --------------------------------------------------------------------------

# content: hook/setup/turn/payoff/cta — cartoon: hook/setup/conflict/twist/punchline/button
BeatRole = Literal["hook", "setup", "turn", "payoff", "cta",
                   "conflict", "twist", "punchline", "button"]


class Character(BaseModel):
    """ตัวละครหนึ่งตัว — appearance_en คือสัญญาว่าหน้าตาจะเหมือนกันทุก shot

    ภาพอ้างอิงของตัวละคร (ขั้น character_design) สร้างจาก appearance_en ตัวนี้
    แล้วถูกส่งเป็น input_references ให้ทุก keyframe ที่ตัวละครปรากฏ
    voice_en ไปอยู่ใน prompt ของโมเดลวิดีโอ เพราะโมเดลเป็นคนสร้างเสียงพูดเอง
    """
    name: str = Field(description="ชื่อเรียกสั้น ๆ ภาษาไทย ใช้อ้างใน speaker ของแต่ละ beat")
    role: str = Field(description="บทบาทในเรื่อง เช่น พ่อ, ลูกชาย, ผู้บรรยาย")
    appearance_en: str = Field(
        description="ลักษณะภายนอกภาษาอังกฤษ เจาะจง: อายุ เพศ หน้า ผม เสื้อผ้า "
                    "ต้องคงที่ทั้งเรื่อง เพราะใช้สร้างภาพอ้างอิง")
    voice_en: str = Field(
        description="ลักษณะเสียงภาษาอังกฤษ เช่น 'warm Thai male voice, mid-40s, calm'")


class Beat(BaseModel):
    idx: int
    role: BeatRole
    speaker: str | None = Field(
        default=None, description="ชื่อตัวละครที่พูดใน beat นี้ ต้องตรงกับ characters "
                                  "หรือ null ถ้าไม่มีใครพูด")
    dialogue: str = Field(
        default="", description="คำพูดจริงภาษาไทยที่ตัวละครจะพูดในคลิป "
                                "ห้ามมีวงเล็บ ห้ามคำกำกับฉาก ว่างได้ถ้าไม่มีใครพูด")
    visual_intent: str = Field(description="อธิบายภาพเป็นภาษาคน ยังไม่ใช่ prompt")
    sfx: str | None = Field(
        default=None, description="เสียงประกอบ/บรรยากาศ ภาษาอังกฤษสั้น ๆ เช่น 'cartoon boing' "
                                  "— โมเดลวิดีโอสร้างเสียงตามนี้")
    duration_s: int = Field(
        ge=1, le=30,
        description="ความยาว shot เป็นวินาที (จำนวนเต็ม) — นี่คือเวลาจริงที่จะสั่งโมเดลวิดีโอ "
                    "ไม่ใช่การประมาณ ต้องพอให้พูด dialogue จบ")

    @field_validator("dialogue")
    @classmethod
    def _no_stage_directions(cls, v: str) -> str:
        for bad in ("(", "[", "**"):
            if bad in v:
                raise ValueError(
                    "dialogue ต้องเป็นคำพูดล้วน ห้ามมีวงเล็บ คำกำกับฉาก หรือ markdown"
                )
        return v.strip()


class Story(BaseModel):
    title: str
    angle: str
    audience: str
    hook_line: str = Field(description="ประโยคเปิด 3 วินาทีแรก ต้องแยกออกมาให้เห็นชัด")
    characters: list[Character] = Field(min_length=1, max_length=4)
    beats: list[Beat] = Field(min_length=3, max_length=12)
    cta: str
    target_duration_s: int = Field(ge=10, le=180)
    tone: str

    @property
    def total_duration_s(self) -> int:
        return sum(b.duration_s for b in self.beats)

    def character(self, name: str | None) -> Character | None:
        return next((c for c in self.characters if c.name == name), None)

    @field_validator("beats")
    @classmethod
    def _first_is_hook(cls, v: list[Beat]) -> list[Beat]:
        if v and v[0].role != "hook":
            raise ValueError("beat แรกต้องเป็น role=hook เสมอ")
        return v

    @field_validator("beats")
    @classmethod
    def _renumber(cls, v: list[Beat]) -> list[Beat]:
        """บังคับ idx ให้เป็น 0..n-1 ตามลำดับที่โมเดลส่งมา

        idx เป็นเลขลำดับภายใน ไม่มีความหมายเชิงเนื้อหา แต่ถ้าปล่อยให้โมเดล
        เลือกเองจะเกิดกรณีที่ stage 'story' นับจาก 1 ส่วน stage 'shots'
        นับจาก 0 แล้ว build_shots หา beat ไม่เจอ — เจอจริงกับ Qwen3.6
        ตัดปัญหาที่ต้นทางดีกว่าไปไล่แก้ทุกที่ที่ใช้ idx
        """
        for i, b in enumerate(v):
            b.idx = i
        return v

    @field_validator("beats")
    @classmethod
    def _speaker_exists(cls, v: list[Beat], info) -> list[Beat]:
        """speaker ต้องอ้างตัวละครที่ประกาศไว้ — ไม่งั้น keyframe จะหาภาพอ้างอิงไม่เจอ
        และ prompt วิดีโอจะบอกโมเดลว่าใครพูดไม่ได้"""
        names = {c.name for c in (info.data.get("characters") or [])}
        if not names:
            return v
        for b in v:
            if b.speaker and b.speaker not in names:
                raise ValueError(
                    f"beat {b.idx} speaker '{b.speaker}' ไม่มีใน characters {sorted(names)}")
            if b.dialogue and not b.speaker:
                raise ValueError(f"beat {b.idx} มี dialogue แต่ไม่ระบุ speaker")
        return v


# --------------------------------------------------------------------------
# 4. Shots  (Gate 1 อนุมัติที่นี่)
# --------------------------------------------------------------------------

Composition = Literal["wide", "medium", "closeup", "extreme_closeup", "overhead", "pov"]
CameraMove = Literal["static", "push_in", "pull_out", "pan_left", "pan_right", "tilt_up", "tilt_down", "handheld"]


class KenBurns(BaseModel):
    """พารามิเตอร์การขยับภาพนิ่งใน animatic — ให้ตรงกับ camera_move ที่วางไว้"""
    zoom_start: float = Field(default=1.0, ge=1.0, le=2.0)
    zoom_end: float = Field(default=1.12, ge=1.0, le=2.0)
    pan_x: float = Field(default=0.0, ge=-0.4, le=0.4, description="สัดส่วนความกว้าง")
    pan_y: float = Field(default=0.0, ge=-0.4, le=0.4)

    @classmethod
    def from_camera(cls, move: CameraMove) -> "KenBurns":
        table = {
            "static": cls(zoom_start=1.02, zoom_end=1.02),
            "push_in": cls(zoom_start=1.0, zoom_end=1.16),
            "pull_out": cls(zoom_start=1.16, zoom_end=1.0),
            "pan_left": cls(zoom_start=1.12, zoom_end=1.12, pan_x=0.14),
            "pan_right": cls(zoom_start=1.12, zoom_end=1.12, pan_x=-0.14),
            "tilt_up": cls(zoom_start=1.12, zoom_end=1.12, pan_y=0.14),
            "tilt_down": cls(zoom_start=1.12, zoom_end=1.12, pan_y=-0.14),
            "handheld": cls(zoom_start=1.06, zoom_end=1.10),
        }
        return table.get(move, cls())


class ShotPlan(BaseModel):
    """สิ่งที่ LLM ต้องคืนมา — ไม่มีเรื่องเวลา เพราะเวลาถูกกำหนดไว้แล้วใน Beat.duration_s"""
    beat_idx: int
    characters: list[str] = Field(
        default_factory=list,
        description="ชื่อตัวละคร (ตาม Story.characters) ที่ปรากฏในภาพ shot นี้ — "
                    "ใช้เลือกภาพอ้างอิงที่จะส่งให้โมเดลภาพ")
    image_prompt: str = Field(description="prompt ภาษาอังกฤษสำหรับสร้างภาพนิ่ง แนวตั้ง 9:16")
    negative_prompt: str | None = None
    composition: Composition
    camera_move: CameraMove
    motion_intent: str = Field(
        description="กล้องขยับยังไงตอนแปลงเป็นวิดีโอ — เก็บไว้ใช้ตอน i2v"
    )
    subject_action: str = Field(
        default="",
        description="ตัวละครทำอะไรในช็อตนี้ ภาษาอังกฤษ ประโยคเดียว "
                    "เช่น 'the father turns to his son and starts explaining'",
    )
    transition_in: Literal["cut", "fade", "whip"] = "cut"


class ShotPlanOut(BaseModel):
    shots: list[ShotPlan] = Field(min_length=1, max_length=24)


class Shot(BaseModel):
    """shot ที่ผสม plan เข้ากับ timeline แล้ว

    start_s/end_s มาจาก Beat.duration_s ที่ปัดให้ตรงชุดความยาวของโมเดลวิดีโอ
    หลังขั้น video_gen จะถูกเขียนทับด้วยความยาวจริงของคลิปที่ได้ (retime_shots)
    เพราะคลิปมีเสียงพูดอยู่ข้างใน ตัดให้พอดี timeline ไม่ได้อีกแล้ว
    """
    id: str
    idx: int
    beat_idx: int
    start_s: float
    end_s: float
    plan: ShotPlan
    ken_burns: KenBurns
    keyframe_path: str | None = None
    clip_path: str | None = None
    regen_count: int = 0
    # ค่าใช้จ่ายจริงที่ provider รายงานกลับมา (usage.cost) สะสมรวม regen
    keyframe_cost_usd: float = 0.0
    clip_cost_usd: float = 0.0
    clip_job_id: str | None = None
    clip_model: str | None = None

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


# --------------------------------------------------------------------------
# 5. QC
# --------------------------------------------------------------------------

class QCIssue(BaseModel):
    shot_id: str | None = None
    severity: Literal["block", "warn"]
    code: str
    detail: str
    suggested_fix: str | None = None


class QCReport(BaseModel):
    verdict: Literal["pass", "fix", "fail"]
    issues: list[QCIssue] = []

    @property
    def blocking(self) -> list[QCIssue]:
        return [i for i in self.issues if i.severity == "block"]


# --------------------------------------------------------------------------
# 6. Metadata / publishing
# --------------------------------------------------------------------------

class PostMeta(BaseModel):
    caption_th: str = Field(max_length=2000)
    hashtags: list[str] = Field(max_length=12)
    cover_time_s: float = Field(ge=0, description="เวลาที่ควรตัดเป็นภาพปก")
    title_yt: str = Field(max_length=100)

    @field_validator("hashtags")
    @classmethod
    def _hash_prefix(cls, v: list[str]) -> list[str]:
        return [h if h.startswith("#") else "#" + h.lstrip("#") for h in v]


# --------------------------------------------------------------------------
# สถานะของ run
# --------------------------------------------------------------------------

class RunState(StrEnum):
    DRAFT = "draft"
    CONCEPT = "concept"
    STORY_DRAFT = "story_draft"
    STORY_REVIEW = "story_review"          # GATE 0
    SHOT_PLANNING = "shot_planning"
    CHARACTER_DESIGN = "character_design"
    KEYFRAMING = "keyframing"
    ANIMATIC = "animatic"
    STORYBOARD_REVIEW = "storyboard_review"  # GATE 1
    VIDEO_GEN = "video_gen"
    FINAL_RENDER = "final_render"
    QC = "qc"
    FINAL_REVIEW = "final_review"           # GATE 2
    APPROVED = "approved"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    REJECTED = "rejected"
    FAILED = "failed"


GATES = {RunState.STORY_REVIEW, RunState.STORYBOARD_REVIEW, RunState.FINAL_REVIEW}
