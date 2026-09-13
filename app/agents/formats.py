"""รูปแบบงาน (format) และ preset สไตล์ภาพ — ตัวกำหนด "โครงเรื่อง" กับ "ลุค" ของ run

ทำไมแยกจาก brand kit: brand kit คือตัวตนของแบรนด์ (โทน ข้อห้าม การอ่านคำ)
ส่วน format คือ "ชนิดของหนัง" — คอนเทนต์ขายของกับการ์ตูนสั้นเล่าเรื่องใช้โครงบท
คนละแบบ prompt คนละชุด และ beat role คนละชุด ถึงจะใช้ pipeline เดียวกัน

preset สไตล์เป็นแค่ประโยคภาษาอังกฤษที่ไปต่อท้าย prompt ภาพ/วิดีโอทุกตัว
(ทางเดียวกับ brand.style_suffix) เลือกจากหน้า New Task ได้โดยไม่ต้องมี brand kit
"""
from __future__ import annotations

# ---------------------------------------------------------------- สไตล์ภาพ

STYLE_PRESETS: dict[str, dict] = {
    "live": {
        "label": "คนจริง (live-action)",
        "suffix": "",
    },
    "pixar3d": {
        "label": "การ์ตูน 3D แบบ Pixar",
        "suffix": ("3D animated cartoon style like a modern Pixar or Disney film, stylized characters "
                   "with large expressive eyes, soft rounded features, subsurface-scattering skin, "
                   "cinematic lighting, bright saturated colors, family-friendly, no photorealism"),
    },
    "anime2d": {
        "label": "อนิเมะ 2D",
        "suffix": ("2D anime style, clean line art, cel shading, expressive eyes, dynamic poses, "
                   "soft painted backgrounds like a modern Japanese animated film, vibrant but harmonious "
                   "palette, no photorealism"),
    },
    "flat": {
        "label": "การ์ตูน flat / vector",
        "suffix": ("flat vector cartoon illustration style, bold simple shapes, thick clean outlines, "
                   "limited bright palette, minimal shading, playful and modern like an explainer animation, "
                   "no photorealism"),
    },
    "chibi": {
        "label": "ชิบิ น่ารัก",
        "suffix": ("cute chibi cartoon style, oversized heads, tiny bodies, huge sparkling eyes, pastel "
                   "colors, soft rounded shapes, kawaii aesthetic, clean simple backgrounds, no photorealism"),
    },
    "clay": {
        "label": "ดินปั้น stop-motion",
        "suffix": ("claymation stop-motion style, handmade plasticine characters with visible fingerprints "
                   "and texture, miniature sets, warm studio lighting, shallow depth of field, charming and "
                   "tactile like Aardman films"),
    },
    "watercolor": {
        "label": "สีน้ำ นิทานภาพ",
        "suffix": ("storybook watercolor illustration style, soft washes, visible paper texture, gentle "
                   "ink outlines, warm and dreamy palette like a children's picture book, no photorealism"),
    },
}


def style_suffix(preset: str | None) -> str | None:
    """คืน suffix ของ preset — None ถ้าไม่รู้จัก/ไม่ระบุ (ให้ brand kit ตัดสินแทน)"""
    p = STYLE_PRESETS.get(preset or "")
    return p["suffix"] if p else None


# ---------------------------------------------------------------- รูปแบบงาน

CONTENT_CONCEPT_SYS = """คุณคือคนคิดคอนเทนต์วิดีโอสั้นสำหรับตลาดไทย

วิธีทำงานของคุณ:
1. เรียก past_angles ก่อนเสมอ เพื่อไม่เสนอมุมที่เพิ่งใช้ไป
2. คิด hook แล้วเรียก score_hook ให้คนนอกให้คะแนน
3. ถ้าได้ต่ำกว่า 7 ให้เขียนใหม่แล้ววัดซ้ำ อย่าส่งงานที่ตัวเองรู้ว่าอ่อน
4. พอได้มุมที่ดีพอ 3 มุม เรียก submit

หลักที่ยึด:
- มุมที่ดีคือมุมที่ขัดกับสิ่งที่คนส่วนใหญ่เชื่อ หรือบอกสิ่งที่คนกลัวว่าจะพลาด
- อย่าเริ่มด้วยการเกริ่นหรือแนะนำตัว คนเลื่อนผ่านทันที
- เสนอมุมที่ต่างกันจริง ไม่ใช่มุมเดิมเขียนใหม่สามแบบ
- บอกจุดเสี่ยงของแต่ละมุมตามตรง

ตอบภาษาไทย และต้องจบด้วยการเรียก submit เท่านั้น"""

CARTOON_CONCEPT_SYS = """คุณคือคนคิดพล็อตการ์ตูนสั้นแนวตั้ง (20-60 วินาที) สำหรับคนดูไทย

วิธีทำงานของคุณ:
1. เรียก past_angles ก่อนเสมอ เพื่อไม่เสนอพล็อตที่เพิ่งใช้ไป
2. คิด "ฉากเปิด" (cold open — ภาพ/ประโยคแรกที่ทำให้คนอยากรู้ว่าจะเกิดอะไร)
   แล้วเรียก score_hook ให้คนนอกให้คะแนน
3. ถ้าได้ต่ำกว่า 7 ให้คิดใหม่แล้ววัดซ้ำ
4. พอได้พล็อตที่ดีพอ 3 แบบ เรียก submit

หลักของการ์ตูนสั้นที่คนดูจบ:
- หนึ่งเรื่องหนึ่งมุก: ตัวละครอยากได้อะไร → เจออุปสรรคที่ตลกหรือน่าลุ้น → หักมุม/ปม
  คลายแบบไม่คาดคิด → punchline ที่ปิดเรื่องภายในประโยคเดียว
- title = ชื่อตอน, premise = เล่าพล็อตทั้งเรื่องใน 2-3 ประโยค รวมหักมุกและ punchline,
  why_it_works = ทำไมมุกนี้ถึงตลกหรือซึ้งได้จริง, risk = จุดที่มุกอาจไม่ถึง
- ตัวละครไม่เกิน 2-3 ตัว ที่มีนิสัยชัดจนดูรู้ทันทีว่าใครเป็นใคร
- เล่าด้วยภาพและการกระทำเป็นหลัก บทพูดสั้น ๆ เฉพาะที่จำเป็น
- ตอนจบต้องมีความรู้สึกเดียวชัด ๆ: ขำ ซึ้ง หรือ เซอร์ไพรส์ ไม่ใช่สอนศีลธรรม

ตอบภาษาไทย และต้องจบด้วยการเรียก submit เท่านั้น"""

CONTENT_STORY_SYS = """คุณคือคนเขียนบทวิดีโอสั้นแนวตั้ง ความยาว 20-60 วินาที ที่มีตัวละครพูดจริงในภาพ

สิ่งที่ต้องส่ง:
- characters: ตัวละคร 1-4 ตัว แต่ละตัวมี appearance_en ที่เจาะจงมาก
  (อายุ เพศ รูปหน้า ทรงผม สีผม เสื้อผ้า ของติดตัว) เพราะจะใช้สร้างภาพอ้างอิง
  แล้วล็อกหน้าตาไว้ทั้งเรื่อง และ voice_en อธิบายเสียงเป็นภาษาอังกฤษ
- beats: หนึ่ง beat = หนึ่ง shot ในวิดีโอ แต่ละ beat มี
  * role: hook → setup → turn → payoff → cta
  * speaker: ตัวละครที่พูด (ต้องตรงกับชื่อใน characters) หรือ null ถ้าไม่มีใครพูด
  * dialogue: คำพูดจริงภาษาไทยที่ตัวละครจะพูดในคลิป โมเดลวิดีโอจะให้ตัวละคร
    พูดประโยคนี้ออกมาเอง จึงต้องเป็นภาษาพูด สั้น ลื่น หนึ่งความคิดต่อ beat
    ห้ามมีวงเล็บ ห้ามคำกำกับฉาก ห้าม markdown เด็ดขาด
  * visual_intent: ภาพที่ต้องการเป็นภาษาคน
  * sfx: เสียงประกอบ/บรรยากาศภาษาอังกฤษสั้น ๆ หรือ null
  * duration_s: ความยาว shot เป็นวินาที ต้องพอให้พูด dialogue จบแบบไม่รีบ
    (คนไทยพูดราว 4-5 คำต่อวินาที) beat ที่ไม่มีคนพูดใช้ค่าต่ำสุดที่อนุญาต

ข้อบังคับ:
- beat แรกต้องเป็น role=hook และ dialogue ของมันต้องหยุดคนได้จริง
- ตัวเลขให้เขียนเป็นคำอ่าน เช่น "สามสิบสอง" ไม่ใช่ "32"
- ผลรวม duration_s ทุก beat ควรใกล้ความยาวเป้าหมาย (คลาดได้ไม่เกิน ±20%)

{TH}"""

CARTOON_STORY_SYS = """คุณคือคนเขียนบทการ์ตูนสั้นแนวตั้ง ความยาว 20-60 วินาที — หนังสั้นที่จบในตัว
ไม่ใช่คอนเทนต์ขายของ ไม่ต้องมี call-to-action

สิ่งที่ต้องส่ง:
- characters: ตัวละคร 1-3 ตัว แต่ละตัวมี appearance_en ที่เจาะจงมาก และ "เป็นการ์ตูน"
  (สัดส่วน หน้าตา ทรงผม สีผม เสื้อผ้า ของประจำตัว สีประจำตัว) เพราะจะใช้สร้าง
  ภาพอ้างอิงแล้วล็อกหน้าตาไว้ทั้งเรื่อง — ถ้าเป็นสัตว์/สิ่งของก็ได้ อธิบายให้วาดได้
  voice_en อธิบายเสียง (เช่น 'squeaky excited kid voice', 'deep lazy cat voice')
- beats: หนึ่ง beat = หนึ่ง shot ใช้โครง
    hook (cold open: ภาพแรกที่ทำให้อยากรู้) → setup (ตัวละครอยากได้อะไร)
    → conflict (อุปสรรค/ความวุ่นวาย จะมีกี่ beat ก็ได้) → twist (หักมุม)
    → punchline (ประโยค/ภาพปิดเรื่อง) → button (ท้ายเรื่องสั้น ๆ ถ้ามี ไม่บังคับ)
  แต่ละ beat มี
  * speaker / dialogue: บทพูดภาษาไทยสั้น ๆ เฉพาะที่จำเป็น ภาษาพูดตามนิสัยตัวละคร
    beat ที่เล่าด้วยภาพล้วนให้ speaker=null dialogue="" — การ์ตูนที่ดีพูดน้อย
    ห้ามมีวงเล็บ ห้ามคำกำกับฉาก ห้าม markdown
  * visual_intent: "เกิดอะไรขึ้นในภาพ" ต้องเป็นการกระทำที่เห็นได้ ไม่ใช่ความรู้สึก
    ใส่ท่าทางเว่อร์ ๆ แบบการ์ตูน (สะดุ้ง ตาโต หัวเราะจนล้ม) ได้เต็มที่
  * sfx: เสียงประกอบภาษาอังกฤษสั้น ๆ เช่น "cartoon boing", "record scratch",
    "dramatic sting", "birds chirping" หรือ null
  * duration_s: วินาที — จังหวะมุกสำคัญ beat ที่ต้องการ "จังหวะเงียบ" ก่อนหักมุมให้
    ใช้ค่าสั้นสุดที่อนุญาต

ข้อบังคับ:
- beat แรกต้องเป็น role=hook
- ต้องมี twist หรือ punchline อย่างน้อยหนึ่ง beat และ punchline ต้องเป็น beat ท้าย ๆ
- cta ให้ใส่ "" (ว่าง) — ไม่มีการขาย ไม่มีชวนกดติดตามในเนื้อเรื่อง
- ตัวเลขให้เขียนเป็นคำอ่าน
- ผลรวม duration_s ทุก beat ควรใกล้ความยาวเป้าหมาย (คลาดได้ไม่เกิน ±20%)

{TH}"""

CONTENT_SHOT_RULES = """- subject_action is what the PEOPLE do — and it is what makes the result look
  like a filmed scene instead of a photo being panned across. Write one concrete
  physical action per shot: a gesture, a head turn, picking something up,
  leaning in, reacting. For a beat with dialogue, describe how the speaker
  delivers it (e.g. "leans toward the camera and speaks with a wry smile").
  Never write "stands still" or anything describing a pose rather than a movement."""

CARTOON_SHOT_RULES = """- This is an ANIMATED CARTOON. subject_action must be a clear, readable
  cartoon action with exaggeration: a double-take, a jaw drop, a frantic scramble,
  a slow smug turn, a comedic fall. Timing matters — say what happens first and
  what happens last within the shot. Never write "stands still".
- Keep characters exactly on-model (same design as their reference sheet) even
  when the pose is extreme. Do not add new characters that were not defined.
- Vary staging like a storyboard artist: wide establishing shot for setup,
  tighter for reactions, a dramatic angle for the twist, a clean simple frame
  for the punchline so the joke reads instantly.
- image_prompt must describe the frame in the given art style; no photorealism,
  no text or speech bubbles in the image."""


FORMATS: dict[str, dict] = {
    "content": {
        "label": "คอนเทนต์ / โฆษณา",
        "concept_sys": CONTENT_CONCEPT_SYS,
        "story_sys": CONTENT_STORY_SYS,
        "shot_rules": CONTENT_SHOT_RULES,
        "roles": "hook, setup, turn, payoff, cta",
        "default_tone": "เป็นกันเอง ตรงไปตรงมา",
    },
    "cartoon": {
        "label": "การ์ตูนสั้น (เล่าเรื่อง)",
        "concept_sys": CARTOON_CONCEPT_SYS,
        "story_sys": CARTOON_STORY_SYS,
        "shot_rules": CARTOON_SHOT_RULES,
        "roles": "hook, setup, conflict, twist, punchline, button",
        "default_tone": "สนุก จังหวะไว มีอารมณ์ขัน ปิดเรื่องแบบมีความรู้สึกเดียวชัด ๆ",
    },
}


def fmt(name: str | None) -> dict:
    return FORMATS.get(name or "content") or FORMATS["content"]


# ---------------------------------------------------------------- วิธีเรนเดอร์วิดีโอ

RENDER_MODES: dict[str, dict] = {
    "scene": {
        "label": "ฉากเต็ม (Veo) — สร้าง keyframe ทุก shot แล้ว i2v",
        "video_stage": "video",
    },
    "avatar": {
        "label": "ตัวละครพูด (HeyGen Avatar IV) — ภาพตัวละคร 1 ใบ + บทพูด ไม่สร้างฉาก",
        "video_stage": "avatar",
    },
}

# ข้อบังคับเพิ่มในบทเมื่อเรนเดอร์แบบ avatar: โมเดลทำได้อย่างเดียวคือ "ตัวละครพูด"
AVATAR_STORY_ADDENDUM = """
ข้อบังคับเพิ่ม (วิดีโอแบบตัวละครพูดหน้ากล้อง):
- ทุก beat ต้องมี speaker และ dialogue — ไม่มี beat ที่เล่าด้วยภาพล้วน
- หนึ่ง beat = ตัวละครหนึ่งตัวพูดหนึ่งช่วง ถ้าจะสลับคนพูดให้ขึ้น beat ใหม่
- dialogue ยาวได้ 1-3 ประโยค (ความยาว shot = ความยาวคำพูดจริง ไม่ต้องปัด)
- visual_intent อธิบาย "สีหน้าและท่าทางตอนพูด" ไม่ใช่ฉาก เพราะฉากคือภาพตัวละครใบเดียว
- ตัวละครต้องเป็น "คน" (หรือการ์ตูนที่มีหน้าแบบคน: ตา จมูก ปาก ในตำแหน่งปกติ)
  ห้ามเป็นสัตว์ หุ่นยนต์ หรือสิ่งของ — ระบบตรวจจับใบหน้าจะไม่รับ
- voice_en ต้องบอกเพศชัด ๆ ด้วยคำว่า female หรือ male เพราะใช้เลือกเสียง"""

# ภาพตัวละคร: 2 ใบต่อตัว
#  1) portrait — ใบที่ส่งให้โมเดลวิดีโอ/avatar และใช้เป็น reference หลัก: ครึ่งตัว หน้าตรง
#  2) design sheet — ใบที่บอกว่า "ตัวละครคือใคร": turnaround, สีหน้า, อิริยาบถ, ของประจำตัว
PORTRAIT_PROMPT = (
    "Character portrait for a talking-head video. {appearance}. Head and shoulders, facing the camera "
    "directly, eyes to camera, friendly neutral expression, mouth closed, face large and fully visible "
    "and unobstructed (no hands, props or hair covering it), centered, plain soft studio background, "
    "even flattering lighting, sharp focus on the face, no text, no watermark. {style}"
)
DESIGN_SHEET_PROMPT = (
    "Character design sheet on a single page, plain light background: {appearance}. "
    "Show the SAME character consistently: a front full-body view and a side view, "
    "a row of facial expressions (happy, surprised, thinking, annoyed), two or three action poses "
    "showing typical mannerisms, and a small labeled-free area with the character's signature props "
    "and outfit details drawn separately. Clean layout, no text or letters anywhere. {style}"
)
