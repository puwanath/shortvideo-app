



# shortvideo — ระบบผลิตวิดีโอสั้นภาษาไทยแบบ storyboard-first

โจทย์หนึ่งบรรทัด → LLM คิดมุม/พล็อต → เขียนบท + ตัวละคร + บทพูด + ความยาวต่อ shot
→ ภาพอ้างอิงตัวละคร (portrait + design sheet) → storyboard → animatic → **คนอนุมัติ**
→ วิดีโอที่**โมเดลสร้างเสียงพูดไทยเอง** → QC → **คนอนุมัติ** → โพสต์ YouTube / Facebook / TikTok

FastAPI + arq worker + Postgres + Redis, ffmpeg ทำงานสื่อทั้งหมด, LLM บนเครื่อง (vLLM)
สำหรับงานข้อความ, OpenRouter สำหรับภาพ/วิดีโอ หน้า UI เป็นไฟล์เดียว

ผลจริง (2026-09-13): คลิป 24 วินาที เสร็จใน ~10 นาที ราคา **$1.3–1.5** ต่อคลิป
(การ์ตูน 3D/อนิเมะ + Veo) หรือ **~$0.03 + $0.05/วินาที** ในโหมด Avatar

---

## เริ่มใช้ใน 5 นาที

```bash
cp .env.example .env      # ใส่ OPENROUTER_API_KEY และ VLLM_BASE_URL
docker compose up -d      # postgres + redis + api(:8000) + worker
docker compose ps         # api ต้อง healthy ก่อน worker ถึงจะสตาร์ท (worker ไม่สร้างตาราง)
```

### 1) เปิด provider บนบัญชี OpenRouter

ที่ <https://openrouter.ai/settings/privacy> → allowed providers ต้องมี
**`google-vertex`** (Veo) และ **`heygen`** (Avatar IV) — ไม่เปิดจะได้ 404
"No allowed providers" ตอน video_gen ทั้งที่ทุกอย่างก่อนหน้าผ่านหมด
และเช็กว่า API key ไม่ได้ตั้ง credit limit ต่ำ (`GET /api/v1/key` → `limit_remaining`)

### 2) ใส่ routing profile (จำเป็น — ยังไม่มีตัว seed)

ทุก task จะล้มทันทีด้วย `routing profile ไม่มี stage 'concept'` จนกว่าจะมีแถวนี้
ครบทั้ง 9 stage:

```sql
INSERT INTO routing_profile (id, name, stages, animatic_only, is_default, created_at, updated_at)
VALUES (gen_random_uuid(), 'default', '{
  "concept":  {"primary": {"provider": "vllm_local", "model": "Qwen3.6-35B-A3B-FP8", "params": {"max_tokens": 12000}}, "timeout_s": 600},
  "story":    {"primary": {"provider": "vllm_local", "model": "Qwen3.6-35B-A3B-FP8", "params": {"max_tokens": 12000}}, "timeout_s": 600},
  "shots":    {"primary": {"provider": "vllm_local", "model": "Qwen3.6-35B-A3B-FP8", "params": {"max_tokens": 12000}}, "timeout_s": 600},
  "meta":     {"primary": {"provider": "vllm_local", "model": "Qwen3.6-35B-A3B-FP8", "params": {"max_tokens": 12000}}, "timeout_s": 600},
  "qc":       {"primary": {"provider": "vllm_local", "model": "Qwen3.6-35B-A3B-FP8", "params": {"max_tokens": 12000, "vision": true}}, "timeout_s": 900},
  "character":{"primary": {"provider": "openrouter", "model": "openai/gpt-image-2.5-flare", "params": {"aspect": "9:16", "size": "1024x1536"}},
               "fallbacks": [{"provider": "openrouter", "model": "bytedance-seed/seedream-5-0-lite", "params": {"aspect": "9:16"}}], "timeout_s": 1800},
  "keyframe": {"primary": {"provider": "openrouter", "model": "openai/gpt-image-2.5-flare", "params": {"aspect": "9:16", "size": "1024x1536"}},
               "fallbacks": [{"provider": "openrouter", "model": "bytedance-seed/seedream-5-0-lite", "params": {"aspect": "9:16"}}], "timeout_s": 1800},
  "video":    {"primary": {"provider": "openrouter", "model": "google/veo-3.1-lite", "params": {"aspect": "9:16", "resolution": "720p"}}, "fallbacks": [], "timeout_s": 1800},
  "avatar":   {"primary": {"provider": "openrouter", "model": "heygen/avatar-iv", "params": {"aspect": "9:16", "resolution": "720p"}}, "fallbacks": [], "timeout_s": 1800}
}'::json, false, true, now(), now());
```

`video` **ไม่มี fallback โดยตั้งใจ** — โมเดลอื่น (seedance/kling/wan) พูดไทยไม่รู้เรื่อง
ถ้า Veo ปฏิเสธจะ fail ให้เห็นแล้วกด Resume ดีกว่าแอบถอยไปได้คลิปที่ใช้ไม่ได้
ทุก stage ใส่ `fallbacks` / `timeout_s` เพิ่มได้ `qc` ต้องมี `params.vision: true`
ถ้าโมเดลบนเครื่องดูภาพเป็น ไม่งั้นระบบจะข้าม VLM แล้วเตือนใน QC

### 3) สั่งงาน

เปิด <http://localhost:8000/> → **+ New Task** หรือ

```bash
curl -X POST localhost:8000/tasks -H 'content-type: application/json' -d '{
  "brief": "แมวอ้วนขี้เกียจพยายามขโมยปลาย่างจากหมาที่หลับอยู่ วางแผนซับซ้อนแต่พลาดเพราะเรื่องโง่ ๆ",
  "target_duration_s": 24, "format": "cartoon", "style_preset": "anime2d", "render": "scene"
}'
# แนบภาพสินค้า (โฆษณา) ใช้ multipart
curl -X POST localhost:8000/tasks/upload -F brief='โฆษณาขวดน้ำ GreenSip ฝาไม้ไผ่ เก็บเย็น 24 ชม.' \
     -F format=content -F render=avatar -F style_preset=live -F products=@bottle.png
```

pipeline เดินเองจนถึง **Gate 0** (บท) → กด Approve → **Gate 1** (storyboard + animatic
ราคาถึงตรงนี้ ~$0.05–0.25) → Approve · generate video → **Gate 2** (วิดีโอ + QC) → Approve → โพสต์

---

## เลือกอะไรได้บ้างตอนสร้าง task

| ตัวเลือก | ค่า | ผล |
|---|---|---|
| **Format** | `content` | โครงคอนเทนต์/โฆษณา: hook → setup → turn → payoff → **cta** |
| | `cartoon` | โครงหนังสั้น: hook (cold open) → setup → conflict → **twist → punchline** → button, ไม่มี CTA, พูดน้อย, มี SFX ต่อ beat |
| **Render** | `scene` | สร้าง keyframe ทุก shot แล้ว image-to-video ด้วย **Veo 3.1** (พูดไทยเอง) — ฉากเต็ม ตัวละครหลายตัว สินค้าอยู่ในภาพได้ |
| | `avatar` | **HeyGen Avatar IV**: ภาพตัวละคร 1 ใบ + บทพูด → talking-head ยาวตามคำพูดจริง ไม่สร้างฉาก ($0.03 ค่าภาพทั้ง run) — ต้องเป็น "หน้าคน" และทุก beat ต้องมีคนพูด |
| **Art style** | `live` `pixar3d` `anime2d` `flat` `chibi` `clay` `watercolor` | ประโยคสไตล์ต่อท้าย prompt ภาพ/วิดีโอทุกตัว (ชนะ brand kit) |
| **Product photos** | ไฟล์ภาพหลายรูป | LLM บนเครื่อง "เห็น" ภาพตอนเขียนบท และเป็น reference ทุก keyframe ในโหมด scene |
| `target_duration_s` | 10–180 | เป้าความยาว — บทจะแบ่ง beat ให้รวมใกล้ค่านี้ |
| `budget_cap_usd` | ดีฟอลต์ 5 | เพดานเงินต่อ run เกินแล้ว fail ทันที |

ดู `GET /formats` สำหรับรายการปัจจุบัน เพิ่ม format/สไตล์ = แก้ dict ใน `app/agents/formats.py`

---

## หน้า UI

หน้าเดียว (`app/static/index.html`) สไตล์ระบบ Apple: ฟอนต์ระบบ + Noto Sans Thai,
light/dark ตามเครื่องหรือกดเลือกที่มุมขวาบน (Auto / ☀︎ / ☾), แกรเดียนต์น้ำเงิน→ม่วง

- **ซ้าย** รายการ run (ป้าย Cartoon / Avatar) · **กลาง** ชื่อเรื่อง + stepper 16 ขั้น +
  งบ/ความยาว + ค่าใช้จ่ายต่อ stage + พรีวิว (final / animatic / keyframe / **คลิปรายตัว**
  กดจากฟิล์มสตริป) + ปุ่ม Approve / Reject / **Resume** / **⇩ Download** / Post…
- **ขวา** แท็บ Script (ตัวละคร + beat + บทพูด) · Shots (เวลา, cost, job id) · QC · **Post** ·
  Log (สด — commit ทุก event)
- **Post**: เพิ่มบัญชี (วาง token) → ติ๊กหลายบัญชี → เลือก private/unlisted/public → Post now
  → ประวัติพร้อมลิงก์และข้อผิดพลาด

รีเฟรชเองทุก 2.5 วินาทีโดยไม่ทับ `<video>` ที่กำลังเล่นและไม่รีเซ็ต scroll

---

## โพสต์ขึ้นแพลตฟอร์ม

| แพลตฟอร์ม | วิธี | สิ่งที่ต้องเตรียม |
|---|---|---|
| YouTube Shorts | resumable upload + `containsSyntheticMedia: true` (AI disclosure) | OAuth token scope `youtube.upload` + `GOOGLE_CLIENT_ID/SECRET` ใน `.env` (refresh) |
| Facebook Reels | `start` → `rupload.facebook.com` → `finish` → poll `video_status=ready` | **Page** token (`pages_manage_posts` ฯลฯ) + `page_id`; `private` = เก็บเป็น DRAFT |
| TikTok | Direct Post: `init` → PUT chunk → poll `PUBLISH_COMPLETE`, `is_aigc: true` | user token `video.publish` + `TIKTOK_CLIENT_KEY/SECRET`; แอปยังไม่ audit → บังคับ SELF_ONLY |

`POST /runs/{id}/publish {target_ids:[…], privacy}` หนึ่ง job ต่อบัญชี idempotent ต่อ
(run, target, hash ไฟล์) โพสต์ล้มเหลว run กลับไป `approved` ไม่ใช่ `failed`
ตอนนี้ต้องวาง token เองใน UI — ยังไม่มี OAuth connect flow

---

## Resume เมื่อล้มกลางทาง

run ที่ `failed` มีปุ่ม **Resume from …** (`POST /runs/{id}/resume`) กลับไป state ที่ตาย
(`Run.failed_state`) แล้วเดินต่อ ของที่จ่ายไปแล้วไม่จ่ายซ้ำ: คลิปที่เจนเสร็จก่อนล้ม
ถูกบันทึกพร้อม cost ทันที (`run_video` เก็บของที่สำเร็จ*ก่อน*โยน error) และรอบต่อไป
ข้าม shot ที่มีไฟล์แล้ว เช่นเดียวกับ keyframe และภาพตัวละคร

เจอบ่อย: `402 Insufficient credits` = เติมเครดิต, `403 Key limit exceeded` = เพดานที่ตั้งบน
key เอง (แก้ที่หน้า keys ไม่ใช่เติมเงิน), Veo `completed with no output` = โดน filter สุ่ม
resume ซ้ำได้ไม่โดนคิดเงิน

---

## โครงสร้าง

```
app/
  schemas.py            สัญญาข้อมูลระหว่าง stage (Pydantic) — Story/Character/Beat/Shot/QC
  models.py             ตาราง Postgres (Task มี format/style_preset/render/product_refs)
  config.py             ค่าจาก env

  providers/
    base.py             Protocol + dataclass กลาง — agent ห้าม import provider ตรง ๆ
    openrouter.py       LLM / image / video (async job) / TTS / STT — ตรวจกับ API จริงทุกข้อ
    local.py            vLLM ในเครื่อง (+ Thai Whisper, ไม่ได้ใช้แล้ว)
    image_local.py      สร้างภาพด้วย GPU ในเครื่อง (ไม่ได้ใช้แล้ว)
    registry.py         routing + fallback chain + budget guard + stage_can_see

  agents/
    formats.py          format (content/cartoon), render mode (scene/avatar), style presets, prompt ทุกชุด
    stages.py           ขั้น one-shot: บท, shot plan, ภาพตัวละคร 2 ใบ, keyframe, วิดีโอ, QC-VLM, แคปชัน
    llm.py              structured() — บังคับ JSON ตาม schema พร้อม repair loop
    loop.py             เครื่องยนต์ agent loop + เบรกทุกชนิด
    concept_agent.py    agentic: ดูงานเก่า → ให้คะแนน hook → แก้ → ส่ง
    qc_agent.py         agentic: ตรวจ → แก้ prompt → สร้างใหม่ → ตรวจซ้ำ (จำกัดรอบ/เงิน)

  media/
    render.py           ต่อคลิปที่มีเสียงพูด (normalize ภาพ+เสียง, anullsrc, loudnorm 2 รอบ)
    animatic.py         Ken Burns + concat (ไม่มีเสียง/ซับในสายนี้)
    probe.py            QC deterministic ด้วย ffprobe/ffmpeg
    thai_text.py, ass.py, audio.py   สายซับ/TTS เดิม — ไม่ได้ใช้ใน pipeline แต่ demo ยังใช้

  publishers/           youtube.py · facebook.py · tiktok.py + base (idempotency, error kinds)
  orchestrator/runner.py FSM + advance() + approve() + resume()
  db.py / queue.py      engine+Session / enqueue — แยกไว้กัน circular import
  main.py               FastAPI (API + เสิร์ฟ UI ท้ายสุด)
  worker.py             arq: advance / approve / resume / regenerate / publish + sweep cron
  static/index.html     UI

scripts/
  demo_animatic.py      media pipeline ทั้งสาย offline (ffmpeg + libass + ฟอนต์ไทย)
  test_agent_loop.py    เบรกของ agent loop 18 ข้อ — เทสต์เดียวที่มี
```

ทดสอบ offline: `pip install pythainlp pydantic` แล้วรันสองสคริปต์ข้างบน ไม่ต้องมี key

---

## การตัดสินใจที่ควรรู้ก่อนแก้โค้ด

**เวลามาจากบท และต้องเป็นค่าที่โมเดลวิดีโอรับได้ตั้งแต่แรก** (โหมด scene) —
`Beat.duration_s` ที่ LLM กำหนดคือเวลาจริง `video_durations()` ถามชุดที่โมเดลรับ
(Veo: 4/6/8) บอก LLM ตอนเขียนบท `build_shots()` ปัด*ขึ้น*อีกชั้น เพราะคลิปมีเสียงพูด
อยู่ข้างใน ตัดท้ายทิ้งไม่ได้ หลัง `video_gen` `retime_shots()` เขียน timeline ทับด้วย
ความยาวจริง โหมด avatar ไม่ต้องปัดเลย — ยาวตามคำพูด

**เสียงพูดไทย: Veo กับ HeyGen เท่านั้น** — วัดด้วย ASR: seedance/kling/wan ประกาศ
8–10 ภาษาไม่มีไทย และผลออกมาเป็นคำไร้ความหมาย Veo 3.1 ถอดตรงบท; HeyGen ใช้เสียง
"Multilingual" (Kore/Orus) ถอดตรงบททุกคำ ระบบเลือกหญิง/ชายจาก `voice_en`
`video_prompt()` เอาบทพูดขึ้นก่อนเสมอ เพราะ i2v ให้น้ำหนักคำต้น ๆ

**ตัวละครล็อกด้วยภาพอ้างอิงต่อตัว** — portrait + design sheet ถูกส่งเป็น
`input_references` เฉพาะตัวที่อยู่ใน shot พร้อมคำอธิบายหน้าตาซ้ำใน prompt
(ภาพคุมหน้า คำอธิบายคุมเสื้อผ้า) วัดแล้ว: เสื้อ สร้อย หมวก ตรงกันทุก shot

**ความสามารถของโมเดลต้องเช็ก ไม่ใช่เดา** — API หลายตัวรับฟิลด์ที่ไม่รู้จักแล้วทิ้ง
เงียบ ๆ (`reference_images` ผิดชื่อ, `guided_json` ใน vLLM, `generate_audio` กับโมเดล
ที่ไม่ทำเสียง) adapter จึงส่งเฉพาะฟิลด์ที่ catalog ประกาศ และ `stage_can_see`
ตัดสินว่า QC จะส่งภาพให้ VLM หรือแค่บอกตรง ๆ ว่าไม่ได้ตรวจ

**ทุก transition และทุก event commit ลง DB ทันที** ไม่ใช่ตอนจบ job — ถ้า worker
ตายเรารู้ว่าตายที่ไหน UI เห็นสด และ `sweep_stuck_runs` ปลุก run ที่ค้างเกิน 30 นาที

**agent มีแค่สองที่ และอยู่ข้างใน node ของ DAG** — `concept` กับ `qc` เป็น
tool-calling loop เพราะจำนวนรอบไม่รู้ล่วงหน้า ที่เหลือเป็น DAG จึง resume ได้และคุมงบได้
เบรกที่บังคับ: `max_steps` · เพดานเงิน · เพดาน regen ต่อ shot · หยุดเมื่อแก้แล้วไม่ดีขึ้น
· จบด้วย terminal tool เท่านั้น · validate อาร์กิวเมนต์ที่ตัว loop

**cost คือค่าจริงจาก provider** (`usage.cost`) บันทึกเป็น `Generation` หนึ่งแถวต่อหนึ่ง
call พร้อม shot id และ job id ไม่ประเมินเอง `GET /runs/{id}` มี `cost_by_stage`

**QC deterministic ก่อน VLM เสมอ** และเกณฑ์ animatic กับวิดีโอจริงต่างกัน
(animatic ไม่มีเสียง ไม่ตรวจ freeze)

---

## ข้อควรรู้ตอน deploy

- **uid ของ container ต้องตรงกับเจ้าของ `./var/runs`** ดีฟอลต์ 1000 ถ้าไม่ใช่ให้
  `APP_UID=$(id -u) APP_GID=$(id -g) docker compose build`
- **`.env` ต้องมีจริง** compose อ้างด้วย `env_file`
- **โค้ดถูก copy เข้า image** แก้แล้วต้อง `docker compose build && docker compose up -d api worker`
- **DB เก่า `create_all` ไม่ ALTER** — ถ้า DB สร้างก่อน 2026-09-13:
  ```sql
  ALTER TABLE task ADD COLUMN format varchar(20) DEFAULT 'content', ADD COLUMN style_preset varchar(30),
                   ADD COLUMN render varchar(20) DEFAULT 'scene', ADD COLUMN product_refs json;
  ALTER TABLE run  ADD COLUMN characters json, ADD COLUMN failed_state varchar(40);
  ALTER TABLE publish_target ADD COLUMN config json;
  ALTER TABLE publish_attempt ADD COLUMN url varchar(400);
  ```
- **minio อยู่หลัง profile `storage`** เพราะโค้ดยังไม่ใช้ S3
- **ffmpeg ใน image เป็น 7.x** ทดสอบแล้ว animatic ออกมาเหมือน 6.1

---

## ยังไม่ได้ทำ

- **ส่งเสียง TTS ของเราเองให้ HeyGen** — API รับ `audio_url` แต่ต้องเป็น HTTPS สาธารณะ
  รอ MinIO/S3 (ทำแล้วจะใช้เสียงไทยจาก qwen TTS ที่คุมน้ำเสียงได้เอง)
- **สินค้าในโหมด avatar** — Avatar IV รับภาพหน้าเดียว โฆษณาที่ต้องเห็นสินค้าใช้โหมด scene
- **Instagram Reels** — ต้องมีไฟล์บน URL สาธารณะ (รอ S3 เช่นกัน)
- **OAuth connect flow + cron refresh token** — ตอนนี้วาง token เอง
- **MinIO/S3**, **alembic**, **daily budget ระดับระบบ**, **webhook receiver**,
  **AI disclosure บน Facebook** (Graph API ไม่มีช่อง)

---

## เรื่อง OpenRouter API ที่ตรวจกับของจริงแล้ว (อย่าแก้จากความจำ)

| เรื่อง | ที่ถูก |
|---|---|
| หา model วิดีโอ/ภาพ | `GET /videos/models` / `GET /images/models` — ไม่โผล่ใน `/models` |
| image-to-video | `frame_images: [{type, image_url, frame_type: "first_frame"}]` |
| reference image | `input_references` (frame_images ชนะถ้ามีทั้งคู่) — avatar-iv รับภาพทางนี้เท่านั้น |
| avatar-iv | `prompt` = บทพูด, **`voice_id` ต้องอยู่ใน `provider.options.heygen`** (ระดับบนโดนปฏิเสธแม้อยู่ใน passthrough), ไม่มี `duration`/`generate_audio`, ต้องตรวจเจอหน้าคน |
| Veo | `google-vertex` ต้องอยู่ใน allowed providers; 4/6/8 วิ; `generate_audio: true`; filter สุ่ม → resume |
| ฟิลด์ที่โมเดลไม่รู้จัก | โดน reject ทั้งคำขอ — adapter ส่งเฉพาะที่ catalog ประกาศ |
| status / ผลลัพธ์ | `pending`/`in_progress`/`completed`/`failed`; `unsigned_urls[0]` ต้องแนบ auth; `usage.cost` = ค่าจริง |
| ภาพ | `gpt-image-2.5-flare` ไม่มี `seed`; `input_references` สูงสุด 16 ใบ; $0.004–0.015/ใบ |
| เครดิต | `402` = เครดิตหมด (job วิดีโอจองวงเงินล่วงหน้า) · `403 Key limit exceeded` = เพดานของ key |

---

## ผู้พัฒนา

| ชื่อ | บทบาท |
|---|---|
| **DevPooh** | Developer — ออกแบบและพัฒนาระบบทั้งหมด |
