FROM python:3.12-slim

# ffmpeg ต้องมี libass สำหรับซับไทย และ fonts-thai สำหรับ fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg fonts-thai-tlwg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHAINLP_DATA_DIR=/opt/pythainlp

# uid/gid ต้องตรงกับเจ้าของ ./var/runs บนโฮสต์ ไม่งั้น worker เขียนไฟล์ลง
# bind mount ไม่ได้ (ส่ง --build-arg APP_UID=$(id -u) ถ้าโฮสต์ไม่ใช่ 1000)
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g "$APP_GID" app 2>/dev/null || true \
 && useradd -u "$APP_UID" -g "$APP_GID" -m -s /bin/bash app

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ดึงข้อมูลตัดคำของ pythainlp มาตอน build ไม่ใช่ตอน request แรก
# วางไว้นอก HOME เพราะ warm-up รันเป็น root แต่ตอนใช้งานรันเป็น app
RUN mkdir -p "$PYTHAINLP_DATA_DIR" \
 && python -c "from pythainlp.tokenize import word_tokenize; word_tokenize('ทดสอบ')" \
 && chmod -R a+rX "$PYTHAINLP_DATA_DIR"

COPY app ./app
COPY assets ./assets
COPY scripts ./scripts

# STORAGE_ROOT ดีฟอลต์ของ compose — ต้องมีอยู่จริงและ app เขียนได้
RUN mkdir -p /data/runs && chown -R app:app /data /app

USER app
EXPOSE 8000

# ไม่ใส่ HEALTHCHECK ตรงนี้ เพราะ image เดียวกันถูกใช้ทั้ง api และ worker
# ซึ่ง worker ไม่ได้เปิดพอร์ต HTTP — healthcheck อยู่ใน compose รายบริการแทน
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
