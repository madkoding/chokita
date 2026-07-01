FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    alsa-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY scripts/download_models.sh /tmp/download_models.sh
RUN bash /tmp/download_models.sh

COPY . .

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser ${APP_HOME}
USER appuser

CMD ["python", "-m", "src.main"]
