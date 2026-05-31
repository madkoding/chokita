FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    PIPER_BIN=/usr/local/bin/piper

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    tk \
    libgl1 \
    libglib2.0-0 \
    alsa-utils \
    ffmpeg \
    curl \
    ca-certificates \
    xauth \
    && rm -rf /var/lib/apt/lists/*

RUN case "${TARGETARCH}" in \
      amd64) PIPER_ARCH="x86_64" ;; \
      arm64) PIPER_ARCH="aarch64" ;; \
      *) echo "Unsupported architecture: ${TARGETARCH}" && exit 1 ;; \
    esac \
    && curl -L "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_${PIPER_ARCH}.tar.gz" -o /tmp/piper.tar.gz \
    && mkdir -p /opt/piper \
    && tar -xzf /tmp/piper.tar.gz -C /opt/piper --strip-components=1 \
    && ln -s /opt/piper/piper /usr/local/bin/piper \
    && rm /tmp/piper.tar.gz

WORKDIR ${APP_HOME}

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser ${APP_HOME}
USER appuser

CMD ["python", "-m", "src.main"]
