FROM python:3.12-slim

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
    alsa-utils \
    ffmpeg \
    curl \
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN case "${TARGETARCH}" in \
      amd64) PIPER_ARCH="amd64" ;; \
      arm64) PIPER_ARCH="arm64" ;; \
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

COPY scripts/download_models.sh /tmp/download_models.sh
RUN bash /tmp/download_models.sh

COPY . .

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser ${APP_HOME}
USER appuser

CMD ["python", "-m", "src.main"]
