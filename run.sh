#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# --- args ---
ASSUME_YES=0
for arg in "$@"; do
    [ "$arg" = "--yes" ] && ASSUME_YES=1
done

# --- helpers ---
info()  { echo "=> $*"; }
ok()    { echo "[OK] $*"; }
warn()  { echo "[WARN] $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }

confirm() {
    [ "$ASSUME_YES" = 1 ] && return 0
    local reply
    read -r -p "$1 [y/N] " reply
    case "$reply" in [yY]|[yY][eE][sS]) return 0 ;; *) return 1 ;; esac
}

# --- 1. Python >=3.12 ---
PYTHON=""
for try in python3.12 python3.13 python3.14 python3; do
    command -v "$try" >/dev/null 2>&1 || continue
    ver=$("$try" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
    case "$ver" in 3.1[2-9]|3.[2-9]*) PYTHON="$try"; break ;; esac
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] Se necesita Python >=3.12." >&2
    case "$(uname -s)" in
        Darwin) echo "  brew install python@3.12" ;;
        Linux)  echo "  sudo apt install python3.12 python3.12-venv (o tu gestor)" ;;
    esac >&2
    exit 1
fi
ok "Python: $($PYTHON --version)"

# --- 2. Venv ---
want_ver=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ ! -d .venv ]; then
    $PYTHON -m venv .venv
    ok "Venv creado ($want_ver)"
else
    venv_ver=$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
    if [ "$venv_ver" != "$want_ver" ]; then
        warn "Venv desactualizado ($venv_ver), recreando..."
        rm -rf .venv
        $PYTHON -m venv .venv
        ok "Venv recreado ($want_ver)"
    fi
fi

if [ ! -f .venv/bin/pip ]; then
    .venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 || {
        warn "pip no disponible via ensurepip."
        error "Instala python3.12-venv (o la version correspondiente) y reintenta."
    }
    ok "pip instalado en el venv."
fi

# --- 3. Deps del sistema (solo ffmpeg + curl + unzip) ---
install_sysdeps() {
    local pm="$1" install_cmd="$2" pkgs="$3"
    local missing=""
    for pkg in $pkgs; do
        case "$pm" in
            apt-get) dpkg -s "$pkg" >/dev/null 2>&1 || missing="$missing $pkg" ;;
            dnf)     rpm -q "$pkg" >/dev/null 2>&1 || missing="$missing $pkg" ;;
            pacman)  pacman -Qi "$pkg" >/dev/null 2>&1 || missing="$missing $pkg" ;;
            brew)    brew list "$pkg" >/dev/null 2>&1 || missing="$missing $pkg" ;;
        esac
    done
    if [ -n "$missing" ]; then
        info "Faltan paquetes del sistema:$missing"
        if confirm "Instalar con $install_cmd?"; then
            $install_cmd $missing
            ok "Paquetes instalados."
        else
            warn "Instalacion manual: $install_cmd $pkgs"
        fi
    fi
}

case "$(uname -s)" in
    Linux)
        if command -v apt-get >/dev/null 2>&1; then
            install_sysdeps apt-get "sudo apt-get install -y -qq" \
                "build-essential ffmpeg curl unzip python3.12-dev zstd"
        elif command -v dnf >/dev/null 2>&1; then
            install_sysdeps dnf "sudo dnf install -y" \
                "gcc-c++ ffmpeg curl unzip python3.12-devel zstd"
        elif command -v pacman >/dev/null 2>&1; then
            install_sysdeps pacman "sudo pacman -S --needed --noconfirm" \
                "base-devel ffmpeg curl unzip zstd"
        else
            warn "No se detecto gestor de paquetes (apt/dnf/pacman)."
            warn "Instala manualmente: build-essential/gcc-c++, ffmpeg, curl, unzip, zstd"
        fi
        ;;
    Darwin)
        command -v brew >/dev/null 2>&1 || error "Homebrew no instalado: https://brew.sh"
        install_sysdeps brew "brew install" "ffmpeg curl unzip zstd"
        ;;
esac

# --- 4. Ollama: instalar ---
if ! command -v ollama >/dev/null 2>&1; then
    info "Ollama no instalado."
    if confirm "Instalar Ollama (curl -fsSL https://ollama.com/install.sh | sh)?"; then
        command -v curl >/dev/null 2>&1 || error "curl necesario para instalar Ollama."
        curl -fsSL https://ollama.com/install.sh | sh
        ok "Ollama instalado."
    else
        error "Instala Ollama manualmente: https://ollama.com/download"
    fi
fi

# --- 5. Ollama: iniciar ---
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_BASE_URL

ollama_alive() { curl -sf "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; }

if ! ollama_alive; then
    info "Ollama no responde. Iniciando en segundo plano..."
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    for i in $(seq 1 30); do
        sleep 1
        ollama_alive && { ok "Ollama iniciado."; break; }
    done
    if ! ollama_alive; then
        warn "Ollama no respondio tras 30s. Revisa /tmp/ollama.log"
        error "Inicialo manualmente: ollama serve"
    fi
fi
ok "Ollama conectado en $OLLAMA_BASE_URL"

# --- 6. Pull de modelos Ollama ---
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:1.5b}"
OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"

for model in "$OLLAMA_MODEL" "$OLLAMA_EMBED_MODEL"; do
    if ollama list 2>/dev/null | grep -q "$model"; then
        ok "Modelo $model presente."
    else
        info "Descargando $model..."
        ollama pull "$model"
        ok "Modelo $model descargado."
    fi
done

# --- 7. Modelos Piper ---
command -v curl >/dev/null 2>&1 || error "curl no instalado."

if [ ! -f models/es_ES-sharvard-medium.onnx ] || [ ! -f models/es_ES-sharvard-medium.onnx.json ]; then
    info "Faltan modelos Piper. Descargando..."
    bash scripts/download_models.sh
fi

[ -f models/es_ES-sharvard-medium.onnx ] || error "Piper .onnx no descargado en models/"
[ -f models/es_ES-sharvard-medium.onnx.json ] || warn "Piper .json no encontrado. La TTS podria fallar."
ok "Modelos Piper listos."

# --- 8. Deps Python ---
.venv/bin/python -m pip install -q -r requirements.txt
ok "Dependencias Python instaladas."

# --- 9. Lanzar servidor web ---
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8080}"
info "Servidor web en http://$WEB_HOST:$WEB_PORT"
exec .venv/bin/python -m uvicorn src.web:app --host "$WEB_HOST" --port "$WEB_PORT"
