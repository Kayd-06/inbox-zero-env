# ============================================================
# InboxZeroEnv – Production Docker Image
# Python 3.11 slim, installs dependencies, runs inference.py
# ============================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="InboxZeroEnv Contributors"
LABEL description="OpenEnv-compatible email triage RL environment"
LABEL version="1.0.0"

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── HF Space non-root setup (UID 1000) ──────────────────────
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ── Python dependencies (layer-cached) ──────────────────────
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────
COPY --chown=user:user env/          ./env/
COPY --chown=user:user data/         ./data/
COPY --chown=user:user inference.py  ./inference.py
COPY --chown=user:user openenv.yaml  ./openenv.yaml

# ── Ensure the package is importable ────────────────────────
ENV PYTHONPATH=/app

# ── Runtime environment variables (override at run time) ────
# These are intentionally left empty — pass them via docker run -e
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

# ── Health check: verify the env can be imported ─────────────
RUN python -c "from env import InboxZeroEnv; env = InboxZeroEnv('easy'); print('✓ InboxZeroEnv import OK')"

# ── Default command ──────────────────────────────────────────
CMD ["python", "inference.py"]
