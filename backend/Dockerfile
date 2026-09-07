# syntax=docker/dockerfile:1

FROM python:3.13-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr,
# and stop pip from caching wheels inside the image layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install only the minimal OS packages needed to build Python wheels
# (some ML dependencies, e.g. torch/transformers, need a C compiler for
# certain platforms/versions). Removed in the same layer to keep the
# image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency manifest first so Docker can cache the
# (slow) `pip install` layer and skip it on rebuilds that only change
# application code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application source. This layer changes often, but the
# dependency layer above stays cached across rebuilds.
COPY app/ ./app/

# Create a dedicated non-root user/group to run the application, and a
# writable cache directory for Hugging Face model downloads. `appuser`
# has no home directory, so we point HF_HOME at an explicit location
# under /app that we can chown.
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --no-create-home appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appgroup /app/.cache

ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

USER appuser

EXPOSE 8000

# No secrets or .env files are baked into the image (see .dockerignore).
# Configuration (HF_TOKEN, HF_MODEL_ID, ALLOWED_ORIGINS, etc.) is read
# from the environment at runtime via app/config.py, so it must be
# supplied with `docker run -e ...` / `--env-file` / your orchestrator's
# secret mechanism.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
