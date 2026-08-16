# --- Base image ---
# buster is EOL (Debian 10); use bookworm (Debian 12, current stable)
FROM python:3.11-slim-bookworm

# --- System dependencies (rarely changes, keep early) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Workdir ---
WORKDIR /app

# --- Install Python deps FIRST, before anything else that changes often ---
# This layer only re-runs if requirements.txt itself changes, regardless of
# later edits to ENV vars, CMD, or application code.
# Install torch separately, as its own cached layer, using the CPU-only wheel index.
# This is much smaller than the default torch wheel (~200MB vs ~500MB+, since it skips
# bundled CUDA libraries you don't need on a CPU-only container) and, being a separate
# layer, a network failure here won't force other packages to redownload on rebuild.
RUN pip install --no-cache-dir --timeout=1000 --retries=5 \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=3000 --retries=5 -r requirements.txt

# --- Environment (safe to edit freely below this point without busting the pip cache) ---
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1

# --- Copy application code ---
COPY . /app

# NOTE: BC_predictor_model.keras and BC_scaler.pkl must be at the project root
# alongside this Dockerfile so the COPY above picks them up.
#
# NOTE: The LLM weights (local_models/llama3.2-breastcancer-assistant, ~6GB) are
# intentionally NOT expected to be baked into this image. Either:
#   (a) mount them as a volume at runtime, e.g.
#       docker run -v C:\path\to\local_models:/app/local_models ...
#   (b) or let model_loader.py download them from the Hub on first container start
#       (requires network access and, if the repo is private, an HF_TOKEN env var
#       passed via `docker run -e HF_TOKEN=...`)
# Baking a 6GB model into the image itself will make builds slow and the image huge.

# --- Port ---
# Streamlit's default port is 8501, not 8080
EXPOSE 8501

# --- Healthcheck (optional but useful for orchestration/liveness checks) ---
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# --- Run the Streamlit app ---
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]