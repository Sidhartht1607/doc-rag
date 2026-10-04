# Document Q&A service.  Not built in CI yet -- see README "Docker" for the exact commands.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    OMP_NUM_THREADS=1
WORKDIR /app

# requirements.txt is exported from uv.lock (`uv export --no-dev --no-hashes --no-emit-project`).
# It lists CUDA wheels for torch on Linux (~3 GB) that this CPU-only service never uses, so install
# the CPU build of torch first and filter the CUDA packages out of the rest.
COPY requirements.txt .
RUN TORCH="$(grep -E '^torch==' requirements.txt | cut -d' ' -f1 | cut -d';' -f1)" \
 && pip install "$TORCH" --index-url https://download.pytorch.org/whl/cpu \
 && grep -v -E '^(nvidia-|triton|cuda-|torch==)' requirements.txt > requirements.cpu.txt \
 && pip install -r requirements.cpu.txt

# bake both models into the image so the container starts fast and works offline
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-mpnet-base-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY rag_app ./rag_app
COPY eval ./eval
ENV PYTHONPATH=/app

# The source PDF is copyrighted and not in git: mount it when you run the container.
# The index is built from it on first start and cached in /app/data/index (mount a volume to keep it).
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "rag_app.api:app", "--host", "0.0.0.0", "--port", "8000"]
