# Multi-stage: bake the embedding model into the image so first request is fast.
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# system deps: `file` for type detection, tesseract for OCR, poppler for pdf
RUN apt-get update && apt-get install -y --no-install-recommends \
      file tesseract-ocr libtesseract-dev poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install fastapi "uvicorn[standard]" python-multipart

# --- optional: bake BGE-M3 in (uncomment for production; ~2GB, slow build) ---
# RUN pip install FlagEmbedding \
#  && python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
# ENV EMBEDDER=bge-m3

COPY docparse ./docparse
COPY service ./service

ENV EMBEDDER=auto RERANKER=auto INDEX_DIR=/data/index
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
