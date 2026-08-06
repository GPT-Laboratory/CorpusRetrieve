FROM python:3.12-slim AS runtime

ARG VERSION=1.0.0
ARG INSTALL_BGE=false

LABEL org.opencontainers.image.title="CorpusRetrieve" \
      org.opencontainers.image.description="Convert document corpora into portable retrieval artifacts" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      tesseract-ocr \
      tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY requirements-embeddings.txt .
RUN if [ "$INSTALL_BGE" = "true" ]; then pip install -r requirements-embeddings.txt; fi

COPY corpusretrieve ./corpusretrieve
COPY docparse ./docparse
COPY service ./service
COPY VERSION pyproject.toml README.md ./

RUN useradd --create-home --uid 10001 corpusretrieve
USER corpusretrieve

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl --fail http://127.0.0.1:8000/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
