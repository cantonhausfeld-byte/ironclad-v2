FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by some Python packages (e.g. lightgbm shared libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only what's needed to install dependencies first (improves layer caching)
COPY pyproject.toml ./
COPY ironclad/ ./ironclad/
COPY cli/ ./cli/

RUN pip install --no-cache-dir -e ".[api]"

# Data directory mounted at runtime; create empty placeholder so the import
# path exists even without a bind-mount (tests, smoke checks).
RUN mkdir -p /data/ironclad

ENV IRONCLAD_DATA_DIR=/data/ironclad
ENV IRONCLAD_LOG_FORMAT=json
ENV IRONCLAD_LOG_LEVEL=INFO

# Default: run the FastAPI server on port 8000.
# Override CMD to run the CLI instead: docker run ... ironclad backfill
EXPOSE 8000
CMD ["uvicorn", "ironclad.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
