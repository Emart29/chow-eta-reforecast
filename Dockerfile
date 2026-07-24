# Serving image for the ETA re-forecasting API.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install the lean serving dependencies first for better layer caching.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code and the pre-trained serving bundle. The bundle is trained
# offline and committed, so the image needs no training step at build time
# (training on a small cloud instance is slow and needs the full ML stack).
COPY src ./src
COPY api ./api
COPY models/serving_bundle.joblib ./models/serving_bundle.joblib

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT','8000')}/health\")"

# Bind to the platform-provided PORT (Render, Railway, Fly) with a local default.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
