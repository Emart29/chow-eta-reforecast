# Serving image for the ETA re-forecasting API.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install the lean serving dependencies first for better layer caching.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code.
COPY src ./src
COPY api ./api

# Simulate data and train the serving bundle at build time, so the container
# starts ready to serve without a separate training step.
RUN python -m src.simulate.lifecycle --n-orders 20000 \
 && python -c "from src.serving.bundle import build_and_save; build_and_save()"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
