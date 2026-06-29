# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Install the package first (better layer caching)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[app]"

COPY app ./app

EXPOSE 8000
# Default: serve the prediction API. Override `command` for the dashboard.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
