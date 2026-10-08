# Build stage with Python & ffmpeg
FROM python:3.11-slim

# System dependencies: ffmpeg is required for audio preview decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for layer caching
COPY requirements-deploy.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and weights
COPY . /app

ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONIOENCODING=utf-8
ENV PORT=8050

EXPOSE 8050

# Launch server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8050"]
