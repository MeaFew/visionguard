FROM python:3.11-slim-bookworm AS python-base

WORKDIR /app

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["python", "scripts/train_yolo.py"]
