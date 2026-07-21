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

COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# Run as a non-root user.
RUN useradd --system --create-home --shell /usr/sbin/nologin appuser
USER appuser

CMD ["python", "scripts/demo_inference.py", "--help"]
