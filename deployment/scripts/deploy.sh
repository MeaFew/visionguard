#!/usr/bin/env bash
set -euo pipefail

# VisionGuard deployment script for Linux

INSTALL_DIR="/opt/visionguard"
SERVICE_FILE="deployment/systemd/visionguard.service"
# Path of the exported ONNX model, relative to the repo root. Must match the
# --model argument in ${SERVICE_FILE}.
MODEL_PATH="runs/detect/models/real_train/weights/best.onnx"

echo "==> Deploying VisionGuard to ${INSTALL_DIR}"

# Fail fast: the service's ExecStart points at this model file, but runs/ is
# excluded from the generic copy below, so the model must exist to be copied.
if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: ${MODEL_PATH} not found." >&2
    echo "Export it first, e.g.: python scripts/export_onnx.py" >&2
    exit 1
fi

# Create dedicated group and user if not present (the systemd unit runs as
# visionguard:visionguard).
if ! getent group visionguard >/dev/null 2>&1; then
    sudo groupadd --system visionguard
fi
if ! id -u visionguard >/dev/null 2>&1; then
    sudo useradd --system --no-create-home --gid visionguard visionguard
fi

# Install directory — copy project files EXCLUDING build artifacts, VCS, and
# large/local-only data so the deployment is lean and free of host-local state.
# (A plain `cp -r .` previously copied .git, data/, __pycache__, .venv, etc.)
sudo mkdir -p "${INSTALL_DIR}"
tar --owner=0 --group=0 \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
    --exclude='data/raw' \
    --exclude='data/processed' \
    --exclude='runs' \
    --exclude='.cache' \
    --exclude='*.egg-info' \
    -cf - . | sudo tar -xf - -C "${INSTALL_DIR}"
sudo chown -R visionguard:visionguard "${INSTALL_DIR}"

# Install the ONNX model explicitly: runs/ is excluded above (training
# artifacts are large and host-local), but the service needs this one file.
sudo install -D -o visionguard -g visionguard -m 0644 \
    "${MODEL_PATH}" "${INSTALL_DIR}/${MODEL_PATH}"

# Build C++ service
echo "==> Building C++ inference service"
cd "${INSTALL_DIR}/cpp"
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/opt/onnxruntime/lib/cmake/onnxruntime"
make -j"$(nproc)"

# Install systemd service
sudo cp "${INSTALL_DIR}/${SERVICE_FILE}" /etc/systemd/system/visionguard.service
sudo systemctl daemon-reload
sudo systemctl enable visionguard
sudo systemctl restart visionguard

echo "==> Deployment complete"
echo "Check status: sudo systemctl status visionguard"
echo "Check logs:   sudo journalctl -u visionguard -f"
