#!/usr/bin/env bash
set -euo pipefail

# VisionGuard deployment script for Linux

INSTALL_DIR="/opt/visionguard"
SERVICE_FILE="deployment/systemd/visionguard.service"

echo "==> Deploying VisionGuard to ${INSTALL_DIR}"

# Create user if not exists
if ! id -u visionguard >/dev/null 2>&1; then
    sudo useradd --system --no-create-home --group nogroup visionguard
fi

# Install directory
sudo mkdir -p "${INSTALL_DIR}"
sudo cp -r . "${INSTALL_DIR}/"
sudo chown -R visionguard:nogroup "${INSTALL_DIR}"

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
