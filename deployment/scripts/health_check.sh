#!/usr/bin/env bash
set -euo pipefail

# VisionGuard health check script

SERVICE="visionguard"
ADDRESS="localhost:50051"
LOG_DIR="/var/log/visionguard"

check_service() {
    if systemctl is-active --quiet "${SERVICE}"; then
        echo "[OK] Service ${SERVICE} is running"
    else
        echo "[FAIL] Service ${SERVICE} is not running"
        return 1
    fi
}

check_port() {
    if nc -z "${ADDRESS%:*}" "${ADDRESS##*:}" >/dev/null 2>&1; then
        echo "[OK] gRPC port ${ADDRESS} is listening"
    else
        echo "[FAIL] gRPC port ${ADDRESS} is not listening"
        return 1
    fi
}

check_disk() {
    local usage
    usage=$(df "${LOG_DIR}" 2>/dev/null | awk 'NR==2 {print $5}' | tr -d '%')
    if [[ -n "${usage}" && "${usage}" -lt 90 ]]; then
        echo "[OK] Disk usage: ${usage}%"
    else
        echo "[WARN] Disk usage high: ${usage}%"
    fi
}

rotate_logs() {
    if [[ -d "${LOG_DIR}" ]]; then
        find "${LOG_DIR}" -name "*.log" -size +100M -exec gzip {} \;
        find "${LOG_DIR}" -name "*.gz" -mtime +30 -delete
    fi
}

main() {
    echo "==> VisionGuard Health Check $(date -Iseconds)"
    check_service
    check_port
    check_disk
    rotate_logs
    echo "==> Done"
}

main "$@"
