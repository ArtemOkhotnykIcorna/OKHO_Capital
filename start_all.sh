#!/usr/bin/env bash
set -e

echo "======================================="
echo " OKHO SYSTEM BOOTSTRAP"
echo "======================================="

echo "[1/4] Checking swap..."
SWAP_ON=$(swapon --show | wc -l)
if [ "$SWAP_ON" -eq 0 ]; then
  echo "⚠️  Swap not enabled. Enabling swapfile..."
  swapon /swapfile || echo "❌ Failed to enable swap"
else
  echo "✅ Swap OK"
fi

echo "[2/4] Reloading systemd..."
systemctl daemon-reload

SERVICES=(
  okho-prod-bot
  okho-prod-dash
  okho-dev-bot
  okho-dev-dash
  okho-lowtp-bot
  okho-lowtp-dash
  okho-nodoge-bot
  okho-nodoge-dash
  okho-trail-bot
  okho-trail-dash
)

echo "[3/4] Starting all services..."
for svc in "${SERVICES[@]}"; do
  echo "→ starting $svc"
  systemctl start "$svc" || echo "❌ failed: $svc"
done

echo "[4/4] Status summary:"
echo "---------------------------------------"
systemctl --no-pager --type=service | grep okho- || true
echo "---------------------------------------"
echo "🚀 OKHO SYSTEM STARTED"
