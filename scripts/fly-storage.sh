#!/usr/bin/env bash
# Create persistent storage for this app (Fly Volume). Validates fly.toml.
# Run from repo root after: fly auth login
#
# Usage:
#   ./scripts/fly-storage.sh
# Environment overrides:
#   FLY_APP, FLY_PRIMARY_REGION, FLY_VOLUME_NAME, FLY_VOLUME_SIZE_GB, FLY_CONFIG
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.fly/bin:${PATH}"

: "${FLY_APP:=elon-twit-analysis}"
: "${FLY_PRIMARY_REGION:=nrt}"
: "${FLY_VOLUME_NAME:=bot_data}"
: "${FLY_VOLUME_SIZE_GB:=3}"
: "${FLY_CONFIG:=${ROOT}/fly.toml}"

if ! command -v fly >/dev/null 2>&1; then
  echo "flyctl not found. Install: curl -L https://fly.io/install.sh | sh"
  echo "Then: export PATH=\"\${HOME}/.fly/bin:\${PATH}\""
  exit 1
fi

if ! fly auth whoami >/dev/null 2>&1; then
  echo "Not logged in. Run: fly auth login"
  exit 1
fi

echo "==> Validating ${FLY_CONFIG}"
fly config validate -c "${FLY_CONFIG}"

volume_exists() {
  fly volumes list -a "${FLY_APP}" --json 2>/dev/null | python3 -c "
import json, sys
want = sys.argv[1]
raw = sys.stdin.read().strip() or '[]'
data = json.loads(raw)
items = data if isinstance(data, list) else data.get('volumes', data.get('Volumes', []))
for v in items:
    name = v.get('Name') or v.get('name')
    if name == want:
        sys.exit(0)
sys.exit(1)
" "${FLY_VOLUME_NAME}"
}

echo "==> Volumes for app ${FLY_APP}"
fly volumes list -a "${FLY_APP}" || true

if volume_exists; then
  echo "==> Volume '${FLY_VOLUME_NAME}' already exists; skipping create."
else
  echo "==> Creating volume '${FLY_VOLUME_NAME}' (${FLY_VOLUME_SIZE_GB}GB, region ${FLY_PRIMARY_REGION})"
  fly volumes create "${FLY_VOLUME_NAME}" \
    -a "${FLY_APP}" \
    -r "${FLY_PRIMARY_REGION}" \
    -s "${FLY_VOLUME_SIZE_GB}" \
    -y
fi

if [[ "${SET_DATA_DIR_SECRET:-}" == "1" ]]; then
  echo "==> SET_DATA_DIR_SECRET=1: setting DATA_DIR=/data (overrides fly.toml [env] for that key)"
  fly secrets set DATA_DIR=/data -a "${FLY_APP}"
else
  echo "==> DATA_DIR comes from fly.toml [env]. To use a secret instead: SET_DATA_DIR_SECRET=1 ./scripts/fly-storage.sh"
fi

echo "==> Done. Mount: fly.toml [[mounts]] source=${FLY_VOLUME_NAME} -> /data"
echo "    Deploy: fly deploy"
echo "    One volume per Machine: fly scale count app=1"
