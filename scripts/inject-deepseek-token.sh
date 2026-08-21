#!/usr/bin/env bash
# Injects a DeepSeek Bearer token directly into the running container.
# No redeploy needed — the proxy picks it up on the next request.
#
# How to get the token from Chrome:
#   1. Go to chat.deepseek.com (logged in)
#   2. DevTools → Network → any request to /api/v0/
#   3. Copy the value after "Authorization: Bearer " from the request headers

set -euo pipefail

SSH_HOST="blog"
APP_UUID="ubrad02htpc0h2rmducrk1nb"
API="http://127.0.0.1:8000/api/v1"

echo "=== DeepSeek token injection ==="
read -rsp "Paste Bearer token: " DS_TOKEN
echo

ssh "$SSH_HOST" bash -s -- "$DS_TOKEN" "$APP_UUID" "$API" <<'REMOTE'
DS_TOKEN="$1"
APP_UUID="$2"
API="$3"

CONTAINER=$(sudo -n docker ps --format "{{.Names}}" | grep "^$APP_UUID" | head -1)
if [ -z "$CONTAINER" ]; then
  echo "ERROR: container not found (is deepseek-proxy running?)"
  exit 1
fi
echo "Container: $CONTAINER"

# Write token into the running container
sudo -n docker exec "$CONTAINER" bash -c "echo '$DS_TOKEN' > /root/.deepseek_token && chmod 600 /root/.deepseek_token"

# Validate it immediately
CODE=$(sudo -n docker exec "$CONTAINER" \
  python3 -c "
import requests, os
t = open('/root/.deepseek_token').read().strip()
r = requests.get('https://chat.deepseek.com/api/v0/users/current',
    headers={'Authorization': f'Bearer {t}', 'User-Agent': 'Mozilla/5.0'},
    timeout=10)
print(r.status_code, r.json().get('code','?'))
" 2>&1)
echo "Validation: $CODE"

if echo "$CODE" | grep -q "^200 0"; then
  # Also persist it to the env var so a future redeploy keeps it
  TOKEN=$(cat ~/.coolify-api-token)
  curl -s -o /dev/null \
    -X PATCH "$API/applications/$APP_UUID/envs" \
    -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d "{\"key\":\"DEEPSEEK_TOKEN\",\"value\":\"$DS_TOKEN\",\"is_preview\":false}" || true
  echo "Token is valid and saved. No redeploy needed."
else
  echo "WARNING: token did not validate — check it and try again."
  exit 1
fi
REMOTE
