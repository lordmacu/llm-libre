#!/usr/bin/env bash
# Updates DeepSeek credentials in Coolify (deepseek-proxy app on blog).
# Run from your Mac: bash scripts/update-deepseek-creds.sh

set -euo pipefail

SSH_HOST="blog"
APP_UUID="ubrad02htpc0h2rmducrk1nb"
API="http://127.0.0.1:8000/api/v1"

echo "=== DeepSeek credentials update ==="
read -rp  "Email: "    DS_EMAIL
read -rsp "Password: " DS_PASS
echo

echo "Connecting to $SSH_HOST..."

ssh "$SSH_HOST" bash -s -- "$DS_EMAIL" "$DS_PASS" "$APP_UUID" "$API" <<'REMOTE'
DS_EMAIL="$1"
DS_PASS="$2"
APP_UUID="$3"
API="$4"

TOKEN=$(cat ~/.coolify-api-token)
H="Authorization: Bearer $TOKEN"
BASE="$API/applications/$APP_UUID/envs"

patch_env() {
  local key="$1" val="$2" preview="$3"
  STATUS=$(curl -s -o /tmp/coolify_resp.json -w "%{http_code}" \
    -X PATCH "$BASE" \
    -H "$H" -H "Content-Type: application/json" \
    -d "{\"key\":\"$key\",\"value\":\"$val\",\"is_preview\":$preview}")
  if [ "$STATUS" = "201" ]; then
    echo "  OK  $key (preview=$preview)"
  else
    echo "  FAIL $key (preview=$preview) HTTP $STATUS: $(cat /tmp/coolify_resp.json)"
  fi
}

echo "Updating production..."
patch_env "DEEPSEEK_EMAIL"    "$DS_EMAIL" "false"
patch_env "DEEPSEEK_PASSWORD" "$DS_PASS"  "false"

echo "Updating preview..."
patch_env "DEEPSEEK_EMAIL"    "$DS_EMAIL" "true"
patch_env "DEEPSEEK_PASSWORD" "$DS_PASS"  "true"

# The redeploy is not optional. The proxy caches its session token at
# /root/.deepseek_token INSIDE the container, which has no volume, so new
# credentials do nothing until that container is destroyed and the process logs
# in again.
echo "Triggering redeploy..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "$H" "$API/deploy?uuid=$APP_UUID&force=false")
echo "  Redeploy: HTTP $STATUS"

# Verify against the CONTAINER, not against Coolify. On 2026-08-19 the API
# reported the new email while the running container still carried the old one:
# a deployment had been queued before the env-var save committed, so the new
# container came up with stale values and cached a token for the muted account.
# Coolify agreeing with you is not evidence that the process restarted with it.
echo "Waiting for the new container..."
for _ in $(seq 1 40); do
  sleep 5
  C=$(sudo -n docker ps --format "{{.Names}}" | grep "^$APP_UUID" | head -1)
  [ -n "$C" ] || continue
  RUNNING=$(sudo -n docker inspect "$C" --format "{{range .Config.Env}}{{println .}}{{end}}" \
            | grep "^DEEPSEEK_EMAIL=" | cut -d= -f2-)
  if [ "$RUNNING" = "$DS_EMAIL" ]; then
    echo "  OK  container $C is running as $RUNNING"
    exit 0
  fi
done
echo "  FAIL the running container still reports DEEPSEEK_EMAIL=$RUNNING"
echo "       Re-run this script: the deploy most likely raced the env save."
exit 1
REMOTE
