#!/usr/bin/env bash
# Configures the Telegram alert channel for llm-libre (rate-limit notices).
# Run from your Mac:  bash scripts/setup-telegram-alerts.sh
#
# Create the bot first with @BotFather (/newbot) and have its token at hand.
# The token is read with a hidden prompt, sent straight to Coolify, and never
# written to this repo, to your shell history, or to a file.

set -euo pipefail

SSH_HOST="blog"
APP_UUID="nhh7ouv2zh3lla6ia9vxv34q"          # llm-libre
API="http://127.0.0.1:8000/api/v1"

echo "=== Telegram alerts for llm-libre ==="
echo
read -rsp "Bot token (from @BotFather, hidden): " TG_TOKEN
echo
[ -n "$TG_TOKEN" ] || { echo "No token given. Nothing was changed."; exit 1; }

# 1. Is the token real? getMe is the cheapest possible check and it fails loudly
#    rather than leaving a dead channel configured.
echo "Checking the token..."
ME=$(curl -s -m 15 "https://api.telegram.org/bot${TG_TOKEN}/getMe")
if ! echo "$ME" | grep -q '"ok":true'; then
  echo "  The token was rejected by Telegram. Nothing was changed."
  echo "  Response: $(echo "$ME" | head -c 200)"
  exit 1
fi
BOT_USER=$(echo "$ME" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["username"])')
echo "  OK  the bot is @${BOT_USER}"

# 2. Discover the chat id instead of making the operator hunt for it. Telegram
#    only reveals it once the bot has been spoken to -- a bot cannot message
#    someone who never opened a conversation with it.
echo
echo "Now open Telegram, go to @${BOT_USER} and send it any message (e.g. 'hola')."
read -rp "Press Enter once you have sent it..."

CHAT_ID=""
for _ in $(seq 1 10); do
  UPDATES=$(curl -s -m 15 "https://api.telegram.org/bot${TG_TOKEN}/getUpdates")
  # python3, not sed: BSD sed (macOS, where this runs) does not support `\?`,
  # so the pattern silently matched nothing and the script reported "no message
  # seen" on a perfectly working bot -- a failure indistinguishable from the
  # operator not having sent one. It also handles negative ids (groups) and
  # updates that carry no `message` at all.
  CHAT_ID=$(echo "$UPDATES" | python3 -c '
import json, sys
try:
    updates = json.load(sys.stdin).get("result", [])
except Exception:
    sys.exit(0)
ids = [u[k]["chat"]["id"] for u in updates
       for k in ("message", "channel_post", "edited_message") if k in u]
print(ids[-1] if ids else "")
')
  [ -n "$CHAT_ID" ] && break
  echo "  ...no message seen yet, retrying"
  sleep 3
done

if [ -z "$CHAT_ID" ]; then
  echo "  Could not see any message. Send one to @${BOT_USER} and re-run."
  echo "  (If the bot was added to a GROUP, send the message there instead.)"
  exit 1
fi
echo "  OK  chat id: ${CHAT_ID}"

# 3. Prove the channel works END TO END before storing it. Configuring a channel
#    that cannot actually deliver is the failure this whole script exists to
#    avoid -- it looks identical to "no alerts have fired yet".
echo
echo "Sending a test message..."
SENT=$(curl -s -m 15 -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
  -H "Content-Type: application/json" \
  -d "{\"chat_id\":\"${CHAT_ID}\",\"text\":\"llm-libre: alert channel configured. You will get a message here whenever a rate limit takes a model out of routing.\"}")
if ! echo "$SENT" | grep -q '"ok":true'; then
  echo "  Telegram accepted the token but refused to send. Nothing was stored."
  echo "  Response: $(echo "$SENT" | head -c 200)"
  exit 1
fi
echo "  OK  check Telegram, the message should be there"

# 4. Only now store it.
echo
echo "Storing in Coolify..."
ssh "$SSH_HOST" bash -s -- "$TG_TOKEN" "$CHAT_ID" "$APP_UUID" "$API" <<'REMOTE'
TG_TOKEN="$1"; CHAT_ID="$2"; APP_UUID="$3"; API="$4"
H="Authorization: Bearer $(cat ~/.coolify-api-token)"
BASE="$API/applications/$APP_UUID/envs"

# POST creates, PATCH updates, and Coolify does NOT accept either for the other
# case: PATCH on a key that does not exist yet answers 404 "Environment variable
# not found". These keys are new by definition the first time this runs, so
# PATCH-only silently configured nothing at all -- verified against the live API
# on 2026-08-19. POST first, fall back to PATCH for a re-run.
put() {
  BODY="{\"key\":\"$1\",\"value\":\"$2\",\"is_preview\":$3}"
  STATUS=$(curl -s -o /tmp/tg_resp.json -w "%{http_code}" -X POST "$BASE" \
    -H "$H" -H "Content-Type: application/json" -d "$BODY")
  if [ "$STATUS" != "201" ] && [ "$STATUS" != "200" ]; then
    STATUS=$(curl -s -o /tmp/tg_resp.json -w "%{http_code}" -X PATCH "$BASE" \
      -H "$H" -H "Content-Type: application/json" -d "$BODY")
  fi
  case "$STATUS" in
    201|200) echo "  OK  $1 (preview=$3)" ;;
    *) echo "  FAIL $1 (preview=$3) HTTP $STATUS: $(cat /tmp/tg_resp.json)" ;;
  esac
}
put TELEGRAM_BOT_TOKEN "$TG_TOKEN" false
put TELEGRAM_CHAT_ID   "$CHAT_ID"  false
put TELEGRAM_BOT_TOKEN "$TG_TOKEN" true
put TELEGRAM_CHAT_ID   "$CHAT_ID"  true
REMOTE

echo
echo "Stored. The alerts start once llm-libre restarts -- it reads these at"
echo "startup (notify.from_env). Deploy when the gateway is quiet:"
echo
echo "  ssh $SSH_HOST 'curl -s -H \"Authorization: Bearer \$(cat ~/.coolify-api-token)\" \\"
echo "    \"$API/deploy?uuid=$APP_UUID\"'"
