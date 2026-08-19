"""Outbound alerts, currently Telegram.

WHY THIS EXISTS. A rate limit removes a route from routing (the cooldown makes
it ineligible, see proxy.Cooldowns), and until now that happened silently: the
only trace was a row in `events` and a number in /v1/ranking that somebody had
to go and look at. The question "why is my good model not being used right now"
had no answer that arrived on its own.

TWO RULES, both learned the hard way elsewhere in this gateway:

1. Notifying must never be able to break serving. Telegram being slow, down, or
   misconfigured is not a reason for a client request to fail or hang, so every
   send is fire-and-forget and every failure is swallowed into a log line. A
   monitoring channel that can take down the thing it monitors is worse than no
   channel.

2. Silence when unconfigured, not a crash. With no token the notifier is a no-op
   object rather than None, so callers never branch on whether alerting exists.
"""
import asyncio
import logging
import os

import httpx

log = logging.getLogger(__name__)

# How long a single send may take. Short on purpose: this runs alongside real
# traffic on a saturated machine, and an alert that is thirty seconds late is
# worthless anyway.
TIMEOUT_S = 10.0


class Notifier:
    """No-op base. `Telegram` overrides `send`; everything else keeps working
    unchanged when alerting is not configured."""

    enabled = False

    def notify(self, text: str) -> None:
        """Fire-and-forget. Returns immediately; never raises."""


class Telegram(Notifier):
    enabled = True

    def __init__(self, token: str, chat_id: str,
                 client: httpx.AsyncClient | None = None):
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._chat_id = chat_id
        self._client = client

    def notify(self, text: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop (a test, a synchronous script): there is nothing to detach
            # onto, and blocking here would be exactly the coupling rule 1
            # forbids. Dropping the alert is the correct trade.
            log.debug("telegram: no running loop, alert dropped: %.80s", text)
            return
        loop.create_task(self._send(text))

    async def _send(self, text: str) -> None:
        try:
            client = self._client or httpx.AsyncClient()
            r = await client.post(
                self._url, timeout=TIMEOUT_S,
                json={"chat_id": self._chat_id, "text": text,
                      "parse_mode": "HTML", "disable_web_page_preview": True})
            if r.status_code != 200:
                log.warning("telegram: HTTP %s: %.200s", r.status_code, r.text)
        except Exception as e:            # noqa: BLE001 -- see rule 1 above
            log.warning("telegram: could not send (%s: %s)", type(e).__name__, e)


def from_env() -> Notifier:
    """A configured Telegram notifier, or the silent no-op when the environment
    does not carry both halves. Half-configured counts as unconfigured: a token
    with nowhere to send it cannot alert anyone, and starting up as if it could
    is how a monitoring channel ends up silently dead."""
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if token and chat_id:
        log.info("alerts: telegram enabled, chat %s", chat_id)
        return Telegram(token, chat_id)
    if token or chat_id:
        log.warning("alerts: telegram is HALF configured (token=%s chat_id=%s); "
                    "alerting stays off", bool(token), bool(chat_id))
    return Notifier()
