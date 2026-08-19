import httpx
import pytest

from llm_libre.models import Capabilities, Route
from llm_libre.notify import Notifier, Telegram, from_env
from llm_libre.proxy import ALL_CAPABILITIES, CHAT, IMAGES, Proxy
from llm_libre.providers import Provider
from llm_libre.storage import Storage


class _Spy(Notifier):
    """Captures instead of sending, so the tests assert on WHAT would be said."""
    enabled = True

    def __init__(self):
        self.sent: list[str] = []

    def notify(self, text: str) -> None:
        self.sent.append(text)


def _proxy(notifier):
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 1000, 100))], 1.0)
    prov = {"kilo": Provider("kilo", "free", "openai", "https://k.test", "",
                             "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={})))
    return Proxy(prov, store, http, notifier=notifier)


def _429(seconds=None):
    headers = {"Retry-After": str(seconds)} if seconds is not None else {}
    return httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://k.test"))


# --- the anti-flood rule, which is the one that matters in production


def test_only_the_refusal_that_actually_excludes_a_route_alerts():
    """A route already out of routing sends no second alert.

    Measured 2026-08-19: z-ai/glm-5.2 took 55 refusals across 11 real
    exhaustions. Alerting per 429 rather than per exclusion is a 5x flood of
    messages that all say the same thing.
    """
    spy = _Spy()
    p = _proxy(spy)
    p._punish_429("kilo/a:free", 1000.0, _429(300), scope=CHAT)
    assert len(spy.sent) == 1
    for extra in range(1, 6):           # still inside the cooldown
        p._punish_429("kilo/a:free", 1000.0 + extra, _429(300), scope=CHAT)
    assert len(spy.sent) == 1, "a route already excluded must not alert again"


def test_a_route_that_came_back_and_is_refused_again_alerts_again():
    """The de-duplication is per EXCLUSION, not per route for all time."""
    spy = _Spy()
    p = _proxy(spy)
    p._punish_429("kilo/a:free", 1000.0, _429(60), scope=CHAT)
    p._punish_429("kilo/a:free", 5000.0, _429(60), scope=CHAT)   # cooldown expired
    assert len(spy.sent) == 2


def test_the_alert_names_the_capability_that_was_lost():
    """An image limit does not take chat down, and the message must not imply it
    did -- that is the whole point of scoped cooldowns."""
    spy = _Spy()
    p = _proxy(spy)
    p._punish_429("kilo/a:free", 1000.0, _429(300), scope=IMAGES)
    assert "images" in spy.sent[0]
    assert "chat and images" not in spy.sent[0]

    spy2 = _Spy()
    p2 = _proxy(spy2)
    p2._punish_429("kilo/a:free", 1000.0, _429(300), scope=ALL_CAPABILITIES)
    assert "chat and images" in spy2.sent[0]


def test_the_alert_says_whether_the_return_time_is_theirs_or_ours():
    spy = _Spy()
    p = _proxy(spy)
    p._punish_429("kilo/a:free", 1000.0, _429(600), scope=CHAT)
    assert "the provider stated it" in spy.sent[0]

    spy2 = _Spy()
    p2 = _proxy(spy2)
    p2._punish_429("kilo/a:free", 1000.0, _429(None), scope=CHAT)
    assert "estimated by us" in spy2.sent[0]


# --- notifying must never be able to break serving


def test_a_notifier_that_explodes_does_not_break_the_cooldown():
    class _Broken(Notifier):
        enabled = True

        def notify(self, text):
            raise RuntimeError("telegram is down")

    p = _proxy(_Broken())
    p._punish_429("kilo/a:free", 1000.0, _429(300), scope=CHAT)
    # The punishment is what protects routing; the alert is commentary.
    assert p.cooldowns.until("kilo/a:free", CHAT) == 1300.0


def test_alerting_is_silent_and_harmless_when_unconfigured(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    n = from_env()
    assert n.enabled is False
    n.notify("nobody hears this")       # must not raise


def test_half_configured_counts_as_unconfigured(monkeypatch):
    """A token with nowhere to send it cannot alert anyone; starting up as if it
    could is how a monitoring channel ends up silently dead."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert from_env().enabled is False


def test_it_is_telegram_when_both_halves_are_present(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    n = from_env()
    assert isinstance(n, Telegram)
    assert n.enabled is True


async def test_telegram_posts_what_the_bot_api_expects():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await Telegram("TOKEN", "42", client=client)._send("hola")
    assert seen["url"] == "https://api.telegram.org/botTOKEN/sendMessage"
    import json
    body = json.loads(seen["body"])
    assert body["chat_id"] == "42"
    assert body["text"] == "hola"


async def test_a_telegram_outage_is_swallowed():
    def handler(request):
        raise httpx.ConnectError("no route to host")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await Telegram("TOKEN", "42", client=client)._send("hola")   # must not raise
