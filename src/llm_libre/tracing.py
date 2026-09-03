"""One id per client request, followed from the caller down to the provider.

The `events` table already groups a failover chain under a `request_id`, but it
mints that id here, so it can only ever answer questions about the inside of
this gateway. The number an operator actually starts from is the one the CALLER
measured -- "the app waited 33 seconds" -- and without a shared id there is no
way to split that into the hop that classified the request, the routing, and
the provider that took 30 of those seconds.

So the id is the caller's when the caller brought one, and it travels onward on
the provider request, which lets the proxy behind this gateway log the same
string. Three logs, one grep.

The id is echoed into a response header and into log lines, which makes it
attacker-controlled text: [sanitise] is what keeps a newline or a kilobyte of
junk from being reflected into either.
"""
import contextvars
import logging
import re
import uuid

log = logging.getLogger("llm_libre.tracing")

#: The header the id travels on, in and out. Spelled the way every reverse
#: proxy already spells it, so an id set upstream of this gateway is picked up
#: rather than replaced.
HEADER = "X-Request-Id"

#: Printable, no whitespace, short. Wide enough for a uuid4 hex, a cuid, or an
#: app-side "t-1a2b3c" -- narrow enough that nothing in it can forge a log line.
_SAFE = re.compile(r"\A[A-Za-z0-9_.:-]{1,64}\Z")

_current: contextvars.ContextVar[str] = contextvars.ContextVar(
    "llm_libre_request_id", default="")


def new_id() -> str:
    """A fresh id. Prefixed so a glance at a log says who minted it: `gw-` means
    the caller brought none, which is itself worth knowing when a trace has a
    hole in it."""
    return "gw-" + uuid.uuid4().hex[:16]


def sanitise(raw: str | None) -> str:
    """The caller's id if it is usable, a fresh one otherwise.

    Never raises and never returns empty: every request gets an id, so no code
    path downstream has to handle the absence of one.
    """
    if raw and _SAFE.match(raw):
        return raw
    return new_id()


def current() -> str:
    """The id of the request being served on this task, or empty outside one
    (the probing scheduler, a test calling into the proxy directly)."""
    return _current.get()


def bind(value: str):
    """Set the id for this task. Returns the token to reset with, so a server
    that reuses tasks cannot leak one request's id into the next."""
    return _current.set(value)


def reset(token) -> None:
    _current.reset(token)
