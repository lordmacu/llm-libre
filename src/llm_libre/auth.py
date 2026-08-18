import time

# How many tracked buckets a limiter will hold. Reached only by traffic from
# many distinct sources, which is exactly when the dict itself becomes the
# problem: without a cap, remembering every IP that ever knocked IS the memory
# exhaustion the limiter exists to prevent. On overflow the oldest-idle buckets
# are dropped -- those callers get a fresh allowance, which is the right way to
# fail: permissive about who is counted, never permissive about how much any one
# of them may do.
MAX_TRACKED = 10_000


class RateLimiter:
    """Sliding window over the last 60 seconds, keyed by whatever you pass in.

    Named for what it does rather than for what it counts, because it now counts
    two different things: API keys (60/min by default) and client IPs on the
    paths that have no key to count. Assumes a single worker process, which the
    Dockerfile pins (`--workers 1`); with more, each would keep its own window
    and the effective limit would multiply.
    """

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if len(self._hits) > MAX_TRACKED:
            self._evict(now)
        hits = [t for t in self._hits.get(key, ()) if now - t < 60.0]
        if len(hits) >= self.per_minute:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True

    def _evict(self, now: float) -> None:
        # Drop every bucket with nothing left inside its window. That alone is
        # usually enough; if it is not (all buckets active), the dict is left as
        # it is rather than evicting someone who is currently being limited --
        # dropping an active bucket would hand a heavy caller a clean slate.
        self._hits = {k: v for k, v in self._hits.items()
                      if any(now - t < 60.0 for t in v)}



def client_ip(request) -> str:
    """Best available identity for an unauthenticated caller.

    In production every external request arrives through a Cloudflare tunnel, so
    the socket address is always 127.0.0.1 and would make one shared bucket for
    the whole internet. `CF-Connecting-IP` is what Cloudflare puts the real one
    in, and it is trustworthy HERE specifically because the origin is not
    reachable except through that tunnel -- a header nobody can inject is a
    header worth reading.

    Falls back to `X-Forwarded-For`'s first entry and then to the socket. The
    fallbacks ARE spoofable by anything with direct access to the port, which is
    accepted: a caller that can reach the port directly is already inside, and
    the limit is a guard against strangers on the internet, not a security
    boundary against the host itself.
    """
    headers = request.headers
    ip = headers.get("cf-connecting-ip")
    if not ip:
        forwarded = headers.get("x-forwarded-for", "")
        ip = forwarded.split(",")[0].strip()
    if not ip:
        ip = getattr(getattr(request, "client", None), "host", "") or "unknown"
    return ip
