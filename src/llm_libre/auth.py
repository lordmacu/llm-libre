from collections import defaultdict


class PerKeyRateLimiter:
    """Simple in-memory sliding window. Assumes a single worker process."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str, now: float) -> bool:
        hits = [t for t in self._hits[key] if now - t < 60.0]
        if len(hits) >= self.per_minute:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True
