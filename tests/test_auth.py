from llm_libre.auth import PerKeyRateLimiter


def test_allows_up_to_the_limit():
    limiter = PerKeyRateLimiter(per_minute=3)
    assert [limiter.allow("k", 0.0) for _ in range(4)] == [True, True, True, False]


def test_the_window_frees_up_after_a_minute():
    limiter = PerKeyRateLimiter(per_minute=1)
    assert limiter.allow("k", 0.0) is True
    assert limiter.allow("k", 30.0) is False
    assert limiter.allow("k", 61.0) is True


def test_each_key_has_its_own_quota():
    limiter = PerKeyRateLimiter(per_minute=1)
    assert limiter.allow("a", 0.0) is True
    assert limiter.allow("b", 0.0) is True
