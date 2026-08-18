from llm_libre.models import Capabilities, Metrics, Route, RouteRequest
from llm_libre.router import order_routes


def r(model, provider="kilo", tier="free", tools=True, vision=False, context=100000):
    return Route(provider, model, tier,
                 Capabilities(tools=tools, vision=vision, context=context, max_output=4096))


def m(quality=0.8, reliability=0.9, ttft=500, cooldown=0.0, measured_at=1000.0):
    return Metrics(quality, reliability, ttft, cooldown, measured_at)


def test_routes_without_tool_support_are_dropped_when_tools_are_asked_for():
    routes = [r("con:free", tools=True), r("sin:free", tools=False)]
    out = order_routes(routes, {}, RouteRequest(needs_tools=True), now=0.0)
    assert [x.model_id for x in out] == ["con:free"]


def test_routes_without_vision_are_dropped_when_vision_is_asked_for():
    routes = [r("ve:free", vision=True), r("ciego:free", vision=False)]
    out = order_routes(routes, {}, RouteRequest(needs_vision=True), now=0.0)
    assert [x.model_id for x in out] == ["ve:free"]


def test_routes_with_insufficient_context_are_dropped():
    routes = [r("grande:free", context=200000), r("chico:free", context=8000)]
    out = order_routes(routes, {}, RouteRequest(min_context=100000), now=0.0)
    assert [x.model_id for x in out] == ["grande:free"]


def test_it_orders_by_descending_score():
    routes = [r("malo:free"), r("bueno:free")]
    metrics = {"kilo/malo:free": m(quality=0.3), "kilo/bueno:free": m(quality=0.95)}
    out = order_routes(routes, metrics, RouteRequest(), now=0.0)
    assert [x.model_id for x in out] == ["bueno:free", "malo:free"]


def test_paid_routes_always_go_last_even_when_they_score_better():
    routes = [r("MiniMax-M3", provider="minimax", tier="paid"), r("flojo:free")]
    metrics = {"minimax/MiniMax-M3": m(quality=1.0, reliability=1.0, ttft=100),
               "kilo/flojo:free": m(quality=0.2, reliability=0.3, ttft=5000)}
    out = order_routes(routes, metrics, RouteRequest(), now=0.0)
    assert [x.tier for x in out] == ["free", "paid"]


def test_allow_paid_false_removes_the_paid_routes():
    routes = [r("MiniMax-M3", provider="minimax", tier="paid"), r("g:free")]
    out = order_routes(routes, {}, RouteRequest(allow_paid=False), now=0.0)
    assert [x.tier for x in out] == ["free"]


def test_it_excludes_routes_in_cooldown_but_not_expired_ones():
    routes = [r("castigada:free"), r("vencida:free")]
    metrics = {"kilo/castigada:free": m(cooldown=500.0),
               "kilo/vencida:free": m(cooldown=50.0)}
    out = order_routes(routes, metrics, RouteRequest(), now=100.0)
    assert [x.model_id for x in out] == ["vencida:free"]


def test_an_explicit_model_filters_but_keeps_both_providers():
    routes = [r("comun:free", provider="kilo"),
              r("comun:free", provider="openrouter"),
              r("otro:free", provider="kilo")]
    out = order_routes(routes, {}, RouteRequest(model="comun:free"), now=0.0)
    assert len(out) == 2
    assert {x.provider for x in out} == {"kilo", "openrouter"}


def test_no_candidates_returns_an_empty_list():
    out = order_routes([r("sin:free", tools=False)], {},
                       RouteRequest(needs_tools=True), now=0.0)
    assert out == []


def test_a_route_without_metrics_uses_the_neutral_ones_not_zeros():
    # kilo/conocida:free scores 0.3*0.5*latency_factor(1500) = 0.075 on balanced.
    # kilo/nueva:free (no metrics) scores with NEUTRAL_METRICS (0.6, 0.8, 1500):
    # 0.6*0.8*latency_factor(1500) = 0.24, so it beats the known one and comes first.
    # If the fallback were Metrics(0,0,0,0) it would score 0 and land LAST: the
    # order would invert and this assert would fail, which is exactly what this
    # test must detect.
    #
    # `measured_at=None` on the known route is deliberate (fix round 3, B2b): both
    # routes end up equally "unmeasured" so that what decides the order is the
    # score, which is the only thing this test wants to protect. The
    # measured-before-assumed criterion has its own tests, below.
    routes = [r("conocida:free"), r("nueva:free")]
    metrics = {"kilo/conocida:free": m(quality=0.3, reliability=0.5, ttft=1500,
                                       measured_at=None)}
    out = order_routes(routes, metrics, RouteRequest(), now=0.0)
    assert [x.model_id for x in out] == ["nueva:free", "conocida:free"]


# --- Fix round 3, B2 (Blocking), half (b): a route that never went through the
#     quality battery carries a neutral assumption (0.6), not a measurement. It
#     cannot be preferred over a route whose quality WAS measured: that is what
#     kept a freshly appeared model -- fast and unevaluated -- at the top for up
#     to 25h. It has to stay reachable, though, or it would never be measured. ---

def test_a_never_probed_route_goes_after_one_with_measured_quality():
    routes = [r("nueva:free"), r("medida:free")]
    metrics = {
        # The measured one scores WORSE on balanced (0.35*0.9*f(500) = 0.24) than
        # the new one with the neutrals (0.6*0.9*f(200) = 0.48): if the order were
        # by score alone, the new one would win. It must lose anyway.
        "kilo/medida:free": m(quality=0.35, ttft=500, measured_at=1000.0),
        "kilo/nueva:free": m(quality=0.6, ttft=200, measured_at=None),
    }
    out = order_routes(routes, metrics, RouteRequest(), now=0.0)
    assert [x.model_id for x in out] == ["medida:free", "nueva:free"]


def test_a_never_probed_route_stays_in_the_chain_so_it_can_be_measured():
    routes = [r("nueva:free"), r("medida:free")]
    metrics = {"kilo/medida:free": m(quality=0.9, measured_at=1000.0)}
    out = order_routes(routes, metrics, RouteRequest(), now=0.0)
    assert "nueva:free" in [x.model_id for x in out]


def test_between_two_never_probed_routes_the_score_still_decides():
    routes = [r("lenta:free"), r("rapida:free")]
    metrics = {"kilo/lenta:free": m(ttft=5000, measured_at=None),
               "kilo/rapida:free": m(ttft=100, measured_at=None)}
    out = order_routes(routes, metrics, RouteRequest(), now=0.0)
    assert [x.model_id for x in out] == ["rapida:free", "lenta:free"]


# --- Task 13: `priority`, a concept DISTINCT from `tier` and from `profile`. ---

def _rp(model, priority, provider="kilo", tier="free", tools=True):
    return Route(provider, model, tier,
                 Capabilities(tools=tools, vision=False, context=100000, max_output=4096),
                 priority=priority)


def test_priority_orders_within_the_same_tier_above_the_score():
    # gpt-5 (priority 0) scores WORSE than the usual free route (priority 1, the
    # default) and still has to beat it: priority decides before the score within
    # the same tier.
    routes = [_rp("chatgpt:free", 0, provider="chatgpt"), _rp("normal:free", 1)]
    metrics = {"chatgpt/chatgpt:free": m(quality=0.3, reliability=0.3, ttft=3000),
               "kilo/normal:free": m(quality=0.99, reliability=0.99, ttft=50)}
    out = order_routes(routes, metrics, RouteRequest(), now=0.0)
    assert [x.model_id for x in out] == ["chatgpt:free", "normal:free"]


def test_priority_does_not_break_the_paid_routes_go_last_invariant():
    # THE CASE THAT MATTERS: a PAID route with priority 0 (the highest possible)
    # and a perfect score against a mediocre free route with the default priority.
    # Money is the reason: paid goes last ALWAYS, priority cannot buy that place.
    routes = [_rp("MiniMax-M3", 0, provider="minimax", tier="paid"),
              _rp("mediocre:free", 100)]
    metrics = {"minimax/MiniMax-M3": m(quality=1.0, reliability=1.0, ttft=50),
               "kilo/mediocre:free": m(quality=0.2, reliability=0.3, ttft=5000)}
    out = order_routes(routes, metrics, RouteRequest(), now=0.0)
    assert [x.tier for x in out] == ["free", "paid"]
    assert [x.model_id for x in out] == ["mediocre:free", "MiniMax-M3"]


# --- Finding 2 of the Task 13 review: `priority` had no escape hatch for a
#     persistently broken route -- with chatgpt at priority 0, order_routes() kept
#     putting it first ALWAYS, without looking at its health, making the cooldown
#     (round 8: triggered by a 429 immediately or by a PROBE confirming the route
#     is broken, see proxy.Proxy._suspect) the ONLY way out of that trap: it is
#     filtered BEFORE priority matters. ---

# --- Finding 5 of the review: `priority` and "never measured" could be swapped
#     in the sort key without any existing test noticing -- precisely the rung
#     that decides the real rollout: on a fresh deploy, ALL of chatgpt's routes
#     start unmeasured while Kilo already carries measurements from the
#     production database. If the order were (unmeasured, priority) instead of
#     (priority, unmeasured), Kilo (measured) would beat chatgpt (priority 0,
#     unmeasured) on deploy day -- the opposite of what priority:0 promises. ---

def test_priority_decides_before_measured_versus_unmeasured():
    # chatgpt: priority 0 (the best) but NEVER MEASURED (quality_measured_at
    # None, the real state on deploy day). kilo: priority 1 (worse) but WITH a
    # real measurement -- the real state of a production database already
    # running. If the order were (unmeasured, priority), kilo would win.
    routes = [_rp("chatgpt:free", 0, provider="chatgpt"), _rp("normal:free", 1)]
    metrics = {
        "chatgpt/chatgpt:free": m(measured_at=None),
        "kilo/normal:free": m(measured_at=1000.0),
    }
    out = order_routes(routes, metrics, RouteRequest(), now=0.0)
    assert [x.model_id for x in out] == ["chatgpt:free", "normal:free"]


def test_a_route_in_cooldown_is_skipped_even_with_the_highest_priority():
    routes = [_rp("chatgpt:free", 0, provider="chatgpt"), _rp("normal:free", 1)]
    metrics = {
        # chatgpt has the best priority AND the best score -- and still cannot
        # win while it is in cooldown.
        "chatgpt/chatgpt:free": m(quality=0.99, reliability=0.99, ttft=50,
                                  cooldown=500.0),
        "kilo/normal:free": m(quality=0.2, reliability=0.3, ttft=5000),
    }
    out = order_routes(routes, metrics, RouteRequest(), now=100.0)
    assert [x.model_id for x in out] == ["normal:free"]


def test_an_explicit_model_is_served_even_when_it_is_not_the_highest_priority():
    # "Choosing a model by hand already works and must keep working" (brief,
    # point 4): asking for a real id bypasses the ordering entirely, priority
    # included.
    routes = [_rp("prioritario:free", 0, provider="chatgpt"),
              _rp("elegido-a-mano:free", 1)]
    out = order_routes(routes, {}, RouteRequest(model="elegido-a-mano:free"), now=0.0)
    assert [x.model_id for x in out] == ["elegido-a-mano:free"]


# --- Task 13 follow-up: chatgpt became a DISCOVERED catalogue with declared
#     capabilities (tools:false ALWAYS). This pins the behaviour the generic
#     tools/explicit-id filter already covered, using a route with the real SHAPE
#     of a chatgpt route (priority 0, tools=False). ---

def test_a_request_with_tools_never_routes_to_chatgpt():
    chatgpt = _rp("gpt-5-3-mini", 0, provider="chatgpt", tools=False)
    kilo = _rp("con-tools:free", 1, tools=True)
    out = order_routes([chatgpt, kilo], {}, RouteRequest(needs_tools=True), now=0.0)
    assert [x.model_id for x in out] == ["con-tools:free"]


def test_a_request_without_tools_prefers_chatgpt_for_its_priority():
    # The flip side: without requiring tools, chatgpt still goes first despite
    # tools=False, because nobody is asking for them.
    chatgpt = _rp("gpt-5-3-mini", 0, provider="chatgpt", tools=False)
    kilo = _rp("normal:free", 1, tools=True)
    out = order_routes([chatgpt, kilo], {}, RouteRequest(), now=0.0)
    assert [x.model_id for x in out] == ["gpt-5-3-mini", "normal:free"]


def test_a_hand_picked_chatgpt_id_still_routes_straight_through():
    chatgpt = _rp("gpt-5-3-mini", 0, provider="chatgpt", tools=False)
    kilo = _rp("otro:free", 1)
    out = order_routes([chatgpt, kilo], {}, RouteRequest(model="gpt-5-3-mini"), now=0.0)
    assert [x.model_id for x in out] == ["gpt-5-3-mini"]


def test_a_genuinely_discovered_chatgpt_route_routes_straight_when_asked_for():
    # Integrates catalog.normalize (where chatgpt DISCOVERS its ids, with
    # default_capabilities) with router.order_routes: it exercises the whole
    # path, not hand-built routes.
    from llm_libre.catalog import normalize
    defaults = Capabilities(tools=False, vision=False, context=128000, max_output=8192)
    discovered = normalize(
        "chatgpt",
        {"data": [{"id": "gpt-5-3-mini", "description": "GPT-5.3 Mini"},
                  {"id": "gpt-5-5", "description": "GPT-5.5"}]},
        priority=0, default_capabilities=defaults)
    out = order_routes(discovered, {}, RouteRequest(model="gpt-5-3-mini"), now=0.0)
    assert [x.model_id for x in out] == ["gpt-5-3-mini"]
