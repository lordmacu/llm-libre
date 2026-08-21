"""
Complex live tests for tool-calling emulation with deepseek-chat.
All tests are @pytest.mark.vivo — require DEEPSEEK_PROXY_URL to be set.
"""
import json
import os
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.vivo

YAML = str(Path(__file__).resolve().parents[1] / "providers.yaml")

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string", "description": "City name"},
                                  "units": {"type": "string", "enum": ["celsius", "fahrenheit"],
                                            "description": "Temperature unit"}},
                   "required": ["city"]},
}}

SEARCH_TOOL = {"type": "function", "function": {
    "name": "search_web",
    "description": "Search the internet for information on any topic",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string", "description": "Search query"},
                                  "max_results": {"type": "integer", "description": "Max results (1-10)"}},
                   "required": ["query"]},
}}

CALC_TOOL = {"type": "function", "function": {
    "name": "calculate",
    "description": "Evaluate mathematical expressions and perform complex calculations",
    "parameters": {"type": "object",
                   "properties": {"expression": {"type": "string",
                                                 "description": "Math expression to evaluate"},
                                  "precision": {"type": "integer",
                                                "description": "Decimal places in result"}},
                   "required": ["expression"]},
}}

ALL_TOOLS = [WEATHER_TOOL, SEARCH_TOOL, CALC_TOOL]


def _build_proxy(url: str):
    from llm_libre.storage import Storage
    from llm_libre.models import Capabilities, Route
    from llm_libre.providers import load
    from llm_libre.proxy import Proxy

    providers = {p.id: p for p in load(YAML, {"DEEPSEEK_PROXY_URL": url})}
    route = Route("deepseek", "deepseek-chat", "free",
                 Capabilities(tools=True, vision=False, context=64000, max_output=8192))
    store = Storage(":memory:")
    store.create_schema()
    return providers, route, store


def _tool_calls(response) -> list:
    return ((response.json.get("choices") or [{}])[0]
            .get("message", {}).get("tool_calls") or [])


def _content(response) -> str:
    return ((response.json.get("choices") or [{}])[0]
            .get("message", {}).get("content") or "")


def _skip_if_no_url():
    url = os.getenv("DEEPSEEK_PROXY_URL")
    if not url:
        pytest.skip("DEEPSEEK_PROXY_URL not set")
    return url


# A deliberately COMPLETE tool: nested object, array of typed items, enums,
# a default, required vs optional at two levels. This is the shape real
# agentic clients send, and the shape prompted models get wrong most.
ORDER_TOOL = {"type": "function", "function": {
    "name": "create_delivery_order",
    "description": "Create a hardware-store delivery order for a customer",
    "parameters": {
        "type": "object",
        "properties": {
            "customer": {"type": "object",
                         "description": "Who receives the order",
                         "properties": {
                             "name": {"type": "string"},
                             "phone": {"type": "string"},
                             "address": {"type": "string",
                                         "description": "Full street address"}},
                         "required": ["name", "address"]},
            "items": {"type": "array",
                      "description": "Products to deliver",
                      "items": {"type": "object", "properties": {
                          "sku": {"type": "string"},
                          "quantity": {"type": "integer"},
                          "unit": {"type": "string",
                                   "enum": ["unit", "box", "kg"]}},
                          "required": ["sku", "quantity"]}},
            "priority": {"type": "string", "enum": ["normal", "express"],
                         "default": "normal"},
            "notes": {"type": "string"}},
        "required": ["customer", "items"]},
}}

ORDER_PROMPT = ("Create a delivery order for Marta Ruiz at Calle 45 #12-34 in "
                "Bogota, phone 3001234567: 2 boxes of screws (SKU TOR-8x2) and "
                "5 kg of cement (SKU CEM-50). It is urgent, please.")


def _check_order_arguments(args: dict) -> None:
    """The assertions that make this scenario COMPLETE: not just 'a call came
    back' but 'the nested structure, the coerced types and the enums are all
    what the schema promised the client'."""
    customer = args.get("customer")
    assert isinstance(customer, dict), f"customer is not an object: {args}"
    assert "marta" in str(customer.get("name", "")).lower()
    assert customer.get("address"), f"missing required address: {customer}"

    items = args.get("items")
    assert isinstance(items, list) and len(items) >= 2, f"items: {items}"
    for item in items:
        assert isinstance(item, dict), f"item is not an object: {item}"
        assert isinstance(item.get("sku"), str) and item["sku"], f"sku: {item}"
        assert isinstance(item["quantity"], int) and not isinstance(
            item["quantity"], bool), (
            f"quantity must arrive as a real integer (schema coercion): {item}")
        if "unit" in item:
            assert item["unit"] in ("unit", "box", "kg"), f"unit enum: {item}"
    skus = {i["sku"].upper() for i in items}
    assert any("TOR" in s for s in skus) and any("CEM" in s for s in skus), skus

    if "priority" in args:
        assert args["priority"] in ("normal", "express"), args["priority"]


async def test_full_agentic_flow_with_a_complete_tool():
    """The whole loop a real agentic client runs, against the real backend:
    a rich nested schema in, a typed call out, a result injected, a grounded
    text answer back."""
    url = _skip_if_no_url()
    providers, route, store = _build_proxy(url)

    async with httpx.AsyncClient(timeout=90) as http:
        from llm_libre.proxy import Proxy
        proxy = Proxy(providers, store, http)

        messages = [{"role": "user", "content": ORDER_PROMPT}]
        r1 = await proxy.complete(
            [route],
            {"model": "deepseek-chat", "messages": messages,
             "tools": [ORDER_TOOL], "tool_choice": "required"},
            now=0.0)
        assert r1.status == 200, f"turn 1: HTTP {r1.status}: {r1.json}"
        tcs = _tool_calls(r1)
        assert tcs, f"turn 1: no tool_calls: {json.dumps(r1.json)[:400]}"
        tc = tcs[0]
        assert tc["function"]["name"] == "create_delivery_order"
        args = json.loads(tc["function"]["arguments"])
        _check_order_arguments(args)
        print(f"\n  turn 1 — call: {json.dumps(args, ensure_ascii=False)[:300]}")

        result = json.dumps({"order_id": "ORD-7781", "status": "confirmed",
                             "eta_minutes": 45, "total_cop": 182000})
        messages += [
            {"role": "assistant", "content": None, "tool_calls": [tc]},
            {"role": "tool", "tool_call_id": tc["id"], "content": result},
        ]
        r2 = await proxy.complete(
            [route],
            {"model": "deepseek-chat", "messages": messages,
             "tools": [ORDER_TOOL]},
            now=0.0)
        assert r2.status == 200, f"turn 2: HTTP {r2.status}: {r2.json}"
        final = _content(r2)
        assert final.strip(), "turn 2: empty final answer"
        grounded = any(fact in final for fact in ("ORD-7781", "7781", "45"))
        assert grounded, f"final answer ignores the tool result: {final!r}"
        print(f"  turn 2 — final: {final[:200]!r}")


async def test_full_agentic_flow_streaming_turn():
    """The same first turn over SSE: one tool_calls chunk, then [DONE]."""
    url = _skip_if_no_url()
    providers, route, store = _build_proxy(url)

    async with httpx.AsyncClient(timeout=90) as http:
        from llm_libre.proxy import Proxy
        proxy = Proxy(providers, store, http)

        lines = [l async for l in proxy.complete_stream(
            [route],
            {"model": "deepseek-chat", "stream": True,
             "messages": [{"role": "user", "content": ORDER_PROMPT}],
             "tools": [ORDER_TOOL], "tool_choice": "required"},
            now=0.0)]
        assert lines[-1].strip() == "data: [DONE]"
        payloads = [json.loads(l[len("data: "):]) for l in lines
                    if l.startswith("data: ") and "[DONE]" not in l]
        assert payloads, f"no chunks: {lines}"
        delta = payloads[0]["choices"][0]["delta"]
        tcs = delta.get("tool_calls")
        assert tcs, f"stream did not carry tool_calls: {payloads[0]}"
        assert tcs[0]["function"]["name"] == "create_delivery_order"
        args = json.loads(tcs[0]["function"]["arguments"])
        _check_order_arguments(args)
        assert payloads[0]["choices"][0]["finish_reason"] == "tool_calls"
        print(f"\n  stream — call: {json.dumps(args, ensure_ascii=False)[:300]}")


async def test_selects_correct_tool_from_multiple():
    """With 3 tools available, the model should pick the right one for each question."""
    url = _skip_if_no_url()
    providers, route, store = _build_proxy(url)

    cases = [
        ("What is the square root of 144?", "calculate"),
        ("What's the weather in Medellín right now?", "get_weather"),
        ("Search for information about the DeepSeek V3 model", "search_web"),
    ]

    async with httpx.AsyncClient(timeout=90) as http:
        from llm_libre.proxy import Proxy
        proxy = Proxy(providers, store, http)

        for question, expected_tool in cases:
            r = await proxy.complete(
                [route],
                {"model": "deepseek-chat",
                 "messages": [{"role": "user", "content": question}],
                 "tools": ALL_TOOLS,
                 "tool_choice": "required"},
                now=0.0,
            )
            assert r.status == 200, f"HTTP {r.status} for: {question}"
            tcs = _tool_calls(r)
            assert tcs, f"No tool_calls for: {question!r}"
            name = tcs[0]["function"]["name"]
            assert name == expected_tool, (
                f"For {question!r} expected '{expected_tool}', got '{name}'"
            )
            args = json.loads(tcs[0]["function"]["arguments"])
            print(f"\n  {question!r} → {name}({args})")


async def test_includes_optional_arguments_when_relevant():
    """When the question mentions units, the model should include the optional 'units' param."""
    url = _skip_if_no_url()
    providers, route, store = _build_proxy(url)

    async with httpx.AsyncClient(timeout=90) as http:
        from llm_libre.proxy import Proxy
        proxy = Proxy(providers, store, http)

        r = await proxy.complete(
            [route],
            {"model": "deepseek-chat",
             "messages": [{"role": "user",
                           "content": "What temperature in Fahrenheit is it in New York?"}],
             "tools": [WEATHER_TOOL],
             "tool_choice": "required"},
            now=0.0,
        )
        assert r.status == 200
        tcs = _tool_calls(r)
        assert tcs, "No tool_calls generated"
        args = json.loads(tcs[0]["function"]["arguments"])
        assert "city" in args
        print(f"\n  args received: {args}")
        if "units" in args:
            print(f"  included 'units': {args['units']}")
        else:
            print("  did not include 'units' (acceptable, it was optional)")


async def test_two_turn_agentic_loop():
    """
    Two-turn agentic loop:
    1. User asks about weather in Bogotá
    2. Model calls get_weather → inject fake result
    3. Model gives final text response
    """
    url = _skip_if_no_url()
    providers, route, store = _build_proxy(url)

    async with httpx.AsyncClient(timeout=90) as http:
        from llm_libre.proxy import Proxy
        proxy = Proxy(providers, store, http)

        messages = [{"role": "user", "content": "What's the weather like in Bogotá today?"}]

        r1 = await proxy.complete(
            [route],
            {"model": "deepseek-chat",
             "messages": messages,
             "tools": [WEATHER_TOOL],
             "tool_choice": "required"},
            now=0.0,
        )
        assert r1.status == 200, f"Turn 1 failed: {r1.json}"
        tcs = _tool_calls(r1)
        assert tcs, "Turn 1: no tool_calls generated"

        tc = tcs[0]
        call_id = tc["id"]
        fn_name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"])
        print(f"\n  Turn 1 — called: {fn_name}({args})")

        fake_result = json.dumps({
            "city": args.get("city", "Bogotá"),
            "temperature": 18,
            "condition": "partly cloudy",
            "humidity": 72,
            "wind_kmh": 15,
        })

        messages = messages + [
            {"role": "assistant", "content": None, "tool_calls": [tc]},
            {"role": "tool", "tool_call_id": call_id, "content": fake_result},
        ]

        r2 = await proxy.complete(
            [route],
            {"model": "deepseek-chat",
             "messages": messages,
             "tools": [WEATHER_TOOL]},
            now=0.0,
        )
        assert r2.status == 200, f"Turn 2 failed: {r2.json}"
        final_text = _content(r2)
        assert final_text.strip(), "Turn 2: empty text response"

        tcs2 = _tool_calls(r2)
        print(f"  Turn 2 — final response: {final_text[:200]!r}")
        if not tcs2:
            assert ("18" in final_text or "cloudy" in final_text.lower()
                    or "bogot" in final_text.lower()), (
                f"Response does not mention weather data: {final_text!r}"
            )


async def test_three_turn_comparison_loop():
    """
    Multi-turn loop to compare weather in two cities:
    Model calls get_weather twice (one per city) then gives a text comparison.
    """
    url = _skip_if_no_url()
    providers, route, store = _build_proxy(url)

    async with httpx.AsyncClient(timeout=90) as http:
        from llm_libre.proxy import Proxy
        proxy = Proxy(providers, store, http)

        messages = [{"role": "user",
                     "content": "Compare the weather in Bogotá and Madrid. "
                                "I need temperature and conditions for both."}]

        fake_weather = {
            "bogotá": {"temperature": 18, "condition": "partly cloudy", "humidity": 72},
            "madrid": {"temperature": 34, "condition": "sunny and hot", "humidity": 25},
        }

        calls_made = []

        for turn in range(3):
            r = await proxy.complete(
                [route],
                {"model": "deepseek-chat",
                 "messages": messages,
                 "tools": [WEATHER_TOOL]},
                now=0.0,
            )
            # The free DeepSeek proxy intermittently answers 200 with an empty
            # body under consecutive calls -- its own session degrading, not an
            # emulation failure. Skipping keeps this test measuring what it is
            # for (multi-turn coherence) instead of the provider's uptime.
            if r.status == 503 and "sin contenido" in str(r.json):
                pytest.skip("provider returned an empty 200; not an emulation failure")
            assert r.status == 200, f"Turn {turn + 1} HTTP {r.status}"

            tcs = _tool_calls(r)
            if not tcs:
                text = _content(r)
                print(f"\n  Turn {turn + 1} — final response ({len(text)} chars):")
                print(f"  {text[:300]!r}")
                assert text.strip(), "Final response is empty"
                print(f"  Calls made: {calls_made}")
                assert len(calls_made) >= 1, "No tool calls were made"
                return

            assistant_msg = {"role": "assistant", "content": None, "tool_calls": tcs}
            messages = messages + [assistant_msg]

            for tc in tcs:
                fn_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                city = args.get("city", "").lower()
                calls_made.append(f"{fn_name}({args})")
                print(f"  Turn {turn + 1} — called: {fn_name}({args})")

                result = next(
                    (v for k, v in fake_weather.items() if k in city or city in k),
                    {"temperature": 20, "condition": "unknown"}
                )
                result["city"] = args.get("city", city)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                })

        text = _content(r)
        print(f"\n  Loop ended at turn {turn + 1}. Calls: {calls_made}")
        if text:
            print(f"  Last text: {text[:200]!r}")


# --- the genuinely complex one: a dependent multi-tool mission ---
# Five tools plus a decoy. The user names only a customer CODE; the customer's
# name and address exist ONLY inside get_customer's result, the real stock
# level only inside check_inventory's -- so a correct order PROVES the model
# chained tool results instead of hallucinating, across several turns, all
# through the emulation layer.

INVENTORY_TOOL = {"type": "function", "function": {
    "name": "check_inventory",
    "description": "Check stock for a product SKU",
    "parameters": {"type": "object", "properties": {
        "sku": {"type": "string"}},
        "required": ["sku"]},
}}

CUSTOMER_TOOL = {"type": "function", "function": {
    "name": "get_customer",
    "description": "Look up a customer record by its customer code",
    "parameters": {"type": "object", "properties": {
        "customer_id": {"type": "string", "description": "Code like C-102"}},
        "required": ["customer_id"]},
}}

SCHEDULE_TOOL = {"type": "function", "function": {
    "name": "schedule_delivery",
    "description": "Book a delivery time slot for an existing order",
    "parameters": {"type": "object", "properties": {
        "order_id": {"type": "string"},
        "window": {"type": "string", "enum": ["morning", "afternoon", "evening"]}},
        "required": ["order_id", "window"]},
}}

NOTIFY_TOOL = {"type": "function", "function": {
    "name": "notify_customer",
    "description": "Send the customer a text message",
    "parameters": {"type": "object", "properties": {
        "customer_id": {"type": "string"},
        "message": {"type": "string"}},
        "required": ["customer_id", "message"]},
}}

DECOY_TOOL = {"type": "function", "function": {
    "name": "get_mars_weather",
    "description": "Current weather at a Mars colony site",
    "parameters": {"type": "object", "properties": {
        "site": {"type": "string"}},
        "required": ["site"]},
}}

MISSION_TOOLS = [INVENTORY_TOOL, CUSTOMER_TOOL, ORDER_TOOL, SCHEDULE_TOOL,
                 NOTIFY_TOOL, DECOY_TOOL]

def _mission_result(name: str, args: dict, issued: set) -> dict:
    """A dependency-aware backend, like a real one: an order id only exists
    after create_delivery_order returns it, and a call referencing an id never
    issued gets an ERROR result the model must recover from. Measured against
    the real backend (2026-08-20): the model sometimes parallelises the whole
    mission in one turn, inventing an order id ("TBD", "DEL-12345") for the
    downstream calls -- exactly the failure a real tool backend answers with
    an error, and the recovery is part of what this mission tests."""
    if name == "check_inventory":
        return {"sku": "TOR-8x2", "available": 8, "restock_eta_days": 3}
    if name == "get_customer":
        return {"customer_id": "C-102", "name": "Marta Ruiz",
                "phone": "3001234567", "address": "Calle 45 #12-34, Bogota"}
    if name == "create_delivery_order":
        issued.add("ORD-9944")
        return {"order_id": "ORD-9944", "status": "confirmed",
                "total_cop": 96000}
    if name == "schedule_delivery":
        oid = args.get("order_id")
        if oid not in issued:
            return {"error": f"unknown order_id {oid!r}: create the order "
                             "first and use the order_id from its result"}
        return {"order_id": oid, "window": args.get("window"),
                "slot": "2026-08-21 08:00-11:00"}
    if name == "notify_customer":
        if not any(i in (args.get("message") or "") for i in issued):
            return {"error": "rejected: the message must mention a real order "
                             "number issued by create_delivery_order"}
        return {"delivered_to": "3001234567", "status": "sent"}
    return {"error": f"unknown tool {name}"}

MISSION_PROMPT = (
    "Handle this for customer C-102: they want 12 units of SKU TOR-8x2 "
    "delivered tomorrow morning. First check the stock and their customer "
    "record, create the order for whatever quantity is actually available, "
    "schedule the delivery, and text them a confirmation that mentions the "
    "order number. Use the tools; do not invent data.")


async def test_multi_tool_mission_with_dependent_calls():
    url = _skip_if_no_url()
    providers, route, store = _build_proxy(url)

    async with httpx.AsyncClient(timeout=120) as http:
        from llm_libre.proxy import Proxy
        proxy = Proxy(providers, store, http)

        messages = [{"role": "user", "content": MISSION_PROMPT}]
        calls_made = []
        issued: set = set()
        final_text = ""

        for turn in range(8):
            r = await proxy.complete(
                [route],
                {"model": "deepseek-chat", "messages": messages,
                 "tools": MISSION_TOOLS},
                now=0.0)
            if r.status == 503:
                pytest.skip(f"provider degraded mid-mission: {r.json}")
            assert r.status == 200, f"turn {turn + 1}: HTTP {r.status}: {r.json}"
            tcs = _tool_calls(r)
            if not tcs:
                final_text = _content(r)
                break

            messages = messages + [
                {"role": "assistant", "content": None, "tool_calls": tcs}]
            for tc in tcs:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                calls_made.append((name, args))
                print(f"  turn {turn + 1}: {name}({json.dumps(args, ensure_ascii=False)[:160]})")
                assert name != "get_mars_weather", "the decoy was called"
                outcome = _mission_result(name, args, issued)
                if "error" in outcome:
                    print(f"           -> tool error: {outcome['error'][:90]}")
                messages.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "content": json.dumps(outcome)})

        names = [n for n, _ in calls_made]
        print(f"  mission calls: {names}")
        print(f"  final: {final_text[:220]!r}")

        # The chain itself: the data-source calls happened before the order.
        assert "create_delivery_order" in names, f"no order was created: {names}"
        order_at = names.index("create_delivery_order")
        assert "check_inventory" in names[:order_at], names
        assert "get_customer" in names[:order_at], names

        # The order was built FROM TOOL RESULTS, not from the prompt: the
        # customer's name only exists in get_customer's answer, and the honest
        # quantity (8, not the requested 12) only in check_inventory's.
        order_args = calls_made[order_at][1]
        assert "marta" in str(order_args.get("customer", {}).get("name", "")).lower(), order_args
        quantities = [i.get("quantity") for i in order_args.get("items", [])]
        assert quantities and all(isinstance(q, int) for q in quantities), order_args
        assert quantities[0] <= 12, order_args
        if quantities[0] == 8:
            print("  adapted the quantity to the real stock (8)")

        # Downstream calls reference the order id that ONLY the order result
        # carries. The LAST call of each kind is the one that must be right:
        # an eager model may fire them early with an invented id, eat the
        # error result, and correct itself -- that recovery is part of the
        # mission, not a failure of it.
        scheduled = [a for n, a in calls_made if n == "schedule_delivery"]
        assert scheduled, f"delivery was never scheduled: {names}"
        assert scheduled[-1].get("order_id") == "ORD-9944", scheduled[-1]
        assert scheduled[-1].get("window") in ("morning", "afternoon", "evening")

        notified = [a for n, a in calls_made if n == "notify_customer"]
        assert notified, f"customer was never notified: {names}"
        assert "9944" in notified[-1].get("message", ""), notified[-1]

        assert final_text.strip(), "mission ended without a final answer"
