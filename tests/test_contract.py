import logging

from llm_libre.contract import (REQUIRED_CAPABILITIES, VERSION, Auth,
                                ProviderContract, parse_health)


def _caps(**overrides):
    caps = {k: False for k in REQUIRED_CAPABILITIES}
    caps.update(overrides)
    return caps


def _doc(**overrides):
    doc = {
        "status": "ok",
        "provider": "chatgpt",
        "version": "2.5.0",
        "contract": VERSION,
        "auth": {"mode": "account", "plan": "go",
                 "subscription_active": True,
                 "expires_at": "2026-09-06T00:28:46Z"},
        "capabilities": _caps(chat=True, streaming=True, vision=True, images=True),
    }
    doc.update(overrides)
    return doc


def test_a_compliant_document_is_parsed():
    c = parse_health("chatgpt", _doc())
    assert isinstance(c, ProviderContract)
    assert c.version == VERSION
    assert c.provider == "chatgpt"
    assert c.auth == Auth(mode="account", plan="go", subscription_active=True,
                          expires_at="2026-09-06T00:28:46Z")
    assert c.capabilities["images"] is True
    assert c.capabilities["translate"] is False


def test_a_document_without_the_contract_key_is_not_a_contract(caplog):
    # The pre-contract proxies. This is the NORMAL case during rollout, so it
    # must be silent: a warning per provider per sweep would train the operator
    # to ignore the log.
    doc = _doc()
    del doc["contract"]
    with caplog.at_level(logging.WARNING):
        assert parse_health("chatgpt", doc) is None
    assert caplog.records == []


def test_a_different_contract_version_is_refused_and_logged(caplog):
    with caplog.at_level(logging.WARNING):
        assert parse_health("chatgpt", _doc(contract=VERSION + 1)) is None
    assert "version" in caplog.text


def test_a_missing_capability_key_refuses_the_whole_document(caplog):
    caps = _caps(chat=True)
    del caps["translate"]
    with caplog.at_level(logging.WARNING):
        assert parse_health("chatgpt", _doc(capabilities=caps)) is None
    assert "translate" in caplog.text


def test_a_non_boolean_capability_refuses_the_whole_document(caplog):
    # chatgpt-proxy's pre-contract block carried English prose in these fields
    # ("automatic (override with web_search: true/false)"). A truthy string
    # would silently read as True, which is exactly the wrong direction.
    with caplog.at_level(logging.WARNING):
        assert parse_health("chatgpt", _doc(capabilities=_caps(search="automatic"))) is None
    assert "search" in caplog.text


def test_unknown_capability_keys_are_ignored():
    # How a new capability ships before the gateway learns to use it.
    c = parse_health("chatgpt", _doc(capabilities=_caps(chat=True, video=True)))
    assert set(c.capabilities) == set(REQUIRED_CAPABILITIES)


def test_a_malformed_auth_block_degrades_to_unknown_without_losing_capabilities():
    c = parse_health("chatgpt", _doc(auth="account"))
    assert c.auth.mode == "unknown"
    assert c.capabilities["chat"] is True


def test_an_unrecognised_auth_mode_degrades_to_unknown(caplog):
    with caplog.at_level(logging.WARNING):
        c = parse_health("chatgpt", _doc(auth={"mode": "subscriber"}))
    assert c.auth.mode == "unknown"
    assert "subscriber" in caplog.text


def test_a_body_that_is_not_an_object_is_not_a_contract():
    assert parse_health("chatgpt", ["ok"]) is None
    assert parse_health("chatgpt", None) is None
