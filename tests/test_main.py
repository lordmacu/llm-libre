import os

import pytest

# `llm_libre.main` runs `state = build_state()` (real I/O: it reads
# proveedores.yaml and opens the database) as soon as it is imported -- that is
# what lets `uvicorn llm_libre.main:app` work as an entrypoint. So that FIRST
# import does not depend on the ambient environment of whoever runs the suite
# (who may or may not have a .env loaded), a minimal environment that touches no
# disk is set BEFORE the import: RUTA_DB=":memory:" avoids the production default
# (/datos/llm-libre.sqlite3, not writable outside Docker) and any key at all
# avoids the very fail-fast this file tests below.
os.environ["RUTA_DB"] = ":memory:"
os.environ.setdefault("LLM_LIBRE_API_KEYS", "startup-key-for-tests")

from llm_libre import main  # noqa: E402


def test_build_state_without_keys_fails_loudly(monkeypatch):
    """Without LLM_LIBRE_API_KEYS the process must not start silently: better an
    immediate, explicit failure than a container that looks healthy (/health) and
    rejects 100% of requests with a 401."""
    monkeypatch.delenv("LLM_LIBRE_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="LLM_LIBRE_API_KEYS"):
        main.build_state()


def test_build_state_with_empty_or_comma_only_keys_also_fails(monkeypatch):
    # ",, ,"  -> after stripping each piece, no usable key: the same case as the
    # variable being absent, not a different one.
    monkeypatch.setenv("LLM_LIBRE_API_KEYS", " , ,  ,")
    with pytest.raises(RuntimeError, match="LLM_LIBRE_API_KEYS"):
        main.build_state()


def test_build_state_with_configured_keys_works(monkeypatch):
    monkeypatch.setenv("LLM_LIBRE_API_KEYS", "one-key, another-key")
    state = main.build_state()
    assert state.api_keys == {"one-key", "another-key"}
    assert state.providers  # it loaded the repo's real YAML
    assert state.store is not None
    assert state.proxy is not None
