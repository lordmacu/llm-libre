"""Tests for the installer's pure logic.

Nothing here starts a container or touches the network. What is tested is the
part where an installer silently does the wrong thing: parsing what the user
picked, and editing a .env that holds credentials.
"""
import pytest

from llm_libre.installer import (PROVIDERS, host_gateway_url, parse_selection,
                                 read_env, write_env)


# ── Selection parsing ────────────────────────────────────────────────────────

def test_all_selects_everything():
    assert parse_selection("all", 5) == [1, 2, 3, 4, 5]
    assert parse_selection("todos", 5) == [1, 2, 3, 4, 5]


def test_commas_and_spaces_both_work():
    assert parse_selection("1,3", 5) == [1, 3]
    assert parse_selection("1 3", 5) == [1, 3]
    assert parse_selection(" 3 , 1 ", 5) == [1, 3]


def test_ranges_work():
    assert parse_selection("2-4", 5) == [2, 3, 4]
    assert parse_selection("1,3-5", 5) == [1, 3, 4, 5]


def test_duplicates_collapse():
    assert parse_selection("2,2,2", 5) == [2]


def test_out_of_range_numbers_are_dropped_not_installed():
    """Installing provider 9 of 5 would be worse than ignoring it."""
    assert parse_selection("9", 5) == []
    assert parse_selection("0", 5) == []
    assert parse_selection("1,9", 5) == [1]


@pytest.mark.parametrize("raw", ["", "   ", "abc", "1,abc", "3-1", "-", None])
def test_nonsense_selects_nothing_so_the_prompt_repeats(raw):
    assert parse_selection(raw, 5) == []


# ── .env editing ─────────────────────────────────────────────────────────────

def test_writing_preserves_comments_and_unrelated_keys(tmp_path):
    """This file holds credentials a user may have written by hand. Rewriting
    it wholesale would delete them without saying so."""
    env = tmp_path / ".env"
    env.write_text("# my notes\nKEEP=yes\nCHANGE=old\n")
    write_env(env, {"CHANGE": "new", "ADD": "1"})
    text = env.read_text()
    assert "# my notes" in text
    assert "KEEP=yes" in text
    assert "CHANGE=new" in text and "CHANGE=old" not in text
    assert "ADD=1" in text


def test_writing_creates_the_file_when_absent(tmp_path):
    env = tmp_path / ".env"
    write_env(env, {"A": "1"})
    assert read_env(env) == {"A": "1"}


def test_a_credentials_file_is_not_world_readable(tmp_path):
    env = tmp_path / ".env"
    write_env(env, {"SECRET": "hunter2"})
    assert oct(env.stat().st_mode)[-3:] == "600"


def test_reading_ignores_comments_and_blanks(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# c\n\nA=1\n  B = 2  \nnot-an-assignment\n")
    assert read_env(env) == {"A": "1", "B": "2"}


def test_reading_a_missing_file_is_empty_not_an_error(tmp_path):
    assert read_env(tmp_path / "nope") == {}


# ── The provider registry ────────────────────────────────────────────────────

def test_every_provider_has_a_unique_key_and_port():
    keys = [p.key for p in PROVIDERS]
    ports = [p.port for p in PROVIDERS]
    assert len(set(keys)) == len(keys)
    assert len(set(ports)) == len(ports)


def test_the_url_variable_matches_what_providers_yaml_reads():
    """The installer writes these; the gateway reads them. If the two names
    drift, the install "succeeds" and the gateway sees no providers."""
    import yaml

    declared = {p["id"]: p.get("base_url_env") for p in
                yaml.safe_load(open("providers.yaml"))["providers"]}
    for provider in PROVIDERS:
        expected = declared.get(provider.key)
        if expected:
            assert provider.url_env == expected, provider.key


def test_every_provider_offers_at_least_one_way_in():
    for provider in PROVIDERS:
        assert provider.modes, provider.key
        for mode in provider.modes:
            if mode.key == "anonymous":
                assert mode.prompts == ()
            else:
                assert mode.prompts, f"{provider.key}/{mode.key} asks for nothing"


def test_only_mistral_offers_a_free_mode():
    """Measured this session: with every credential stripped, only Mistral
    answered. Offering a free mode elsewhere would promise what fails."""
    free = {p.key for p in PROVIDERS if any(m.key == "anonymous" for m in p.modes)}
    assert free == {"mistral"}


def test_secrets_are_marked_so_they_are_never_echoed():
    for provider in PROVIDERS:
        for mode in provider.modes:
            for env_var, _question, secret in mode.prompts:
                if "PASSWORD" in env_var or "TOKEN" in env_var or "SESSION" in env_var:
                    assert secret, f"{env_var} would be typed in the clear"


def test_email_prompts_are_not_hidden():
    """Hiding an email helps nobody and makes typos invisible."""
    for provider in PROVIDERS:
        for mode in provider.modes:
            for env_var, _q, secret in mode.prompts:
                if env_var.endswith("EMAIL"):
                    assert not secret


# ── Container-to-container URL ───────────────────────────────────────────────

def test_the_gateway_reaches_proxies_by_host_gateway_not_localhost():
    """`127.0.0.1` inside the gateway container is the gateway itself, so the
    link would point at nothing."""
    url = host_gateway_url(8894)
    assert url == "http://host.docker.internal:8894/v1"
    assert "127.0.0.1" not in url and "localhost" not in url


def test_the_link_url_keeps_the_v1_suffix():
    """providers.yaml documents the /v1 as mandatory."""
    assert host_gateway_url(8890).endswith("/v1")
