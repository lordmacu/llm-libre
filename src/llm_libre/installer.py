"""Interactive installer: from a bare machine to a working gateway.

Run it with `python3 scripts/install.py`. It walks the whole path -- Docker,
the orchestrator, the providers you pick, their credentials, the wiring, and a
real prompt at the end to prove it works -- narrating every step.

TWO RULES THIS FILE FOLLOWS, because an installer that breaks them is worse
than no installer:

1. STANDARD LIBRARY ONLY. It runs before anything is installed, so it cannot
   import a package the user does not have yet. That is why there is no `rich`,
   no `click`, no `requests`.

2. NOTHING IS CLAIMED THAT WAS NOT CHECKED. Every step verifies its own result
   before printing a success line, and the final smoke test sends a real prompt
   through the gateway rather than checking that a container is "up". A
   container can be up and the provider still refuse every request.
"""
from __future__ import annotations

import getpass
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


# ── The provider registry ────────────────────────────────────────────────────

@dataclass(frozen=True)
class OtpFlow:
    """A two-step email login the PROXY performs, not the installer.

    Several providers have no password at all: you give an email, they send a
    six-digit code, you send it back. That cannot be answered up front like a
    token can -- the code does not exist until the first call is made -- so
    this flow runs AFTER the container is up, against the proxy's own
    endpoints, and the proxy stores the session it gets back.

    Distinct from the OTP *worker* some proxies also configure: that one reads
    the emailed code automatically to RENEW an expiring session. This is the
    initial login, and a human reads the code.
    """
    request_path: str             # asks the vendor to email a code
    verify_path: str              # exchanges the code for a session
    code_field: str = "otp"       # what the verify endpoint calls the code
    email_field: str = "email"


@dataclass(frozen=True)
class AuthMode:
    """One way to authenticate a provider.

    `prompts` maps the variable the proxy reads to what the installer must ask,
    and is answered BEFORE the container starts. `otp` is the alternative: a
    flow that can only run once the proxy is answering.

    `secret` fields are read with getpass so they never reach the terminal
    history or a screen recording.
    """
    key: str                      # "token" | "password" | "otp" | "anonymous"
    label: str
    prompts: tuple = ()           # (env_var, question, is_secret)
    otp: OtpFlow | None = None


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    repo: str
    port: int                     # host port; container always listens on 8000
    modes: tuple
    notes: str = ""
    compose_dir: str = "."        # where the compose file lives inside the repo
    extra_env: dict = field(default_factory=dict)

    @property
    def url_env(self) -> str:
        """The variable llm-libre reads to find this proxy (see providers.yaml)."""
        return f"{self.key.upper()}_PROXY_URL"


TOKEN_ONLY = "This provider has no anonymous mode: without credentials every request is refused."

PROVIDERS: tuple = (
    Provider(
        key="mistral", label="Mistral (Le Chat)",
        repo="https://github.com/lordmacu/mistral-proxy.git", port=8894,
        modes=(
            AuthMode("anonymous", "Free, no account (chat only)"),
            AuthMode("password", "Email and password", (
                ("MISTRAL_EMAIL", "Mistral email", False),
                ("MISTRAL_PASSWORD", "Mistral password", True))),
            AuthMode("token", "Session token", (
                ("MISTRAL_SESSION_TOKEN", "Ory session token", True),)),
        ),
        notes=("The only provider whose chat answers with NO account -- measured, "
               "not assumed. Vision, images, audio, search and conversations all "
               "need credentials."),
    ),
    Provider(
        key="chatgpt", label="ChatGPT",
        repo="https://github.com/lordmacu/chatgpt-proxy.git", port=8890,
        modes=(
            AuthMode("password", "Email and password", (
                ("EMAIL", "ChatGPT email", False),
                ("PASSWORD", "ChatGPT password", True))),
            AuthMode("token", "Access token", (
                ("CHATGPT_ACCESS_TOKEN", "Access token", True),)),
        ),
        notes=TOKEN_ONLY + " Richest provider of the five: 11 capabilities with an active plan.",
    ),
    Provider(
        key="grok", label="Grok (xAI)",
        repo="https://github.com/lordmacu/grok-proxy.git", port=8893,
        compose_dir="docker-api",
        modes=(
            AuthMode("otp", "Email and a code they send you", otp=OtpFlow(
                request_path="/auth/otp/send",
                verify_path="/auth/otp/verify",
                code_field="code")),
            AuthMode("password", "Email and password", (
                ("GROK_EMAIL", "Grok email", False),
                ("GROK_PASSWORD", "Grok password", True))),
            AuthMode("token", "Session token you already have", (
                ("GROK_SESSION_TOKEN", "Grok session token", True),)),
        ),
        notes=(TOKEN_ONLY + " Accounts created through Twitter or Google have no "
               "password -- use the emailed code for those."),
    ),
    Provider(
        key="perplexity", label="Perplexity",
        repo="https://github.com/lordmacu/perplexity-proxy.git", port=8891,
        compose_dir="docker-api",
        modes=(
            AuthMode("otp", "Email and a code they send you", otp=OtpFlow(
                request_path="/perplexity/auth/request-otp",
                verify_path="/perplexity/auth/verify-otp",
                code_field="otp")),
            AuthMode("token", "Session cookie you already have", (
                ("PERPLEXITY_SESSION", "__Secure-next-auth.session-token", True),)),
        ),
        notes=(TOKEN_ONLY + " Perplexity has no password: the code emailed to you "
               "IS the login. The cookie option is for one you already extracted."),
    ),
    Provider(
        key="deepseek", label="DeepSeek",
        repo="https://github.com/lordmacu/deepseek.git", port=8892,
        modes=(
            AuthMode("password", "Email and password", (
                ("DEEPSEEK_EMAIL", "DeepSeek email", False),
                ("DEEPSEEK_PASSWORD", "DeepSeek password", True))),
            AuthMode("token", "Bearer token", (
                ("DEEPSEEK_TOKEN", "DeepSeek token", True),)),
        ),
        notes=TOKEN_ONLY,
    ),
)

BY_KEY = {p.key: p for p in PROVIDERS}


# ── Narration ────────────────────────────────────────────────────────────────

class UI:
    """Every step announces itself, succeeds or fails out loud, and says why.

    Colour is optional on purpose: NO_COLOR and a non-tty both disable it, so
    the output stays readable in a log file or a CI job.
    """

    def __init__(self, stream=None, colour: bool | None = None):
        self.out = stream or sys.stdout
        if colour is None:
            colour = self.out.isatty() and not os.getenv("NO_COLOR")
        self.colour = colour

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.colour else text

    def title(self, text: str) -> None:
        self.out.write("\n" + self._c("1", text) + "\n")
        self.out.write(self._c("2", "─" * len(text)) + "\n")

    def step(self, text: str) -> None:
        self.out.write(self._c("36", "→ ") + text + "\n")

    def ok(self, text: str) -> None:
        self.out.write(self._c("32", "✓ ") + text + "\n")

    def warn(self, text: str) -> None:
        self.out.write(self._c("33", "! ") + text + "\n")

    def fail(self, text: str) -> None:
        self.out.write(self._c("31", "✗ ") + text + "\n")

    def info(self, text: str) -> None:
        self.out.write("  " + self._c("2", text) + "\n")


# ── Shell helpers ────────────────────────────────────────────────────────────

def run(cmd: list, cwd: Path | None = None, check: bool = True,
        capture: bool = True, timeout: int = 900) -> subprocess.CompletedProcess:
    """Run a command, and on failure raise with the output attached.

    A silent failure in an installer is the worst outcome: the user is left
    with a half-built system and no idea which step broke.
    """
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        capture_output=capture, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"`{' '.join(cmd)}` failed ({proc.returncode})\n{detail}")
    return proc


def have(binary: str) -> bool:
    return shutil.which(binary) is not None


def compose_command() -> list:
    """`docker compose` (v2) or `docker-compose` (v1), whichever exists."""
    if have("docker"):
        probe = subprocess.run(["docker", "compose", "version"],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            return ["docker", "compose"]
    if have("docker-compose"):
        return ["docker-compose"]
    raise RuntimeError("no Docker Compose found (neither `docker compose` nor `docker-compose`)")


# ── .env handling ────────────────────────────────────────────────────────────

def read_env(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def write_env(path: Path, updates: dict) -> None:
    """Merge `updates` into a .env, keeping comments and unrelated keys.

    Rewriting the file wholesale would silently drop anything the user had put
    there by hand, which for a file holding credentials is not a small loss.
    """
    lines = path.read_text().splitlines() if path.exists() else []
    seen = set()
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    path.write_text("\n".join(out).rstrip() + "\n")
    path.chmod(0o600)      # it holds credentials


# ── HTTP without dependencies ────────────────────────────────────────────────

def http_json(url: str, payload: dict | None = None, headers: dict | None = None,
              timeout: int = 30) -> tuple:
    """Returns (status, parsed-or-text). Never raises for an HTTP status."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode(errors="replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        status = exc.code
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body


def wait_for_health(url: str, ui: UI, attempts: int = 60, delay: float = 5.0) -> bool:
    """Poll until a real HTTP answer arrives.

    Deliberately does NOT accept a connection failure as progress: an earlier
    version of this project's tooling treated `000` as success and reported a
    service live while it was mid-restart.
    """
    for attempt in range(1, attempts + 1):
        status, _ = http_json(url, timeout=8)
        if status == 200:
            return True
        if attempt % 6 == 0:
            ui.info(f"still waiting ({attempt * int(delay)}s)…")
        time.sleep(delay)
    return False


def host_gateway_url(port: int) -> str:
    """How a container reaches another container's published port on this host.

    `host.docker.internal` resolves on Docker Desktop and, since 20.10, on
    Linux too when the container is started with
    `--add-host=host.docker.internal:host-gateway` -- which the gateway's
    compose file does. One name for both platforms beats branching on OS and
    getting it wrong on the third one.
    """
    return f"http://host.docker.internal:{port}/v1"


# ── Steps ────────────────────────────────────────────────────────────────────

def ensure_docker(ui: UI, assume_yes: bool = False) -> None:
    """Docker and Compose present, or install them where that is safe to do.

    Only Linux gets an automated install, and only from Docker's own script,
    and only after saying so. On macOS and Windows the equivalent is a desktop
    application with a licence prompt -- installing that behind the user's back
    would be presumptuous, so it stops and says what to install.
    """
    ui.step("Checking Docker")
    if have("docker"):
        version = run(["docker", "--version"]).stdout.strip()
        ui.ok(f"Docker present — {version}")
    else:
        system = platform.system()
        if system != "Linux":
            ui.fail(f"Docker is not installed, and this is {system}.")
            ui.info("Install Docker Desktop from https://docs.docker.com/get-docker/ "
                    "and run this installer again.")
            raise SystemExit(1)
        ui.warn("Docker is not installed. It can be installed from get.docker.com "
                "(Docker's official script, run with sudo).")
        if not (assume_yes or confirm(ui, "Install Docker now?", default=True)):
            raise SystemExit("Docker is required. Nothing was changed.")
        ui.step("Installing Docker — this takes a few minutes")
        run(["bash", "-c", "curl -fsSL https://get.docker.com | sudo sh"],
            capture=False, timeout=1800)
        ui.ok("Docker installed")

    ui.step("Checking Docker Compose")
    compose = compose_command()
    ui.ok(f"Compose present — {' '.join(compose)}")

    ui.step("Checking that the Docker daemon is reachable")
    probe = run(["docker", "info"], check=False)
    if probe.returncode != 0:
        ui.fail("Docker is installed but the daemon is not answering.")
        ui.info("Start Docker Desktop, or on Linux: sudo systemctl start docker")
        ui.info("If this is a fresh Linux install, you may need to log out and back "
                "in for your user to join the `docker` group.")
        raise SystemExit(1)
    ui.ok("Docker daemon answering")


def confirm(ui: UI, question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"  {question} {suffix} ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes", "s", "si", "sí"):
            return True
        if answer in ("n", "no"):
            return False
        ui.warn("Answer y or n.")


def choose_providers(ui: UI) -> list:
    """Ask which providers to install. Nothing is installed by default."""
    ui.title("Which providers do you want to add?")
    for index, provider in enumerate(PROVIDERS, 1):
        free = "free mode available" if any(m.key == "anonymous" for m in provider.modes) \
            else "account required"
        ui.out.write(f"  {index}. {provider.label:22} port {provider.port}   ({free})\n")
        if provider.notes:
            ui.info(provider.notes)
    ui.out.write("\n")
    while True:
        raw = input("  Numbers separated by commas (or `all`): ").strip().lower()
        chosen = parse_selection(raw, len(PROVIDERS))
        if chosen:
            return [PROVIDERS[i - 1] for i in chosen]
        ui.warn("Pick at least one, e.g. `1,3` or `all`.")


def parse_selection(raw: str, total: int) -> list:
    """Parse `all`, `1,3`, `1 3`, `2-4`. Returns sorted unique 1-based indices.

    Split out of the prompt so it can be tested without a terminal -- selection
    parsing is where an installer silently does the wrong thing.
    """
    raw = (raw or "").strip().lower()
    if raw in ("all", "todos", "*"):
        return list(range(1, total + 1))
    picked = set()
    for chunk in raw.replace(",", " ").split():
        if "-" in chunk[1:]:
            start, _, end = chunk.partition("-")
            try:
                lo, hi = int(start), int(end)
            except ValueError:
                return []
            if lo > hi:
                return []
            for value in range(lo, hi + 1):
                if 1 <= value <= total:
                    picked.add(value)
            continue
        try:
            value = int(chunk)
        except ValueError:
            return []
        if 1 <= value <= total:
            picked.add(value)
    return sorted(picked)


def choose_mode(ui: UI, provider: Provider) -> AuthMode:
    """Free or authenticated, and which kind of credential.

    A provider with a single mode is not asked about -- offering a choice with
    one option wastes the user's attention and implies alternatives exist.
    """
    if len(provider.modes) == 1:
        mode = provider.modes[0]
        ui.info(f"{provider.label}: {mode.label} (the only mode this provider supports)")
        return mode
    ui.out.write(f"\n  How do you want to use {provider.label}?\n")
    for index, mode in enumerate(provider.modes, 1):
        ui.out.write(f"    {index}. {mode.label}\n")
    while True:
        raw = input("  Choice: ").strip()
        try:
            index = int(raw)
        except ValueError:
            index = 0
        if 1 <= index <= len(provider.modes):
            return provider.modes[index - 1]
        ui.warn(f"Pick a number between 1 and {len(provider.modes)}.")


def collect_credentials(ui: UI, mode: AuthMode) -> dict:
    """Ask for what this mode needs. Secrets never echo to the terminal."""
    values = {}
    for env_var, question, secret in mode.prompts:
        while True:
            value = getpass.getpass(f"  {question}: ") if secret else input(f"  {question}: ")
            value = value.strip()
            if value:
                values[env_var] = value
                break
            ui.warn("Cannot be empty.")
    return values


def start_gateway(ui: UI, root: Path, api_key: str) -> None:
    """Build and start llm-libre itself, then wait for a real /health."""
    ui.title("Orchestrator (llm-libre)")
    env_path = root / ".env"
    if not env_path.exists() and (root / ".env.example").exists():
        ui.step("Creating .env from .env.example")
        env_path.write_text((root / ".env.example").read_text())
    write_env(env_path, {"LLM_LIBRE_API_KEYS": api_key})
    ui.ok("API key written to .env")

    ui.step("Building the image (first run downloads layers, be patient)")
    compose = compose_command()
    run(compose + ["build"], cwd=root, capture=False, timeout=1800)
    ui.ok("Image built")

    ui.step("Starting the container")
    run(compose + ["up", "-d"], cwd=root, capture=False)
    ui.ok("Container started")

    ui.step("Waiting for the gateway to answer /health")
    if not wait_for_health("http://127.0.0.1:8101/health", ui):
        ui.fail("The gateway did not answer. Logs: docker compose logs --tail 50")
        raise SystemExit(1)
    ui.ok("Gateway healthy on http://127.0.0.1:8101")


def install_provider(ui: UI, provider: Provider, workdir: Path, mode: AuthMode,
                     credentials: dict) -> str:
    """Clone, configure and start one provider.

    Returns "ok" (it answered a real prompt), "unverified" (it is running but
    did not answer) or "failed" (it never came up).

    The middle case exists on purpose. A provider whose quota is spent today
    answers nothing and is still worth linking: the gateway handles exactly
    that with cooldowns and failover, and excluding it would be a worse
    outcome than including it with a warning. Only a container that never came
    up is skipped, because there is nothing to link.
    """
    ui.title(f"{provider.label}")
    target = workdir / provider.key

    if target.exists():
        ui.step(f"Repository already at {target} — pulling")
        run(["git", "pull", "--ff-only"], cwd=target, check=False)
    else:
        ui.step(f"Cloning {provider.repo}")
        run(["git", "clone", "--depth", "1", provider.repo, str(target)], capture=False)
    ui.ok("Source ready")

    compose_dir = target / provider.compose_dir
    env_path = compose_dir / ".env"
    if credentials:
        write_env(env_path, {**credentials, **provider.extra_env})
        ui.ok(f"Credentials written to {env_path} (mode 600)")
    else:
        write_env(env_path, dict(provider.extra_env) or {"ANONYMOUS": "1"})
        ui.ok("No credentials needed — running in free mode")

    ui.step("Building and starting the container")
    compose = compose_command()
    run(compose + ["up", "-d", "--build"], cwd=compose_dir, capture=False, timeout=1800)

    ui.step(f"Waiting for {provider.label} on port {provider.port}")
    if not wait_for_health(f"http://127.0.0.1:{provider.port}/health", ui, attempts=40):
        ui.fail(f"{provider.label} did not answer on {provider.port}.")
        ui.info(f"Logs: cd {compose_dir} && docker compose logs --tail 50")
        return "failed"

    if mode.otp is not None and not run_otp_login(ui, provider, mode.otp):
        ui.warn(f"{provider.label} is up but not logged in. Chat will fail until "
                f"you authenticate; the container keeps running.")

    status, body = http_json(f"http://127.0.0.1:{provider.port}/health")
    if isinstance(body, dict) and "capabilities" in body:
        live = sorted(k for k, v in body["capabilities"].items() if v)
        ui.ok(f"{provider.label} up — {len(live)}/11 capabilities: {', '.join(live)}")
        auth = (body.get("auth") or {}).get("mode")
        if auth == "anonymous" and mode.key != "anonymous":
            ui.warn("It reports mode `anonymous`: the credentials were not picked "
                    "up. The prompt below will show whether it can answer at all.")
    else:
        ui.ok(f"{provider.label} up on port {provider.port}")

    return "ok" if verify_provider(ui, provider) else "unverified"


def verify_provider(ui: UI, provider: Provider, timeout: int = 180) -> bool:
    """Send a real prompt to THIS proxy and require a real answer.

    This exists because the cheaper checks lie in a specific, dangerous way.
    `/health` answering 200 only proves the web server is running. And the
    contract's `auth.mode` is a LOCAL read of whether credential variables are
    set -- it says nothing about whether they are correct. A mistyped password
    produces a healthy container reporting `account`, and the failure surfaces
    later, to a user who was told the install succeeded.

    So the check is the only one that cannot be faked: ask it something and see
    whether it answers. Costs one request against the account, which is the
    right price for knowing.
    """
    ui.step(f"Verifying {provider.label} answers a real prompt")
    status, body = http_json(
        f"http://127.0.0.1:{provider.port}/v1/chat/completions",
        payload={"model": "auto",
                 "messages": [{"role": "user", "content": "responde solo: ok"}]},
        timeout=timeout)

    if status == 401 or status == 403:
        ui.fail(f"{provider.label} refused the request ({status}): the credentials "
                f"are wrong or expired.")
        return False
    if status != 200:
        detail = str(body)[:200]
        ui.fail(f"{provider.label} answered {status}: {detail}")
        return False
    try:
        answer = body["choices"][0]["message"]["content"].strip()
    except Exception:
        ui.fail(f"{provider.label} answered 200 with an unusable body: {str(body)[:200]}")
        return False
    if not answer:
        # A 200 carrying nothing is how a refused or muted account looks from
        # the outside. Accepting it would report a working provider that
        # answers every real question with silence.
        ui.fail(f"{provider.label} answered 200 but with empty content — the "
                f"account is likely rate limited or muted.")
        return False

    ui.ok(f"{provider.label} answered: {answer[:60]!r}")
    return True


def run_otp_login(ui: UI, provider: Provider, flow: OtpFlow) -> bool:
    """The two-step email login, run against the proxy once it is answering.

    Why here and not with the other prompts: the code does not exist until the
    first call is made, so it cannot be collected up front like a token. The
    proxy stores whatever session it gets back -- the installer never sees or
    writes it, which is one fewer place a credential lives.
    """
    base = f"http://127.0.0.1:{provider.port}"
    ui.step(f"Logging in to {provider.label} with an emailed code")

    email = input("  Email: ").strip()
    if not email:
        ui.fail("An email is required.")
        return False

    status, body = http_json(f"{base}{flow.request_path}",
                             payload={flow.email_field: email}, timeout=60)
    if status not in (200, 201):
        ui.fail(f"Could not request the code ({status}): {str(body)[:200]}")
        return False
    ui.ok(f"Code requested — check {email}")

    for attempt in (1, 2, 3):
        code = input("  Code you received: ").strip()
        if not code:
            ui.warn("Cannot be empty.")
            continue
        status, body = http_json(
            f"{base}{flow.verify_path}",
            payload={flow.email_field: email, flow.code_field: code}, timeout=60)
        if status in (200, 201):
            ui.ok(f"{provider.label} authenticated")
            return True
        ui.warn(f"The code was not accepted ({status}): {str(body)[:160]}")
        if attempt < 3:
            ui.info("Codes expire quickly — if this one has, ask for a new one "
                    "and run the installer again.")
    ui.fail("Three codes rejected. Leaving this provider unauthenticated.")
    return False


def link_provider(ui: UI, root: Path, provider: Provider) -> None:
    """Point the gateway at this proxy."""
    ui.step(f"Linking {provider.label} to the orchestrator")
    write_env(root / ".env", {provider.url_env: host_gateway_url(provider.port)})
    ui.ok(f"{provider.url_env}={host_gateway_url(provider.port)}")


def restart_gateway(ui: UI, root: Path) -> None:
    """Reload the gateway so it picks up the new providers.

    Necessary, not decorative: `providers.load()` runs once at startup, so a
    proxy added afterwards stays invisible until the process restarts.
    """
    ui.step("Restarting the orchestrator so it sees the new providers")
    compose = compose_command()
    run(compose + ["up", "-d"], cwd=root, capture=False)
    if not wait_for_health("http://127.0.0.1:8101/health", ui, attempts=40):
        ui.fail("The gateway did not come back after the restart.")
        raise SystemExit(1)
    ui.ok("Orchestrator restarted")


def smoke_test(ui: UI, api_key: str) -> bool:
    """Send a real prompt end to end.

    The point of finishing here rather than at "containers are up": a container
    can be perfectly healthy while every request it forwards is refused. Only
    an answer proves the install.
    """
    ui.title("Final check — a real prompt through the gateway")
    headers = {"Authorization": f"Bearer {api_key}"}

    status, body = http_json("http://127.0.0.1:8101/v1/models", headers=headers)
    if status != 200 or not isinstance(body, dict):
        ui.fail(f"/v1/models answered {status}: {str(body)[:200]}")
        return False
    routes = body.get("data") or []
    ui.ok(f"The gateway is publishing {len(routes)} model(s)")
    if not routes:
        ui.fail("No routes: the providers are up but none produced a usable model.")
        return False

    ui.step("Asking: “responde solo: hola”")
    status, body = http_json(
        "http://127.0.0.1:8101/v1/chat/completions",
        payload={"model": "auto",
                 "messages": [{"role": "user", "content": "responde solo: hola"}]},
        headers=headers, timeout=180)
    if status != 200 or not isinstance(body, dict):
        ui.fail(f"The request answered {status}: {str(body)[:300]}")
        return False
    try:
        answer = body["choices"][0]["message"]["content"].strip()
    except Exception:
        ui.fail(f"Unexpected response shape: {str(body)[:300]}")
        return False
    served_by = body.get("model", "?")
    ui.ok(f"Answer: {answer!r}  (served by {served_by})")
    return True


# ── Entry point ──────────────────────────────────────────────────────────────

def generate_api_key() -> str:
    import secrets
    return "llmlibre-" + secrets.token_urlsafe(24)


def main(argv: list | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="llm-libre-install",
        description="Install llm-libre and the providers you choose, end to end.")
    parser.add_argument("--workdir", default=str(Path.home() / "llm-libre-providers"),
                        help="where provider repositories are cloned")
    parser.add_argument("--api-key", default=None,
                        help="gateway API key (generated if omitted)")
    parser.add_argument("--yes", action="store_true",
                        help="do not ask for confirmation on installs")
    parser.add_argument("--skip-docker-check", action="store_true")
    args = parser.parse_args(argv)

    ui = UI()
    root = Path(__file__).resolve().parents[2]

    ui.title("llm-libre installer")
    ui.info(f"Repository:  {root}")
    ui.info(f"Providers:   {args.workdir}")

    try:
        if not args.skip_docker_check:
            ensure_docker(ui, assume_yes=args.yes)

        api_key = args.api_key or read_env(root / ".env").get("LLM_LIBRE_API_KEYS") \
            or generate_api_key()
        start_gateway(ui, root, api_key)

        chosen = choose_providers(ui)
        ui.ok(f"Selected: {', '.join(p.label for p in chosen)}")

        workdir = Path(args.workdir).expanduser()
        workdir.mkdir(parents=True, exist_ok=True)

        # Credentials are collected for ALL providers first, then containers are
        # built one by one. Asking mid-build would leave the user watching a
        # progress bar waiting to be asked for a password.
        plans = []
        for provider in chosen:
            mode = choose_mode(ui, provider)
            plans.append((provider, mode, collect_credentials(ui, mode)))

        installed, unverified = [], []
        for provider, mode, credentials in plans:
            outcome = install_provider(ui, provider, workdir, mode, credentials)
            if outcome == "failed":
                ui.warn(f"{provider.label} was skipped; the rest continue.")
                continue
            link_provider(ui, root, provider)
            installed.append(provider)
            if outcome == "unverified":
                unverified.append(provider)
                ui.warn(f"{provider.label} is linked but never answered. It stays "
                        f"in the pool -- the gateway will fail over past it until "
                        f"it recovers.")

        if not installed:
            ui.fail("No provider came up. Nothing to link.")
            return 1

        restart_gateway(ui, root)

        ok = smoke_test(ui, api_key)
        ui.title("Done" if ok else "Finished with warnings")
        ui.info(f"Gateway:  http://127.0.0.1:8101")
        ui.info(f"API key:  {api_key}")
        ui.info(f"Providers installed: {', '.join(p.label for p in installed)}")
        if unverified:
            ui.warn(f"Never answered a prompt: {', '.join(p.label for p in unverified)}. "
                    f"Most likely wrong credentials or a spent quota.")
        if not ok:
            ui.warn("The containers are up but the test prompt did not come back. "
                    "Check credentials with: curl -s http://127.0.0.1:8101/health")
        return 0 if ok else 2

    except KeyboardInterrupt:
        ui.warn("\nInterrupted. Anything already installed keeps running.")
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        ui.fail(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
