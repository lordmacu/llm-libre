"""Content-addressed store for binaries a provider generated for us.

Why this exists at all: a provider's asset URL is not something a client can be
handed. Mistral's are Azure Blob SAS links and grok's are its own asset host --
they EXPIRE, some need headers the client does not have, and every one of them
names the provider, which breaks the single promise this gateway makes ("change
only base_url"). A client that stores the URL we return and opens it a week
later must still get the image.

So: the bytes are fetched once, stored here, and re-served from our own domain.

ASSETS, not images, on purpose. grok can produce documents as well, and the only
thing this module knows about a blob is its bytes and its content type -- which
is exactly the right amount of knowledge to also cover whatever a provider
generates next.

Content-addressed by SHA-256:
  - identical bytes stored twice cost one file (two providers asked for the same
    prompt, or a client retried);
  - the id is unguessable, which is what makes `GET /v1/assets/{id}` safe to
    expose without a key -- and it has to be exposed without one, or no `<img>`
    tag and no markdown preview would ever render it;
  - the id cannot collide with a path, so it is safe to build a filename from
    (see `_path`, which still validates rather than trusting that).

The bytes live on disk (under the same persistent volume as the database) and
the metadata in SQLite, so retention can be enforced by age with one query
rather than by walking a directory.
"""
from __future__ import annotations

import base64
import hashlib
import re
import time
from pathlib import Path

# A hostile or broken provider must not be able to fill the disk one response at
# a time. 10 MB is far above any image either upstream produces and far below
# anything that would matter on a 230 GB volume.
MAX_BYTES = 10 * 1024 * 1024

# How long a generated asset stays retrievable. Long enough that a client can
# store the URL and come back to it, bounded so the volume does not grow without
# limit. Pruned by the same probing cycle that prunes telemetry.
RETENTION_S = 30 * 24 * 3600.0

_ID = re.compile(r"^[0-9a-f]{64}$")

# What we are willing to hand back. A provider returning something unexpected
# gets stored as a plain download rather than as, say, text/html -- serving
# attacker-influenced HTML from our own domain would be a stored-XSS hole on
# whatever else that domain hosts.
_SAFE_TYPES = {
    "image/png", "image/jpeg", "image/webp", "image/gif", "image/svg+xml",
    "application/pdf", "text/plain", "text/csv", "application/json",
    "application/zip",
}
_FALLBACK_TYPE = "application/octet-stream"

# image/svg+xml is in the list above because it is a legitimate generated format,
# but an SVG can carry script: it is served with a Content-Disposition that stops
# the browser rendering it inline. See `content_disposition`.
_NEVER_INLINE = {"image/svg+xml"}


def normalise_type(content_type: str | None) -> str:
    """The stored content type: the provider's, if we are willing to serve it."""
    if not content_type:
        return _FALLBACK_TYPE
    base = content_type.split(";")[0].strip().lower()
    return base if base in _SAFE_TYPES else _FALLBACK_TYPE


def content_disposition(content_type: str) -> str:
    """`inline` for things a browser should render, `attachment` for the rest.

    An SVG is an image everywhere except in a browser, where it is a document
    that can run script. Serving one inline from our own origin would let a
    provider's output execute there.
    """
    return "attachment" if content_type in _NEVER_INLINE else "inline"


class AssetStore:
    def __init__(self, directory: str, con):
        self._dir = Path(directory)
        self._con = con
        # A directory that cannot be created must NOT stop the process. Image
        # generation is one endpoint of five, and refusing to start over it
        # would take chat, ranking and health down with it -- the same class of
        # failure as a container that looks healthy but serves nothing, only
        # inverted. Unusable means `put` returns None and `get` returns None,
        # which the images endpoint already handles by falling back to the
        # provider's own URL.
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self.usable = True
        except OSError:
            self.usable = False

    def put(self, data: bytes, content_type: str | None, now: float) -> str | None:
        """Store bytes, return their id. None if they are empty or oversized.

        Returning None rather than raising: the caller is in the middle of a
        successful generation, and failing to localise one asset should degrade
        to "hand back what the provider said" -- never turn a working response
        into an error.
        """
        if not self.usable or not data or len(data) > MAX_BYTES:
            return None
        asset_id = hashlib.sha256(data).hexdigest()
        path = self._path(asset_id)
        if not path.exists():
            # Written to a temporary name and moved into place, so a crash
            # mid-write cannot leave a truncated file under a hash that claims
            # to describe complete content.
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        self._con.execute(
            """INSERT INTO assets (id, content_type, bytes, created_at) VALUES (?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET created_at = excluded.created_at""",
            (asset_id, normalise_type(content_type), len(data), now))
        self._con.commit()
        return asset_id

    def get(self, asset_id: str) -> tuple[bytes, str] | None:
        if not self.usable or not _ID.match(asset_id or ""):
            return None
        row = self._con.execute(
            "SELECT content_type FROM assets WHERE id = ?", (asset_id,)).fetchone()
        if not row:
            return None
        path = self._path(asset_id)
        if not path.exists():
            return None
        return path.read_bytes(), row[0]

    def prune(self, before: float) -> int:
        """Delete assets older than `before`, from disk AND from the table.

        The file goes first: a row with no file is a 404 (recoverable), a file
        with no row is invisible and leaks disk forever.
        """
        if not self.usable:
            return 0
        rows = self._con.execute(
            "SELECT id FROM assets WHERE created_at < ?", (before,)).fetchall()
        for (asset_id,) in rows:
            self._path(asset_id).unlink(missing_ok=True)
        self._con.execute("DELETE FROM assets WHERE created_at < ?", (before,))
        self._con.commit()
        return len(rows)

    def _path(self, asset_id: str) -> Path:
        # Validated, not trusted: the id reaches here from a URL path segment.
        # Without this, "../../etc/passwd" would be a filename.
        if not _ID.match(asset_id or ""):
            raise ValueError("invalid asset id")
        return self._dir / asset_id


async def localise(payload: dict, store: AssetStore, http, public_base: str,
                   response_format: str | None, now: float | None = None) -> dict:
    """Rewrite a provider's image response so every URL points at us.

    Given `{"data": [{"url": "https://<provider>/..."}]}`, each URL is fetched
    ONCE, stored, and replaced by `{public_base}/v1/assets/{id}` -- or by
    `b64_json` when the caller asked for that, which is OpenAI's own way of
    saying "do not make me do a second request".

    Only URLs that came back from a provider WE called are fetched. Nothing here
    ever touches a URL from the client's request: that would turn this endpoint
    into an open fetcher of arbitrary internal addresses.

    Degrades rather than fails. If a download errors, or the asset is too large,
    or there is no `public_base` configured, that entry is left EXACTLY as the
    provider sent it: a usable-but-expiring URL beats a 500, and the client can
    still see something. `payload` is never mutated -- a retry upstream must not
    find a half-rewritten body.
    """
    entries = payload.get("data")
    if not isinstance(entries, list):
        return payload
    now = time.time() if now is None else now
    want_b64 = response_format == "b64_json"
    out = []
    for entry in entries:
        if not isinstance(entry, dict):
            out.append(entry)
            continue
        url = entry.get("url")
        if not url or not str(url).lower().startswith(("http://", "https://")):
            out.append(entry)
            continue
        data, content_type = await _download(http, url)
        if data is None:
            out.append(entry)
            continue
        if want_b64:
            new = {k: v for k, v in entry.items() if k != "url"}
            new["b64_json"] = base64.b64encode(data).decode()
            out.append(new)
            continue
        asset_id = store.put(data, content_type, now)
        if asset_id is None or not public_base:
            out.append(entry)
            continue
        out.append({**entry, "url": f"{public_base.rstrip('/')}/v1/assets/{asset_id}"})
    return {**payload, "data": out}


async def _download(http, url: str) -> tuple[bytes | None, str | None]:
    try:
        resp = await http.get(url, timeout=60, follow_redirects=True)
    except Exception:
        return None, None
    if resp.status_code != 200:
        return None, None
    data = resp.content
    if not data or len(data) > MAX_BYTES:
        return None, None
    return data, resp.headers.get("content-type")
