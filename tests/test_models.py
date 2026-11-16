from __future__ import annotations

import http.server
import threading
from pathlib import Path

import pytest

from vupscale import models
from vupscale.models import MODEL_ALIASES, ModelError, cached_path, resolve_model

PAYLOAD = b"fake-model-weights" * 4096


class _Handler(http.server.BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self):  # noqa: N802
        type(self).hits += 1
        if self.path.startswith("/missing"):
            self.send_error(404)
            return
        body = PAYLOAD
        self.send_response(200)
        # /truncated lies about the size: the client must notice the short body.
        declared = len(body) * 2 if self.path.startswith("/truncated") else len(body)
        self.send_header("Content-Length", str(declared))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.hits = 0
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_catalog_is_consistent():
    assert "realesrgan-x4" in MODEL_ALIASES
    for alias, info in MODEL_ALIASES.items():
        assert alias == info.alias == alias.lower()
        assert info.url.startswith("https://")
        assert info.filename.endswith(".pth")
        assert info.scale in (1, 2, 4)


def test_local_path(tmp_path: Path):
    weights = tmp_path / "local.pth"
    weights.write_bytes(b"x")
    assert resolve_model(weights) == weights
    assert resolve_model(str(weights)) == weights


def test_missing_file_mentions_aliases(tmp_path: Path):
    with pytest.raises(ModelError, match="realesrgan-x4"):
        resolve_model("not-a-real-alias")
    with pytest.raises(ModelError, match="not found"):
        resolve_model(tmp_path / "nope.pth")


def test_url_download_is_cached(server: str, isolated_cache: Path):
    url = f"{server}/weights/tiny.pth"
    path = resolve_model(url, quiet=True)
    assert path == isolated_cache / "models" / "tiny.pth"
    assert path.read_bytes() == PAYLOAD
    assert not list(path.parent.glob("*.part"))
    assert resolve_model(url, quiet=True) == path
    assert _Handler.hits == 1  # second call came from the cache


def test_alias_resolves_through_the_cache(server: str, monkeypatch: pytest.MonkeyPatch):
    info = models.ModelInfo("test-alias", f"{server}/alias_model.pth", 2, "test")
    monkeypatch.setitem(MODEL_ALIASES, "test-alias", info)
    path = resolve_model("Test-Alias", quiet=True)  # aliases are case-insensitive
    assert path.name == "alias_model.pth" and path.read_bytes() == PAYLOAD


def test_truncated_download_leaves_nothing_behind(server: str):
    url = f"{server}/truncated/model.pth"
    with pytest.raises(ModelError):
        resolve_model(url, quiet=True)
    assert not cached_path(url).exists()
    assert not list(cached_path(url).parent.glob("*.part"))


def test_http_error_is_reported(server: str):
    with pytest.raises(ModelError, match="failed to download"):
        resolve_model(f"{server}/missing/model.pth", quiet=True)


def test_opaque_urls_do_not_collide():
    a = cached_path("https://example.com/download?id=1")
    b = cached_path("https://example.com/download?id=2")
    assert a != b and a.suffix == ".pth"
