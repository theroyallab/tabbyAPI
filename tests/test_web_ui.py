"""The built-in chat client: served at /ui unless disabled, and its sampling
panel keeps up with the sampler params the server supports."""

import json

from fastapi.testclient import TestClient
from loguru import logger

from common import auth
from common.sampling import UNSUPPORTED_PARAMS, BaseSamplerRequest
from common.tabby_config import config
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.server import setup_app
from endpoints.ui.router import STATIC_DIR

# Sampler fields the panel leaves out on purpose
NOT_IN_PANEL = {
    # Dynamic temperature; exllamav3 doesn't implement it
    "min_temp",
    "max_temp",
    # Only used by mirostat_mode, which exllamav3 doesn't implement
    "mirostat_tau",
    "mirostat_eta",
    # llama.cpp spelling of dry_range, which the panel has
    "dry_penalty_last_n",
    # Chat completions take BOS handling from the template
    "add_bos_token",
    # The client doesn't display logprobs
    "logprobs",
    "top_logprobs",
}


def panel_keys():
    groups = json.loads((STATIC_DIR / "samplers.json").read_text(encoding="utf8"))["groups"]
    return [param["key"] for group in groups for param in group["params"]]


def test_panel_covers_supported_samplers():
    expected = set(BaseSamplerRequest.model_fields) - set(UNSUPPORTED_PARAMS) - NOT_IN_PANEL
    missing = expected - set(panel_keys())
    assert not missing, f"add these to endpoints/ui/static/samplers.json: {sorted(missing)}"


def test_panel_keys_are_request_fields():
    keys = panel_keys()
    assert len(keys) == len(set(keys)), "duplicate keys in samplers.json"

    unknown = set(keys) - set(ChatCompletionRequest.model_fields)
    assert not unknown, f"not chat completion request fields: {sorted(unknown)}"


def test_ui_served(monkeypatch):
    monkeypatch.setattr(config.network, "web_ui", True)
    client = TestClient(setup_app())

    response = client.get("/ui", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"

    response = client.get("/ui/")
    assert response.status_code == 200
    assert "<title>TabbyAPI Chat</title>" in response.text
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-cache"

    for asset in ("app.js", "effects.js", "theme.js", "style.css", "themes.css", "samplers.json"):
        assert client.get(f"/ui/{asset}").status_code == 200

    # The vendored Markdown renderer ships with its license
    assert '<script src="vendor/markdown-it.min.js">' in client.get("/ui/").text
    assert client.get("/ui/vendor/markdown-it.min.js").status_code == 200
    license_text = client.get("/ui/vendor/markdown-it.LICENSE.txt").text
    for package in ("markdown-it", "entities", "linkify-it", "mdurl", "punycode.js", "uc.micro"):
        assert package in license_text, package

    # Not part of the API schema
    assert not any(path.startswith("/ui") for path in client.get("/openapi.json").json()["paths"])


def test_ui_disabled(monkeypatch):
    monkeypatch.setattr(config.network, "web_ui", False)
    client = TestClient(setup_app())

    assert client.get("/ui/").status_code == 404
    assert client.get("/ui", follow_redirects=False).status_code == 404


def startup_lines(monkeypatch, admin_key, disable_auth):
    monkeypatch.setattr(config.network, "web_ui", True)
    monkeypatch.setattr(auth, "DISABLE_AUTH", disable_auth)
    monkeypatch.setattr(auth, "AUTH_KEYS", auth.AuthKeys(api_key="api", admin_key=admin_key))

    lines = []
    sink = logger.add(lambda message: lines.append(str(message)), format="{message}")
    try:
        setup_app("127.0.0.1", 5000)
    finally:
        logger.remove(sink)
    return [line for line in lines if line.startswith("Chat client at")]


def test_startup_link_carries_admin_key(monkeypatch):
    # In the fragment, so the browser never sends it; quoted, since a
    # hand-written key can hold any characters
    lines = startup_lines(monkeypatch, "se/cret key", disable_auth=False)
    assert lines == ["Chat client at http://127.0.0.1:5000/ui/#key=se%2Fcret%20key\n"]


def test_startup_link_without_auth(monkeypatch):
    lines = startup_lines(monkeypatch, "secret", disable_auth=True)
    assert lines == ["Chat client at http://127.0.0.1:5000/ui/\n"]
