"""/metrics is off unless network.enable_metrics says otherwise."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.metrics import MetricsManager
from common.tabby_config import config
from endpoints.core.router import router


def test_counting_follows_the_option(monkeypatch):
    # The generation loop skips its per-piece metrics calls on this, so a
    # disabled server does not pay for them
    monkeypatch.setattr(config.network, "enable_metrics", False)
    assert MetricsManager.enabled() is False

    monkeypatch.setattr(config.network, "enable_metrics", True)
    assert MetricsManager.enabled() is True


def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr(config.network, "enable_metrics", False)

    response = client().get("/metrics")

    assert response.status_code == 404
    assert "not protected by API key" not in response.text


def test_serves_the_prometheus_text_format_when_enabled(monkeypatch):
    monkeypatch.setattr(config.network, "enable_metrics", True)

    response = client().get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "tabbyapi:tokens_predicted_total" in response.text
