"""/metrics is off unless network.enable_metrics says otherwise, and takes an API key."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common import auth, model
from common.metrics import MetricsManager
from common.tabby_config import config
from endpoints.core.router import router


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(auth, "DISABLE_AUTH", True)
    monkeypatch.setattr(model, "container", None)

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_disabled_by_default(client, monkeypatch):
    monkeypatch.setattr(config.network, "enable_metrics", False)

    response = client.get("/metrics")

    assert response.status_code == 404
    assert "enable_metrics" in response.text


def test_serves_the_prometheus_text_format_when_enabled(client, monkeypatch):
    monkeypatch.setattr(config.network, "enable_metrics", True)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert "tabbyapi:tokens_predicted_total" in response.text
    assert response.headers["process-start-time-unix"] == str(
        int(MetricsManager.process_start_time)
    )


def test_needs_an_api_key_unless_auth_is_disabled(client, monkeypatch):
    monkeypatch.setattr(config.network, "enable_metrics", True)
    monkeypatch.setattr(auth, "DISABLE_AUTH", False)

    assert client.get("/metrics").status_code == 401
