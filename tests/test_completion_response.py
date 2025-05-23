import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from endpoints.server import setup_app

app = setup_app()
client = TestClient(app)


@pytest.fixture
def mock_model_container():
    with patch("common.model.container") as mock_container:

        async def mock_generate(*_a, **_kw):
            return {
                "text": "Hello world",
                "prompt_tokens": 2,
                "generated_tokens": 1,
                "finish_reason": "stop",
            }

        mock_container.generate.side_effect = mock_generate
        mock_container.model_dir.name = "test-model"
        yield mock_container


@pytest.fixture
def enable_completion():
    with patch("common.tabby_config.config") as mock_config:
        mock_config.developer.disable_request_streaming = True
        mock_config.network.disable_auth = True
        yield mock_config
        
@pytest.fixture(autouse=True)
def setup_auth():
    # Mock auth keys to avoid NoneType error
    with patch("common.auth.AUTH_KEYS") as mock_auth_keys:
        mock_auth_keys.verify_key.return_value = True
        with patch("common.auth.DISABLE_AUTH", True):
            yield mock_auth_keys


def test_completion_response_fields(mock_model_container, enable_completion):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 1},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "text_completion"
    assert isinstance(data["created"], int)
    assert data["model"] == "test-model"
    assert data["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }


def test_completion_nested_token_prompt(mock_model_container, enable_completion):
    mock_model_container.decode_tokens.return_value = "Hello"
    resp = client.post(
        "/v1/completions",
        json={"prompt": [[1, 2, 3]], "max_tokens": 1},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200
