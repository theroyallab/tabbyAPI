import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from endpoints.server import setup_app

app = setup_app()
client = TestClient(app)

@pytest.fixture
def mock_model_container():
    with patch("common.model.container") as mock_container:
        mock_container.supports_logprob_extraction.return_value = True
        mock_container.encode_tokens.return_value = list(range(5))

        params = MagicMock()
        params.max_seq_len = 5
        model_card = MagicMock(parameters=params)
        mock_container.model_info.return_value = model_card

        async def mock_generate(*_a, **_kw):
            return {
                "text": "Hello world",
                "prompt_tokens": 2,
                "generated_tokens": 0,
                "prompt_token_strings": ["Hello", "world"],
                "prompt_token_logprobs": [None, -2.0],
                "offset": [0, 6],
                "top_logprobs": [None, {"world": -2.0}],
                "finish_reason": "stop",
            }

        mock_container.generate.side_effect = mock_generate
        mock_container.model_dir.name = "test-model"
        yield mock_container

@pytest.fixture
def dummy_config():
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

def test_best_of_not_supported(mock_model_container, dummy_config):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "best_of": 2},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 422

def test_best_of_less_than_n(mock_model_container, dummy_config):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "best_of": 1, "n": 2},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 422

def test_best_of_with_stream(mock_model_container, dummy_config):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "best_of": 1, "stream": True},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 422
