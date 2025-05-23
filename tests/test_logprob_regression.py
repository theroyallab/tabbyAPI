import pytest
from unittest.mock import patch, MagicMock
from collections import OrderedDict
from fastapi.testclient import TestClient

from common import tabby_config
from endpoints.server import setup_app
from endpoints.OAI.utils.completion import _create_response

# Configure logprob settings for the tests
tabby_config.config.developer.enable_logprob = True
tabby_config.config.developer.disable_request_streaming = True
tabby_config.config.network.disable_auth = True

app = setup_app()
client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_model_container():
    """Provide a mocked model container with logprob support."""
    with patch("common.model.container") as mock_container:
        mock_container.supports_logprob_extraction.return_value = True
        mock_container.encode_tokens.return_value = list(range(2))
        mock_container.decode_tokens.return_value = "prompt"

        params = MagicMock()
        params.max_seq_len = 10
        model_card = MagicMock(parameters=params)
        mock_container.model_info.return_value = model_card

        async def mock_generate(*_a, **_kw):
            return {
                "text": "",
                "prompt_tokens": 2,
                "generated_tokens": 0,
                "prompt_token_strings": ["Hello", "world"],
                "prompt_token_logprobs": [None, -0.5],
                "offset": [0, 6],
                "top_logprobs": [None, {"world": -0.5}],
                "finish_reason": "stop",
            }

        mock_container.generate.side_effect = mock_generate
        mock_container.compute_sequence_logprobs.side_effect = mock_generate
        mock_container.model_dir.name = "test-model"
        yield mock_container


@pytest.fixture(autouse=True)
def setup_auth():
    with patch("common.auth.AUTH_KEYS") as mock_auth_keys:
        mock_auth_keys.verify_key.return_value = True
        with patch("common.auth.DISABLE_AUTH", True):
            yield mock_auth_keys


def test_use_sequence_data_structures():
    """_create_response should preserve token order as lists."""
    generation = {
        "text": "Hello world",
        "prompt_tokens": 0,
        "generated_tokens": 2,
        "token_probs": OrderedDict([("Hello", -0.1), (" world", -0.2)]),
        "logprobs": [{"Hello": -0.1}, {" world": -0.2}],
        "offset": [0, 6],
    }
    response = _create_response("abc", generation, "test-model")
    logprobs = response.choices[0].logprobs
    assert isinstance(logprobs.tokens, list)
    assert isinstance(logprobs.token_logprobs, list)
    assert logprobs.tokens == ["Hello", " world"]
    assert logprobs.token_logprobs == [-0.1, -0.2]
    assert logprobs.top_logprobs == [{"Hello": -0.1}, {" world": -0.2}]
    assert logprobs.text_offset == [0, 6]


def test_text_offset_start_indices(mock_model_container):
    """Offsets should be start positions of each token."""
    resp = client.post(
        "/v1/logprob",
        json={"prompt": "Hello world"},
        headers={"X-API-Key": "key"},
    )
    assert resp.status_code == 200
    data = resp.json()["choices"][0]["logprobs"]
    assert data["text_offset"] == [0, 6]

    tokens = data["tokens"]
    computed = []
    current = 0
    for t in tokens:
        computed.append(current)
        current += len(t)
    assert data["text_offset"] == computed


def test_top_logprobs_alignment(mock_model_container):
    resp = client.post(
        "/v1/logprob",
        json={"prompt": "Hello world", "logprobs": 1},
        headers={"X-API-Key": "key"},
    )
    assert resp.status_code == 200
    top_logprobs = resp.json()["choices"][0]["logprobs"]["top_logprobs"]
    for entry, token in zip(top_logprobs, ["Hello", "world"]):
        assert token in entry


def test_completion_prompt_logprobs_echo(mock_model_container):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello world", "max_tokens": 0, "logprobs": 2, "echo": True},
        headers={"X-API-Key": "key"},
    )
    assert resp.status_code == 200
    data = resp.json()["choices"][0]["logprobs"]
    assert data["tokens"] == ["Hello", "world"]


def test_streaming_logprobs_rejected(mock_model_container):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 1, "logprobs": 1, "stream": True},
        headers={"X-API-Key": "key"},
    )
    assert resp.status_code == 501
