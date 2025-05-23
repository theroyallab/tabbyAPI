import asyncio
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import httpx

from common import tabby_config
from endpoints.server import setup_app

# Directly modify the config
tabby_config.config.developer.enable_logprob = True
tabby_config.config.developer.disable_request_streaming = True
tabby_config.config.developer.logprob_timeout_seconds = 60.0
tabby_config.config.network.disable_auth = True

# Now create the app
app = setup_app()
client = TestClient(app)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def mock_model_container():
    """
    Stubbed model container that supports log-prob extraction.
    """
    # We need to ensure this fixture runs before any tests
    with patch("common.model.container") as mock_container:
        # Enable logprob support
        mock_container.supports_logprob_extraction.return_value = True
        mock_container.encode_tokens.return_value = list(range(5))  # 5 tokens
        
        # For token ID decoding
        mock_container.decode_tokens.return_value = "Decoded token text"

        # Setup model info
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
                "prompt_token_logprobs": [
                    None,
                    -2.0,
                ],  # First token has None for logprob
                "offset": [0, 6],  # Character offsets in the original text
                "top_logprobs": [
                    None,
                    {"world": -2.0, "planet": -3.5},
                ],  # Top alternatives
                "finish_reason": "stop",
            }

        mock_container.generate.side_effect = mock_generate
        mock_container.compute_sequence_logprobs.side_effect = mock_generate
        mock_container.model_dir.name = "test-model"
        
        # Yield the mock to make it available to tests that need to modify it
        yield mock_container


@pytest.fixture
def enable_logprob_endpoint():
    """
    Marker fixture for tests that require the logprob endpoint to be enabled.
    The actual patching happens at the module level.
    """
    # No need to do anything here since we've patched at module level
    yield None
        
@pytest.fixture(autouse=True)
def setup_auth():
    # Mock auth keys to avoid NoneType error
    with patch("common.auth.AUTH_KEYS") as mock_auth_keys:
        mock_auth_keys.verify_key.return_value = True
        with patch("common.auth.DISABLE_AUTH", True):
            yield mock_auth_keys


# --------------------------------------------------------------------------- #
# Endpoint disabled
# --------------------------------------------------------------------------- #
def test_logprob_endpoint_disabled():
    # Temporarily disable the logprob endpoint
    original_value = tabby_config.config.developer.enable_logprob
    try:
        tabby_config.config.developer.enable_logprob = False
        
        # Need to create a new client
        temp_app = setup_app()
        temp_client = TestClient(temp_app)
        
        resp = temp_client.post(
            "/v1/logprob",
            json={"prompt": "Hello world"},
            headers={"X-API-Key": "test_api_key"},
        )
        assert resp.status_code == 501
        assert "not enabled" in resp.json()["error"]["message"]
    finally:
        # Restore the original value
        tabby_config.config.developer.enable_logprob = original_value


# --------------------------------------------------------------------------- #
# Happy-path (n is omitted or 1)
# --------------------------------------------------------------------------- #
def test_logprob_basic(mock_model_container, enable_logprob_endpoint):
    resp = client.post(
        "/v1/logprob",
        json={"prompt": "Hello world"},
        headers={"X-API-Key": "test_api_key"},
    )
    # Print error for debugging
    print(f"Status: {resp.status_code}, Response: {resp.text}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "logprob"
    assert len(data["choices"]) == 1

    toks = data["choices"][0]["logprobs"]["tokens"]
    logps = data["choices"][0]["logprobs"]["token_logprobs"]
    offsets = data["choices"][0]["logprobs"]["text_offset"]
    top_logps = data["choices"][0]["logprobs"]["top_logprobs"]

    assert toks == ["Hello", "world"]
    assert logps == [None, -2.0]  # First token has None for logprob
    assert offsets == [0, 6]  # Character positions in original text
    assert top_logps[0] is None  # First token has None for top logprobs
    assert "world" in top_logps[1]  # Second token has "world" in alternatives
    assert "planet" in top_logps[1]  # Second token has "planet" in alternatives
    assert (
        data["choices"][0]["logprobs"]["sum"] == -2.0
    )  # Sum only includes non-None values
    assert data["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 0,
        "total_tokens": 2,
    }


def test_completion_prompt_logprobs(mock_model_container, enable_logprob_endpoint):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello world", "max_tokens": 0, "logprobs": 2, "echo": True},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["text"] == "Hello world"
    assert data["choices"][0]["logprobs"]["tokens"] == ["Hello", "world"]


def test_completion_prompt_logprobs_no_echo(
    mock_model_container, enable_logprob_endpoint
):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello world", "max_tokens": 0, "logprobs": 2},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["text"] == ""
    assert data["choices"][0]["logprobs"]["tokens"] == ["Hello", "world"]


def test_completion_logprobs_exceeds_limit(
    mock_model_container, enable_logprob_endpoint
):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello world", "max_tokens": 0, "logprobs": 6},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 422  # Pydantic validation produces 422 status code
    # The message may vary, but we're only checking the status code


def test_completion_logprobs_zero_rejected(mock_model_container, enable_logprob_endpoint):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello world", "max_tokens": 0, "logprobs": 0},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 422  # Pydantic validation produces 422 status code
    # The message may vary, but we're only checking the status code


def test_completion_echo(mock_model_container, enable_logprob_endpoint):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello world", "max_tokens": 1, "echo": True},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["text"] == "Hello worldHello world"
    
def test_completion_echo_with_token_ids(mock_model_container, enable_logprob_endpoint):
    # Mock the decode_tokens method
    mock_model_container.decode_tokens.return_value = "Decoded token text"
    
    resp = client.post(
        "/v1/completions",
        json={"prompt": [1, 2, 3], "max_tokens": 1, "echo": True},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["text"] == "Decoded token textHello world"
    
    # Verify decode_tokens was called with the token IDs
    mock_model_container.decode_tokens.assert_called_once_with([1, 2, 3])
    
def test_completion_echo_with_batch_prompts(mock_model_container, enable_logprob_endpoint):
    # Our implementation joins the prompts with a newline if they're a list of strings
    # Let's verify the text includes the list-joined prompt followed by the output
    
    resp = client.post(
        "/v1/completions",
        json={
            "prompt": ["Prompt 1", "Prompt 2"], 
            "max_tokens": 1, 
            "echo": True,
            "n": 1  # Use n=1 to simplify the test
        },
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["choices"]) == 1
    # Verify it contains the joined prompt and the output
    assert "Prompt 1\nPrompt 2" in data["choices"][0]["text"]
    assert "Hello world" in data["choices"][0]["text"]


def test_completion_echo_logprobs_comprehensive(
    mock_model_container, enable_logprob_endpoint
):
    """Test comprehensive echo+logprobs behavior with all arrays."""
    async def mock_generate(*_a, **_kw):
        return {
            "text": " generated",  # Single generated token
            "prompt_tokens": 2,
            "generated_tokens": 1,
            "finish_reason": "stop",
        }

    async def mock_logprobs(prompt: str, params=None):
        return {
            "text": prompt + " generated",
            "prompt_tokens": 3,  # "Hello", "world", "generated"
            "generated_tokens": 0,
            "prompt_token_strings": ["Hello", "world", "generated"],
            "prompt_token_logprobs": [None, -1.0, -0.5],  # First token has no context
            "top_logprobs": [
                None,                    # No alternatives for first token
                {"world": -1.0, "there": -1.2},  # Alternatives for "world"
                {"generated": -0.5, "output": -0.7}  # Alternatives for "generated"
            ],
            "offset": [0, 6, 12],  # Character positions
            "finish_reason": "stop",
        }

    mock_model_container.generate.side_effect = mock_generate
    mock_model_container.compute_sequence_logprobs.side_effect = mock_logprobs

    # Test echo=True with logprobs
    resp = client.post(
        "/v1/completions",
        json={
            "prompt": "Hello world", 
            "max_tokens": 1, 
            "logprobs": 2, 
            "echo": True
        },
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    
    choice = data["choices"][0]
    logprobs = choice["logprobs"]
    
    # Verify text includes prompt when echo=True
    assert choice["text"] == "Hello world generated"
    
    # Verify all logprob arrays include prompt tokens when echo=True
    assert logprobs["tokens"] == ["Hello", "world", "generated"]
    assert logprobs["token_logprobs"] == [None, -1.0, -0.5]
    assert logprobs["text_offset"] == [0, 6, 12]
    
    # Verify top_logprobs includes prompt alternatives
    assert len(logprobs["top_logprobs"]) == 3
    assert logprobs["top_logprobs"][0] is None  # First token has no alternatives
    assert logprobs["top_logprobs"][1]["world"] == -1.0
    assert logprobs["top_logprobs"][2]["generated"] == -0.5
    
    # Test echo=False with logprobs for comparison
    resp_no_echo = client.post(
        "/v1/completions",
        json={
            "prompt": "Hello world", 
            "max_tokens": 1, 
            "logprobs": 2, 
            "echo": False
        },
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp_no_echo.status_code == 200
    data_no_echo = resp_no_echo.json()
    
    choice_no_echo = data_no_echo["choices"][0]
    logprobs_no_echo = choice_no_echo["logprobs"]
    
    # Verify text excludes prompt when echo=False
    assert choice_no_echo["text"] == " generated"
    
    # Verify logprob arrays exclude prompt tokens when echo=False
    assert logprobs_no_echo["tokens"] == ["generated"]  # Only generated token
    assert logprobs_no_echo["token_logprobs"] == [-0.5]  # Only generated logprob
    assert logprobs_no_echo["text_offset"] == [12]  # Only generated offset
    assert len(logprobs_no_echo["top_logprobs"]) == 1  # Only generated alternatives


# --------------------------------------------------------------------------- #
# Validation / error handling
# --------------------------------------------------------------------------- #
def test_logprob_n_missing_defaults_to_one(
    mock_model_container, enable_logprob_endpoint
):
    resp = client.post(
        "/v1/logprob",
        json={"prompt": "Hello world"},  # n omitted ⇒ 1
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200


def test_logprob_n_greater_than_one_rejected(enable_logprob_endpoint):
    resp = client.post(
        "/v1/logprob",
        json={"prompt": "Hello world", "n": 2},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 422


def test_logprob_n_zero_rejected(enable_logprob_endpoint):
    resp = client.post(
        "/v1/logprob",
        json={"prompt": "Hello world", "n": 0},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 422


def test_logprob_with_max_tokens_rejected(enable_logprob_endpoint):
    resp = client.post(
        "/v1/logprob",
        json={"prompt": "Hello world", "max_tokens": 1},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 422


def test_model_not_supporting_logprob(mock_model_container, enable_logprob_endpoint):
    # Temporarily disable logprob support in the mock container
    original_return = mock_model_container.supports_logprob_extraction.return_value
    mock_model_container.supports_logprob_extraction.return_value = False
    
    try:
        resp = client.post(
            "/v1/logprob",
            json={"prompt": "Hello world"},
            headers={"X-API-Key": "test_api_key"},
        )
        assert resp.status_code == 501
        assert "not support logprob extraction" in resp.json()["error"]["message"]
    finally:
        # Restore the original return value
        mock_model_container.supports_logprob_extraction.return_value = original_return


def test_logprob_timeout(mock_model_container, enable_logprob_endpoint):
    # Make the compute_sequence_logprobs method raise a TimeoutError
    mock_model_container.compute_sequence_logprobs.side_effect = asyncio.TimeoutError("Operation timed out")
    
    resp = client.post(
        "/v1/logprob",
        json={"prompt": "Hello world"},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 408
    assert "timed out" in resp.json()["error"]["message"]
    
    # Reset the side effect for other tests
    async def mock_generate(*_a, **_kw):
        return {
            "text": "Hello world",
            "prompt_tokens": 2,
            "generated_tokens": 0,
            "prompt_token_strings": ["Hello", "world"],
            "prompt_token_logprobs": [
                None,
                -2.0,
            ],  # First token has None for logprob
            "offset": [0, 6],  # Character offsets in the original text
            "top_logprobs": [
                None,
                {"world": -2.0, "planet": -3.5},
            ],  # Top alternatives
            "finish_reason": "stop",
        }
    mock_model_container.compute_sequence_logprobs.side_effect = mock_generate


def test_logprob_prompt_too_long(mock_model_container, enable_logprob_endpoint):
    mock_model_container.encode_tokens.return_value = list(range(10))
    mock_model_container.model_info.return_value.parameters.max_seq_len = 5
    resp = client.post(
        "/v1/logprob",
        json={"prompt": "A" * 10},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 413


# --------------------------------------------------------------------------- #
# Concurrency smoke-test (still one choice per request)
# --------------------------------------------------------------------------- #
def test_logprob_concurrent(mock_model_container, enable_logprob_endpoint):
    # We're using a synchronous test client, not a real async test
    responses = []
    for i in range(3):
        resp = client.post(
            "/v1/logprob",
            json={"prompt": f"Hello {i}"},
            headers={"X-API-Key": "test_api_key"},
        )
        responses.append(resp)

    for resp in responses:
        assert resp.status_code == 200
        assert len(resp.json()["choices"]) == 1


# --------------------------------------------------------------------------- #
# Enhanced parameter validation tests
# --------------------------------------------------------------------------- #
def test_completion_logprobs_stream_rejected(
    mock_model_container, enable_logprob_endpoint
):
    """Test that streaming with logprobs is rejected for completions endpoint."""
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 1, "logprobs": 1, "stream": True},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 501
    response_data = resp.json()
    # Error can be in detail field, message field, or nested under error.message
    error_msg = (
        response_data.get("detail") or 
        response_data.get("message", "") or 
        response_data.get("error", {}).get("message", "")
    )
    assert "Streaming with logprobs is not supported" in error_msg


def test_completion_top_logprobs_stream_rejected(
    mock_model_container, enable_logprob_endpoint
):
    """Test that streaming with top_logprobs is also rejected for completions."""
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 1, "top_logprobs": 2, "stream": True},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 501
    response_data = resp.json()
    error_msg = (
        response_data.get("detail") or 
        response_data.get("message", "") or 
        response_data.get("error", {}).get("message", "")
    )
    assert "Streaming with logprobs is not supported" in error_msg


def test_completion_both_logprobs_params_stream_rejected(
    mock_model_container, enable_logprob_endpoint
):
    """Test that streaming is rejected when both logprobs parameters are provided."""
    resp = client.post(
        "/v1/completions",
        json={
            "prompt": "Hello", 
            "max_tokens": 1, 
            "logprobs": 1, 
            "top_logprobs": 2, 
            "stream": True
        },
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 501
    response_data = resp.json()
    error_msg = (
        response_data.get("detail") or 
        response_data.get("message", "") or 
        response_data.get("error", {}).get("message", "")
    )
    assert "Streaming with logprobs is not supported" in error_msg


def test_completion_stream_without_logprobs_works(
    mock_model_container, enable_logprob_endpoint
):
    """Test that streaming works when no logprobs are requested."""
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 1, "stream": True},
        headers={"X-API-Key": "test_api_key"},
    )
    # This should work (streaming without logprobs)
    assert resp.status_code == 200


def test_completion_non_stream_with_logprobs_works(
    mock_model_container, enable_logprob_endpoint
):
    """Test that non-streaming requests with logprobs work correctly."""
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 1, "logprobs": 1},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200


def test_completion_non_stream_with_top_logprobs_works(
    mock_model_container, enable_logprob_endpoint
):
    """Test that non-streaming requests with top_logprobs work correctly."""
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 1, "top_logprobs": 2},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200


def test_chat_completion_stream_with_logprobs_works(
    mock_model_container, enable_logprob_endpoint
):
    """Test that chat completions support streaming with logprobs."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
            "top_logprobs": 2,
            "stream": True
        },
        headers={"X-API-Key": "test_api_key"},
    )
    # Chat completions should support streaming with logprobs
    assert resp.status_code == 200


def test_chat_completion_non_stream_with_logprobs_works(
    mock_model_container, enable_logprob_endpoint
):
    """Test that chat completions support non-streaming with logprobs."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
            "top_logprobs": 2
        },
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200


def test_logprob_parameter_edge_cases(
    mock_model_container, enable_logprob_endpoint
):
    """Test edge cases for logprob parameters."""
    # Test logprobs=False, top_logprobs=0 (should not be considered as requesting logprobs)
    resp = client.post(
        "/v1/completions",
        json={
            "prompt": "Hello", 
            "max_tokens": 1, 
            "logprobs": False,
            "top_logprobs": 0,
            "stream": True
        },
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200  # Should work since no logprobs requested
    
    # Test logprobs=None, top_logprobs=None
    resp = client.post(
        "/v1/completions",
        json={
            "prompt": "Hello", 
            "max_tokens": 1, 
            "stream": True
        },
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200  # Should work


def test_logprob_long_prompt_offsets(mock_model_container, enable_logprob_endpoint):
    """Test logprobs with long prompts to verify offset handling."""
    tokens = [f"t{i}" for i in range(100)]

    async def mock_generate(*_a, **_kw):
        return {
            "text": "x" * 100,
            "prompt_tokens": 100,
            "generated_tokens": 0,
            "prompt_token_strings": tokens,
            "prompt_token_logprobs": [0.0] * 100,
            "offset": list(range(100)),
            "top_logprobs": [None] * 100,
            "finish_reason": "stop",
        }

    mock_model_container.compute_sequence_logprobs.side_effect = mock_generate
    mock_model_container.generate.side_effect = mock_generate
    mock_model_container.encode_tokens.return_value = list(range(100))
    mock_model_container.model_info.return_value.parameters.max_seq_len = 200

    resp = client.post(
        "/v1/logprob",
        json={"prompt": "A" * 100},
        headers={"X-API-Key": "test_api_key"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["choices"][0]["logprobs"]["tokens"]) == 100
    assert len(data["choices"][0]["logprobs"]["text_offset"]) == 100