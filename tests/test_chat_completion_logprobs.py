import asyncio
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import httpx
import json

from common import tabby_config
from endpoints.server import setup_app

# Enable logprob support for testing
tabby_config.config.developer.enable_logprob = True
tabby_config.config.developer.disable_request_streaming = False  # Keep streaming enabled for stream tests
tabby_config.config.developer.logprob_timeout_seconds = 60.0
tabby_config.config.network.disable_auth = True

# Setup the app
app = setup_app()
client = TestClient(app)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def mock_model_container():
    """
    Stubbed model container that supports log-prob extraction for chat completion testing.
    """
    with patch("common.model.container") as mock_container:
        # Enable logprob support
        mock_container.supports_logprob_extraction.return_value = True
        mock_container.encode_tokens.return_value = list(range(5))  # 5 tokens
        
        # For token ID decoding
        mock_container.decode_tokens.return_value = "Decoded token text"

        # Setup model info
        params = MagicMock()
        params.max_seq_len = 100
        model_card = MagicMock(parameters=params)
        mock_container.model_info.return_value = model_card

        # Mock generate for non-streaming responses
        async def mock_generate(*_a, **kwargs):
            # Extract logprobs from the data object
            data = _a[2] if len(_a) > 2 else None  # data is the 3rd positional argument
            logprobs_enabled = getattr(data, 'logprobs', None)
            top_logprobs_count = getattr(data, 'top_logprobs', None)
            
            # Handle both new and old logprobs parameter styles
            if top_logprobs_count is not None:
                logprobs_count = top_logprobs_count
            elif isinstance(logprobs_enabled, int):
                logprobs_count = logprobs_enabled
            elif logprobs_enabled is True:
                logprobs_count = 5  # Default when logprobs=True but no top_logprobs specified
            elif logprobs_enabled is False or logprobs_enabled is None:
                logprobs_count = 0
            else:
                logprobs_count = 0
            
            # Sample response with logprobs
            response = {
                "text": "I'm an AI assistant.",
                "prompt_tokens": 10,
                "generated_tokens": 5,
                "finish_reason": "stop",
            }
            
            # Add logprobs if requested (even if no alternatives are requested)
            if logprobs_enabled is not None and logprobs_enabled is not False:
                # These are the token strings that should align with logprobs
                response["generated_token_strings_list"] = ["I'm", " an", " AI", " assistant", "."]
                
                # The logprob for each token
                response["token_logprobs_list"] = [-1.5, -0.8, -1.2, -0.5, -0.3]
                
                # Alternative logprobs for each position - limit to requested count
                all_alternatives = [
                    {"I": -2.0, "Hello": -2.5, "Hey": -3.0},
                    {" a": -1.5, " the": -2.0, " this": -2.2},
                    {" language": -2.0, " chat": -2.5, " helpful": -3.0},
                    {" model": -1.0, " bot": -1.5, " tool": -2.0},
                    {"!": -0.8, ",": -1.2, "?": -1.5}
                ]
                
                # Limit each position to the requested number of alternatives
                limited_alternatives = []
                for alt_dict in all_alternatives:
                    if logprobs_count == 0:
                        # No alternatives when logprobs_count is 0
                        limited_alternatives.append({})
                    else:
                        # Sort by logprob (highest first) and take the top N
                        sorted_items = sorted(alt_dict.items(), key=lambda x: x[1], reverse=True)[:logprobs_count]
                        limited_alternatives.append(dict(sorted_items))
                
                response["alternative_logprobs_list"] = limited_alternatives
            
            return response

        # Mock generate_gen for streaming responses
        async def mock_generate_gen(*_a, **kwargs):
            # Extract logprobs from the data object
            data = _a[2] if len(_a) > 2 else None
            logprobs_enabled = getattr(data, 'logprobs', None)
            top_logprobs_count = getattr(data, 'top_logprobs', None)
            
            # Handle both new and old logprobs parameter styles
            if top_logprobs_count is not None:
                logprobs_count = top_logprobs_count
            elif isinstance(logprobs_enabled, int):
                logprobs_count = logprobs_enabled
            elif logprobs_enabled is True:
                logprobs_count = 5  # Default when logprobs=True but no top_logprobs specified
            elif logprobs_enabled is False or logprobs_enabled is None:
                logprobs_count = 0
            else:
                logprobs_count = 0
            tokens = ["I'm", " an", " AI", " assistant", "."]
            
            for i, token in enumerate(tokens):
                event = {
                    "text": token,
                    "index": 0,  # First choice
                }
                
                # Add logprobs if requested
                if logprobs_count > 0:
                    event["token_logprob"] = [-1.5, -0.8, -1.2, -0.5, -0.3][i]
                    
                    # Alternative logprobs for current token
                    alternatives = [
                        {"I": -2.0, "Hello": -2.5, "Hey": -3.0},
                        {" a": -1.5, " the": -2.0, " this": -2.2},
                        {" language": -2.0, " chat": -2.5, " helpful": -3.0},
                        {" model": -1.0, " bot": -1.5, " tool": -2.0},
                        {"!": -0.8, ",": -1.2, "?": -1.5}
                    ][i]
                    
                    # Only include top N alternatives based on logprobs_count
                    if logprobs_count > 0:
                        # Sort alternatives by logprob value (highest first) and take top N
                        sorted_alts = dict(sorted(alternatives.items(), key=lambda x: x[1], reverse=True)[:logprobs_count])
                        event["alternative_logprobs"] = sorted_alts
                
                yield event
            
            # Final event with finish reason
            yield {
                "finish_reason": "stop",
                "index": 0,
                "prompt_tokens": 10,
                "generated_tokens": 5
            }

        # Set up the mock methods
        mock_container.generate.side_effect = mock_generate
        mock_container.generate_gen.side_effect = mock_generate_gen
        mock_container.compute_sequence_logprobs.side_effect = mock_generate
        mock_container.model_dir.name = "test-model"
        
        # Mock prompt template render method (async)
        async def mock_prompt_render(template_vars):
            return "Mocked prompt from template"
        
        # Mock prompt template extract_metadata method (async)
        async def mock_extract_metadata(template_vars):
            metadata = MagicMock()
            metadata.stop_strings = []
            metadata.tool_starts = None
            return metadata
        
        mock_container.prompt_template.render.side_effect = mock_prompt_render
        mock_container.prompt_template.extract_metadata.side_effect = mock_extract_metadata
        
        # Mock get_special_tokens method
        mock_container.get_special_tokens.return_value = {
            "bos_token": "<s>",
            "eos_token": "</s>",
        }
        
        yield mock_container


@pytest.fixture(autouse=True)
def setup_auth():
    # Mock auth keys to avoid NoneType error
    with patch("common.auth.AUTH_KEYS") as mock_auth_keys:
        mock_auth_keys.verify_key.return_value = True
        with patch("common.auth.DISABLE_AUTH", True):
            yield mock_auth_keys


# --------------------------------------------------------------------------- #
# Non-streaming chat completion tests
# --------------------------------------------------------------------------- #
def test_chat_completion_with_logprobs():
    """Test that chat completion returns logprobs when requested."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
            "top_logprobs": 2
        },
        headers={"X-API-Key": "test_api_key"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify basic response structure
    assert "choices" in data
    assert len(data["choices"]) == 1
    assert "logprobs" in data["choices"][0]
    
    # Verify logprobs content
    logprobs = data["choices"][0]["logprobs"]
    assert "content" in logprobs
    
    # Check token count matches the expected number
    assert len(logprobs["content"]) == 5  # Should have 5 tokens
    
    # Verify each token has the right structure
    for token_info in logprobs["content"]:
        assert "token" in token_info
        assert "logprob" in token_info
        assert isinstance(token_info["logprob"], float)
        
        # Verify top_logprobs
        assert "top_logprobs" in token_info
        
        # We requested 2 top_logprobs
        assert len(token_info["top_logprobs"]) <= 2
        
        # Verify structure of each top_logprob entry
        for alt_logprob in token_info["top_logprobs"]:
            assert "token" in alt_logprob
            assert "logprob" in alt_logprob
            assert isinstance(alt_logprob["logprob"], float)


def test_chat_completion_with_logprobs_zero_top():
    """Test that chat completion works with logprobs but no top_logprobs."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
            "top_logprobs": 0
        },
        headers={"X-API-Key": "test_api_key"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify logprobs
    logprobs = data["choices"][0]["logprobs"]
    
    # Check each token has empty top_logprobs list
    for token_info in logprobs["content"]:
        assert token_info["top_logprobs"] == []


def test_chat_completion_with_no_logprobs():
    """Test that chat completion doesn't include logprobs when not requested."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers={"X-API-Key": "test_api_key"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify no logprobs
    assert "logprobs" not in data["choices"][0]


# --------------------------------------------------------------------------- #
# Streaming chat completion tests
# --------------------------------------------------------------------------- #
def test_chat_completion_stream_with_logprobs():
    """Test that streaming chat completion returns logprobs for each token."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
            "top_logprobs": 2,
            "stream": True
        },
        headers={"X-API-Key": "test_api_key"},
    )
    
    assert response.status_code == 200
    
    # Parse SSE stream
    chunks = []
    for line in response.iter_lines():
        if line.startswith(b"data: "):
            if line.startswith(b"data: [DONE]"):
                break
            chunk_data = json.loads(line.decode("utf-8")[6:])  # Skip "data: "
            chunks.append(chunk_data)
    
    # Check we have the expected number of content chunks (5 tokens + 1 finish chunk)
    assert len(chunks) > 0
    
    # Verify token chunks have logprobs
    token_chunks = [c for c in chunks if c["choices"][0].get("delta", {}).get("content")]
    
    for chunk in token_chunks:
        assert "logprobs" in chunk["choices"][0]
        logprobs = chunk["choices"][0]["logprobs"]
        
        # Each streaming chunk has only one token
        assert len(logprobs["content"]) == 1
        
        token_info = logprobs["content"][0]
        assert "token" in token_info
        assert "logprob" in token_info
        assert isinstance(token_info["logprob"], float)
        
        # Check top_logprobs
        assert "top_logprobs" in token_info
        assert len(token_info["top_logprobs"]) <= 2


def test_chat_completion_stream_with_logprobs_max_top():
    """Test streaming chat completion with maximum allowed top_logprobs."""
    # Assuming 5 is the maximum value based on test data
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
            "top_logprobs": 5,
            "stream": True
        },
        headers={"X-API-Key": "test_api_key"},
    )
    
    assert response.status_code == 200
    
    # Parse SSE stream
    for line in response.iter_lines():
        if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
            chunk_data = json.loads(line.decode("utf-8")[6:])
            
            # Skip chunks without content
            if not chunk_data["choices"][0].get("delta", {}).get("content"):
                continue
                
            # Check the first content chunk
            logprobs = chunk_data["choices"][0]["logprobs"]
            token_info = logprobs["content"][0]
            
            # Top logprobs should have at most 5 entries (or fewer if not enough alternatives)
            assert len(token_info["top_logprobs"]) <= 5
            break


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_chat_completion_logprobs_token_alignment():
    """Test that tokens align correctly with their logprobs."""
    with patch("common.model.container") as mock_container:
        # Create a custom mock for this test
        mock_container.supports_logprob_extraction.return_value = True
        
        # Expected token sequence
        expected_tokens = ["Hello", " world", "!"]
        
        async def mock_gen():
            return {
                "text": "Hello world!",
                "prompt_tokens": 5,
                "generated_tokens": 3,
                "generated_token_strings_list": expected_tokens,
                "token_logprobs_list": [-1.0, -0.5, -0.2],
                "alternative_logprobs_list": [
                    {"Hi": -2.0},
                    {" there": -1.5},
                    {".": -1.0}
                ],
                "finish_reason": "stop",
            }
        
        mock_container.generate.side_effect = mock_gen
        mock_container.model_info.return_value = MagicMock(parameters=MagicMock(max_seq_len=100))
        mock_container.model_dir.name = "test-model"
        
        # Create app with our custom mock
        with patch("common.model.container", mock_container):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "logprobs": True,
                    "top_logprobs": 1
                },
                headers={"X-API-Key": "test_api_key"},
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Get the logprobs
            logprobs = data["choices"][0]["logprobs"]
            
            # Check token alignment
            assert len(logprobs["content"]) == len(expected_tokens)
            
            for i, token_info in enumerate(logprobs["content"]):
                # Token string should match what the backend provided
                assert token_info["token"] == expected_tokens[i]


def test_chat_completion_empty_response():
    """Test logprobs with an empty response."""
    with patch("common.model.container") as mock_container:
        mock_container.supports_logprob_extraction.return_value = True
        
        async def mock_gen():
            return {
                "text": "",
                "prompt_tokens": 5,
                "generated_tokens": 0,
                "generated_token_strings_list": [],
                "token_logprobs_list": [],
                "alternative_logprobs_list": [],
                "finish_reason": "stop",
            }
        
        mock_container.generate.side_effect = mock_gen
        mock_container.model_info.return_value = MagicMock(parameters=MagicMock(max_seq_len=100))
        mock_container.model_dir.name = "test-model"
        
        with patch("common.model.container", mock_container):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": ""}],
                    "logprobs": True,
                    "top_logprobs": 1
                },
                headers={"X-API-Key": "test_api_key"},
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Check that logprobs is present but content is empty
            logprobs = data["choices"][0]["logprobs"]
            assert logprobs["content"] == []


def test_chat_completion_special_tokens():
    """Test logprobs with special tokens."""
    with patch("common.model.container") as mock_container:
        mock_container.supports_logprob_extraction.return_value = True
        
        # Include some special tokens
        special_tokens = ["<s>", "User:", "Assistant:", "</s>"]
        
        async def mock_gen():
            return {
                "text": "Response with special tokens",
                "prompt_tokens": 5,
                "generated_tokens": 4,
                "generated_token_strings_list": special_tokens,
                "token_logprobs_list": [-0.1, -0.2, -0.3, -0.4],
                "alternative_logprobs_list": [
                    {"<pad>": -1.0},
                    {"System:": -1.5},
                    {"Human:": -1.2},
                    {"<eos>": -1.0}
                ],
                "finish_reason": "stop",
            }
        
        mock_container.generate.side_effect = mock_gen
        mock_container.model_info.return_value = MagicMock(parameters=MagicMock(max_seq_len=100))
        mock_container.model_dir.name = "test-model"
        
        with patch("common.model.container", mock_container):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "logprobs": True,
                    "top_logprobs": 1
                },
                headers={"X-API-Key": "test_api_key"},
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Check special tokens are present in logprobs
            logprobs = data["choices"][0]["logprobs"]
            tokens = [item["token"] for item in logprobs["content"]]
            
            for special_token in special_tokens:
                assert special_token in tokens


def test_chat_completion_with_invalid_logprobs_value():
    """Test that providing an invalid logprobs value returns an error."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": "invalid"  # Should be boolean
        },
        headers={"X-API-Key": "test_api_key"},
    )
    
    # Should return a validation error
    assert response.status_code == 422


def test_chat_completion_with_negative_top_logprobs():
    """Test that providing a negative top_logprobs value returns an error."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
            "top_logprobs": -1  # Invalid negative value
        },
        headers={"X-API-Key": "test_api_key"},
    )
    
    # Should return a validation error
    assert response.status_code == 422


def test_chat_completion_with_too_large_top_logprobs():
    """Test that providing a top_logprobs value larger than allowed works properly."""
    # If max allowed is 5 but user requests 10, it should be capped at 5 (not error)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
            "top_logprobs": 10  # Larger than mock implementation supports
        },
        headers={"X-API-Key": "test_api_key"},
    )
    
    # Should still succeed, just capped at max
    assert response.status_code == 200
    data = response.json()
    
    # Each token should have at most 5 alternatives
    logprobs = data["choices"][0]["logprobs"]
    for token_info in logprobs["content"]:
        assert len(token_info["top_logprobs"]) <= 5
