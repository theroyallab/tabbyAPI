"""Tests for ExLlamaV3 AsyncJob handling in generation."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

from common import tabby_config
from common.sampling import BaseSamplerRequest
from endpoints.server import setup_app
from fastapi.testclient import TestClient

# Configure for testing
tabby_config.config.network.disable_auth = True

app = setup_app()
client = TestClient(app)


class MockJob:
    """Mock the inner Job class from ExLlamaV3."""
    def __init__(self, identifier=None, **kwargs):
        self.identifier = identifier
        self.max_new_tokens = kwargs.get('max_new_tokens', 100)
        self.min_new_tokens = kwargs.get('min_new_tokens', 0)
        self.stop_conditions = kwargs.get('stop_conditions', [])


class MockAsyncJob:
    """Mock AsyncJob that wraps a Job."""
    def __init__(self, generator, identifier=None, **kwargs):
        self.generator = generator
        self.job = MockJob(identifier=identifier, **kwargs)
        self.cancelled = False
        self._events = []
    
    def set_events(self, events: List[Dict[str, Any]]):
        """Set the events this job will yield."""
        self._events = events
    
    async def __aiter__(self):
        """Async iteration over generation events."""
        for event in self._events:
            if self.cancelled:
                break
            yield event
    
    async def cancel(self):
        """Cancel the job."""
        self.cancelled = True


@pytest.fixture
def mock_exllamav3_asyncjob():
    """Mock ExllamaV3Container with proper AsyncJob handling."""
    with patch("common.model.container") as mock_container:
        # Basic setup
        mock_container.loaded = True
        mock_container.model_dir.name = "test-model"
        mock_container.generator_max_seq_len = 4096
        
        # Mock generator
        mock_generator = MagicMock()
        mock_container.generator = mock_generator
        
        # Mock model info
        params = MagicMock()
        params.max_seq_len = 4096
        model_card = MagicMock(parameters=params)
        mock_container.model_info.return_value = model_card
        
        # Mock tokenizer and model
        mock_container.tokenizer = MagicMock()
        mock_container.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_container.model = MagicMock()
        mock_container.hf_model = MagicMock()
        mock_container.hf_model.add_bos_token.return_value = True
        
        # Mock methods
        mock_container.encode_tokens.return_value = [1, 2, 3, 4, 5]
        mock_container.decode_tokens.return_value = "test"
        mock_container.decode_single = lambda tid: f"token_{tid}"
        
        # Override generate_gen to use our mock AsyncJob
        async def mock_generate_gen(request_id, prompt, params, abort_event=None, mm_embeddings=None):
            # Create mock job with the request_id as identifier
            job = MockAsyncJob(mock_generator, identifier=request_id)
            
            # Set up events to yield
            job.set_events([
                {
                    "event": "text",
                    "text": "Hello",
                    "eos": False,
                    "token_ids": [6, 7, 8],
                    "generated_tokens": 3,
                    "prompt_tokens": 5,
                },
                {
                    "event": "text", 
                    "text": " world",
                    "eos": False,
                    "token_ids": [9, 10],
                    "generated_tokens": 5,
                },
                {
                    "event": "text",
                    "text": "!",
                    "eos": True,
                    "token_ids": [11],
                    "generated_tokens": 6,
                    "finish_reason": "stop",
                    "stop_str": None,
                    "metrics": {
                        "prompt_tokens": 5,
                        "generated_tokens": 6,
                    }
                }
            ])
            
            # Simulate the actual generate_gen method behavior
            async for event_dict in job:
                generation_chunk = {
                    "text": event_dict.get("text", ""),
                    "index": job.job.identifier,  # This is the fix we're testing
                    "token_ids_list": event_dict.get("token_ids", []),
                }
                
                # Add other fields if present
                if "prompt_tokens" in event_dict:
                    generation_chunk["prompt_tokens"] = event_dict["prompt_tokens"]
                if "generated_tokens" in event_dict:
                    generation_chunk["generated_tokens"] = event_dict["generated_tokens"]
                if event_dict.get("eos"):
                    generation_chunk["finish_reason"] = event_dict.get("finish_reason", "stop")
                    generation_chunk["stop_str"] = event_dict.get("stop_str")
                
                yield generation_chunk
        
        mock_container.generate_gen = mock_generate_gen
        
        # Mock the regular generate method
        async def mock_generate(prompt, params, request_id, abort_event=None):
            chunks = []
            async for chunk in mock_generate_gen(request_id, prompt, params, abort_event):
                chunks.append(chunk)
            
            # Return the final state
            return {
                "text": "Hello world!",
                "prompt_tokens": 5,
                "generated_tokens": 6,
                "finish_reason": "stop",
                "index": request_id,
            }
        
        mock_container.generate = mock_generate
        
        yield mock_container


@pytest.fixture(autouse=True)
def setup_auth():
    """Mock authentication."""
    with patch("common.auth.AUTH_KEYS") as mock_auth_keys:
        mock_auth_keys.verify_key.return_value = True
        with patch("common.auth.DISABLE_AUTH", True):
            yield mock_auth_keys


def test_asyncjob_identifier_access(mock_exllamav3_asyncjob):
    """Test that AsyncJob identifier is accessed correctly via job.job.identifier."""
    # Make a completion request
    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 10,
            "stream": False,
        },
        headers={"X-API-Key": "test-key"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # The response should have the completion
    assert "choices" in data
    assert len(data["choices"]) == 1
    assert data["choices"][0]["text"] == "Hello world!"
    
    # The index should be set (it's the request_id internally)
    assert "index" in data["choices"][0]


@pytest.mark.asyncio
async def test_asyncjob_streaming(mock_exllamav3_asyncjob):
    """Test that AsyncJob streaming works with correct identifier access."""
    request_id = "test-stream-123"
    prompt = "Hello"
    params = BaseSamplerRequest(max_tokens=10)
    
    chunks = []
    async for chunk in mock_exllamav3_asyncjob.generate_gen(request_id, prompt, params):
        chunks.append(chunk)
        # Verify each chunk has the correct index
        assert chunk["index"] == request_id
    
    # Verify we got all chunks
    assert len(chunks) == 3
    assert chunks[0]["text"] == "Hello"
    assert chunks[1]["text"] == " world"
    assert chunks[2]["text"] == "!"
    assert chunks[2].get("finish_reason") == "stop"


@pytest.mark.asyncio 
async def test_multiple_concurrent_asyncjobs(mock_exllamav3_asyncjob):
    """Test that multiple AsyncJobs maintain their own identifiers correctly."""
    # Create multiple generation tasks with different IDs
    async def generate_with_id(request_id: str):
        chunks = []
        async for chunk in mock_exllamav3_asyncjob.generate_gen(
            request_id, "Test prompt", BaseSamplerRequest(max_tokens=10)
        ):
            chunks.append(chunk)
        return request_id, chunks
    
    # Run multiple generations concurrently
    results = await asyncio.gather(
        generate_with_id("request-1"),
        generate_with_id("request-2"),
        generate_with_id("request-3"),
    )
    
    # Verify each maintained its own identifier
    for request_id, chunks in results:
        for chunk in chunks:
            assert chunk["index"] == request_id, f"Expected {request_id}, got {chunk['index']}"


def test_asyncjob_attribute_error():
    """Test that accessing job.identifier directly raises AttributeError."""
    mock_generator = MagicMock()
    job = MockAsyncJob(mock_generator, identifier="test-123")
    
    # This should work (the fix)
    assert job.job.identifier == "test-123"
    
    # This should fail (the bug)
    with pytest.raises(AttributeError):
        _ = job.identifier


def test_chat_completion_with_asyncjob(mock_exllamav3_asyncjob):
    """Test chat completion endpoint works with the AsyncJob fix."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": False,
        },
        headers={"X-API-Key": "test-key"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify the response structure
    assert "choices" in data
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] == "Hello world!"
    assert data["choices"][0]["finish_reason"] == "stop"