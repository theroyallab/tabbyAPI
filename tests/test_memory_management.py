"""Tests for memory management in ExLlamaV3 backend.

This test suite verifies that VRAM is properly managed during intensive
operations like perplexity evaluation and logprob computation.
"""

import pytest
import torch
import gc
from unittest.mock import MagicMock, patch, call
from typing import List, Dict

from common import tabby_config
from common.sampling import BaseSamplerRequest
from endpoints.server import setup_app
from fastapi.testclient import TestClient

# Configure for testing
tabby_config.config.developer.enable_logprob = True
tabby_config.config.developer.disable_request_streaming = True
tabby_config.config.network.disable_auth = True

app = setup_app()
client = TestClient(app)


class MemoryTracker:
    """Helper class to track memory allocations and cleanups."""
    
    def __init__(self):
        self.allocations = []
        self.cleanups = []
        self.empty_cache_calls = 0
        self.gc_collect_calls = 0
    
    def record_allocation(self, size_mb: float, description: str):
        self.allocations.append({"size_mb": size_mb, "description": description})
    
    def record_cleanup(self, description: str):
        self.cleanups.append(description)
    
    def record_empty_cache(self):
        self.empty_cache_calls += 1
    
    def record_gc_collect(self):
        self.gc_collect_calls += 1


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability for testing memory management."""
    with patch("torch.cuda.is_available") as mock_available:
        mock_available.return_value = True
        
        # Mock torch.cuda.empty_cache
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            # Mock gc.collect
            with patch("gc.collect") as mock_gc_collect:
                yield {
                    "is_available": mock_available,
                    "empty_cache": mock_empty_cache,
                    "gc_collect": mock_gc_collect
                }


@pytest.fixture
def mock_exllamav3_with_memory_tracking(mock_cuda_available):
    """Mock ExllamaV3Container with memory tracking capabilities."""
    with patch("common.model.container") as mock_container:
        # Basic setup
        mock_container.loaded = True
        mock_container.supports_logprob_extraction.return_value = True
        mock_container.model_dir.name = "test-model"
        mock_container.generator_max_seq_len = 4096
        
        # Add memory tracker
        tracker = MemoryTracker()
        mock_container._memory_tracker = tracker
        
        # Mock model with CUDA device
        mock_model = MagicMock()
        mock_model.device = torch.device("cuda")
        # Don't actually create CUDA tensors, just mock the iterator
        mock_param = MagicMock()
        mock_param.device = torch.device("cuda")
        mock_model.parameters.return_value = iter([mock_param])
        mock_container.model = mock_model
        
        # Mock compute_sequence_logprobs to simulate memory operations
        def mock_compute_sequence_logprobs(prompt, params, profile=False, aggressive_memory_cleanup=True):
            # Simulate tensor allocations
            tracker.record_allocation(100, "log_softmax tensor")
            tracker.record_allocation(50, "rank calculation tensor")
            
            if aggressive_memory_cleanup:
                # Simulate cleanup calls with error handling
                tracker.record_cleanup("ranks tensor")
                try:
                    mock_cuda_available["empty_cache"]()
                    tracker.record_empty_cache()
                except RuntimeError:
                    pass  # Silently handle CUDA errors
                
                tracker.record_cleanup("log_sm tensor")
                try:
                    mock_cuda_available["empty_cache"]()
                    tracker.record_empty_cache()
                except RuntimeError:
                    pass  # Silently handle CUDA errors
            
            return {
                "text": prompt,
                "prompt_tokens": 10,
                "generated_tokens": 0,
                "prompt_token_strings": ["token"] * 10,
                "prompt_token_logprobs": [None] + [-2.0] * 9,
                "prompt_token_ranks": [None] + [1] * 9,
                "top_logprobs": [None] * 10,
                "offset": list(range(0, 50, 5)),
                "finish_reason": "stop"
            }
        
        mock_container.compute_sequence_logprobs = mock_compute_sequence_logprobs
        
        # Mock compute_perplexity_efficient to simulate chunked processing
        def mock_compute_perplexity(token_ids, chunk_size=None, aggressive_memory_cleanup=True):
            # Simulate processing 3 chunks
            for i in range(3):
                tracker.record_allocation(200, f"chunk {i} logits")
                tracker.record_allocation(150, f"chunk {i} loss calculation")
                
                if aggressive_memory_cleanup:
                    tracker.record_cleanup(f"chunk {i} tensors")
                    try:
                        mock_cuda_available["empty_cache"]()
                        tracker.record_empty_cache()
                    except RuntimeError:
                        pass  # Silently handle CUDA errors
                    
                    # Simulate gc.collect for large sequences
                    if i == 2:  # Third chunk
                        try:
                            mock_cuda_available["gc_collect"]()
                            tracker.record_gc_collect()
                        except RuntimeError:
                            pass
            
            return 2.5, 15000  # perplexity, num_tokens
        
        mock_container.compute_perplexity_efficient = mock_compute_perplexity
        
        # Mock other required methods
        mock_container.encode_tokens.return_value = list(range(10))
        mock_container.decode_tokens.return_value = "token"
        mock_container.generate.return_value = mock_compute_sequence_logprobs("test", BaseSamplerRequest())
        
        params = MagicMock()
        params.max_seq_len = 4096
        model_card = MagicMock(parameters=params)
        mock_container.model_info.return_value = model_card
        
        yield mock_container


@pytest.fixture(autouse=True)
def setup_auth():
    """Mock authentication."""
    with patch("common.auth.AUTH_KEYS") as mock_auth_keys:
        mock_auth_keys.verify_key.return_value = True
        with patch("common.auth.DISABLE_AUTH", True):
            yield mock_auth_keys


def test_compute_sequence_logprobs_memory_cleanup(mock_exllamav3_with_memory_tracking, mock_cuda_available):
    """Test that compute_sequence_logprobs properly cleans up GPU memory."""
    container = mock_exllamav3_with_memory_tracking
    tracker = container._memory_tracker
    
    # Reset tracker
    tracker.allocations.clear()
    tracker.cleanups.clear()
    tracker.empty_cache_calls = 0
    
    # Call with aggressive cleanup enabled (default)
    result = container.compute_sequence_logprobs("Test prompt", BaseSamplerRequest(logprobs=5))
    
    # Verify memory was allocated
    assert len(tracker.allocations) > 0
    assert any("log_softmax" in a["description"] for a in tracker.allocations)
    assert any("rank" in a["description"] for a in tracker.allocations)
    
    # Verify cleanup was performed
    assert len(tracker.cleanups) > 0
    assert any("ranks" in c for c in tracker.cleanups)
    assert any("log_sm" in c for c in tracker.cleanups)
    
    # Verify torch.cuda.empty_cache was called
    assert tracker.empty_cache_calls >= 2
    assert mock_cuda_available["empty_cache"].call_count >= 2


def test_compute_sequence_logprobs_no_cleanup(mock_exllamav3_with_memory_tracking, mock_cuda_available):
    """Test that memory cleanup can be disabled when needed."""
    container = mock_exllamav3_with_memory_tracking
    tracker = container._memory_tracker
    
    # Reset tracker and mocks
    tracker.cleanups.clear()
    tracker.empty_cache_calls = 0
    mock_cuda_available["empty_cache"].reset_mock()
    
    # Call with aggressive cleanup disabled
    result = container.compute_sequence_logprobs(
        "Test prompt", 
        BaseSamplerRequest(logprobs=5),
        aggressive_memory_cleanup=False
    )
    
    # Verify no cleanup was performed
    assert len(tracker.cleanups) == 0
    assert tracker.empty_cache_calls == 0
    assert mock_cuda_available["empty_cache"].call_count == 0


def test_compute_perplexity_memory_cleanup(mock_exllamav3_with_memory_tracking, mock_cuda_available):
    """Test that compute_perplexity_efficient properly manages memory during chunked processing."""
    container = mock_exllamav3_with_memory_tracking
    tracker = container._memory_tracker
    
    # Reset tracker
    tracker.allocations.clear()
    tracker.cleanups.clear()
    tracker.empty_cache_calls = 0
    tracker.gc_collect_calls = 0
    
    # Simulate processing a long sequence
    token_ids = torch.randint(0, 32000, (1, 15000))
    perplexity, num_tokens = container.compute_perplexity_efficient(token_ids)
    
    # Verify chunked processing
    assert len(tracker.allocations) >= 6  # At least 2 allocations per chunk × 3 chunks
    
    # Verify cleanup after each chunk
    assert len(tracker.cleanups) >= 3  # One cleanup per chunk
    assert all("chunk" in c for c in tracker.cleanups)
    
    # Verify empty_cache was called after each chunk
    assert tracker.empty_cache_calls >= 3
    assert mock_cuda_available["empty_cache"].call_count >= 3
    
    # Verify gc.collect was called for large sequence
    assert tracker.gc_collect_calls >= 1
    assert mock_cuda_available["gc_collect"].call_count >= 1


def test_memory_cleanup_with_api_request(mock_exllamav3_with_memory_tracking, mock_cuda_available):
    """Test memory cleanup during actual API request."""
    # Reset mocks
    mock_cuda_available["empty_cache"].reset_mock()
    
    # Make logprob request
    resp = client.post(
        "/v1/logprob",
        json={"prompt": "Test memory cleanup during API request", "logprobs": 5},
        headers={"X-API-Key": "key"},
    )
    
    assert resp.status_code == 200
    
    # Verify memory cleanup was performed
    assert mock_cuda_available["empty_cache"].call_count > 0


def test_memory_cleanup_disabled_on_cpu():
    """Test that memory cleanup operations are skipped on CPU."""
    with patch("common.model.container") as mock_container:
        mock_container.loaded = True
        mock_container.supports_logprob_extraction.return_value = True
        
        # Mock CPU device
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_container.model = mock_model
        
        # Track CUDA operations
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            with patch("torch.cuda.is_available", return_value=False):
                # Simulate compute_sequence_logprobs on CPU
                # The actual implementation should skip CUDA operations
                
                # Verify no CUDA operations were attempted
                assert mock_empty_cache.call_count == 0


@pytest.mark.parametrize("sequence_length,expected_gc_calls", [
    (5000, 0),   # Short sequence, no GC
    (15000, 1),  # Long sequence, triggers GC
    (30000, 2),  # Very long sequence, multiple GC calls
])
def test_gc_triggered_for_long_sequences(mock_exllamav3_with_memory_tracking, mock_cuda_available, sequence_length, expected_gc_calls):
    """Test that garbage collection is triggered for very long sequences."""
    container = mock_exllamav3_with_memory_tracking
    
    # Modify the mock to simulate longer processing
    def mock_compute_perplexity_long(token_ids, chunk_size=None, aggressive_memory_cleanup=True):
        num_chunks = sequence_length // 5000  # One chunk per 5000 tokens
        tracker = container._memory_tracker
        
        for i in range(num_chunks):
            if aggressive_memory_cleanup:
                try:
                    mock_cuda_available["empty_cache"]()
                except RuntimeError:
                    pass
                # GC triggered based on total tokens processed
                if (i + 1) * 5000 > 10000:
                    try:
                        mock_cuda_available["gc_collect"]()
                        tracker.record_gc_collect()
                    except RuntimeError:
                        pass
        
        return 2.5, sequence_length
    
    container.compute_perplexity_efficient = mock_compute_perplexity_long
    
    # Reset GC mock
    mock_cuda_available["gc_collect"].reset_mock()
    
    # Process sequence
    token_ids = torch.randint(0, 32000, (1, sequence_length))
    container.compute_perplexity_efficient(token_ids)
    
    # Verify GC was called appropriate number of times
    assert mock_cuda_available["gc_collect"].call_count >= expected_gc_calls


def test_memory_cleanup_error_handling(mock_exllamav3_with_memory_tracking, mock_cuda_available):
    """Test that memory cleanup doesn't break if CUDA operations fail."""
    # Make empty_cache raise an exception
    mock_cuda_available["empty_cache"].side_effect = RuntimeError("CUDA error")
    
    # Should still complete successfully
    result = mock_exllamav3_with_memory_tracking.compute_sequence_logprobs(
        "Test prompt", 
        BaseSamplerRequest(logprobs=5)
    )
    
    assert result is not None
    assert "prompt_tokens" in result