#!/usr/bin/env python3
"""Test script to verify VRAM cleanup during perplexity evaluation."""

import torch
import gc
import time
from typing import List

def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def monitor_memory_during_computation():
    """Monitor GPU memory usage during a simulated perplexity computation."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory test")
        return
    
    device = torch.device("cuda")
    
    print("=== Testing VRAM cleanup during perplexity evaluation ===\n")
    
    # Simulate processing chunks like in compute_perplexity_efficient
    chunk_size = 2048
    num_chunks = 5
    vocab_size = 32000
    
    memory_usage = []
    
    # Initial memory
    initial_mem = get_gpu_memory_usage()
    print(f"Initial GPU memory: {initial_mem:.1f} MB")
    memory_usage.append(initial_mem)
    
    # Test without cleanup
    print("\n1. Testing WITHOUT memory cleanup:")
    for i in range(num_chunks):
        # Simulate forward pass
        chunk_ids = torch.randint(0, vocab_size, (1, chunk_size), device=device)
        logits = torch.randn(1, chunk_size, vocab_size, device=device)
        
        # Simulate loss calculation
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk_ids[:, 1:].contiguous()
        
        # Calculate cross entropy (simplified)
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction='sum'
        )
        
        # Get loss value
        loss_value = loss.item()
        
        mem_after = get_gpu_memory_usage()
        print(f"  Chunk {i+1}: {mem_after:.1f} MB (+{mem_after - initial_mem:.1f} MB)")
        memory_usage.append(mem_after)
    
    no_cleanup_peak = max(memory_usage) - initial_mem
    
    # Clear everything
    del chunk_ids, logits, shift_logits, shift_labels, loss
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(0.5)  # Give time for cleanup
    
    # Test with cleanup
    print("\n2. Testing WITH memory cleanup:")
    memory_usage = []
    initial_mem = get_gpu_memory_usage()
    print(f"Reset to: {initial_mem:.1f} MB")
    memory_usage.append(initial_mem)
    
    for i in range(num_chunks):
        # Simulate forward pass
        chunk_ids = torch.randint(0, vocab_size, (1, chunk_size), device=device)
        logits = torch.randn(1, chunk_size, vocab_size, device=device)
        
        # Simulate loss calculation
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk_ids[:, 1:].contiguous()
        
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction='sum'
        )
        
        loss_value = loss.item()
        
        # Clean up tensors from this chunk (as in optimized code)
        del logits, shift_logits, shift_labels, loss, chunk_ids
        
        # Clear CUDA cache after each chunk
        torch.cuda.empty_cache()
        
        mem_after = get_gpu_memory_usage()
        print(f"  Chunk {i+1}: {mem_after:.1f} MB (+{mem_after - initial_mem:.1f} MB)")
        memory_usage.append(mem_after)
    
    with_cleanup_peak = max(memory_usage) - initial_mem
    
    print(f"\n=== Results ===")
    print(f"Peak memory increase WITHOUT cleanup: {no_cleanup_peak:.1f} MB")
    print(f"Peak memory increase WITH cleanup: {with_cleanup_peak:.1f} MB")
    print(f"Memory saved: {no_cleanup_peak - with_cleanup_peak:.1f} MB")
    
    if with_cleanup_peak < no_cleanup_peak * 0.5:
        print("\n✓ Memory cleanup is working effectively!")
    else:
        print("\n⚠ Memory cleanup may need improvement")
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()

def test_logprobs_memory_cleanup():
    """Test memory cleanup in compute_sequence_logprobs."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory test")
        return
    
    device = torch.device("cuda")
    
    print("\n=== Testing VRAM cleanup in compute_sequence_logprobs ===\n")
    
    # Simulate a long sequence
    seq_len = 4096
    vocab_size = 32000
    
    initial_mem = get_gpu_memory_usage()
    print(f"Initial GPU memory: {initial_mem:.1f} MB")
    
    # Simulate log_softmax computation
    ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    logits = torch.randn(1, seq_len, vocab_size, device=device)
    
    print(f"After creating logits: {get_gpu_memory_usage():.1f} MB")
    
    # Compute log_softmax
    log_sm = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    del logits  # Free original logits
    
    mem_after_logsm = get_gpu_memory_usage()
    print(f"After log_softmax: {mem_after_logsm:.1f} MB")
    
    # Simulate rank calculation
    chosen = ids[0]
    chosen_next = chosen[1:].unsqueeze(-1)
    gathered = log_sm[0, :-1].gather(1, chosen_next).squeeze(-1)
    ranks = (log_sm[0, :-1] >= gathered.unsqueeze(-1)).sum(-1).add_(1)
    
    mem_after_ranks = get_gpu_memory_usage()
    print(f"After rank calculation: {mem_after_ranks:.1f} MB")
    
    # Clean up (as in optimized code)
    del ranks
    torch.cuda.empty_cache()
    
    mem_after_cleanup1 = get_gpu_memory_usage()
    print(f"After first cleanup: {mem_after_cleanup1:.1f} MB")
    
    # Clean up log_sm
    del log_sm, gathered, chosen_next
    torch.cuda.empty_cache()
    
    final_mem = get_gpu_memory_usage()
    print(f"After full cleanup: {final_mem:.1f} MB")
    
    print(f"\nMemory freed: {mem_after_ranks - final_mem:.1f} MB")
    
    # Final cleanup
    del ids, chosen
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    monitor_memory_during_computation()
    test_logprobs_memory_cleanup()
    
    print("\n✓ Memory cleanup tests completed!")