"""
Batch perplexity evaluation implementation for ExLlamaV3.
Processes entire sequences at once instead of token-by-token.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from loguru import logger


class BatchPerplexityEvaluator:
    """Efficient batch perplexity computation for ExLlamaV3."""
    
    def __init__(self, model_container):
        self.container = model_container
        self.model = model_container.model
        self.tokenizer = model_container.tokenizer
        
    @torch.inference_mode()
    def compute_batch_perplexity(
        self, 
        token_ids: torch.Tensor,
        max_length: Optional[int] = None
    ) -> Tuple[float, int]:
        """
        Compute perplexity for a batch of sequences efficiently.
        
        Args:
            token_ids: Tensor of shape (batch_size, seq_len) or (seq_len,)
            max_length: Maximum context length to use
            
        Returns:
            Tuple of (perplexity, num_tokens_evaluated)
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            
        batch_size, seq_len = token_ids.shape
        device = self.model.device if hasattr(self.model, 'device') else torch.device('cuda')
        
        # Move to model device if needed
        if token_ids.device != device:
            token_ids = token_ids.to(device)
            
        # Use model's max length if not specified
        if max_length is None:
            max_length = self.container.generator_max_seq_len
            
        total_loss = 0.0
        total_tokens = 0
        
        # Process in chunks if sequence is too long
        for start_idx in range(0, seq_len - 1, max_length):
            end_idx = min(start_idx + max_length, seq_len)
            chunk_ids = token_ids[:, start_idx:end_idx]
            
            # Get model outputs for the chunk
            logits = self.model.forward(
                chunk_ids,
                {
                    "attn_mode": "flash_attn_nc",
                    "position": start_idx,
                    "last_token_only": False,  # Need all positions
                }
            )
            
            # Shift for next-token prediction
            # logits: (batch, seq_len, vocab_size)
            # We predict token i+1 from position i
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk_ids[:, 1:].contiguous()
            
            # Use cross-entropy for efficient computation
            # This computes -log P(correct token) for each position
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )
            
            num_tokens = shift_labels.numel()
            total_loss += loss.item()
            total_tokens += num_tokens
            
        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity, total_tokens
    
    def compute_dataset_perplexity(
        self,
        texts: List[str],
        batch_size: int = 1,
        max_length: Optional[int] = None,
        show_progress: bool = True
    ) -> float:
        """
        Compute perplexity over a dataset of texts.
        
        Args:
            texts: List of text strings to evaluate
            batch_size: Number of texts to process together
            max_length: Maximum context length
            show_progress: Whether to show progress
            
        Returns:
            Overall perplexity
        """
        total_loss = 0.0
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            batch_ids = []
            for text in batch_texts:
                ids = self.tokenizer.encode(text, add_bos=True)
                if ids.dim() > 1:
                    ids = ids.squeeze(0)
                batch_ids.append(ids)
                
            # Pad sequences to same length
            max_len = max(ids.size(0) for ids in batch_ids)
            padded_ids = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
            
            for j, ids in enumerate(batch_ids):
                padded_ids[j, :ids.size(0)] = ids
                
            # Compute perplexity for batch
            batch_ppl, batch_tokens = self.compute_batch_perplexity(
                padded_ids, max_length
            )
            
            # Accumulate (in log space for numerical stability)
            total_loss += torch.log(torch.tensor(batch_ppl)) * batch_tokens
            total_tokens += batch_tokens
            
            if show_progress and (i + batch_size) % 10 == 0:
                logger.info(f"Processed {i + batch_size}/{len(texts)} texts")
                
        # Final perplexity
        avg_loss = total_loss / total_tokens
        final_perplexity = torch.exp(avg_loss).item()
        
        return final_perplexity


# Integration with existing ExLlamaV3Container
def add_batch_perplexity_method(container_class):
    """Add batch perplexity evaluation to ExLlamaV3Container."""
    
    def compute_batch_perplexity(self, token_ids, max_length=None):
        evaluator = BatchPerplexityEvaluator(self)
        return evaluator.compute_batch_perplexity(token_ids, max_length)
    
    container_class.compute_batch_perplexity = compute_batch_perplexity
    
    return container_class