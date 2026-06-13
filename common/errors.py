class ContextLengthExceededError(Exception):
    pass

def validate_context_requirements(
    context_len: int,
    max_seq_len: int,
    max_tokens: int,
    cache_max_num_tokens: int = None,
    max_rq_tokens: int = None,
    allocation_boundary: int = None,
):
    if context_len > max_seq_len:
        raise ContextLengthExceededError(
            f"Prompt length {context_len} is greater than max_seq_len {max_seq_len}"
        )
    if cache_max_num_tokens is not None and context_len + max_tokens > cache_max_num_tokens:
        raise ContextLengthExceededError(
            f"Prompt length {context_len} + max_tokens {max_tokens} is greater than cache size {cache_max_num_tokens}"
        )
