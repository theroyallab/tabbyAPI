from collections import deque

import torch

from exllamav3 import Filter

from common.logger import xlogger


def build_whitespace_bitmask(tokenizer) -> torch.Tensor:
    """Packed int32 bitmask (bit set = token is pure whitespace) over the vocab.

    A token counts as whitespace when its piece is empty or contains only
    whitespace, counting SentencePiece's "▁" as a space. Special tokens are
    never treated as whitespace.
    """

    vocab_size = tokenizer.actual_vocab_size
    piece_map = None
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and hasattr(inner, "get_vocab"):
        try:
            piece_map = {
                token_id: piece
                for piece, token_id in inner.get_vocab().items()
                if token_id < vocab_size
            }
        except Exception:
            piece_map = None
    if piece_map is None:
        piece_map = getattr(tokenizer, "extended_id_to_piece", None) or {}
    words = (vocab_size + 31) // 32
    mask = torch.zeros(words, dtype=torch.int32)
    for token_id, piece in piece_map.items():
        if piece is None or piece.startswith("<") and piece.endswith(">"):
            continue
        flat = piece.replace("▁", " ")
        if flat != "" and flat.strip() == "":
            mask[token_id >> 5] |= 1 << (token_id & 31)
    return mask


class WhitespaceStallGuardFilter(Filter):
    """Re-rails a grammar-constrained generation out of whitespace stalls.

    A grammar can only forbid tokens; it cannot compel the model to continue
    or stop. When the remaining legal continuations include unlimited
    whitespace, a model that "feels done" can emit whitespace forever,
    burning the token budget with output that never satisfies the grammar.

    This wrapper watches accepted tokens. If the model emits `stall_tokens`
    consecutive pure-whitespace tokens while the inner grammar filter is
    still incomplete, the next logit mask additionally excludes whitespace
    tokens, forcing the sampler onto a legal content token (or EOS). This
    advances the grammar instead of idling in a legal whitespace attractor.

    If excluding whitespace would empty the allowed set (the grammar truly
    requires whitespace), the unmasked grammar mask is returned unchanged.
    """

    def __init__(self, inner: Filter, whitespace_bitmask: torch.Tensor, stall_tokens: int = 8):
        if stall_tokens < 1:
            raise ValueError("stall_tokens must be at least 1")

        # Assign inner before super().__init__: the base class's is_active
        # assignment flows through the property setter, which mirrors into
        # the inner filter.
        self.inner = inner
        self._is_active = False
        self._recent: deque[int] = deque(maxlen=stall_tokens)
        self._stalled = False
        super().__init__(
            inner.tokenizer,
            inner.trigger_token,
            inner.prefix_str,
            inner.eos_after_completed,
        )
        self.whitespace_bitmask = whitespace_bitmask
        self.stall_tokens = stall_tokens
        # Per-token boolean view of the packed bitmask, for dense masks.
        bits = (
            (whitespace_bitmask.unsqueeze(-1) >> torch.arange(32, dtype=torch.int32)) & 1
        ).bool().reshape(-1)
        self._ws_token_bool = bits

    @property
    def is_active(self):
        return self._is_active

    @is_active.setter
    def is_active(self, value):
        was = self._is_active
        self._is_active = value
        self.inner.is_active = value
        if value and not was:
            # Activating from a trigger: pre-trigger history is not part of
            # the constrained sequence.
            self._recent.clear()
            self._stalled = False

    # -- token bookkeeping -------------------------------------------------

    def _is_whitespace(self, token: int) -> bool:
        words = self.whitespace_bitmask
        return bool(token >= 0 and (words[token >> 5].item() >> (token & 31)) & 1)

    def _update_stall_state(self):
        self._stalled = (
            len(self._recent) == self.stall_tokens
            and not self.inner.is_completed()
            and all(self._is_whitespace(t) for t in self._recent)
        )

    # -- Filter interface ---------------------------------------------------

    def feed(self, token: int) -> bool:
        eos = self.inner.feed(token)
        self._recent.append(token)
        self._update_stall_state()
        return eos

    def rewind(self, num_tokens: int):
        self.inner.rewind(num_tokens)
        for _ in range(min(num_tokens, len(self._recent))):
            self._recent.pop()
        self._update_stall_state()

    def rollback_tokens(self, num_tokens: int) -> bool:
        eos = self.inner.rollback_tokens(num_tokens)
        for _ in range(min(num_tokens, len(self._recent))):
            self._recent.pop()
        self._update_stall_state()
        return eos

    def reset(self):
        self.inner.reset()
        self._recent.clear()
        self._stalled = False

    def is_completed(self) -> bool:
        return self.inner.is_completed()

    def use_background_worker(self) -> bool:
        return self.inner.use_background_worker()

    def attach(self, job):
        super().attach(job)
        self.inner.attach(job)

    def get_next_logit_mask(self) -> torch.Tensor:
        mask = self.inner.get_next_logit_mask()
        if not self._stalled:
            return mask

        if mask.dtype == torch.int32:
            # Packed bitmask: bit set = allowed. Clear whitespace bits.
            words = self.whitespace_bitmask
            guarded = mask & ~words[: mask.shape[-1]].to(mask.dtype)
            if (guarded != 0).any():
                return guarded
            xlogger.debug(
                "Whitespace stall guard active but every legal continuation "
                "is whitespace; leaving the grammar mask unchanged."
            )
            return mask

        # Dense additive mask: 0 = allowed, -inf = masked.
        guarded = mask.clone()
        ws = self._ws_token_bool[: guarded.shape[-1]]
        guarded[..., ws] = float("-inf")
        if torch.isfinite(guarded).any():
            return guarded
        xlogger.debug(
            "Whitespace stall guard active but every legal continuation "
            "is whitespace; leaving the grammar mask unchanged."
        )
        return mask