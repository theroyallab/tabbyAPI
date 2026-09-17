import unittest
from types import SimpleNamespace

import torch

from backends.exllamav3.whitespace_guard import (
    WhitespaceStallGuardFilter,
    build_whitespace_bitmask,
)


VOCAB = 8
PIECES = {
    0: "<eos>",
    1: "hello",
    2: " ",
    3: "\n",
    4: "▁",
    5: "▁▁",
    6: ",",
    7: "x",
}
WS_IDS = {2, 3, 4, 5}


def fake_tokenizer():
    return SimpleNamespace(actual_vocab_size=VOCAB, extended_id_to_piece=dict(PIECES))


def packed_mask(allowed_ids):
    mask = torch.zeros((VOCAB + 31) // 32, dtype=torch.int32)
    for token_id in allowed_ids:
        mask[token_id >> 5] |= 1 << (token_id & 31)
    return mask


def dense_mask(allowed_ids):
    mask = torch.zeros((1, VOCAB), dtype=torch.half)
    for token_id in range(VOCAB):
        if token_id not in allowed_ids:
            mask[0, token_id] = float("-inf")
    return mask


class FakeInner:
    def __init__(self, mask, trigger_token=None):
        self.tokenizer = None
        self.trigger_token = trigger_token
        self.prefix_str = None
        self.eos_after_completed = True
        self.is_active = trigger_token is None
        self._mask = mask
        self._completed = False
        self._eos_after_feed = False
        self.fed = []
        self.rewound = []
        self.rolled_back = []
        self.attached = None
        self.reset_count = 0

    def feed(self, token):
        self.fed.append(token)
        return self._eos_after_feed

    def rewind(self, num_tokens):
        self.rewound.append(num_tokens)

    def rollback_tokens(self, num_tokens):
        self.rolled_back.append(num_tokens)
        return False

    def reset(self):
        self.reset_count += 1

    def is_completed(self):
        return self._completed

    def use_background_worker(self):
        return True

    def attach(self, job):
        self.attached = job

    def get_next_logit_mask(self):
        return self._mask


class BuildBitmaskTests(unittest.TestCase):
    def test_whitespace_pieces_flagged(self):
        mask = build_whitespace_bitmask(fake_tokenizer())
        flagged = {i for i in range(VOCAB) if (mask[i >> 5].item() >> (i & 31)) & 1}
        self.assertEqual(flagged, WS_IDS)

    def test_missing_piece_map_is_safe(self):
        tok = SimpleNamespace(actual_vocab_size=4, extended_id_to_piece={})
        mask = build_whitespace_bitmask(tok)
        self.assertEqual(mask.abs().sum().item(), 0)


class StallGuardTests(unittest.TestCase):
    def guard(self, inner, stall_tokens=8):
        return WhitespaceStallGuardFilter(inner, build_whitespace_bitmask(fake_tokenizer()), stall_tokens)

    def test_inactive_below_threshold(self):
        inner = FakeInner(packed_mask({1, 2, 6}))
        guard = self.guard(inner)
        for token in [2, 3, 4, 5, 2, 3, 4]:  # 7 whitespace tokens
            guard.feed(token)
        self.assertFalse(guard._stalled)
        self.assertTrue(torch.equal(guard.get_next_logit_mask(), inner._mask))

    def test_activates_at_threshold_and_clears_whitespace(self):
        inner = FakeInner(packed_mask({1, 2, 3, 6}))
        guard = self.guard(inner)
        for token in [2, 3, 4, 5, 2, 3, 4, 3]:  # 8 whitespace tokens
            guard.feed(token)
        self.assertTrue(guard._stalled)
        mask = guard.get_next_logit_mask()
        allowed = {i for i in range(VOCAB) if (mask[i >> 5].item() >> (i & 31)) & 1}
        self.assertEqual(allowed, {1, 6})  # whitespace (2,3) railed out

    def test_non_whitespace_token_resets_run(self):
        inner = FakeInner(packed_mask({1, 2, 6}))
        guard = self.guard(inner)
        for token in [2, 3, 4, 5, 2, 3, 4, 1, 2, 3]:
            guard.feed(token)
        self.assertFalse(guard._stalled)

    def test_never_stalls_when_inner_completed(self):
        inner = FakeInner(packed_mask({1, 2, 6}))
        inner._completed = True
        guard = self.guard(inner)
        for token in WS_IDS:
            guard.feed(token)
        for token in WS_IDS:
            guard.feed(token)
        self.assertFalse(guard._stalled)

    def test_leaves_mask_when_all_legal_tokens_are_whitespace(self):
        inner = FakeInner(packed_mask({2, 3}))
        guard = self.guard(inner)
        for token in WS_IDS:
            guard.feed(token)
        for token in WS_IDS:
            guard.feed(token)
        self.assertTrue(guard._stalled)
        self.assertTrue(torch.equal(guard.get_next_logit_mask(), inner._mask))

    def test_rewind_retracts_stall(self):
        inner = FakeInner(packed_mask({1, 2, 6}))
        guard = self.guard(inner)
        for token in [*WS_IDS, *WS_IDS]:  # 8 ws tokens
            guard.feed(token)
        guard.rewind(3)  # back to 5 ws
        self.assertEqual(inner.rewound, [3])
        self.assertFalse(guard._stalled)
        for token in [4, 5, 2]:  # back to 8 ws
            guard.feed(token)
        self.assertTrue(guard._stalled)

    def test_rollback_tokens_delegates_and_retracts(self):
        inner = FakeInner(packed_mask({1, 2, 6}))
        guard = self.guard(inner)
        for token in [*WS_IDS, *WS_IDS]:  # 8 ws tokens
            guard.feed(token)
        self.assertTrue(guard._stalled)
        self.assertFalse(guard.rollback_tokens(2))
        self.assertEqual(inner.rolled_back, [2])
        self.assertFalse(guard._stalled)

    def test_feed_eos_condition_propagates(self):
        inner = FakeInner(packed_mask({1, 2, 6}))
        inner._eos_after_feed = True
        guard = self.guard(inner)
        self.assertTrue(guard.feed(2))
        inner._eos_after_feed = False
        self.assertFalse(guard.feed(3))

    def test_is_active_mirrors_into_inner(self):
        inner = FakeInner(packed_mask({1, 2, 6}), trigger_token=1)
        guard = self.guard(inner)
        self.assertFalse(guard.is_active)
        self.assertFalse(inner.is_active)
        guard.is_active = True
        self.assertTrue(inner.is_active)
        # Pre-trigger history is cleared on activation
        for token in WS_IDS:
            guard.feed(token)
        guard.is_active = False
        guard.is_active = True
        self.assertFalse(guard._stalled)

    def test_reset_clears_state(self):
        inner = FakeInner(packed_mask({1, 2, 6}))
        guard = self.guard(inner)
        for token in WS_IDS:
            guard.feed(token)
        guard.reset()
        self.assertEqual(inner.reset_count, 1)
        self.assertFalse(guard._stalled)

    def test_attach_reaches_inner(self):
        inner = FakeInner(packed_mask({1, 2, 6}))
        guard = self.guard(inner)
        job = SimpleNamespace(generator=SimpleNamespace(padded_vocab_size=32))
        guard.attach(job)
        self.assertIs(inner.attached, job)

    def test_dense_mask_path(self):
        inner = FakeInner(dense_mask({1, 2, 3, 6}))
        guard = self.guard(inner)
        for token in [*WS_IDS, *WS_IDS]:  # 8 ws tokens
            guard.feed(token)
        mask = guard.get_next_logit_mask()
        allowed = {i for i in range(VOCAB) if torch.isfinite(mask[0, i])}
        self.assertEqual(allowed, {1, 6})


if __name__ == "__main__":
    unittest.main()