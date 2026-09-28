import unittest
from types import SimpleNamespace
from unittest.mock import patch

from exllamav3.modules.gated_delta_net import GDNLayerState
from exllamav3.modules.ple import PLELayerState
from exllamav3.modules.sliding_attn import SWALayerState

from backends.exllamav3.model import ExllamaV3Container


def gdn_states(slots, history):
    module = SimpleNamespace(
        fdim_qkv=10240,
        conv_kernel_size=4,
        num_v_heads=48,
        k_head_dim=128,
        v_head_dim=128,
    )
    return {i: GDNLayerState(module, slots, history, 0) for i in range(48)}


class RecurrentSlotCostTests(unittest.TestCase):
    def log(self, cache):
        container = ExllamaV3Container.__new__(ExllamaV3Container)
        container.cache = cache
        with patch("backends.exllamav3.model.xlogger.info") as info:
            container.log_recurrent_slot_cost()
        return info

    def test_gdn_four_slots(self):
        states = gdn_states(4, 4)
        self.assertTrue(all(s.recurrent_state.is_meta for s in states.values()))
        info = self.log(SimpleNamespace(recurrent_layers=states, num_slots=4, max_history=4))
        info.assert_called_once_with(
            "Recurrent state storage: 4 slots, 2910 MiB total "
            "(728 MiB per slot), max_history: 4. "
            "Single-user setups can set max_batch_size: 1."
        )

    def test_gdn_single_slot_has_no_reduction_hint(self):
        info = self.log(
            SimpleNamespace(recurrent_layers=gdn_states(1, 4), num_slots=1, max_history=4)
        )
        info.assert_called_once_with(
            "Recurrent state storage: 1 slot, 728 MiB total (728 MiB per slot), max_history: 4."
        )

    def test_no_history_is_silent(self):
        self.log(
            SimpleNamespace(recurrent_layers=gdn_states(4, 0), num_slots=4, max_history=0)
        ).assert_not_called()

    def test_no_recurrent_layers_is_silent(self):
        for cache in (SimpleNamespace(), SimpleNamespace(recurrent_layers={})):
            with self.subTest(cache=cache):
                self.log(cache).assert_not_called()

    def test_sliding_attention_uses_storage_not_history_times_checkpoint(self):
        module = SimpleNamespace(
            kv_state_size=4352, sliding_window=4096, num_kv_heads=4, head_dim=128
        )
        state = SWALayerState(module, 4, 4, 0)
        self.assertTrue(state.k_state.is_meta)
        self.assertNotEqual(state.storage_size(), 4 * 5 * state.get_checkpoint_size())
        info = self.log(SimpleNamespace(recurrent_layers={0: state}, num_slots=4, max_history=4))
        info.assert_called_once_with(
            "Recurrent state storage: 4 slots, 34 MiB total "
            "(8 MiB per slot), max_history: 4. "
            "Single-user setups can set max_batch_size: 1."
        )

    def test_mixed_state_storage_is_not_labeled_as_vram(self):
        states = gdn_states(4, 4)
        module = SimpleNamespace(
            conv_state_len=4,
            ple_embedding=SimpleNamespace(context_len=4),
            hc_mult=4,
            hidden_size=256,
        )
        states[48] = PLELayerState(module, 4, 4, 0)
        info = self.log(SimpleNamespace(recurrent_layers=states, num_slots=4, max_history=4))
        info.assert_called_once()
        message = info.call_args.args[0]
        total_mib = sum(state.storage_size() for state in states.values()) / 1024**2
        self.assertIn(f"{total_mib:.0f} MiB total", message)
        self.assertNotIn("VRAM", message)
        self.assertNotIn("states", message)


if __name__ == "__main__":
    unittest.main()
