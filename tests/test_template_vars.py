import unittest
from types import SimpleNamespace

from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils.chat_completion import resolve_template_vars


def request(**kwargs):
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **kwargs)


def container(default=None, force=None):
    return SimpleNamespace(
        template_vars_default=default or {},
        template_vars_force=force or {},
    )


class ResolveTemplateVarsTests(unittest.TestCase):
    def test_defaults_applied(self):
        out = resolve_template_vars(request(), container(default={"reasoning_effort": "high"}))
        self.assertEqual(out["reasoning_effort"], "high")

    def test_client_template_vars_override_defaults(self):
        out = resolve_template_vars(
            request(template_vars={"reasoning_effort": "no_think"}),
            container(default={"reasoning_effort": "high"}),
        )
        self.assertEqual(out["reasoning_effort"], "no_think")

    def test_top_level_reasoning_effort_mapped(self):
        out = resolve_template_vars(request(reasoning_effort="low"), container())
        self.assertEqual(out["reasoning_effort"], "low")

    def test_top_level_reasoning_effort_overrides_defaults(self):
        out = resolve_template_vars(
            request(reasoning_effort="low"),
            container(default={"reasoning_effort": "high"}),
        )
        self.assertEqual(out["reasoning_effort"], "low")

    def test_explicit_template_var_beats_top_level(self):
        out = resolve_template_vars(
            request(
                reasoning_effort="low",
                template_vars={"reasoning_effort": "high"},
            ),
            container(),
        )
        self.assertEqual(out["reasoning_effort"], "high")

    def test_force_overrides_client(self):
        out = resolve_template_vars(
            request(
                reasoning_effort="no_think",
                template_vars={"enable_thinking": False},
            ),
            container(force={"reasoning_effort": "high", "enable_thinking": True}),
        )
        self.assertEqual(out["reasoning_effort"], "high")
        self.assertTrue(out["enable_thinking"])

    def test_chat_template_kwargs_alias(self):
        out = resolve_template_vars(
            request(chat_template_kwargs={"reasoning_effort": "high"}), container()
        )
        self.assertEqual(out["reasoning_effort"], "high")

    def test_unrelated_vars_pass_through(self):
        out = resolve_template_vars(
            request(template_vars={"custom": 1}),
            container(default={"other": 2}, force={"third": 3}),
        )
        self.assertEqual(out, {"custom": 1, "other": 2, "third": 3})

    def test_enable_thinking_mapped(self):
        out = resolve_template_vars(request(enable_thinking=False), container())
        self.assertIs(out["enable_thinking"], False)

    def test_verbosity_mapped(self):
        out = resolve_template_vars(request(verbosity="low"), container())
        self.assertEqual(out["verbosity"], "low")

    def test_reasoning_object_mapped(self):
        out = resolve_template_vars(
            request(reasoning={"effort": "high", "enabled": True}), container()
        )
        self.assertEqual(out["reasoning_effort"], "high")
        self.assertIs(out["enable_thinking"], True)

    def test_flat_fields_beat_reasoning_object(self):
        out = resolve_template_vars(
            request(
                reasoning={"effort": "high", "enabled": True},
                reasoning_effort="no_think",
                enable_thinking=False,
            ),
            container(),
        )
        self.assertEqual(out["reasoning_effort"], "no_think")
        self.assertIs(out["enable_thinking"], False)

    def test_reasoning_object_max_tokens_ignored(self):
        out = resolve_template_vars(request(reasoning={"max_tokens": 4096}), container())
        self.assertNotIn("reasoning_effort", out)
        self.assertNotIn("enable_thinking", out)
        self.assertNotIn("max_tokens", out)

    def test_reasoning_object_unknown_keys_ignored(self):
        out = resolve_template_vars(
            request(reasoning={"effort": "low", "exclude": True}), container()
        )
        self.assertEqual(out["reasoning_effort"], "low")
        self.assertNotIn("exclude", out)

    def test_template_vars_beat_flat_fields(self):
        out = resolve_template_vars(
            request(
                enable_thinking=False,
                verbosity="low",
                template_vars={"enable_thinking": True, "verbosity": "high"},
            ),
            container(),
        )
        self.assertIs(out["enable_thinking"], True)
        self.assertEqual(out["verbosity"], "high")


if __name__ == "__main__":
    unittest.main()
