import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from common import model
from endpoints.OAI.types.chat_completion import ChatCompletionMessage, ChatCompletionRequest
from common.templating import PromptTemplate
from endpoints.OAI.utils.chat_completion import (
    CONTINUE_FINAL_MESSAGE_TAG,
    _chat_stream_collector,
    _continued_field,
    _cut_prompt_at_continue_tag,
    _mark_continued_final_message,
    _resolve_start_in_reasoning,
    _start_in_reasoning_mode,
    _user_suffix_len,
    apply_chat_template,
)

TAG = CONTINUE_FINAL_MESSAGE_TAG
BARE = TAG.strip()


def continue_request(**kwargs):
    kwargs.setdefault("messages", [ChatCompletionMessage(role="assistant", content="partial")])
    kwargs.setdefault("add_generation_prompt", False)
    kwargs.setdefault("continue_final_message", True)
    return ChatCompletionRequest(**kwargs)


class CutPromptTests(unittest.TestCase):
    def test_preserving_template(self):
        text = "The story begins"
        prompt = f"[INST] Tell me a story.[/INST] {text}{TAG}</s>"
        cut = _cut_prompt_at_continue_tag(prompt, text)
        self.assertEqual(cut, f"[INST] Tell me a story.[/INST] {text}")

    def test_trimming_template(self):
        # Template trimmed the tag's trailing space; user's own trailing
        # whitespace is trimmed the same way
        text = "The story begins "
        prompt = f"<s>user\nhi\nassistant\n{text}{BARE}</s>"
        cut = _cut_prompt_at_continue_tag(prompt, text)
        self.assertEqual(cut, "<s>user\nhi\nassistant\nThe story begins")

    def test_sentinel_dropped_raises_422(self):
        with self.assertRaises(HTTPException) as ctx:
            _cut_prompt_at_continue_tag("template rewrote everything", "partial")
        self.assertEqual(ctx.exception.status_code, 422)

    def test_user_content_containing_tag_literal(self):
        # rindex must find the appended sentinel, not the user's copy
        text = f"discussing {BARE} as a string"
        prompt = f"header {text}{TAG}eos"
        cut = _cut_prompt_at_continue_tag(prompt, text)
        self.assertEqual(cut, f"header {text}")


class MarkContinuedTests(unittest.TestCase):
    def test_requires_add_generation_prompt_false(self):
        data = continue_request(add_generation_prompt=True)
        with self.assertRaises(HTTPException) as ctx:
            _mark_continued_final_message(data)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_requires_messages(self):
        data = continue_request(messages=[])
        with self.assertRaises(HTTPException):
            _mark_continued_final_message(data)

    def test_requires_content(self):
        data = continue_request(messages=[ChatCompletionMessage(role="assistant", content=None)])
        with self.assertRaises(HTTPException):
            _mark_continued_final_message(data)

    def test_appends_tag_to_string_content(self):
        data = continue_request()
        text = _mark_continued_final_message(data)
        self.assertEqual(text, "partial")
        self.assertEqual(data.messages[-1].content, "partial" + TAG)


class StartInReasoningSuffixTests(unittest.TestCase):
    def setUp(self):
        self.container = SimpleNamespace(
            reasoning_start_token="<think>", reasoning_end_token="</think>"
        )
        patcher = patch("endpoints.OAI.utils.chat_completion.model")
        self.mock_model = patcher.start()
        self.mock_model.container = self.container
        self.addCleanup(patcher.stop)

    def test_template_appended_think(self):
        prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        self.assertTrue(_start_in_reasoning_mode(prompt))

    def test_long_user_continuation_extends_window(self):
        filler = "reasoning " * 50  # 500 chars, past the 256 window
        prompt = f"<|im_start|>assistant\n<think>\n{filler}"
        self.assertFalse(_start_in_reasoning_mode(prompt))
        self.assertTrue(_start_in_reasoning_mode(prompt, user_suffix_len=len(filler)))

    def test_closed_think_in_user_suffix_wins(self):
        # User continuation that already closed its think block
        suffix = "<think>\nthoughts</think>\n\nThe answer is"
        prompt = f"<|im_start|>assistant\n{suffix}"
        self.assertFalse(_start_in_reasoning_mode(prompt, user_suffix_len=len(suffix)))

    def test_tag_lookalike_in_user_suffix_ignored(self):
        # Tag-shaped text inside the user suffix must not disqualify
        suffix = "I should use <strong> tags here"
        prompt = f"<|im_start|>assistant\n<think>\n{suffix}"
        self.assertFalse(_start_in_reasoning_mode(prompt))
        self.assertTrue(_start_in_reasoning_mode(prompt, user_suffix_len=len(suffix)))

    def test_template_tag_outside_suffix_still_disqualifies(self):
        suffix = "user text"
        prompt = f"<think>\n<|im_start|>assistant\n{suffix}"
        self.assertFalse(_start_in_reasoning_mode(prompt, user_suffix_len=len(suffix)))


class UserSuffixLenTests(unittest.TestCase):
    def test_continue_plus_prefix(self):
        data = continue_request(response_prefix="Sure, ")
        self.assertEqual(_user_suffix_len(data), len("partial") + len("Sure, "))

    def test_prefix_requires_generation_prompt_or_continue(self):
        data = ChatCompletionRequest(
            messages=[ChatCompletionMessage(role="user", content="hi")],
            add_generation_prompt=False,
            response_prefix="Sure, ",
        )
        self.assertEqual(_user_suffix_len(data), 0)

    def test_plain_request(self):
        data = ChatCompletionRequest(
            messages=[ChatCompletionMessage(role="user", content="hi")],
        )
        self.assertEqual(_user_suffix_len(data), 0)


def assistant(**kwargs):
    return ChatCompletionMessage(role="assistant", **kwargs)


class ContinuedFieldTests(unittest.TestCase):
    def field(self, message, mode=True):
        return _continued_field(continue_request(messages=[message], continue_final_message=mode))

    def test_true_continues_reasoning_without_content(self):
        self.assertEqual(self.field(assistant(reasoning_content="x")), "reasoning_content")
        self.assertEqual(
            self.field(assistant(reasoning_content="x", content="")), "reasoning_content"
        )

    def test_true_continues_content_when_there_is_any(self):
        self.assertEqual(self.field(assistant(reasoning_content="x", content="y")), "content")
        self.assertEqual(self.field(assistant(content="y")), "content")

    def test_only_assistant_messages_have_reasoning(self):
        message = ChatCompletionMessage(role="user", reasoning_content="x")
        self.assertEqual(self.field(message), "content")

    def test_named_field_wins(self):
        message = assistant(reasoning_content="x", content="")
        self.assertEqual(self.field(message, "content"), "content")
        for name in ("reasoning_content", "reasoning", "thinking"):
            self.assertEqual(self.field(assistant(content="y"), name), "reasoning_content")

    def test_not_continuing(self):
        self.assertIsNone(self.field(assistant(reasoning_content="x"), False))

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValueError):
            continue_request(continue_final_message="tool_calls")

    def test_message_reasoning_goes_by_other_names(self):
        for name in ("reasoning", "thinking"):
            message = ChatCompletionMessage(role="assistant", **{name: "x"})
            self.assertEqual(message.reasoning_content, "x")


class MarkReasoningTests(unittest.TestCase):
    def test_tag_goes_on_the_reasoning(self):
        data = continue_request(messages=[assistant(reasoning_content="thoughts")])

        self.assertEqual(_mark_continued_final_message(data), "thoughts")
        self.assertEqual(data.messages[-1].reasoning_content, "thoughts" + TAG)
        self.assertEqual(data.messages[-1].content, "")

    def test_empty_reasoning_can_be_started(self):
        data = continue_request(
            messages=[assistant(reasoning_content="", content="")],
            continue_final_message="reasoning_content",
        )

        self.assertEqual(_mark_continued_final_message(data), "")

    def assert_rejected(self, data):
        with self.assertRaises(HTTPException) as raised:
            _mark_continued_final_message(data)
        self.assertEqual(raised.exception.status_code, 422)

    def test_named_reasoning_must_exist(self):
        self.assert_rejected(
            continue_request(
                messages=[assistant(content="x")], continue_final_message="reasoning_content"
            )
        )

    def test_named_reasoning_needs_an_assistant_message(self):
        self.assert_rejected(
            continue_request(
                messages=[ChatCompletionMessage(role="user", reasoning_content="x")],
                continue_final_message="reasoning_content",
            )
        )

    def test_tool_calls_cant_be_continued(self):
        call = {"id": "1", "function": {"name": "f", "arguments": "{}"}}
        self.assert_rejected(
            continue_request(messages=[assistant(reasoning_content="x", tool_calls=[call])])
        )


def container(**kwargs):
    defaults = dict(
        harmony=False,
        muse_glimmer=False,
        reasoning=True,
        reasoning_start_token="<think>",
        reasoning_end_token="</think>",
        start_in_reasoning="auto",
    )
    return SimpleNamespace(**{**defaults, **kwargs})


class ResolveStartTests(unittest.TestCase):
    def resolve(self, prompt, data, **kwargs):
        with patch.object(model, "container", container(**kwargs)):
            return _resolve_start_in_reasoning(prompt, data)

    def reasoning_request(self, **kwargs):
        return continue_request(messages=[assistant(reasoning_content="x" * 1000)], **kwargs)

    def test_reasoning_continuation(self):
        self.assertTrue(self.resolve("<think>" + "x" * 1000, self.reasoning_request()))

    def test_config_doesnt_apply_to_a_block_known_to_be_open(self):
        data = self.reasoning_request()
        self.assertTrue(self.resolve("<think>x", data, start_in_reasoning="never"))

    def test_response_prefix_can_close_the_block(self):
        data = self.reasoning_request(response_prefix="</think>\n\nAnswer: ")
        self.assertFalse(self.resolve("<think>x</think>\n\nAnswer: ", data))

        data = self.reasoning_request(response_prefix="</think>Hm. <think>")
        self.assertTrue(self.resolve("<think>x</think>Hm. <think>", data))

    def test_reasoning_prefilled_in_content_is_still_detected(self):
        data = continue_request(messages=[assistant(content="<think>\nFirst, ")])
        self.assertTrue(self.resolve("<|assistant|><think>\nFirst, ", data))

    def test_content_continuation_follows_the_config(self):
        data = continue_request()
        self.assertTrue(self.resolve("partial", data, start_in_reasoning="always"))
        self.assertFalse(self.resolve("<think>partial", data, start_in_reasoning="never"))

    def test_channel_formats(self):
        for flag in ("harmony", "muse_glimmer"):
            self.assertTrue(self.resolve("x", self.reasoning_request(), **{flag: True}))
            self.assertFalse(self.resolve("x", continue_request(), **{flag: True}))

    def test_no_reasoning_support(self):
        self.assertFalse(self.resolve("x", self.reasoning_request(), reasoning=False))


# Reads message.content with no default, as the Qwen3 template does
TAKES_REASONING_CONTENT = (
    "{% for m in messages %}<|{{ m.role }}|>"
    "{% if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif %}"
    "{{ m.content.strip() }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|><think>{% endif %}"
)
# Reads only message.thinking, as the gpt-oss template does
TAKES_THINKING = (
    "{% for m in messages %}<|{{ m.role }}|>"
    "{% if m.thinking %}<think>{{ m.thinking }}</think>{% endif %}"
    "{{ m.content }}<|end|>{% endfor %}"
)
# Drops reasoning, and skips it in the generation prompt, as Gemma 4's does
DROPS_REASONING = (
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|><think>\n</think>{% endif %}"
)


class RenderedPromptTests(unittest.IsolatedAsyncioTestCase):
    user = ChatCompletionMessage(role="user", content="hi")

    async def render(self, template, message, **kwargs):
        mc = container(
            prompt_template=PromptTemplate("test", template),
            use_vision=False,
            template_vars_default={},
            template_vars_force={},
            get_special_tokens=lambda: {},
            hf_model=SimpleNamespace(add_bos_token=lambda: False),
            **kwargs.pop("container", {}),
        )
        data = continue_request(messages=[self.user, message], **kwargs)
        with patch.object(model, "container", mc):
            prompt, _ = await apply_chat_template(data)

        self.assertEqual(data.messages[-1], message)
        return prompt

    async def test_reply_cut_off_during_reasoning(self):
        message = assistant(reasoning_content="First,")
        prompt = await self.render(TAKES_REASONING_CONTENT, message)
        self.assertTrue(prompt.endswith("<|assistant|><think>First,"))

    async def test_template_reading_thinking(self):
        message = assistant(reasoning_content="First,", content="")
        prompt = await self.render(TAKES_THINKING, message)
        self.assertTrue(prompt.endswith("<|assistant|><think>First,"))

    async def test_answer_after_reasoning(self):
        message = assistant(reasoning_content="Done.", content="")
        prompt = await self.render(
            TAKES_REASONING_CONTENT, message, continue_final_message="content"
        )
        self.assertTrue(prompt.endswith("<think>Done.</think>"))

    async def test_template_dropping_reasoning(self):
        message = assistant(reasoning_content="\nFirst,", content="")
        prompt = await self.render(DROPS_REASONING, message)
        self.assertEqual(prompt, "<|user|>hi<|end|><|assistant|><think>\nFirst,")

    async def test_block_opened_by_hand_gets_its_newline(self):
        message = assistant(reasoning_content="First,")
        prompt = await self.render(DROPS_REASONING, message)
        self.assertTrue(prompt.endswith("<think>\nFirst,"))

    async def test_tags_in_the_history_are_left_alone(self):
        self.user = ChatCompletionMessage(role="user", content="What does <think> do?")
        message = assistant(reasoning_content="First,")
        prompt = await self.render(DROPS_REASONING, message)
        self.assertTrue(prompt.startswith("<|user|>What does <think> do?<|end|>"))

    async def test_dropped_reasoning_without_reasoning_support(self):
        message = assistant(reasoning_content="First,")
        with self.assertRaises(HTTPException) as raised:
            await self.render(DROPS_REASONING, message, container={"reasoning": False})
        self.assertEqual(raised.exception.status_code, 422)


class ReasoningBudgetTests(unittest.IsolatedAsyncioTestCase):
    """The budget covers the whole block, including what the request continues."""

    async def injections(self, message, budget):
        injected = []

        async def stream_generate(*args, **kwargs):
            yield {"text": " more", "token_ids": [1]}
            yield {"text": ".", "token_ids": [2], "finish_reason": "stop"}

        mc = container(
            tool_format=None,
            tool_calls_in_reasoning=False,
            reasoning_budget_tokens=None,
            reasoning_budget_message="",
            stream_generate=stream_generate,
            encode_tokens=lambda text: text.split(),
            set_generation_phase=lambda *args: True,
            constrain_generation_output=lambda _, text: injected.append(text) or True,
        )
        data = continue_request(messages=[message], reasoning_budget_tokens=budget)

        with patch.object(model, "container", mc):
            await _chat_stream_collector(0, None, "id", "prompt", data, True, streaming_mode=False)

        return injected

    async def test_spent_budget_ends_the_block_at_once(self):
        message = assistant(reasoning_content="one two three four five")
        self.assertEqual(await self.injections(message, 5), ["</think>"])

    async def test_remaining_budget_is_left(self):
        message = assistant(reasoning_content="one two")
        self.assertEqual(await self.injections(message, 5), [])

    async def test_content_continuation_spends_nothing(self):
        message = assistant(content="one two three four five")
        self.assertEqual(await self.injections(message, 5), [])


if __name__ == "__main__":
    unittest.main()
