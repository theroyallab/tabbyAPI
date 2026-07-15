import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

import common.model  # noqa: F401 - resolve import cycle ordering
from endpoints.OAI.types.chat_completion import ChatCompletionMessage, ChatCompletionRequest
from endpoints.OAI.utils.chat_completion import (
    CONTINUE_FINAL_MESSAGE_TAG,
    _cut_prompt_at_continue_tag,
    _mark_continued_final_message,
    _start_in_reasoning_mode,
    _user_suffix_len,
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


if __name__ == "__main__":
    unittest.main()
