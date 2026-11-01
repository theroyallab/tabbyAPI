"""
A message's reasoning under the names clients send it and templates read it:
reasoning_content, reasoning and thinking.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from common.templating import PromptTemplate
from endpoints.OAI.types.chat_completion import ChatCompletionMessage
from endpoints.OAI.utils import chat_completion
from endpoints.OAI.utils.chat_completion import format_messages_with_template

NAMES = ("reasoning_content", "reasoning", "thinking")

TOOL_CALLS = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]


def assistant(**kwargs):
    return ChatCompletionMessage(role="assistant", **kwargs)


def reads(name):
    """A template rendering the reasoning it reads under one name."""

    return (
        "{% for m in messages %}{% if m." + name + " %}[{{ m." + name + " }}]{% endif %}"
        "{{ m.content }}|{% endfor %}"
    )


# As gpt-oss's: refuses content beside thinking on a tool call, and renders
# thinking whenever the key exists, even if it's empty
HARMONY_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.tool_calls and m.content and m.thinking %}"
    "{{ raise_exception('Cannot pass both content and thinking') }}"
    "{% endif %}"
    "{% if 'thinking' in m %}[{{ m.thinking }}]{% endif %}"
    "{% if m.content %}{{ m.content }}{% endif %}|{% endfor %}"
)


class ReadingReasoningTests(unittest.TestCase):
    def test_each_name(self):
        for name in NAMES:
            self.assertEqual(assistant(**{name: "x"}).reasoning_content, "x")

    def test_null_or_empty_name_does_not_hide_another(self):
        for empty in (None, ""):
            for first, second in ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)):
                fields = {NAMES[first]: empty, NAMES[second]: "x"}
                self.assertEqual(assistant(**fields).reasoning_content, "x", fields)

    def test_reasoning_content_comes_first(self):
        message = assistant(reasoning="b", reasoning_content="a", thinking="c")
        self.assertEqual(message.reasoning_content, "a")

    def test_empty_reasoning_stays_empty(self):
        for name in NAMES:
            self.assertEqual(assistant(**{name: ""}).reasoning_content, "")

    def test_absent_reasoning_stays_absent(self):
        self.assertIsNone(assistant(content="x").reasoning_content)

    def test_not_written_back_under_the_other_names(self):
        dumped = assistant(reasoning="x").model_dump(exclude_none=True)
        self.assertEqual(dumped, {"role": "assistant", "reasoning_content": "x"})


class ExposingReasoningTests(unittest.IsolatedAsyncioTestCase):
    async def render(self, template, *messages, harmony=False):
        container = SimpleNamespace(
            prompt_template=PromptTemplate("test", template),
            use_vision=False,
            harmony=harmony,
            get_special_tokens=lambda: {},
        )
        with patch.object(chat_completion.model, "container", container):
            prompt, _, _ = await format_messages_with_template(list(messages))
        return prompt

    async def test_every_message_reaches_a_template_under_any_name(self):
        messages = (
            assistant(reasoning_content="one", content="a"),
            ChatCompletionMessage(role="user", content="u"),
            assistant(reasoning_content="two", content="b"),
        )
        for name in NAMES:
            self.assertEqual(await self.render(reads(name), *messages), "[one]a|u|[two]b|")

    async def test_sent_and_read_names_may_differ(self):
        for sent in NAMES:
            for read in NAMES:
                prompt = await self.render(reads(read), assistant(content="a", **{sent: "x"}))
                self.assertEqual(prompt, "[x]a|")

    async def test_empty_reasoning_gets_no_other_names(self):
        prompt = await self.render(HARMONY_TEMPLATE, assistant(reasoning_content="", content="a"))
        self.assertEqual(prompt, "a|")

    async def test_only_assistants_have_reasoning(self):
        message = ChatCompletionMessage(role="user", reasoning_content="x", content="u")
        self.assertEqual(await self.render(reads("thinking"), message), "u|")

    async def test_harmony_tool_call_keeps_reasoning_over_content(self):
        message = assistant(reasoning_content="x", content="a", tool_calls=TOOL_CALLS)
        self.assertEqual(await self.render(HARMONY_TEMPLATE, message, harmony=True), "[x]|")

    async def test_harmony_tool_call_without_reasoning_keeps_content(self):
        message = assistant(content="a", tool_calls=TOOL_CALLS)
        self.assertEqual(await self.render(HARMONY_TEMPLATE, message, harmony=True), "a|")

    async def test_harmony_message_without_tool_calls_keeps_content(self):
        message = assistant(reasoning_content="x", content="a")
        self.assertEqual(await self.render(HARMONY_TEMPLATE, message, harmony=True), "[x]a|")

    async def test_other_formats_keep_both_on_a_tool_call(self):
        message = assistant(reasoning_content="x", content="a", tool_calls=TOOL_CALLS)
        self.assertEqual(await self.render(reads("thinking"), message), "[x]a|")

    async def test_request_message_is_left_alone(self):
        message = assistant(reasoning_content="x", content="a", tool_calls=TOOL_CALLS)
        await self.render(HARMONY_TEMPLATE, message, harmony=True)
        self.assertEqual(message.content, "a")
