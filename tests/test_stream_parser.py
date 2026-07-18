import unittest

from endpoints.OAI.utils.stream_parser import HarmonyStreamParser, TagStreamParser


def collect(parser, chunks):
    """Feed chunks and aggregate per-channel text."""
    out = {"reasoning": "", "content": "", "tool": ""}
    for chunk in chunks:
        for channel, text in parser.feed(chunk):
            out[channel] += text
    for channel, text in parser.finish():
        out[channel] += text
    return out


def qwen_parser(**kwargs):
    return TagStreamParser(
        reasoning_start="<think>",
        reasoning_end="</think>",
        tool_start="<tool_call>",
        tool_end="</tool_call>",
        **kwargs,
    )


class TagStreamParserTests(unittest.TestCase):
    def test_basic_reasoning_split(self):
        p = qwen_parser(start_in_reasoning=True)
        out = collect(p, ["thinking...", "</think>", "\n\n", "answer"])
        self.assertEqual(out["reasoning"], "thinking...")
        # Whitespace between reasoning and content is preserved once content arrives
        self.assertEqual(out["content"], "\n\nanswer")

    def test_tag_embedded_in_larger_span(self):
        p = qwen_parser(start_in_reasoning=True)
        out = collect(p, ["thinking\n</think>\n\nanswer"])
        self.assertEqual(out["reasoning"], "thinking\n")
        self.assertEqual(out["content"], "\n\nanswer")

    def test_tag_split_across_chunks(self):
        p = qwen_parser(start_in_reasoning=True)
        out = collect(p, ["thinking", "</th", "ink>", "answer"])
        self.assertEqual(out["reasoning"], "thinking")
        self.assertEqual(out["content"], "answer")

    def test_tag_split_one_char_at_a_time(self):
        p = qwen_parser(start_in_reasoning=True)
        out = collect(p, ["a"] + list("</think>") + ["b"])
        self.assertEqual(out["reasoning"], "a")
        self.assertEqual(out["content"], "b")

    def test_partial_tag_lookalike_is_released(self):
        # "<th" could start "<think>" but "<table>" resolves it as plain text
        p = qwen_parser(start_in_reasoning=False)
        out = collect(p, ["use <th", "e table> tag"])
        self.assertEqual(out["content"], "use <the table> tag")

    def test_trailing_partial_tag_flushed_at_finish(self):
        p = qwen_parser(start_in_reasoning=False)
        out = collect(p, ["text ends with <thi"])
        self.assertEqual(out["content"], "text ends with <thi")

    def test_multi_token_reasoning_tags(self):
        # Gemma-style: reasoning starts with a two-token marker
        p = TagStreamParser(
            reasoning_start="<|channel|>thought",
            reasoning_end="<|end|>",
            start_in_reasoning=False,
        )
        out = collect(p, ["<|channel|>", "thought", "hmm...", "<|end|>", "done"])
        self.assertEqual(out["reasoning"], "hmm...")
        self.assertEqual(out["content"], "done")

    def test_start_not_in_reasoning_model_emits_tag(self):
        p = qwen_parser(start_in_reasoning=False)
        out = collect(p, ["<think>deep", "</think>", "final"])
        self.assertEqual(out["reasoning"], "deep")
        self.assertEqual(out["content"], "final")

    def test_tool_call_in_content(self):
        p = qwen_parser(start_in_reasoning=False)
        out = collect(p, ["I'll check.", "<tool_call>", '{"a": 1}', "</tool_call>"])
        self.assertEqual(out["content"], "I'll check.")
        self.assertEqual(out["tool"], '<tool_call>{"a": 1}</tool_call>')

    def test_tool_tag_split_across_chunks(self):
        p = qwen_parser(start_in_reasoning=False)
        out = collect(p, ["go<tool", "_call>x</tool_c", "all>"])
        self.assertEqual(out["content"], "go")
        self.assertEqual(out["tool"], "<tool_call>x</tool_call>")

    def test_tool_call_in_reasoning_default_collected(self):
        p = qwen_parser(start_in_reasoning=True)
        out = collect(p, ["hm <tool_call>x</tool_call> more", "</think>", "done"])
        self.assertEqual(out["tool"], "<tool_call>x</tool_call>")
        self.assertEqual(out["reasoning"], "hm  more")
        self.assertEqual(out["content"], "done")

    def test_tool_call_in_reasoning_disabled(self):
        p = qwen_parser(start_in_reasoning=True, tool_calls_in_reasoning=False)
        out = collect(p, ["hm <tool_call>x</tool_call> more", "</think>", "done"])
        self.assertEqual(out["tool"], "")
        self.assertEqual(out["reasoning"], "hm <tool_call>x</tool_call> more")
        self.assertEqual(out["content"], "done")

    def test_no_end_tag_tool_format(self):
        # Mistral-style: [TOOL_CALLS] opens, nothing closes
        p = TagStreamParser(tool_start="[TOOL_CALLS]", tool_end=None)
        out = collect(p, ["sure ", "[TOOL_CALLS]", '[{"name": "f"}]'])
        self.assertEqual(out["content"], "sure ")
        self.assertEqual(out["tool"], '[TOOL_CALLS][{"name": "f"}]')

    def test_post_reasoning_whitespace_held_and_dropped(self):
        p = qwen_parser(start_in_reasoning=True)
        out = collect(p, ["think", "</think>", "\n\n  \n"])
        self.assertEqual(out["reasoning"], "think")
        self.assertEqual(out["content"], "")

    def test_post_reasoning_whitespace_prepended_to_content(self):
        p = qwen_parser(start_in_reasoning=True)
        out = collect(p, ["think", "</think>", "\n\n", "answer"])
        self.assertEqual(out["content"], "\n\nanswer")

    def test_no_tags_configured(self):
        p = TagStreamParser()
        out = collect(p, ["plain ", "text"])
        self.assertEqual(out["content"], "plain text")

    def test_saw_tag_tracking(self):
        p = qwen_parser(start_in_reasoning=True)
        p.feed("thinking")
        self.assertFalse(p.saw_tag)
        p.feed("</think>")
        self.assertTrue(p.saw_tag)
        p.feed("content")
        self.assertFalse(p.saw_tag)

    def test_streaming_deltas_preserved(self):
        # The exact text must survive chunk-by-chunk emission
        p = qwen_parser(start_in_reasoning=True)
        chunks = ["I ", "think", " so", "\n</think>\n\nYes", ", 4", "."]
        out = collect(p, chunks)
        self.assertEqual(out["reasoning"], "I think so\n")
        self.assertEqual(out["content"], "\n\nYes, 4.")


class HarmonyStreamParserTests(unittest.TestCase):
    # Generation begins after the prompt's "<|start|>assistant", so the text
    # opens with a message header. <|return|> and <|call|> are stop tokens
    # and never appear in the text.

    def test_reasoning_then_final(self):
        p = HarmonyStreamParser()
        out = collect(
            p,
            [
                "<|channel|>analysis<|message|>Let me think.",
                "<|end|>",
                "<|start|>assistant<|channel|>final<|message|>",
                "The answer is 4.",
            ],
        )
        self.assertEqual(out["reasoning"], "Let me think.")
        self.assertEqual(out["content"], "The answer is 4.")
        self.assertEqual(out["tool"], "")

    def test_single_chunk(self):
        p = HarmonyStreamParser()
        out = collect(
            p,
            [
                "<|channel|>analysis<|message|>hmm<|end|>"
                "<|start|>assistant<|channel|>final<|message|>ok"
            ],
        )
        self.assertEqual(out["reasoning"], "hmm")
        self.assertEqual(out["content"], "ok")

    def test_tool_call(self):
        p = HarmonyStreamParser()
        out = collect(
            p,
            [
                "<|channel|>analysis<|message|>Need the weather.<|end|>",
                "<|start|>assistant<|channel|>commentary",
                " to=functions.get_weather <|constrain|>json",
                "<|message|>",
                '{"location": ',
                '"Tokyo"}',
            ],
        )
        self.assertEqual(out["reasoning"], "Need the weather.")
        self.assertEqual(out["content"], "")
        # finish() terminates the open tool message; the header keeps the
        # role text that followed <|start|>
        self.assertEqual(
            out["tool"],
            "assistant<|channel|>commentary to=functions.get_weather <|constrain|>json"
            '<|message|>{"location": "Tokyo"}<|call|>',
        )

    def test_recipient_before_channel(self):
        # The chat template renders the recipient before the channel
        p = HarmonyStreamParser()
        out = collect(
            p,
            ["<|channel|>analysis<|message|>x<|end|>"]
            + ["<|start|>assistant to=functions.f<|channel|>commentary json<|message|>{}"],
        )
        self.assertIn("to=functions.f", out["tool"])
        self.assertTrue(out["tool"].endswith("<|message|>{}<|call|>"))

    def test_commentary_preamble_is_content(self):
        p = HarmonyStreamParser()
        out = collect(
            p,
            [
                "<|channel|>commentary<|message|>Fetching the data now.<|end|>",
                "<|start|>assistant<|channel|>final<|message|>Done.",
            ],
        )
        self.assertEqual(out["content"], "Fetching the data now.Done.")
        self.assertEqual(out["tool"], "")

    def test_structural_token_split_across_chunks(self):
        p = HarmonyStreamParser()
        out = collect(
            p,
            ["<|channel|>analysis<|mess", "age|>a<|e", "nd|><|start|>assistant"]
            + ["<|channel|>final<|message|>b"],
        )
        self.assertEqual(out["reasoning"], "a")
        self.assertEqual(out["content"], "b")

    def test_channel_properties_and_saw_tag(self):
        p = HarmonyStreamParser()
        self.assertFalse(p.in_reasoning or p.in_tool or p.in_content)
        p.feed("<|channel|>analysis<|message|>")
        self.assertTrue(p.saw_tag)
        self.assertTrue(p.in_reasoning)
        p.feed("thinking")
        self.assertFalse(p.saw_tag)
        p.feed("<|end|><|start|>assistant<|channel|>final<|message|>hello")
        self.assertTrue(p.in_content)

    def test_finish_without_tool_message(self):
        p = HarmonyStreamParser()
        p.feed("<|channel|>final<|message|>done")
        self.assertEqual(p.finish(), [])

    def test_truncation_mid_header_discards_header(self):
        # max_new_tokens can cut generation before <|message|>
        p = HarmonyStreamParser()
        out = collect(p, ["<|channel|>anal"])
        self.assertEqual(out, {"reasoning": "", "content": "", "tool": ""})


class HarmonyToolcallFormatTests(unittest.TestCase):
    def parse(self, text):
        from endpoints.OAI.utils.toolcall_formats.harmony import parse_toolcalls

        return parse_toolcalls(text)

    def test_parse_tool_call(self):
        calls = self.parse(
            "<|channel|>commentary to=functions.get_weather <|constrain|>json"
            '<|message|>{"location": "Tokyo"}<|call|>'
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "get_weather")
        self.assertEqual(calls[0].function.arguments, '{"location": "Tokyo"}')

    def test_parse_recipient_before_channel(self):
        calls = self.parse(
            "assistant to=functions.f<|channel|>commentary json<|message|>{}<|call|>"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "f")
        self.assertEqual(calls[0].function.arguments, "{}")

    def test_invalid_json_passed_through(self):
        calls = self.parse(
            '<|channel|>commentary to=functions.f <|constrain|>json<|message|>{"a": <|call|>'
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.arguments, '{"a":')

    def test_no_recipient_no_call(self):
        self.assertEqual(self.parse("<|channel|>commentary<|message|>preamble<|call|>"), [])


class Hy3ToolcallFormatTests(unittest.TestCase):
    def parse(self, text):
        from endpoints.OAI.utils.toolcall_formats.hy3 import parse_toolcalls

        return parse_toolcalls(text)

    def test_parse_tool_call(self):
        calls = self.parse(
            "<tool_calls:opensource>\n"
            "<tool_call:opensource>get_weather<tool_sep:opensource>\n"
            "<arg_key:opensource>location</arg_key:opensource>\n"
            "<arg_value:opensource>Tokyo</arg_value:opensource>\n"
            "</tool_call:opensource>\n"
            "</tool_calls:opensource>"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "get_weather")
        self.assertEqual(calls[0].function.arguments, '{"location": "Tokyo"}')

    def test_parallel_calls(self):
        calls = self.parse(
            "<tool_calls:opensource>\n"
            "<tool_call:opensource>f<tool_sep:opensource>\n"
            "<arg_key:opensource>a</arg_key:opensource>\n"
            "<arg_value:opensource>1</arg_value:opensource>\n"
            "</tool_call:opensource>\n"
            "<tool_call:opensource>g<tool_sep:opensource>\n"
            "<arg_key:opensource>b</arg_key:opensource>\n"
            '<arg_value:opensource>["x", "y"]</arg_value:opensource>\n'
            "</tool_call:opensource>\n"
            "</tool_calls:opensource>"
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].function.name, "f")
        self.assertEqual(calls[0].function.arguments, '{"a": 1}')
        self.assertEqual(calls[1].function.name, "g")
        self.assertEqual(calls[1].function.arguments, '{"b": ["x", "y"]}')

    def test_missing_tool_sep(self):
        calls = self.parse(
            "<tool_calls:opensource>\n"
            "<tool_call:opensource>f\n"
            "<arg_key:opensource>a</arg_key:opensource>\n"
            "<arg_value:opensource>hello world</arg_value:opensource>\n"
            "</tool_call:opensource>\n"
            "</tool_calls:opensource>"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "f")
        self.assertEqual(calls[0].function.arguments, '{"a": "hello world"}')

    def test_no_args(self):
        calls = self.parse(
            "<tool_calls:opensource>\n"
            "<tool_call:opensource>list_files<tool_sep:opensource>\n"
            "</tool_call:opensource>\n"
            "</tool_calls:opensource>"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "list_files")
        self.assertEqual(calls[0].function.arguments, "{}")

    def test_streamed_through_tag_parser(self):
        from endpoints.OAI.utils.toolcall_formats.hy3 import (
            TOOLCALL_START,
            TOOLCALL_END,
        )

        p = TagStreamParser(
            reasoning_start="<think:opensource>",
            reasoning_end="</think:opensource>",
            tool_start=TOOLCALL_START,
            tool_end=TOOLCALL_END,
            start_in_reasoning=True,
        )
        text = (
            "pondering</think:opensource>Checking the weather."
            "<tool_calls:opensource>\n"
            "<tool_call:opensource>get_weather<tool_sep:opensource>\n"
            "<arg_key:opensource>location</arg_key:opensource>\n"
            "<arg_value:opensource>Tokyo</arg_value:opensource>\n"
            "</tool_call:opensource>\n"
            "</tool_calls:opensource>"
        )
        # Feed in small chunks to exercise tag holdback
        out = collect(p, [text[i : i + 7] for i in range(0, len(text), 7)])
        self.assertEqual(out["reasoning"], "pondering")
        self.assertEqual(out["content"], "Checking the weather.")

        calls = self.parse(out["tool"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "get_weather")
        self.assertEqual(calls[0].function.arguments, '{"location": "Tokyo"}')


if __name__ == "__main__":
    unittest.main()
