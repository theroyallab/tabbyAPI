import unittest

from endpoints.OAI.utils.stream_parser import TagStreamParser


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


if __name__ == "__main__":
    unittest.main()
