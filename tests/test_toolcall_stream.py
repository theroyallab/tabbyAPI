"""
Byte-identity tests for the incremental tool_calls delta streamer: the
concatenated "arguments" fragments per tool call must equal the arguments
string produced by the authoritative end-of-stream qwen3_coder parse, for
every chunking of the incoming text.

Tag literals are assembled from chr() so this file (and any transport
inspecting it) never contains raw template tags.
"""

import json
import random
import unittest
from unittest.mock import patch

from endpoints.OAI.utils import toolcall_stream as tcs
from endpoints.OAI.utils.stream_parser import TOOL, TagStreamParser
from endpoints.OAI.utils.toolcall_formats import qwen3_coder
from endpoints.OAI.utils.toolcall_stream import (
    QwenToolCallDeltaStreamer,
    _is_streamable,
)
from endpoints.OAI.utils.tools import get_toolcall_tags, supports_delta_streaming

LT, GT = chr(60), chr(62)
# The wrapper tags must be the ones the server actually routes on, otherwise
# the fixtures silently exercise a format that does not exist (see
# test_wrapper_tags_match_the_format). Wrappers are NOT part of the tool-call
# grammar below; they only decide which channel the text lands on.
TC_S, TC_E = get_toolcall_tags("qwen3_coder")
RS, RE = LT + "|think|" + GT, LT + "|/think|" + GT


def func(name, params, wrapped=True):
    body = "".join(f"{LT}parameter={k}{GT}{v}{LT}/parameter{GT}" for k, v in params)
    block = f"{LT}function={name}{GT}{body}{LT}/function{GT}"
    return f"{TC_S}{block}{TC_E}" if wrapped else block


REASONING = "Let me think about this carefully." + RE


def assemble(case_text, rng, chunker="random", byte_budget=None):
    """
    Route full model output (reasoning + tool text) through TagStreamParser,
    feed TOOL events to the delta streamer in small chunks, and return the
    streamer, the TOOL-channel text (what the server parses at EOS) and the
    delta frames the streamer produced.

    byte_budget aborts a run that emits more argument bytes than the input
    could ever account for: a streamer that loses track of what it already
    sent re-emits the whole value on every chunk, which is quadratic and
    would otherwise show up as a hung test rather than a failed one.
    """

    parser = TagStreamParser(
        reasoning_start=RS,
        reasoning_end=RE,
        tool_start=TC_S,
        tool_end=TC_E,
        start_in_reasoning=True,
    )
    streamer = QwenToolCallDeltaStreamer()
    tool_text = ""
    frames = []

    emitted_bytes = 0

    def take(piece):
        nonlocal emitted_bytes, tool_text
        for channel, sub in piece:
            if channel == TOOL:
                tool_text += sub
                frame = streamer.feed(sub)
                if frame:
                    frames.append(frame)
                    emitted_bytes += sum(
                        len(d.get("function", {}).get("arguments", "")) for d in frame
                    )
                    if byte_budget is not None and emitted_bytes > byte_budget:
                        raise AssertionError(
                            f"streamer emitted {emitted_bytes} bytes for "
                            f"{len(case_text)} bytes of input (budget {byte_budget})"
                        )

    text = REASONING + case_text
    pos = 0
    while pos < len(text):
        if chunker == "random":
            n = rng.randint(1, 7)
        elif chunker == "tiny":
            n = 1
        else:
            n = len(text)
        piece = text[pos : pos + n]
        pos += n
        take(parser.feed(piece))
    take(parser.finish())

    return streamer, tool_text, frames


class ToolCallContractMixin:
    """Shared assertions: streamed fragments must rebuild the EOS parse."""

    def setUp(self):
        self.rng = random.Random(1234)

    # JSON escaping can at worst double the raw byte count; anything beyond a
    # generous multiple means fragments were emitted more than once
    BYTE_BUDGET = 8

    def assert_matches(self, case_text, byte_strict=True, expect_calls=None):
        budget = self.BYTE_BUDGET * len(case_text) + 200
        for chunker in ("random", "tiny", "whole"):
            with self.subTest(chunker=chunker):
                streamer, tool_text, frames = assemble(
                    case_text, self.rng, chunker, byte_budget=budget
                )
                # The server compares against the parse of the TOOL channel
                # text (full_tool), which is exactly what the streamer saw.
                authoritative = qwen3_coder.parse_toolcalls(tool_text)
                expected = [(c.function.name, c.function.arguments) for c in authoritative]
                mine = list(zip(streamer._names, streamer._assembled))
                if expect_calls is not None:
                    # Vacuity guard: fixture text that never reaches the tool
                    # channel would compare [] against [] and pass silently.
                    self.assertEqual(len(expected), expect_calls, "end-of-stream")
                    self.assertEqual(len(mine), expect_calls, "streamed")
                if byte_strict:
                    self.assertEqual(mine, expected)
                else:
                    self.assertEqual(len(mine), len(expected))
                    for (n1, a1), (n2, a2) in zip(mine, expected):
                        self.assertEqual(n1, n2)
                        self.assertEqual(json.loads(a1), json.loads(a2))
                self.assert_frame_contract(frames, expected, byte_strict)

    def assert_frame_shape(self, frames):
        """
        OAI streaming shape, independent of the content: the first delta of an
        index opens the call (id, type, name, an arguments string starting at
        "{"), later deltas carry fragments only, a frame never carries two
        deltas for the same index, and indices appear in order 0, 1, 2 ...
        """
        first_seen = []
        for frame in frames:
            seen_in_frame = set()
            for delta in frame:
                i = delta["index"]
                self.assertNotIn(i, seen_in_frame, "duplicate index in one frame")
                seen_in_frame.add(i)
                self.assertLess(i, len(first_seen) + 1, "index opened out of order")
                if i not in first_seen:
                    first_seen.append(i)
                    self.assertTrue(delta["id"].startswith("call_"))
                    self.assertEqual(delta["type"], "function")
                    self.assertTrue(delta["function"]["name"])
                    self.assertTrue(delta["function"]["arguments"].startswith("{"))
                else:
                    self.assertNotIn("id", delta)
                    self.assertNotIn("name", delta.get("function", {}))
        return first_seen

    def assert_frame_contract(self, frames, expected, byte_strict=True):
        """
        Frame shape plus the assembled payload: the concatenated fragments
        must rebuild the end-of-stream arguments (byte-identical for
        well-formed calls, parse-identical when byte-strictness is waived).
        """
        first_seen = self.assert_frame_shape(frames)
        self.assertEqual(first_seen, list(range(len(expected))))

        concat = {}
        for frame in frames:
            for delta in frame:
                i = delta["index"]
                concat[i] = concat.get(i, "") + delta["function"].get("arguments", "")

        if byte_strict:
            self.assertEqual(concat, {i: args for i, (_, args) in enumerate(expected)})
        else:
            self.assertEqual(len(concat), len(expected))
            for i, (_, args) in enumerate(expected):
                self.assertEqual(json.loads(concat[i]), json.loads(args))


class QwenToolCallDeltaStreamerTests(ToolCallContractMixin, unittest.TestCase):
    # -- well-formed calls: byte-identical to the end-of-stream parse

    def test_single_string_value(self):
        self.assert_matches(func("get_weather", [("city", "\n Amsterdam \n")]), expect_calls=1)

    def test_multi_params(self):
        self.assert_matches(
            func("search", [("query", "\nbest coffee\n"), ("limit", "\n5\n")]),
            expect_calls=1,
        )

    def test_parallel_calls(self):
        self.assert_matches(
            func("get_weather", [("city", "\nAmsterdam\n")])
            + func("get_weather", [("city", "\nTokyo\n")]),
            expect_calls=2,
        )

    def test_json_array_value(self):
        self.assert_matches(
            func("run", [("ports", "\n[8080, 9000]\n"), ("name", "\nweb\n")]),
            expect_calls=1,
        )

    def test_json_object_value(self):
        self.assert_matches(
            func("deploy", [("config", '\n{"replicas": 3, "canary": true}\n')]),
            expect_calls=1,
        )

    def test_number_and_bool_values(self):
        self.assert_matches(func("scale", [("replicas", "\n42\n")]), expect_calls=1)
        self.assert_matches(func("toggle", [("flag", "\ntrue\n")]), expect_calls=1)

    def test_keyword_divergence_streams_as_string(self):
        # "the value" diverges from "true" and must stream; "neutral" too
        self.assert_matches(
            func("note", [("text", "\nthe value\n"), ("other", "\nneutral\n")]),
            expect_calls=1,
        )

    def test_quoted_string_value(self):
        # a value that parses as a JSON string is coerced, not streamed
        self.assert_matches(func("echo", [("s", '\n"hello"\n')]), expect_calls=1)

    def test_unicode_and_escapes(self):
        self.assert_matches(func("write", [("text", "\n你好世界 🌍 ünïcode\n")]), expect_calls=1)
        self.assert_matches(
            func("write", [("code", '\nline1\nsay "hi" back\\slash\ttab\nline3\n')]),
            expect_calls=1,
        )

    def test_multiline_code_value(self):
        code = '\ndef f(x):\n    # comment\n    return {"a": [1, 2], "b": None}\n\nprint(f(1))\n'
        self.assert_matches(func("write_file", [("content", code)]), expect_calls=1)

    def test_empty_value(self):
        self.assert_matches(func("ping", [("note", "\n\n")]), expect_calls=1)

    def test_ambiguous_number_like_values(self):
        self.assert_matches(
            func("install", [("pkg", "\nnumpy==1.26.4\n"), ("v", "\n1.2.3\n")]),
            expect_calls=1,
        )
        self.assert_matches(func("report", [("day", "\n2024-01-02\n")]), expect_calls=1)

    def test_value_containing_close_tag(self):
        # the EOS regex closes at the first occurrence; the streamer must too
        self.assert_matches(
            func(
                "odd",
                [("html", f"\nbefore{LT}/parameter{GT}after\n"), ("after", "\nx\n")],
            ),
            expect_calls=1,
        )

    def test_whitespace_in_value(self):
        self.assert_matches(func("echo", [("s", "\n  leading and trailing  \n")]), expect_calls=1)

    # -- degenerate inputs: must not crash; parse-equal at worst

    def test_bare_unwrapped_call(self):
        # without the wrapper the channel parser never routes to the tool
        # channel, so both the streamer and full_tool stay empty (upstream
        # limitation: stream detection relies on the wrapper tag)
        streamer, tool_text, _frames = assemble(
            func("get_weather", [("city", "\nOslo\n")], wrapped=False), self.rng
        )
        self.assertEqual(streamer._names, [])
        self.assertEqual(tool_text, "")

    def test_duplicate_param_keys(self):
        # EOS parse keeps the last value (dict semantics); the stream emits
        # both keys, which json.loads resolves to the same dict
        self.assert_matches(
            func("odd", [("k", "\nfirst\n"), ("k", "\nsecond\n")]),
            byte_strict=False,
            expect_calls=1,
        )

    # -- fuzz: random values incl. JSON-lookalikes, every chunking

    def test_duplicate_key_log_is_scoped_to_one_call(self):
        # the duplicate-key debug line must fire once per duplicate inside a
        # call, and never for the same key reused across separate calls
        # (that is what per-call _keys state is for)
        def duplicate_logs(case):
            with patch.object(tcs.xlogger, "debug") as debug:
                assemble(case, random.Random(1), "whole")
            return sum("Duplicate parameter" in str(c.args) for c in debug.call_args_list)

        self.assertEqual(duplicate_logs(func("f", [("k", "\na\n"), ("k", "\nb\n")])), 1)
        self.assertEqual(
            duplicate_logs(func("f", [("k", "\na\n")]) + func("f", [("k", "\nb\n")])),
            0,
        )

    def test_fuzz_random_values(self):
        alphabet = 'abc \n\t"\\{}[]0123456789.:,-_你好🌍=truefalse null'
        for i in range(150):
            with self.subTest(case=i):
                params = []
                for p in range(self.rng.randint(1, 3)):
                    v = "".join(self.rng.choice(alphabet) for _ in range(self.rng.randint(0, 40)))
                    if not v.strip():
                        v += "x"
                    params.append((f"p{p}", "\n" + v + "\n"))
                wrapped = self.rng.random() < 0.8
                text = func("fuzz_fn", params, wrapped=wrapped)
                # A wrapped fixture always yields exactly one call, so the
                # comparison can never degenerate to comparing nothing.
                self.assert_matches(text, expect_calls=1 if wrapped else 0)


class FixtureTests(unittest.TestCase):
    """Guards that keep the fixtures honest (a wrong fixture passes vacuously)."""

    def test_wrapper_tags_match_the_format(self):
        # The fixtures must route on the tags the server routes on. If they
        # diverge, the text never reaches the tool channel and every
        # byte-identity assertion silently compares empty against empty.
        self.assertEqual(TC_S, qwen3_coder.TOOLCALL_START)
        self.assertEqual(TC_E, qwen3_coder.TOOLCALL_END)

    def test_fixture_text_reaches_the_tool_channel(self):
        _streamer, tool_text, frames = assemble(
            func("get_weather", [("city", "\nOslo\n")]), random.Random(0)
        )
        self.assertIn(LT + "function=get_weather" + GT, tool_text)
        self.assertTrue(frames)


class MalformedSequenceTests(ToolCallContractMixin, unittest.TestCase):
    """
    Degenerate tag sequences. Every case must neither crash nor corrupt the
    assembly, and must stay byte-identical to the end-of-stream parse unless
    the divergence is explicitly documented by its own test.
    """

    FC = LT + "/function" + GT
    PC = LT + "/parameter" + GT

    def wrap(self, inner):
        return TC_S + inner + TC_E

    def test_stray_function_close_before_any_open(self):
        # EOS drops closes outside a function block; the streamer must not
        # emit an index -1 delta or crash on the empty assembly list
        self.assert_matches(
            self.wrap(self.FC + func("f", [("k", "\nv\n")], wrapped=False)),
            expect_calls=1,
        )

    def test_stray_param_close_outside_function(self):
        self.assert_matches(
            self.wrap(self.PC + func("f", [("k", "\nv\n")], wrapped=False)),
            expect_calls=1,
        )

    def test_stray_param_open_outside_function(self):
        stray = LT + "parameter=z" + GT + "vv" + self.PC
        self.assert_matches(
            self.wrap(stray + func("f", [("k", "\nv\n")], wrapped=False)),
            expect_calls=1,
        )

    def test_stray_close_after_first_call(self):
        inner = (
            func("a", [("x", "\none\n")], wrapped=False)
            + self.FC
            + func("b", [("y", "\ntwo\n")], wrapped=False)
        )
        self.assert_matches(self.wrap(inner), expect_calls=2)

    def test_function_close_completes_an_unclosed_function(self):
        # EOS: an unclosed function is completed by the next function close,
        # with the stray parameter close in between dropped
        inner = func("a", [("x", "\n1\n")], wrapped=False)
        inner = inner[: -len(self.FC)] + self.PC + self.FC
        self.assert_matches(self.wrap(inner), expect_calls=1)

    def test_nested_function_open_is_merged_into_the_outer_call(self):
        # EOS: the inner open is body text of the outer call, whose name wins
        inner = func("a", [("x", "\n1\n")], wrapped=False)
        inner = inner.replace(self.FC, func("b", [("y", "\n2\n")], wrapped=False) + self.FC)
        self.assert_matches(self.wrap(inner), expect_calls=1)

    def test_nested_function_open_without_params(self):
        inner = LT + "function=a" + GT + func("b", [], wrapped=False) + self.FC
        self.assert_matches(self.wrap(inner), expect_calls=1)

    def test_parameter_open_inside_a_value(self):
        # a literal parameter open inside a value is plain text for both
        # the EOS regex (it closes at the first parameter close) and the
        # streamer (the value state only scans for the close tag)
        literal = LT + "parameter=y" + GT
        self.assert_matches(func("a", [("x", "\ntext " + literal + " more\n")]), expect_calls=1)

    def test_noise_around_the_call_inside_the_wrapper(self):
        inner = func("a", [("x", "\n1\n")], wrapped=False)
        self.assert_matches(self.wrap("prose " + inner + " trailing prose"), expect_calls=1)

    def test_lookalike_tags_outside_a_function_are_noise(self):
        inner = (
            "a "
            + LT
            + "function without equals "
            + LT
            + "parameter without close"
            + func("a", [("x", "\n1\n")], wrapped=False)
        )
        self.assert_matches(self.wrap(inner), expect_calls=1)

    def test_names_with_attributes(self):
        # the format regexes allow trailing attributes on both tag names
        inner = (
            LT
            + 'function=run type="string" '
            + GT
            + LT
            + 'parameter=key type="string"'
            + GT
            + "v"
            + self.PC
            + self.FC
        )
        self.assert_matches(self.wrap(inner), expect_calls=1)

    def test_unicode_and_quote_chars_in_names(self):
        self.assert_matches(
            func("f\u00f6\u00f6-\u4f60\u597d", [("\u00e4\u00f6'k", "\nv\n")]),
            expect_calls=1,
        )

    def test_unterminated_tag_at_end_of_stream(self):
        # a dangling partial tag must be dropped, not emitted as fragments
        inner = func("a", [("x", "\n1\n")], wrapped=False) + LT + "par"
        self.assert_matches(self.wrap(inner), expect_calls=1)

    def test_partial_close_tag_at_end_of_a_value(self):
        # a value ending in a partial close tag: the hold logic must release
        # it as value text once the stream ends
        inner = func("a", [("x", "\nends with " + self.PC[:-2])], wrapped=False)
        self.assert_matches(self.wrap(inner), expect_calls=1)

    def test_unclosed_function_at_end_of_stream(self):
        # DOCUMENTED DIVERGENCE: the EOS regex needs a closing tag and drops
        # the call entirely, while the streamer already sent partial
        # arguments that cannot be retracted. verify() must log the
        # difference and never raise.
        inner = func("a", [("x", "\n1\n")], wrapped=False)
        text = self.wrap(inner[: -len(self.FC)])
        for chunker in ("random", "tiny", "whole"):
            with self.subTest(chunker=chunker):
                streamer, tool_text, _frames = assemble(text, self.rng, chunker)
                self.assertEqual(qwen3_coder.parse_toolcalls(tool_text), [])
                self.assertEqual(streamer._names, ["a"])
                self.assertEqual(streamer._assembled, ['{"x": 1'])
                self.assertTrue(streamer.emitted)
                with (
                    patch.object(tcs.xlogger, "error") as error,
                    patch.object(tcs.xlogger, "debug"),
                ):
                    streamer.verify(tool_text, "req")
                error.assert_called_once()


class IsStreamableTests(unittest.TestCase):
    """Decision table for the stream-live / emit-whole switch."""

    BUFFERED = [
        "",
        "t",
        "tr",
        "tru",
        "true",
        "f",
        "fa",
        "fal",
        "fals",
        "false",
        "n",
        "nu",
        "nul",
        "null",
        '"x',
        "[x",
        "{x",
        "5",
        "5x",
        "-5",
        "-hot",
    ]
    STREAMABLE = [
        "tRu",
        "True",
        "truex",
        "nullable",
        "finally",
        "e",
        "x",
        # a leading '.' or '+' can never start a JSON value, so streaming is
        # safe even though the text looks number-ish
        ".5",
        "+1",
        "\u4f60",
        "\U0001f30d",
    ]

    def test_decision_table(self):
        for value in self.BUFFERED:
            with self.subTest(value=value):
                self.assertFalse(_is_streamable(value), value)

    def test_streamable_table(self):
        for value in self.STREAMABLE:
            with self.subTest(value=value):
                self.assertTrue(_is_streamable(value), value)

    def test_keyword_prefixes_flip_at_the_first_divergent_char(self):
        self.assertFalse(_is_streamable("null"))
        self.assertTrue(_is_streamable("nullx"))
        self.assertFalse(_is_streamable("true"))
        self.assertTrue(_is_streamable("true,"))
        self.assertFalse(_is_streamable("false"))
        self.assertTrue(_is_streamable("falsey"))

    def test_every_decision_still_produces_identical_bytes(self):
        for value in self.BUFFERED + self.STREAMABLE:
            if value.strip() == "":
                continue
            with self.subTest(value=value):
                rng = random.Random(99)
                case = func("f", [("k", "\n" + value + "\n")])
                streamer, tool_text, _frames = assemble(case, rng, "tiny")
                expected = [
                    (c.function.name, c.function.arguments)
                    for c in qwen3_coder.parse_toolcalls(tool_text)
                ]
                self.assertEqual(len(expected), 1)
                self.assertEqual(list(zip(streamer._names, streamer._assembled)), expected)


class VerifyTests(unittest.TestCase):
    """
    verify() is log-only: it must classify correctly, never raise and never
    mutate the assembly. Assertions filter the captured log by message, since
    the parser that verify() calls logs on the same logger.
    """

    STREAM_MSG = "Tool-call delta stream"
    SKIP_MSG = "verification skipped"

    def setUp(self):
        self.text = func("f", [("a", "\nx\n")])
        self.good = qwen3_coder.parse_toolcalls(self.text)[0].function.arguments

    def streamer(self, assembled):
        streamer = QwenToolCallDeltaStreamer()
        # one name per assembled call, as _open_function would have produced
        streamer._names = ["f"] * len(assembled)
        streamer._assembled = list(assembled)
        return streamer

    def counts(self, assembled, full_tool=None):
        """verify() with the logger captured -> (error, debug, skipped) counts."""
        with (
            patch.object(tcs.xlogger, "error") as error,
            patch.object(tcs.xlogger, "debug") as debug,
        ):
            self.streamer(assembled).verify(self.text if full_tool is None else full_tool, "req")
        return (
            sum(self.STREAM_MSG in str(c.args) for c in error.call_args_list),
            sum(
                self.STREAM_MSG in str(c.args) and self.SKIP_MSG not in str(c.args)
                for c in debug.call_args_list
            ),
            sum(self.SKIP_MSG in str(c.args) for c in debug.call_args_list),
        )

    def test_byte_equal_is_silent(self):
        self.assertEqual(self.counts([self.good]), (0, 0, 0))

    def test_parse_equal_byte_divergent_logs_debug(self):
        # duplicate keys: the stream emits both, the EOS dict keeps the last
        text = func("f", [("a", "\none\n"), ("a", "\ntwo\n")])
        self.assertEqual(self.counts(['{"a": "one", "a": "two"}'], text), (0, 1, 0))

    def test_parse_divergent_logs_error(self):
        self.assertEqual(self.counts(['{"a": 999}']), (1, 0, 0))

    def test_missing_or_extra_call_logs_error(self):
        for assembled in ([], ["{}", "{}"]):
            with self.subTest(assembled=assembled):
                self.assertEqual(self.counts(assembled), (1, 0, 0))

    def test_truncation_hint_only_when_more_calls_than_parsed(self):
        # a call streamed but never closed is the one divergence that cannot
        # be avoided by streaming; the log line must name that cause
        def messages(assembled):
            with patch.object(tcs.xlogger, "error") as error, patch.object(tcs.xlogger, "debug"):
                self.streamer(assembled).verify(self.text, "req")
            return [str(c.args[0]) for c in error.call_args_list]

        self.assertTrue(
            any("truncated" in m for m in messages(['{"a": "x"', "{}"])),
            "extra streamed calls should be reported as a probable truncation",
        )
        self.assertFalse(
            any("truncated" in m for m in messages(['{"a": 999}'])),
            "an equal call count is a plain mismatch, not a truncation",
        )

    def test_unparseable_authoritative_text_logs_error(self):
        self.assertEqual(self.counts(['{"a": "x"}'], "not even xml"), (1, 0, 0))

    def test_unparseable_assembly_skips_instead_of_raising(self):
        # equal call count, but the streamed arguments are truncated JSON:
        # the comparison itself blows up and must be swallowed
        self.assertEqual(self.counts(["{truncated"]), (0, 0, 1))

    def test_unexpected_exception_is_swallowed(self):
        with patch.object(tcs, "qwen3_coder") as parser:
            parser.parse_toolcalls.side_effect = RuntimeError("boom")
            with patch.object(tcs.xlogger, "debug") as debug:
                self.streamer([self.good]).verify(self.text, "req")
        self.assertEqual(sum(self.SKIP_MSG in str(c.args) for c in debug.call_args_list), 1)

    def test_verify_does_not_mutate_state(self):
        streamer = self.streamer([self.good])
        before = (list(streamer._names), list(streamer._assembled), streamer.emitted)
        streamer.verify(self.text, "req")
        self.assertEqual(
            (list(streamer._names), list(streamer._assembled), streamer.emitted), before
        )


class FrameMergeTests(unittest.TestCase):
    """A single feed() result is a single SSE frame: one entry per index."""

    def test_whole_text_frame_shape(self):
        case = (
            func("a", [("x", "\none\n")])
            + func("b", [("y", "\ntwo\n")])
            + func("c", [("z", "\nthree\n")])
        )
        _streamer, _tool_text, frames = assemble(case, random.Random(3), "whole")
        self.assertEqual(len(frames), 1, "whole text must arrive as one frame")
        frame = frames[0]
        self.assertEqual([d["index"] for d in frame], [0, 1, 2])
        for i, name in enumerate(["a", "b", "c"]):
            self.assertEqual(frame[i]["function"]["name"], name)
            self.assertTrue(frame[i]["id"].startswith("call_"))
            self.assertEqual(frame[i]["type"], "function")

    def test_interleaved_indices_merge_per_index(self):
        # parallel calls with live-streamed values: a frame may interleave
        # indices, but must merge repeats and keep first-seen order
        streamer = QwenToolCallDeltaStreamer()
        case = (
            func("a", [("x", "\nalpha value\n")])
            + func("b", [("y", "\nbeta value\n")])
            + func("c", [("z", "\ngamma value\n")])
        )
        frames = []
        for i in range(0, len(case), 3):
            frame = streamer.feed(case[i : i + 3])
            if frame:
                frames.append(frame)
        for frame in frames:
            indices = [d["index"] for d in frame]
            self.assertEqual(len(indices), len(set(indices)), "duplicate index")
            self.assertEqual(indices, sorted(indices), "indices not ascending")
        assembled = {}
        for frame in frames:
            for delta in frame:
                assembled[delta["index"]] = assembled.get(delta["index"], "") + delta[
                    "function"
                ].get("arguments", "")
        expected = [
            (c.function.name, c.function.arguments) for c in qwen3_coder.parse_toolcalls(case)
        ]
        self.assertEqual(len(assembled), 3)
        for i, (_name, args) in enumerate(expected):
            self.assertEqual(assembled[i], args)


class FuzzHardeningTests(ToolCallContractMixin, unittest.TestCase):
    """
    Wider fuzz: tag-lookalike characters inside values, injected malformed
    tags, and a large single value. Nothing may raise, and everything that is
    structurally readable must still come out byte-identical to the
    end-of-stream parse; the one input that is not readable has its own test.
    """

    ALPHABET = 'ab \n\t"\\{}[]0123456789.:,-_\u4f60\u597d\U0001f30d=truefalse null' + LT + GT + "/"
    STRAY = [
        LT + "/function" + GT,
        LT + "/parameter" + GT,
        LT + "function=",
        LT + "parameter=",
        LT + "parameter=x" + GT,
        LT + "function",
    ]

    def test_fuzz_values_with_tag_chars(self):
        for i in range(200):
            with self.subTest(case=i):
                params = []
                for p in range(self.rng.randint(1, 3)):
                    v = "".join(
                        self.rng.choice(self.ALPHABET) for _ in range(self.rng.randint(0, 40))
                    )
                    if not v.strip():
                        v += "x"
                    params.append((f"p{p}", "\n" + v + "\n"))
                wrapped = self.rng.random() < 0.9
                self.assert_matches(
                    func("fuzz_fn", params, wrapped=wrapped),
                    expect_calls=1 if wrapped else 0,
                )

    def test_fuzz_malformed_tag_injection(self):
        # Injected stray tags are separated by newlines, so a partial tag
        # cannot glue itself onto the tag that follows: the document stays
        # structurally readable and the stream must stay byte-identical to
        # the end-of-stream parse (that is the whole guarantee).
        for i in range(200):
            with self.subTest(case=i):
                parts = [
                    func(
                        "f",
                        [
                            (
                                f"k{p}",
                                "\n"
                                + "".join(
                                    self.rng.choice(self.ALPHABET)
                                    for _ in range(self.rng.randint(0, 12))
                                )
                                + "\n",
                            )
                        ],
                        wrapped=False,
                    )
                    for p in range(self.rng.randint(1, 2))
                ]
                for _ in range(self.rng.randint(1, 3)):
                    parts.insert(
                        self.rng.randrange(len(parts) + 1),
                        "\n" + self.rng.choice(self.STRAY) + "\n",
                    )
                text = TC_S + "".join(parts) + TC_E
                for chunker in ("random", "tiny", "whole"):
                    with self.subTest(chunker=chunker):
                        streamer, tool_text, _frames = assemble(text, self.rng, chunker)
                        expected = [
                            (c.function.name, c.function.arguments)
                            for c in qwen3_coder.parse_toolcalls(tool_text)
                        ]
                        self.assertEqual(list(zip(streamer._names, streamer._assembled)), expected)

    def test_partial_tag_glued_to_a_real_tag(self):
        # DOCUMENTED DIVERGENCE: a dangling "<parameter=" or "<function=" right
        # in front of a real tag makes the tag regexes disagree about the
        # document itself (the open tag swallows the text up to the next ">").
        # Both sides keep parsing. What still has to hold: no crash, a frame
        # stream a client can assemble (opens before fragments, indices in
        # order), and verify() reporting the difference instead of raising.
        glued = [
            LT + "parameter=" + func("f", [("k", "\nv\n")], wrapped=False),
            LT + "function=" + func("f", [("k", "\nv\n")], wrapped=False),
            func("f", [("k", "\nv\n")], wrapped=False) + LT + "function",
        ]
        for i, inner in enumerate(glued):
            for chunker in ("random", "tiny", "whole"):
                with self.subTest(case=i, chunker=chunker):
                    streamer, tool_text, frames = assemble(TC_S + inner + TC_E, self.rng, chunker)
                    self.assert_frame_shape(frames)
                    with patch.object(tcs.xlogger, "error"), patch.object(tcs.xlogger, "debug"):
                        streamer.verify(tool_text, "req")

    def test_large_single_value(self):
        # 100 KB of text in one value: fragment count and assembly sanity
        big = ("long text with spaces and \u4f60\u597d\n" * 4000)[:100_000]
        case = func("write", [("content", "\n" + big + "\n")])
        streamer, tool_text, frames = assemble(
            case, random.Random(5), "random", byte_budget=2 * len(case)
        )  # few escapes in this value: a tight budget is meaningful here
        expected = [
            (c.function.name, c.function.arguments) for c in qwen3_coder.parse_toolcalls(tool_text)
        ]
        self.assertEqual(list(zip(streamer._names, streamer._assembled)), expected)
        self.assertGreater(len(frames), 100, "value should have streamed live")
        self.assertTrue(streamer._assembled[0].startswith('{"content": "'))
        self.assertGreater(len(streamer._assembled[0]), len(big))


if __name__ == "__main__":
    unittest.main()
