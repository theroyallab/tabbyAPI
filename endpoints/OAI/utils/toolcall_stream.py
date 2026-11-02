"""
Incremental streaming of OpenAI tool_calls deltas for the qwen3_coder
pseudo-XML tool call format.

The end-of-stream parser (toolcall_formats/qwen3_coder.py) collects the full
tool text and parses it in one pass, so streaming clients receive the
complete tool call as a single delta after generation finishes and live UIs
cannot show tool generation as it happens.

QwenToolCallDeltaStreamer consumes TOOL-channel text as it arrives (from
TagStreamParser, which keeps the format's own wrapper tags, TOOLCALL_START
and TOOLCALL_END, in the tool channel) and emits OpenAI-style tool_calls
delta fragments:

  - when a function opens: {index, id, type, function: {name, arguments: "{"}}
  - when a parameter closes: {index, function: {arguments: <json fragment>}}
  - when the function closes: {index, function: {arguments: "}"}}

String parameter values are additionally streamed while they are generated
(as a JSON string), as soon as it is provable that the value cannot parse as
a JSON literal (see _is_streamable). Values that coerce_param_value() may
convert (objects, arrays, numbers, booleans, null, quoted strings) are
emitted whole when their parameter closes.

Guarantee: for well-formed calls, the concatenation of all "arguments"
fragments per index is byte-identical to the arguments string produced by
the end-of-stream parser, so the assembled tool call is unchanged. Malformed
model output is handled the way the end-of-stream regexes read it (stray
close tags and parameters outside a function block are dropped, a nested
function open belongs to the outer call). Fragments cannot be retracted, so
three inputs diverge on purpose and are reported by verify():

  - a call that generation never closes (cut off by max_tokens): the regex
    drops the whole call, the client already has the partial arguments
  - duplicate parameter keys: both are streamed, the parser keeps the last
  - a value containing a literal parameter close tag: both sides close the
    value at the first occurrence, but the remainder is read differently
"""

import json
import re
from typing import Optional
from uuid import uuid4

from common.logger import xlogger
from endpoints.OAI.utils.toolcall_formats import qwen3_coder
from endpoints.OAI.utils.toolcall_formats.common import coerce_param_value

_FUNC_OPEN = re.compile(r"<function=([^>\s]+)[^>]*>")
_PARAM_OPEN = re.compile(r"<parameter=([^>\s]+)[^>]*>")

_FUNC_CLOSE = "</function>"
_PARAM_CLOSE = "</parameter>"

# First characters whose stripped value may still json.loads() to something
# other than the exact input string, per coerce_param_value(). Such values
# are emitted whole at parameter close instead of being streamed.
_JSONY_START = set('"{[0123456789-')

# t/f/n: only the exact keywords parse as JSON; streaming may start as soon
# as the accumulated value diverges from them.
_JSON_KEYWORDS = ("true", "false", "null")


def _is_streamable(stripped: str) -> bool:
    """
    True once the stripped value accumulated so far provably cannot parse as
    a JSON literal, i.e. coerce_param_value() will return it as a plain
    string and json.dumps() will quote it exactly as it is streamed.
    """

    if not stripped:
        return False
    if stripped[0] in _JSONY_START:
        return False
    for keyword in _JSON_KEYWORDS:
        if keyword.startswith(stripped):
            return False
    return True


def _esc(text: str) -> str:
    """JSON-escape a text fragment without surrounding quotes."""

    return json.dumps(text, ensure_ascii=False)[1:-1]


class QwenToolCallDeltaStreamer:
    """
    Emits OAI tool_calls deltas incrementally for qwen3_coder pseudo-XML.

    feed() accepts TOOL-channel text and returns a list of delta dicts to be
    sent as one streaming frame (empty list when nothing is ready). emitted
    is True once any fragment was produced; verify() should be called at the
    end of the stream with the full tool text to cross-check the assembled
    calls against the authoritative end-of-stream parser.
    """

    _OUT = 0
    _VALUE = 1

    def __init__(self):
        self.emitted = False

        self._state = self._OUT
        self._buf = ""

        self._index = -1
        self._in_func = False
        self._first_param = True
        self._keys: set[str] = set()

        self._param_key: Optional[str] = None
        self._raw = ""
        self._streaming = False
        self._sent = ""

        self._names: list[str] = []
        self._assembled: list[str] = []

    # -- emission helpers

    def _open_function(self, name: str) -> dict:
        self._index += 1
        self._first_param = True
        self._keys = set()
        self._assembled.append("{")
        self._names.append(name)
        return {
            "index": self._index,
            "id": f"call_{uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": "{"},
        }

    def _param_prefix(self) -> str:
        key = json.dumps(self._param_key, ensure_ascii=False)
        if self._first_param:
            self._first_param = False
            return key + ": "
        return ", " + key + ": "

    def _arg_fragment(self, fragment: str) -> dict:
        self._assembled[self._index] += fragment
        return {"index": self._index, "function": {"arguments": fragment}}

    def _emit(self, deltas: list, delta: dict):
        self.emitted = True

        # Merge consecutive fragments for the same index into one entry so a
        # frame never carries two deltas for the same tool call.
        if deltas and deltas[-1].get("index") == delta.get("index"):
            prev = deltas[-1]
            prev_fn = prev.get("function", {})
            new_fn = delta.get("function", {})
            merged_fn = dict(prev_fn)
            for field, value in new_fn.items():
                if field == "arguments":
                    merged_fn["arguments"] = merged_fn.get("arguments", "") + value
                else:
                    merged_fn[field] = value
            merged = dict(prev)
            merged["function"] = merged_fn
            deltas[-1] = merged
        else:
            deltas.append(delta)

    # -- buffer management

    @staticmethod
    def _partial_open_hold(buf: str) -> int:
        """Length of a trailing incomplete tag that must be kept in OUT."""

        for literal in ("<function=", "<parameter="):
            pos = buf.rfind(literal)
            if pos >= 0 and buf.find(">", pos) < 0:
                return len(buf) - pos
        for literal in ("<function=", "<parameter=", _FUNC_CLOSE, _PARAM_CLOSE):
            for k in range(min(len(literal) - 1, len(buf)), 0, -1):
                if literal.startswith(buf[-k:]):
                    return k
        return 0

    @staticmethod
    def _partial_close_hold(buf: str) -> int:
        """Length of a trailing text run that could still become a close tag."""

        for k in range(min(len(_PARAM_CLOSE) - 1, len(buf)), 0, -1):
            if _PARAM_CLOSE.startswith(buf[-k:]):
                return k
        return 0

    # -- main entry point

    def feed(self, text: str) -> list:
        deltas: list = []
        self._buf += text

        while True:
            if self._state == self._VALUE:
                close = self._buf.find(_PARAM_CLOSE)
                if close < 0:
                    hold = self._partial_close_hold(self._buf)
                    consume = self._buf[: len(self._buf) - hold] if hold else self._buf
                    self._buf = self._buf[len(self._buf) - hold :] if hold else ""
                    self._consume_value(consume, deltas)
                    break

                self._consume_value(self._buf[:close], deltas)
                self._buf = self._buf[close + len(_PARAM_CLOSE) :]
                self._close_param(deltas)
                self._state = self._OUT
                continue

            # _OUT: advance to the earliest structural tag
            func = _FUNC_OPEN.search(self._buf)
            param = _PARAM_OPEN.search(self._buf)
            fclose = self._buf.find(_FUNC_CLOSE)

            candidates = []
            if func is not None and (fclose < 0 or func.start() < fclose):
                candidates.append(("func", func.start(), func.end(), func.group(1)))
            if param is not None and (fclose < 0 or param.start() < fclose):
                candidates.append(("param", param.start(), param.end(), param.group(1)))
            if fclose >= 0:
                candidates.append(("fclose", fclose, fclose + len(_FUNC_CLOSE), None))

            if not candidates:
                hold = self._partial_open_hold(self._buf)
                self._buf = self._buf[len(self._buf) - hold :] if hold else ""
                break

            kind, _start, end, name = min(candidates, key=lambda c: c[1])
            self._buf = self._buf[end:]

            if kind == "func":
                if self._in_func:
                    # A nested open is body text of the outer call for the
                    # end-of-stream regex, whose name is the outer one. Keep
                    # the outer call and drop this tag.
                    continue
                self._in_func = True
                self._emit(deltas, self._open_function(name))
            elif kind == "fclose":
                if not self._in_func:
                    # Stray close: outside a function block the end-of-stream
                    # regex drops it, so consume it silently here too.
                    continue
                self._emit(deltas, self._arg_fragment("}"))
                self._in_func = False
                self._state = self._OUT
            elif not self._in_func:
                # A parameter outside a function block is dropped by the
                # end-of-stream regex; consume it silently.
                continue
            else:
                self._param_key = name.strip()
                if self._param_key in self._keys:
                    # The end-of-stream parser keeps the last occurrence; the
                    # duplicate key we stream still parses to the same dict.
                    xlogger.debug(
                        "Duplicate parameter in tool call stream",
                        {"function": self._names[self._index], "key": self._param_key},
                    )
                self._keys.add(self._param_key)
                self._raw = ""
                self._streaming = False
                self._sent = ""
                self._state = self._VALUE

        return deltas

    # -- value handling

    def _consume_value(self, text: str, deltas: list):
        if not text:
            return

        self._raw += text

        if not self._streaming:
            stripped = self._raw.strip()
            if stripped and _is_streamable(stripped):
                self._streaming = True
                self._sent = stripped
                self._emit(deltas, self._arg_fragment(self._param_prefix() + '"' + _esc(stripped)))
            return

        # Streaming: the stripped prefix of the raw value grows monotonically;
        # trailing whitespace stays unemitted until later text makes it
        # interior, matching the strip() applied by the end-of-stream parser.
        candidate = self._raw.strip()
        new = candidate[len(self._sent) :]
        if new:
            self._sent = candidate
            self._emit(deltas, self._arg_fragment(_esc(new)))

    def _close_param(self, deltas: list):
        if self._streaming:
            # Close the JSON string. The streamed content must equal the
            # stripped raw value; guard against any drift.
            final = json.dumps(self._raw.strip(), ensure_ascii=False)
            streamed = '"' + _esc(self._sent) + '"'
            if final != streamed:
                xlogger.error(
                    "Tool-call delta stream diverged from value at parameter close",
                    {"function": self._names[self._index], "key": self._param_key},
                )
            self._emit(deltas, self._arg_fragment('"'))
        else:
            value = coerce_param_value(self._raw)
            self._emit(
                deltas,
                self._arg_fragment(self._param_prefix() + json.dumps(value, ensure_ascii=False)),
            )

        self._param_key = None
        self._raw = ""
        self._streaming = False
        self._sent = ""

    # -- end-of-stream cross-check

    def verify(self, full_tool: str, request_id: str):
        """
        Compare the streamed assembly against the authoritative end-of-stream
        parse. Log-only: streamed fragments cannot be retracted.

        Byte-identical for well-formed calls. Degenerate inputs (duplicate
        parameter keys) may differ in bytes but must parse to the same calls,
        and a call that generation was cut off inside is reported as an error,
        because the parser drops a call that never closes.
        """

        try:
            authoritative = qwen3_coder.parse_toolcalls(full_tool)
            # strict=False: a length mismatch here is itself a divergence,
            # and is reported by the comparison below rather than raised here
            mine = list(zip(self._names, self._assembled, strict=False))
            theirs = [(c.function.name, c.function.arguments) for c in authoritative]
            if mine == theirs:
                return

            parsed_equal = len(mine) == len(theirs) and all(
                n1 == n2 and json.loads(a1) == json.loads(a2)
                for (n1, a1), (n2, a2) in zip(mine, theirs, strict=False)
            )
            if parsed_equal:
                xlogger.debug(
                    f"Tool-call delta stream differs byte-wise but parses equal "
                    f"for request {request_id} (duplicate parameter keys?)",
                    {"streamed": mine, "authoritative": theirs},
                )
            else:
                # The common cause in practice: generation was cut off inside a
                # call, and the end-of-stream regex drops a function block that
                # never closes. Say so, because the fragments were already sent.
                hint = ""
                if len(mine) > len(theirs):
                    hint = (
                        " (more calls streamed than parsed: the last call was "
                        "probably truncated, and the end-of-stream parser drops "
                        "a function block without a closing tag)"
                    )
                xlogger.error(
                    f"Tool-call delta stream differs from end-of-stream parse "
                    f"for request {request_id}{hint}",
                    {"streamed": mine, "authoritative": theirs},
                )
        except Exception as exc:  # never fail the stream on verification
            xlogger.debug(f"Tool-call delta verification skipped: {exc}")
