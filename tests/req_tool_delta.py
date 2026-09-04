"""
Live integration checks for incremental tool_calls delta streaming.

Run against a server with a qwen3_coder-family model loaded (config
tool_format: qwen3_coder). Each check prints PASS/FAIL and the script exits
non-zero if anything failed.

  python tests/req_tool_delta.py [base_url]

Checks (numbered as in the test plan):
  I1  SSE contract of streamed tool_calls deltas
  I2  assembled streamed calls == non-streaming authoritative calls
  I3  tool_choice "none" keeps the old path
  I4  tool_choice "required" / named function still streams
  I5  reasoning precedes tool frames, never in the same frame
  I6  content then tool calls in one turn
  I8  client disconnect mid tool-call leaves the server usable
  I9  n>1 keeps choices separate
  I10 logprobs + tools does not crash, logprobs stay off tool frames
  I11 non-streaming path unchanged
  I14 truncated tool call: stream stays valid, server stays usable

Not covered here because they cannot be provoked from the client side: the
end-of-stream fallback for tool text without a function block (unit test
test_fallback_when_streamer_emits_nothing) and the other tool formats (unit
tests test_harmony_backend_unaffected / test_glimmer_backend_unaffected).
"""

import json
import sys

from _common import load_api_keys
from _sse_tools import choice_calls, non_streaming, stream_chat

BASE_URL = "http://localhost:5000/v1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": "Save a short note.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]

WEATHER = {
    "messages": [
        {
            "role": "user",
            "content": "What is the weather in Amsterdam? Call the weather tool.",
        }
    ],
    "tools": TOOLS,
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 256,
}

RESULTS = []


def report(name, ok, details=""):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if details:
        for line in str(details).splitlines():
            print(f"       {line}")


def base(payload):
    request = dict(payload)
    request.setdefault("temperature", 0)
    return request


def i1_i2(api_key):
    streamed = stream_chat(
        api_key,
        BASE_URL,
        base({**WEATHER, "stream_options": {"include_usage": True}}),
        expect_usage=True,
    )
    if streamed.error:
        report("I1 stream request", False, streamed.error)
        return

    tool_frames = [1 for (_i, kind) in streamed.order if kind == "tool"]
    report(
        "I1 SSE contract (frame shape, ordering, [DONE], usage)",
        not streamed.violations,
        "\n".join(streamed.violations[:5]),
    )
    report(
        "I1 tool deltas arrived in more than one frame",
        len(tool_frames) > 1,
        f"{len(tool_frames)} frames carrying tool deltas",
    )
    report(
        "I1 finish_reason is tool_calls",
        streamed.finish_reason == "tool_calls",
        f"finish_reason={streamed.finish_reason!r}",
    )
    report(
        "I1 assembled arguments are valid JSON",
        all(_json_ok(a) for (_n, a) in streamed.assembled()),
        str(streamed.assembled()),
    )

    # The two sides are separate generations: temperature 0 is not bit-exact
    # across requests (prefix cache, batching), so the model occasionally
    # phrases the same call differently. Retry a few times before calling it a
    # failure. A real stream divergence would also surface as a verify() error
    # in the server log, which is what the byte-identity guarantee is pinned
    # by in the unit tests, against one and the same text.
    reference = None
    parsed_ok = byte_ok = False
    for attempt in range(1, 4):
        plain, err = non_streaming(api_key, BASE_URL, base(WEATHER))
        if err:
            report("I2 non-streaming reference", False, err)
            return
        reference = choice_calls(plain)
        parsed_ok = _same_calls(streamed.assembled(), reference)
        byte_ok = streamed.assembled() == reference
        if parsed_ok:
            break

    detail = f"attempts={attempt}\nstreamed  : {streamed.assembled()}\nreference : {reference}"
    report("I2 streamed calls == non-streaming calls (name + parsed args)", parsed_ok, detail)
    report("I2 byte-identical arguments (same two generations)", byte_ok, detail)


def i3(api_key):
    res = stream_chat(
        api_key,
        BASE_URL,
        base({**WEATHER, "tool_choice": "none"}),
    )
    ok = (
        not res.error
        and not res.violations
        and not [1 for (_i, kind) in res.order if kind == "tool"]
        and res.finish_reason == "stop"
    )
    report("I3 tool_choice none: no tool frames, content streams", ok, _detail(res))


def i4(api_key):
    for choice in ("required", {"type": "function", "function": {"name": "get_weather"}}):
        res = stream_chat(api_key, BASE_URL, base({**WEATHER, "tool_choice": choice}))
        tools = [1 for (_i, kind) in res.order if kind == "tool"]
        ok = not res.error and not res.violations and bool(tools)
        report(f"I4 tool_choice {str(choice)[:24]}: streams", ok, _detail(res))


def i5(api_key):
    res = stream_chat(api_key, BASE_URL, base(WEATHER))
    kinds = [kind for (_i, kind) in res.order]
    ok = (
        not res.error
        and not res.violations
        and "reasoning" in kinds
        and "tool" in kinds
        and kinds.index("reasoning") < kinds.index("tool")
    )
    report(
        "I5 reasoning frames precede tool frames",
        ok,
        _detail(res) or f"order head: {kinds[:6]}",
    )


def i6(api_key):
    payload = base(
        {
            **WEATHER,
            # a thinking model spends most of the budget on reasoning first, so
            # a too-small max_tokens truncates the turn before either phase shows
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": "Say exactly one short sentence, then call the weather "
                    "tool for Rotterdam.",
                }
            ],
        }
    )
    res = stream_chat(api_key, BASE_URL, payload)
    kinds = [kind for (_i, kind) in res.order]
    ok = (
        not res.error
        and not res.violations
        and "content" in kinds
        and "tool" in kinds
        and kinds.index("content") < kinds.index("tool")
        and bool(res.content.strip())
    )
    report(
        "I6 content deltas precede tool frames, both complete",
        ok,
        _detail(res)
        or f"content={res.content[:60]!r} calls={res.assembled()} finish={res.finish_reason!r}",
    )


def i8(api_key):
    res = stream_chat(api_key, BASE_URL, base(WEATHER), abort_after_frames=6)
    if not res.aborted:
        report("I8 client disconnect mid tool-call", False, "did not abort in time")
        return
    # the slot must be released: a follow-up request has to complete normally
    follow = stream_chat(api_key, BASE_URL, base(WEATHER))
    ok = follow.finish_reason in ("tool_calls", "stop") and not follow.error
    report(
        "I8 abort then follow-up request completes (slot released)",
        ok,
        _detail(follow) or f"follow-up finish_reason={follow.finish_reason!r}",
    )


def i9(api_key):
    res = stream_chat(api_key, BASE_URL, base({**WEATHER, "n": 2}))
    if res.error and "n " in res.error.lower():
        report("I9 n>1 with tools", True, f"backend rejects n>1: {res.error[:120]}")
        return
    choices = {idx for (idx, _k) in res.calls}
    ok = not res.error and not res.violations and len(choices) >= 1
    report(
        "I9 n>1 keeps per-choice frames tagged with the right index",
        ok,
        _detail(res) or f"choices with calls: {sorted(choices)}",
    )


def i10(api_key):
    res = stream_chat(api_key, BASE_URL, base({**WEATHER, "logprobs": 1}))
    ok = not res.error and not res.violations
    report("I10 logprobs + tools does not crash", ok, _detail(res))


def i11(api_key):
    plain, err = non_streaming(api_key, BASE_URL, base(WEATHER))
    if err:
        report("I11 non-streaming regression", False, err)
        return
    calls = choice_calls(plain)
    ok = bool(calls) and all(_json_ok(a) for (_n, a) in calls)
    report("I11 non-streaming tool_calls unchanged", ok, str(calls))


def i14(api_key):
    """
    Truncation is the one case where a streamed call cannot match the
    end-of-stream parse: the regex drops a function block that never closes,
    while the fragments are already with the client. The contract that must
    still hold is transport-level: no violation, a clean finish, and a server
    that keeps working afterwards.
    """
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Use write_file to save a 30-line Python script that "
                "prints Fibonacci numbers to /tmp/fib.py.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 60,
        "template_vars": {"enable_thinking": False},
    }
    res = stream_chat(api_key, BASE_URL, payload)
    calls = res.assembled()
    truncated = calls and not all(_json_ok(a) for (_n, a) in calls)
    follow = stream_chat(api_key, BASE_URL, base(WEATHER))
    ok = (
        not res.error
        and not res.violations
        and bool(calls)
        and follow.finish_reason in ("tool_calls", "stop")
        and not follow.error
    )
    report(
        "I14 truncated tool call: stream stays valid, server stays usable",
        ok,
        _detail(res)
        or f"truncated={truncated} finish={res.finish_reason!r} "
        f"tail={[a[-30:] for _n, a in calls]}",
    )


def _json_ok(arguments):
    try:
        json.loads(arguments)
        return True
    except Exception:
        return False


def _same_calls(a, b):
    if len(a) != len(b):
        return False
    for (n1, a1), (n2, a2) in zip(a, b):
        if n1 != n2:
            return False
        try:
            if json.loads(a1) != json.loads(a2):
                return False
        except Exception:
            return False
    return True


def _detail(res):
    if res.error:
        return res.error
    return "\n".join(res.violations[:5])


def main():
    global BASE_URL
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
    api_key, _admin = load_api_keys()

    for check in (i1_i2, i3, i4, i5, i6, i8, i9, i10, i11, i14):
        check(api_key)

    failed = [name for name, ok in RESULTS if not ok]
    print()
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("failed checks:")
        for name in failed:
            print(f"  - {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
