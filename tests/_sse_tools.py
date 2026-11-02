"""
Shared SSE helper for the streaming tool-call request scripts.

Consumes an OpenAI-compatible `stream: true` chat completion response and
folds `tool_calls` deltas back together per choice/index, so a script can
assert both the transport contract (frame shape and ordering) and the
assembled payload.

Checks collected in `SseResult.violations` are the ones a client actually
relies on:

  - every tool_calls delta carries an `index`
  - the first delta of a call carries `id`, `type`, `function.name` and
    opens the arguments object; later deltas carry fragments only
  - one frame never carries two deltas for the same call
  - no frame mixes `reasoning_content` with `tool_calls`, and logprobs never
    ride on a frame that carries tool-call deltas
  - the finish frame reports the finish reason and does not re-send calls
    that were already streamed
  - `[DONE]` is the last event, and the usage frame is present when asked
    for
"""

import json

import httpx


class SseResult:
    def __init__(self):
        self.frames = []
        self.calls = {}
        self.content = ""
        self.reasoning = ""
        self.order = []
        self.finish_reason = None
        self.usage = None
        self.done = False
        self.error = None
        self.aborted = False
        self.violations = []

    def call_list(self):
        return [self.calls[i] for i in sorted(self.calls)]

    def assembled(self):
        return [(c["name"], c["arguments"]) for c in self.call_list()]


def _check(result, condition, message):
    if not condition:
        result.violations.append(message)


def stream_chat(
    api_key,
    base_url,
    payload,
    timeout=600,
    abort_after_frames=None,
    expect_usage=False,
):
    """
    POST payload with stream=true and consume the SSE response.

    abort_after_frames stops reading after that many data frames, which
    simulates a client hanging up mid-generation.
    """

    result = SseResult()
    request = dict(payload)
    request["stream"] = True
    seen_tool_deltas = False
    first_seen = {}
    frames = 0

    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=request,
        ) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", "replace")
                result.error = f"HTTP {response.status_code}: {body[:400]}"
                return result

            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    # reading stops here, as a client would
                    result.done = True
                    break

                chunk = json.loads(data)
                frames += 1
                result.frames.append(chunk)
                # calls streamed by earlier frames: a finish frame that carries
                # complete calls is only a violation if nothing was streamed
                # before (that is the authoritative fallback path)
                streamed_before_this_frame = seen_tool_deltas

                if chunk.get("usage"):
                    result.usage = chunk["usage"]

                for choice in chunk.get("choices", []):
                    idx = choice.get("index", 0)
                    delta = choice.get("delta") or {}
                    has_reasoning = bool(delta.get("reasoning_content"))
                    has_content = bool(delta.get("content"))
                    tool_calls = delta.get("tool_calls") or []

                    if has_reasoning:
                        result.reasoning += delta["reasoning_content"]
                        result.order.append((idx, "reasoning"))
                    if has_content:
                        result.content += delta["content"]
                        result.order.append((idx, "content"))
                    if tool_calls:
                        result.order.append((idx, "tool"))

                    _check(
                        result,
                        not (has_reasoning and tool_calls),
                        "frame mixes reasoning_content and tool_calls",
                    )
                    _check(
                        result,
                        not (choice.get("logprobs") and tool_calls),
                        "frame carries logprobs next to tool_calls deltas",
                    )

                    indices_in_frame = []
                    for tc in tool_calls:
                        if "index" not in tc:
                            result.violations.append("tool_calls delta without index")
                            continue
                        tidx = tc["index"]
                        indices_in_frame.append(tidx)
                        key = (idx, tidx)
                        fn = tc.get("function") or {}
                        slot = result.calls.setdefault(key, {"id": "", "name": "", "arguments": ""})
                        if key not in first_seen:
                            first_seen[key] = True
                            _check(
                                result,
                                bool(tc.get("id")),
                                f"first delta of call {key} has no id",
                            )
                            _check(
                                result,
                                tc.get("type") == "function",
                                f"first delta of call {key} has no type=function",
                            )
                            _check(
                                result,
                                bool(fn.get("name")),
                                f"first delta of call {key} has no name",
                            )
                            _check(
                                result,
                                fn.get("arguments", "").startswith("{"),
                                f"first delta of call {key} does not open arguments",
                            )
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            if fn.get("name"):
                                slot["name"] = fn["name"]
                        else:
                            _check(
                                result,
                                "id" not in tc,
                                f"call {key} re-sends an id after the open frame",
                            )
                            _check(
                                result,
                                "name" not in fn,
                                f"call {key} re-sends a name after the open frame",
                            )
                        slot["arguments"] += fn.get("arguments", "")

                    if tool_calls:
                        seen_tool_deltas = True

                    _check(
                        result,
                        len(indices_in_frame) == len(set(indices_in_frame)),
                        "two deltas for the same call in one frame",
                    )

                    if choice.get("finish_reason"):
                        result.finish_reason = choice["finish_reason"]
                        _check(
                            result,
                            not (tool_calls and streamed_before_this_frame),
                            "finish frame re-sends complete tool calls",
                        )

                if abort_after_frames is not None and frames >= abort_after_frames:
                    result.aborted = True
                    break

    if not result.aborted and not result.error:
        _check(result, result.done, "stream ended without [DONE]")
    if expect_usage and not result.aborted:
        _check(result, result.usage is not None, "no usage frame although requested")

    return result


def non_streaming(api_key, base_url, payload, timeout=600):
    request = dict(payload)
    request["stream"] = False
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request,
        timeout=timeout,
    )
    if response.status_code != 200:
        return None, f"HTTP {response.status_code}: {response.text[:400]}"
    return response.json(), None


def choice_calls(completion, choice_index=0):
    """tool_calls of a choice as [(name, arguments), ...] for comparison."""
    choice = completion["choices"][choice_index]
    return [
        (c["function"]["name"], c["function"]["arguments"])
        for c in (choice.get("message", {}).get("tool_calls") or [])
    ]
