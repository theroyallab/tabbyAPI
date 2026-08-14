"""Streaming translation for the Anthropic Messages API.

Anthropic's stream is a block-structured event sequence, not the flat chunk
stream the OAI API emits. Each event carries an SSE event name, and content
arrives inside explicitly opened and closed blocks:

    message_start
      content_block_start -> content_block_delta* -> content_block_stop
      ...
    message_delta (stop reason and output token count)
    message_stop

The generation pipeline emits reasoning and content as separate deltas on the
same chunk, so translating means tracking which block is open and closing it
when the channel changes. Exactly one block is open at a time and indices are
assigned in the order blocks are opened.

There is no [DONE] sentinel: message_stop terminates the stream.
"""

import asyncio
import json
import pathlib
from asyncio import CancelledError
from typing import List, Optional
from uuid import uuid4

from fastapi import Request
from sse_starlette import ServerSentEvent

from common.errors import ContextLengthExceededError
from common.logger import xlogger
from common.multimodal import MultimodalEmbeddingWrapper
from common.networking import DisconnectHandler
from endpoints.Anthropic.errors import error_content
from endpoints.Anthropic.types.messages import MessagesRequest, Usage
from endpoints.Anthropic.utils.messages import stop_reason, usage_from_stats
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils.chat_completion import (
    _chat_stream_collector,
    _resolve_start_in_reasoning,
)
from endpoints.OAI.utils.common_ import get_usage_stats

# Block kinds, matching the content block types they open
TEXT = "text"
THINKING = "thinking"
TOOL_USE = "tool_use"


def _event(name: str, payload: dict) -> ServerSentEvent:
    """Build a named SSE event carrying a JSON payload."""

    return ServerSentEvent(data=json.dumps(payload, ensure_ascii=False), event=name)


def _empty_block(block_kind: str) -> dict:
    """
    Build the empty content block a content_block_start announces.

    The thinking block carries an empty signature for the same reason the
    non-streaming path does: there is nothing to authenticate locally, but SDK
    response models require the field.
    """

    if block_kind == THINKING:
        return {"type": THINKING, "thinking": "", "signature": ""}

    return {"type": TEXT, "text": ""}


def _delta(block_kind: str, text: str) -> dict:
    """Build the delta payload for a block kind."""

    if block_kind == THINKING:
        return {"type": "thinking_delta", "thinking": text}

    return {"type": "text_delta", "text": text}


class ContentBlockTracker:
    """
    Tracks the open content block and hands out block indices.

    Blocks are opened lazily: a kind that never produces text never appears in
    the stream, so a response without reasoning has its text at index 0.
    """

    def __init__(self):
        self.open_kind: Optional[str] = None
        self.index = -1

    def _open(self, block_kind: str) -> ServerSentEvent:
        self.index += 1
        self.open_kind = block_kind

        return _event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": self.index,
                "content_block": _empty_block(block_kind),
            },
        )

    def close(self) -> List[ServerSentEvent]:
        """Close the open block, if any."""

        if self.open_kind is None:
            return []

        self.open_kind = None

        return [_event("content_block_stop", {"type": "content_block_stop", "index": self.index})]

    def write(self, block_kind: str, text: str) -> List[ServerSentEvent]:
        """Emit text into a block of the given kind, switching blocks if needed."""

        if not text:
            return []

        events: List[ServerSentEvent] = []

        if self.open_kind != block_kind:
            events += self.close()
            events.append(self._open(block_kind))

        events.append(
            _event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.index,
                    "delta": _delta(block_kind, text),
                },
            )
        )

        return events

    def write_tool_call(self, tool_call: dict) -> List[ServerSentEvent]:
        """
        Emit a complete tool call as its own block.

        The pipeline parses tool calls only once the generation has finished,
        so the arguments arrive whole rather than incrementally. They are sent
        as a single input_json_delta, which is what an accumulating client
        expects: it concatenates the fragments and parses the result.
        """

        function = tool_call.get("function") or {}
        events = self.close()

        self.index += 1
        self.open_kind = TOOL_USE

        events.append(
            _event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.index,
                    "content_block": {
                        "type": TOOL_USE,
                        "id": tool_call.get("id"),
                        "name": function.get("name"),
                        "input": {},
                    },
                },
            )
        )
        events.append(
            _event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": function.get("arguments") or "{}",
                    },
                },
            )
        )

        return events + self.close()


def _message_start(message_id: str, model_name: str, input_tokens: int) -> ServerSentEvent:
    """Build the opening event, which carries the prompt token count."""

    return _event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model_name,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        },
    )


def _message_delta(reason: str, stop_sequence: Optional[str], usage: Usage) -> ServerSentEvent:
    """
    Build the closing metadata event.

    The whole usage object is repeated here, not just the output count: the
    prefix cache split is only known once prefill has run, so message_start
    could not carry it.
    """

    return _event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": stop_sequence},
            "usage": usage.model_dump(),
        },
    )


def _error_event(message: str, error_type: str = "api_error") -> ServerSentEvent:
    """Build an error event for a stream that has already committed HTTP 200."""

    return _event("error", error_content(message, error_type))


async def _next_generation(gen_queue: asyncio.Queue, gen_task: asyncio.Task):
    """
    Await the next chunk, or return None once the collector is finished.

    The collector normally ends the stream by emitting a finish chunk, but a
    plain return would otherwise leave this generator awaiting a chunk that
    will never arrive, holding the connection and the model slot open. Racing
    the queue against the task bounds that wait.
    """

    getter = asyncio.create_task(gen_queue.get())
    await asyncio.wait({getter, gen_task}, return_when=asyncio.FIRST_COMPLETED)

    if getter.done():
        return getter.result()

    # The collector finished; take anything it queued on the way out. Nothing
    # was consumed from the queue, so cancelling the getter cannot drop a chunk
    getter.cancel()

    return gen_queue.get_nowait() if not gen_queue.empty() else None


async def stream_generate_message(
    prompt: str,
    embeddings: MultimodalEmbeddingWrapper,
    data: MessagesRequest,
    converted: ChatCompletionRequest,
    request: Request,
    model_path: pathlib.Path,
    disconnect_handler: DisconnectHandler,
    input_tokens: int,
):
    """Generator translating the generation stream into Anthropic events."""

    gen_queue = asyncio.Queue()
    gen_task: Optional[asyncio.Task] = None
    message_id = f"msg_{uuid4().hex}"
    model_name = model_path.name

    try:
        xlogger.info(
            f"Received Anthropic streaming request {request.state.id}",
            {
                "prompt": prompt,
                "data": data.model_dump(mode="json"),
                "model_path": str(model_path),
            },
        )

        start_in_reasoning_mode = _resolve_start_in_reasoning(prompt, converted)

        # The Messages API has no multi-choice concept, so there is exactly
        # one collector and no need to track which choice a chunk belongs to
        gen_task = asyncio.create_task(
            _chat_stream_collector(
                0,
                gen_queue,
                request.state.id,
                prompt,
                converted,
                start_in_reasoning_mode,
                mm_embeddings=embeddings,
                streaming_mode=True,
                disconnect_handler=disconnect_handler,
            )
        )

        yield _message_start(message_id, model_name, input_tokens)

        blocks = ContentBlockTracker()
        reason = "end_turn"
        stop_sequence = None

        # Stands in until the finish chunk reports what prefill actually did
        final_usage = Usage(input_tokens=input_tokens, output_tokens=0)

        while True:
            generation = await _next_generation(gen_queue, gen_task)

            # The collector finished without a finish chunk; close out the
            # stream with what arrived rather than waiting on nothing
            if generation is None:
                break

            # The collector pushes an exception to the queue if it fails
            if isinstance(generation, Exception):
                raise generation

            for event in blocks.write(THINKING, generation.get("delta_reasoning_content") or ""):
                yield event
            for event in blocks.write(TEXT, generation.get("delta_content") or ""):
                yield event

            # Tool calls are parsed once generation finishes, so they arrive
            # whole on the finish chunk rather than as incremental deltas
            for tool_call in generation.get("delta_tool_calls") or []:
                for event in blocks.write_tool_call(tool_call):
                    yield event

            finish_reason = generation.get("finish_reason")
            if finish_reason:
                reason, stop_sequence = stop_reason(
                    finish_reason,
                    generation.get("eos_reason"),
                    generation.get("stop_str"),
                    data.stop_sequences,
                )

                # The finish chunk is authoritative: it knows the output
                # count and how much of the prompt the prefix cache served
                usage = get_usage_stats(generation)
                if usage:
                    final_usage = usage_from_stats(usage)

                break

        for event in blocks.close():
            yield event

        yield _message_delta(reason, stop_sequence, final_usage)
        yield _event("message_stop", {"type": "message_stop"})

        xlogger.info(f"Finished Anthropic streaming request {request.state.id}")

    except CancelledError:
        raise

    except ContextLengthExceededError as exc:
        yield _error_event(str(exc), "invalid_request_error")

    except Exception as exc:
        xlogger.error("Error during Anthropic message stream", str(exc), details=f"\n{str(exc)}")
        yield _error_event("Message generation aborted. Please check the server console.")

    finally:
        # A client that hangs up mid-stream leaves the collector running
        if gen_task is not None and not gen_task.done():
            gen_task.cancel()

        await disconnect_handler.cleanup()
