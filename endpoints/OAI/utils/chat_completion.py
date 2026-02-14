"""Chat completion utilities for OAI server."""

import asyncio
import json
import pathlib
from asyncio import CancelledError
from typing import List, Optional
from fastapi import HTTPException, Request
from jinja2 import TemplateError
from loguru import logger

from common import model
from common.multimodal import MultimodalEmbeddingWrapper
from common.networking import (
    get_generator_error,
    handle_request_disconnect,
    handle_request_error,
    request_disconnect_loop,
)
from common.utils import unwrap
from endpoints.OAI.types.chat_completion import (
    ChatCompletionLogprobs,
    ChatCompletionLogprob,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionRespChoice,
    ChatCompletionStreamChunk,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
)
from endpoints.OAI.types.common import UsageStats
from endpoints.OAI.types.tools import ToolCall
from endpoints.OAI.utils.completion import _parse_gen_request_id, _stream_collector
from endpoints.OAI.utils.tools import ToolCallProcessor, TOOL_CALL_SCHEMA


def _serialize_stream_chunk(chunk) -> str:
    """Serialize a streaming chunk with OpenAI-compatible field handling.

    Uses exclude_none=True to strip irrelevant null fields (tool_calls,
    tool_call_id, logprobs, usage) while ensuring finish_reason is always
    present on each choice (as null when not set), matching OpenAI's
    observed streaming behavior.
    """
    d = chunk.model_dump(exclude_none=True)
    for choice in d.get("choices", []):
        if "finish_reason" not in choice:
            choice["finish_reason"] = None
    return json.dumps(d, ensure_ascii=False)


def _create_response(
    request_id: str,
    generations: List[dict],
    model_name: Optional[str],
    tool_call_format: str = "json",
):
    """Create a chat completion response from the provided text."""

    choices = []
    for index, generation in enumerate(generations):
        message = ChatCompletionMessage(
            role="assistant", content=unwrap(generation.get("text"), "")
        )

        tool_calls_raw = generation.get("tool_calls")
        if tool_calls_raw:
            parsed = ToolCallProcessor.parse(tool_calls_raw, format=tool_call_format)
            if parsed:
                message.tool_calls = parsed
            else:
                logger.warning(
                    "Tool call text present but parsing returned no results "
                    f"(format={tool_call_format})"
                )

        # Fallback: detect bare XML tool calls in content that were not
        # caught by the two-pass system (model never emitted tool_start)
        if (
            tool_call_format in ("xml", "auto")
            and not message.tool_calls
            and message.content
            and "<function=" in message.content
        ):
            logger.warning(
                "Fallback: Detected bare XML function blocks in content "
                "(tool_start was likely not emitted by model)"
            )
            remaining, parsed = ToolCallProcessor.extract_content_and_tools(
                message.content
            )
            if parsed:
                message.tool_calls = parsed
                message.content = remaining if remaining else None

        logprob_response = None

        token_probs = unwrap(generation.get("token_probs"), {})
        if token_probs:
            logprobs = unwrap(generation.get("logprobs"), [])

            collected_token_probs = []
            for index, token in enumerate(token_probs.keys()):
                top_logprobs = [
                    ChatCompletionLogprob(token=token, logprob=logprob)
                    for token, logprob in logprobs[index].items()
                ]

                collected_token_probs.append(
                    ChatCompletionLogprob(
                        token=token,
                        logprob=token_probs[token],
                        top_logprobs=top_logprobs,
                    )
                )

            logprob_response = ChatCompletionLogprobs(content=collected_token_probs)

        # Set finish reason
        if message.tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = generation.get("finish_reason", "stop")

        choice = ChatCompletionRespChoice(
            index=index,
            finish_reason=finish_reason,
            stop_str=generation.get("stop_str"),
            message=message,
            logprobs=logprob_response,
        )

        choices.append(choice)

    final_generation = generations[-1]
    prompt_tokens = unwrap(final_generation.get("prompt_tokens"), 0)
    completion_tokens = unwrap(final_generation.get("gen_tokens"), 0)

    response = ChatCompletionResponse(
        id=f"cmpl-{request_id}",
        choices=choices,
        model=model_name,
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            prompt_time=final_generation.get("prompt_time"),
            prompt_tokens_per_sec=final_generation.get("prompt_tokens_per_sec"),
            completion_tokens=completion_tokens,
            completion_time=final_generation.get("gen_time"),
            completion_tokens_per_sec=final_generation.get("gen_tokens_per_sec"),
            total_tokens=prompt_tokens + completion_tokens,
            total_time=final_generation.get("total_time"),
        ),
    )

    return response


def _create_stream_chunk(
    request_id: str,
    generation: Optional[dict] = None,
    model_name: Optional[str] = None,
    is_usage_chunk: bool = False,
):
    """Create a chat completion stream chunk from the provided text.

    Note: Tool-call streaming is handled separately by
    _build_tool_call_chunks() which emits the proper three-phase
    OpenAI-standard chunk sequence.
    """

    index = generation.get("index")
    choices = []
    usage_stats = None

    if is_usage_chunk:
        prompt_tokens = unwrap(generation.get("prompt_tokens"), 0)
        completion_tokens = unwrap(generation.get("gen_tokens"), 0)

        usage_stats = UsageStats(
            prompt_tokens=prompt_tokens,
            prompt_time=generation.get("prompt_time"),
            prompt_tokens_per_sec=generation.get("prompt_tokens_per_sec"),
            completion_tokens=completion_tokens,
            completion_time=generation.get("gen_time"),
            completion_tokens_per_sec=generation.get("gen_tokens_per_sec"),
            total_tokens=prompt_tokens + completion_tokens,
            total_time=generation.get("total_time"),
        )
    elif "finish_reason" in generation:
        finish_reason = generation.get("finish_reason")
        choice = ChatCompletionStreamChoice(
            index=index, finish_reason=finish_reason, delta={}
        )
        choices.append(choice)
    else:
        message = ChatCompletionMessage(
            role="assistant", content=unwrap(generation.get("text"), "")
        )

        logprob_response = None

        token_probs = unwrap(generation.get("token_probs"), {})
        if token_probs:
            logprobs = unwrap(generation.get("logprobs"), {})
            top_logprobs = [
                ChatCompletionLogprob(token=token, logprob=logprob)
                for token, logprob in logprobs.items()
            ]

            generated_token = next(iter(token_probs))
            token_prob_response = ChatCompletionLogprob(
                token=generated_token,
                logprob=token_probs[generated_token],
                top_logprobs=top_logprobs,
            )

            logprob_response = ChatCompletionLogprobs(content=[token_prob_response])

        choice = ChatCompletionStreamChoice(
            index=index,
            delta=message,
            logprobs=logprob_response,
        )

        choices.append(choice)

    chunk = ChatCompletionStreamChunk(
        id=f"chatcmpl-{request_id}",
        choices=choices,
        model=unwrap(model_name, ""),
        usage=usage_stats,
    )

    return chunk


def _build_tool_call_chunks(
    tool_calls: List[ToolCall],
    request_id: str,
    model_name: str,
) -> List[ChatCompletionStreamChunk]:
    """Build the OpenAI-standard streaming sequence for tool calls.

    Emits two chunks:
      1. Tool-call chunk: role="assistant", complete tool_calls with
         index/id/type/name/arguments (all data in one chunk).
      2. Finish chunk: empty delta, finish_reason="tool_calls".

    Complete arguments are sent in a single chunk rather than streamed
    incrementally, which is valid per OpenAI's spec (clients concatenate
    argument strings across deltas) and maximizes compatibility with
    clients that may not implement multi-chunk tool-call assembly.

    The tool_calls are placed directly into a ChatCompletionMessage
    (not a raw dict) so Pydantic validates them as ToolCall objects
    with the index field preserved (ToolCall declares index as Optional[int]).
    """
    chunk_id = f"chatcmpl-{request_id}"

    # Set index on each tool call for streaming
    for idx, tc in enumerate(tool_calls):
        tc.index = idx

    # Chunk 1: Complete tool call data
    tool_call_message = ChatCompletionMessage(
        role="assistant",
        tool_calls=tool_calls,
    )
    tool_chunk = ChatCompletionStreamChunk(
        id=chunk_id,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=tool_call_message,
                finish_reason=None,
            )
        ],
        model=model_name,
    )

    # Chunk 2: Finish signal
    # Use model_construct to prevent Pydantic's smart Union from
    # coercing the empty dict {} into ChatCompletionMessage(role="user")
    finish_choice = ChatCompletionStreamChoice.model_construct(
        index=0,
        delta={},
        finish_reason="tool_calls",
        logprobs=None,
    )
    finish_chunk = ChatCompletionStreamChunk(
        id=chunk_id,
        choices=[finish_choice],
        model=model_name,
    )

    return [tool_chunk, finish_chunk]


async def _append_template_metadata(data: ChatCompletionRequest, template_vars: dict):
    """Adding metadata is a one-time process."""

    template_metadata = await model.container.prompt_template.extract_metadata(
        template_vars
    )

    # Stop strings
    if isinstance(data.stop, str):
        data.stop = [data.stop] + template_metadata.stop_strings
    else:
        data.stop.extend(template_metadata.stop_strings)

    # if a tool start is present, append it to stopping strings
    if template_metadata.tool_start:
        data.stop.append(template_metadata.tool_start)


async def format_messages_with_template(
    messages: List[ChatCompletionMessage],
    existing_template_vars: Optional[dict] = None,
):
    """Barebones function to format chat completion messages into a prompt."""

    template_vars = unwrap(existing_template_vars, {})
    mm_embeddings = MultimodalEmbeddingWrapper() if model.container.use_vision else None

    # Convert all messages to a dictionary representation
    message_dicts: List[dict] = []
    for message in messages:
        if isinstance(message.content, list):
            concatenated_content = ""
            for content in message.content:
                if content.type == "text":
                    concatenated_content += content.text
                elif content.type == "image_url" and mm_embeddings:
                    await mm_embeddings.add(content.image_url.url)
                    concatenated_content += mm_embeddings.text_alias[-1]

            # Convert the message content into a concatenated string
            message.content = concatenated_content

        message_dicts.append(message.model_dump(exclude_none=True))

    # Pre-template: convert tool_call arguments from JSON strings to dicts.
    # OpenAI-compatible clients (Kilo, Roo, etc.) send arguments as JSON
    # strings per the OAI spec, but Qwen3-Coder's template calls
    # .items() on arguments which requires a dict/mapping.
    for msg in message_dicts:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        func["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        logger.warning(
                            "Failed to parse tool_call arguments JSON "
                            "string to dict, keeping as string"
                        )

    # Get all special tokens
    special_tokens_dict = model.container.get_special_tokens()

    template_vars.update({"messages": message_dicts, **special_tokens_dict})

    prompt = await model.container.prompt_template.render(template_vars)
    return prompt, mm_embeddings, template_vars


async def apply_chat_template(data: ChatCompletionRequest):
    """
    Compile the prompt and get any additional stop strings from the template.
    Template stop strings can be overriden by sampler overrides if force is true.
    """

    # Locally store tools dict
    tools = data.model_dump()["tools"]

    try:
        data.template_vars.update(
            {
                "add_generation_prompt": data.add_generation_prompt,
                "tools": tools,
                "functions": data.functions,
            }
        )

        prompt, mm_embeddings, template_vars = await format_messages_with_template(
            data.messages, data.template_vars
        )

        # Append response prefix if present
        if data.response_prefix:
            if data.add_generation_prompt:
                prompt += data.response_prefix
            else:
                logger.warning(
                    "Could not add response prefix because "
                    "add_generation_prompt is False"
                )

        # Removes the starting BOS token if the model adds one
        # This is to prevent add_bos_token from adding multiple bos tokens
        bos_token = template_vars.get("bos_token")
        if (
            bos_token
            and model.container.hf_model.add_bos_token()
            and prompt.startswith(bos_token)
        ):
            prompt = prompt.removeprefix(bos_token)

        # Add template metadata
        await _append_template_metadata(data, template_vars)

        return prompt, mm_embeddings

    except KeyError as exc:
        error_message = handle_request_error(
            "Could not find a Conversation from prompt template "
            f"'{model.container.prompt_template.name}'. "
            "Check your spelling?",
        ).error.message

        raise HTTPException(400, error_message) from exc
    except TemplateError as exc:
        error_message = handle_request_error(f"TemplateError: {str(exc)}").error.message

        raise HTTPException(400, error_message) from exc


async def stream_generate_chat_completion(
    prompt: str,
    embeddings: MultimodalEmbeddingWrapper,
    data: ChatCompletionRequest,
    request: Request,
    model_path: pathlib.Path,
):
    """Generator for the generation process."""
    abort_event = asyncio.Event()
    gen_queue = asyncio.Queue()
    gen_tasks: List[asyncio.Task] = []
    tool_start = model.container.prompt_template.metadata.tool_start
    tool_call_format = model.container.prompt_template.metadata.tool_call_format
    disconnect_task = asyncio.create_task(request_disconnect_loop(request))

    try:
        logger.info(f"Received chat completion streaming request {request.state.id}")

        for idx in range(0, data.n):
            task_gen_params = data.model_copy(deep=True)
            request_id = _parse_gen_request_id(data.n, request.state.id, idx)

            gen_task = asyncio.create_task(
                _stream_collector(
                    idx,
                    gen_queue,
                    request_id,
                    prompt,
                    task_gen_params,
                    abort_event,
                    mm_embeddings=embeddings,
                )
            )

            gen_tasks.append(gen_task)

        # Text accumulation for tool calls
        current_generation_text = ""

        # Consumer loop
        while True:
            # Fast path: items already queued — no task overhead
            if not gen_queue.empty():
                generation = gen_queue.get_nowait()
            else:
                # Slow path: queue empty — race get against disconnect
                get_task = asyncio.create_task(gen_queue.get())
                done, _ = await asyncio.wait(
                    [get_task, disconnect_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect_task in done:
                    get_task.cancel()
                    raise CancelledError()
                generation = get_task.result()

            if disconnect_task.done():
                raise CancelledError()

            # Handle options if a tool model is present
            if tool_start:
                if "stop_str" in generation:
                    generations = await generate_tool_calls(
                        prompt,
                        embeddings,
                        data,
                        [generation],
                        request,
                    )

                    # Only one generation present in this case
                    generation = generations[0]

                    # Emit proper three-phase tool-call streaming sequence
                    if "tool_calls" in generation:
                        tool_calls_raw = generation["tool_calls"]
                        parsed = ToolCallProcessor.parse(
                            tool_calls_raw, format=tool_call_format
                        )
                        if parsed:
                            for tc_chunk in _build_tool_call_chunks(
                                parsed,
                                request.state.id,
                                model_path.name,
                            ):
                                yield _serialize_stream_chunk(tc_chunk)

                            # Handle completion and usage after tool calls
                            if (
                                all(task.done() for task in gen_tasks)
                                and gen_queue.empty()
                            ):
                                if (
                                    data.stream_options
                                    and data.stream_options.include_usage
                                ):
                                    usage_chunk = _create_stream_chunk(
                                        request.state.id,
                                        generation,
                                        model_path.name,
                                        is_usage_chunk=True,
                                    )
                                    yield _serialize_stream_chunk(usage_chunk)

                                logger.info(
                                    "Finished chat completion streaming "
                                    f"request {request.state.id}"
                                )
                                yield "[DONE]"
                                break
                            continue

                elif "text" in generation:
                    current_generation_text += generation["text"]

            # Stream collector will push an exception to the queue if it fails
            if isinstance(generation, Exception):
                raise generation

            response = _create_stream_chunk(
                request.state.id,
                generation,
                model_path.name,
            )
            yield _serialize_stream_chunk(response)

            # Check if all tasks are completed
            if all(task.done() for task in gen_tasks) and gen_queue.empty():
                # Send a usage chunk
                if data.stream_options and data.stream_options.include_usage:
                    usage_chunk = _create_stream_chunk(
                        request.state.id,
                        generation,
                        model_path.name,
                        is_usage_chunk=True,
                    )
                    yield _serialize_stream_chunk(usage_chunk)

                logger.info(
                    f"Finished chat completion streaming request {request.state.id}"
                )

                yield "[DONE]"
                break
    except CancelledError:
        # Get out if the request gets disconnected

        handle_request_disconnect("Chat completion generation cancelled by user.")
    except Exception:
        yield get_generator_error(
            "Chat completion aborted. Please check the server console."
        )
    finally:
        abort_event.set()
        disconnect_task.cancel()


async def generate_chat_completion(
    prompt: str,
    embeddings: MultimodalEmbeddingWrapper,
    data: ChatCompletionRequest,
    request: Request,
    model_path: pathlib.Path,
):
    gen_tasks: List[asyncio.Task] = []
    tool_start = model.container.prompt_template.metadata.tool_start
    tool_call_format = model.container.prompt_template.metadata.tool_call_format

    try:
        logger.info(f"Received chat completion request {request.state.id}")

        for idx in range(0, data.n):
            request_id = _parse_gen_request_id(data.n, request.state.id, idx)

            gen_tasks.append(
                asyncio.create_task(
                    model.container.generate(
                        request_id,
                        prompt,
                        data,
                        mm_embeddings=embeddings,
                    )
                )
            )

        generations = await asyncio.gather(*gen_tasks)

        # Check all the generations and see if a tool call is required
        if tool_start:
            generations = await generate_tool_calls(
                prompt, embeddings, data, generations, request
            )

        response = _create_response(
            request.state.id,
            generations,
            model_path.name,
            tool_call_format=tool_call_format,
        )

        logger.info(f"Finished chat completion request {request.state.id}")

        return response
    except Exception as exc:
        error_message = handle_request_error(
            f"Chat completion {request.state.id} aborted. "
            "Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc


async def generate_tool_calls(
    prompt: str,
    embeddings: MultimodalEmbeddingWrapper,
    data: ChatCompletionRequest,
    generations: List[str],
    request: Request,
):
    gen_tasks: List[asyncio.Task] = []
    tool_start = model.container.prompt_template.metadata.tool_start
    tool_call_format = model.container.prompt_template.metadata.tool_call_format

    # Tracks which generations asked for a tool call
    tool_idx: List[int] = []

    # Copy to make sure the parent JSON schema doesn't get modified
    tool_data = data.model_copy(deep=True)

    if tool_call_format in ("xml", "auto"):
        # XML / auto mode: let the model generate its natural output
        # without JSON schema constraint
        logger.debug(
            f"generate_tool_calls: Using '{tool_call_format}' mode "
            f"(no JSON schema constraint)"
        )

        # Remove tool_start from stop strings so the model can emit
        # multiple sequential <tool_call> blocks without stopping early
        if (
            tool_start
            and isinstance(tool_data.stop, list)
            and tool_start in tool_data.stop
        ):
            tool_data.stop = [s for s in tool_data.stop if s != tool_start]
            logger.debug(
                f"generate_tool_calls: Removed '{tool_start}' from "
                f"second-pass stop strings"
            )
    else:
        # JSON mode: constrained generation (existing behavior)
        tool_data.json_schema = TOOL_CALL_SCHEMA

    for idx, gen in enumerate(generations):
        if gen["stop_str"] != tool_start:
            continue

        logger.info(
            f"Detected tool call in chat completion request "
            f"{request.state.id} (format={tool_call_format})"
        )

        # Append the existing generation text if present
        precursor_text = gen.get("full_text")
        if precursor_text:
            prompt = prompt + precursor_text

        # For XML/auto mode: append tool_start back to prompt.
        # The stop string was consumed by the first pass and not included
        # in full_text, but the model expects to continue after <tool_call>.
        # Include a trailing newline to match the canonical template format.
        if tool_call_format in ("xml", "auto"):
            prompt = prompt + tool_start + "\n"
            logger.debug(
                f"generate_tool_calls: Appended '{tool_start}\\n' "
                f"to prompt for XML continuation"
            )

        gen_request_id = gen.get("request_id")
        tool_request_id = f"{gen_request_id}-tool"

        gen_tasks.append(
            asyncio.create_task(
                model.container.generate(
                    tool_request_id,
                    prompt,
                    tool_data,
                    mm_embeddings=embeddings,
                )
            )
        )

        tool_idx.append(idx)

    if len(tool_idx) > 0:
        tool_calls = await asyncio.gather(*gen_tasks)

        # Map tool calls to their appropriate generation
        for gen_idx, tool_call in zip(tool_idx, tool_calls, strict=True):
            raw_text = tool_call["text"]

            if tool_call_format in ("xml", "auto"):
                # Prepend tool_start to reconstruct complete XML for parser
                raw_text = tool_start + "\n" + raw_text
                logger.debug(
                    f"generate_tool_calls: Raw XML tool call output "
                    f"({len(raw_text)} chars): {raw_text[:500]}..."
                )

            generations[gen_idx]["tool_calls"] = raw_text

    return generations
