"""Chat completion utilities for OAI server."""

import asyncio
import json
import pathlib
from asyncio import CancelledError
from typing import List, Optional
from fastapi import HTTPException, Request
from jinja2 import TemplateError
from common.logger import xlogger
import re

from common import model
from common.multimodal import MultimodalEmbeddingWrapper
from common.networking import (
    get_generator_error,
    handle_request_error,
    DisconnectHandler,
)
from common.utils import unwrap
from endpoints.OAI.types.chat_completion import (
    ChatCompletionLogprobs,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionRespChoice,
    ChatCompletionResponse,
)
from endpoints.OAI.types.common import UsageStats
from endpoints.OAI.utils.completion import _parse_gen_request_id
from endpoints.OAI.utils.tools import (
    get_toolcall_tags,
    parse_toolcalls,
)
from endpoints.OAI.utils.common_ import aggregate_usage_stats, get_usage_stats


def _start_in_reasoning_mode(prompt: str) -> bool:
    """
    Utility function to determine if the formatted prompt indicates that inference should
    start in reasoning mode.
    - the system prompt may contain instructions mentioning both tags
    - templates that force-disable thinking may force <think> </think> in the response
    - templates that force-enable thinking may force just <think>
    Best guess: check if the last occurrence of either is <think>, and not much text
    and no other <> tags follow it.
    """
    _think_prefix_max_chars = 256  # Arbitrary hard-cutoff threshold
    _tags_max_length = 32

    st = model.container.reasoning_start_token
    et = model.container.reasoning_end_token
    last_st = prompt.rfind(st)  # or -1
    last_et = prompt.rfind(et)  # or -1
    if last_st <= last_et:
        return False
    i = last_st + len(st)
    if len(prompt) - i > _think_prefix_max_chars:
        return False
    char_op = st[:1]
    char_cl = st[-1:]
    tags_pattern = char_op + r"\S{1," + str(_tags_max_length - 2) + r"}" + char_cl
    if re.search(tags_pattern, prompt[i:]):
        return False
    return True


def _compose_response(
    request_id: str,
    generations: List[dict],
    model_name: Optional[str],
    return_usage,
) -> ChatCompletionResponse:
    """
    Compose a chat completion response from generations collected in non-streaming mode.
    """

    choices = []
    for generation in generations:
        message = ChatCompletionMessage(
            role="assistant",
            content=generation.get("content") or None,
            reasoning_content=generation.get("reasoning_content") or None,
            tool_calls=generation.get("tool_calls") or None,
        )

        choices.append(
            ChatCompletionRespChoice(
                index=generation.get("index"),
                finish_reason=generation.get("finish_reason", "stop"),
                stop_str=generation.get("stop_str"),
                message=message,
                logprobs=generation.get("logprob_response"),
            )
        )

    usl = [get_usage_stats(g) for g in generations]
    usl = [u for u in usl if u is not None]
    response = ChatCompletionResponse(
        id=f"cmpl-{request_id}",
        choices=choices,
        model=model_name,
        usage=(aggregate_usage_stats(usl) if return_usage and usl else None),
    )
    return response


def _compose_serialize_stream_chunk(
    request_id: str,
    generation: Optional[dict] = None,
    model_name: Optional[str] = None,
    suppress_finish: bool = False,
) -> (str, dict, str):
    """
    Compose a chat completion stream chunk from generation produced by _chat_stream_collector

    TODO: Should maybe Pydantic, but need way to selectively avoid None fields in models to comply
          with the spec and de facto standards
    """

    finish_reason = generation.get("finish_reason") or None
    delta_content = generation.get("delta_content")
    delta_reasoning_content = generation.get("delta_reasoning_content")
    delta_tool = generation.get("delta_tool_calls")
    logprobs = generation.get("logprob_response")

    delta = {}
    if delta_content:
        delta["content"] = delta_content
    if delta_reasoning_content:
        delta["reasoning_content"] = delta_reasoning_content
    if delta_tool:
        delta["tool_calls"] = delta_tool

    choice = {
        "index": generation.get("index"),
        "delta": delta,
        "finish_reason": finish_reason if not suppress_finish else None,
    }

    if logprobs:
        choice["logprobs"] = logprobs.model_dump()

    # Only one choice in a streaming chunk
    choices = [choice]
    data = {
        "id": f"chatcmpl-{request_id}",
        "choices": choices,
    }

    if model_name:
        data["model_name"] = model_name

    # Serialize
    s = json.dumps(data, ensure_ascii=False)  # TODO: Investigate ensure_ascii

    # Check if no data
    is_empty = not delta and not (finish_reason and not suppress_finish)
    return s, data, finish_reason, is_empty


def _compose_serialize_stream_usage_chunk(
    request_id: str,
    usage_stats: UsageStats,
    usage_index: int,
    last_finish_reason: str,
    model_name: Optional[str] = None,
) -> (str, dict):
    """
    Compose a usage chunk to send at the end of a strema
    """

    # Make sure we don't break some client with empty choices list
    delta = {}
    choice = {
        "index": usage_index,
        "delta": delta,
        "finish_reason": last_finish_reason,
    }
    choices = [choice]
    data = {
        "id": f"chatcmpl-{request_id}",
        "choices": choices,
        "usage": usage_stats.model_dump(mode="json"),
    }

    if model_name:
        data["model_name"] = model_name

    # Serialize
    s = json.dumps(data, ensure_ascii=False)  # TODO: Investigate ensure_ascii
    return s, data


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
                        # xlogger.debug("Parsed tool call", {"func": func})
                    except (json.JSONDecodeError, ValueError):
                        xlogger.warning(
                            "Failed to parse tool_call arguments JSON "
                            "string to dict, keeping as string",
                            {"args": args},
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
        if model.container.force_enable_thinking:
            data.template_vars.update({"enable_thinking": True})

        prompt, mm_embeddings, template_vars = await format_messages_with_template(
            data.messages, data.template_vars
        )

        # Append response prefix if present
        if data.response_prefix:
            if data.add_generation_prompt:
                prompt += data.response_prefix
            else:
                xlogger.warning(
                    "Could not add response prefix because add_generation_prompt is False"
                )

        # Removes the starting BOS token if the model adds one
        # This is to prevent add_bos_token from adding multiple bos tokens
        bos_token = template_vars.get("bos_token")
        if bos_token and model.container.hf_model.add_bos_token() and prompt.startswith(bos_token):
            prompt = prompt.removeprefix(bos_token)

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


def _parse_tool_calls(
    text: str,
    tool_format: str,
    request_id: str,
) -> list:
    """
    Parse collected tool calls and convert to OAI format.

    Insert tool indices as well. (These are not choice indices; OAI enumerates the tool
    calls within each individual choice for the sake of streaming incomplete tool arg
    deltas, which we don't do here.)
    """

    parsed = parse_toolcalls(text, tool_format)
    for tc_idx, p in enumerate(parsed):
        p.index = tc_idx
    dumped = [p.model_dump(mode="json") for p in parsed]

    if len(parsed):
        xlogger.info(
            f"Parsed {len(parsed)} tool calls in chat completion request {request_id}",
            {"tool_format": tool_format, "parsed": parsed, "dumped": dumped},
            details=f"(format={tool_format})",
        )
    return dumped


async def _chat_stream_collector(
    task_idx: int,
    gen_queue: asyncio.Queue | None,
    request_id: str,
    prompt: str,
    params: ChatCompletionRequest,
    start_in_reasoning_mode: bool,
    mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
    streaming_mode: bool = True,
    disconnect_handler: DisconnectHandler = None,
):
    """
    Starts a request on the backend and collects generations while tracking phase, for a single
    choice.

    In streaming mode, emits chunks of text to be emitted as deltas to the client, divided into
    reasoning/content/tool phases. Tool calls are parsed together at the end of stream, so the
    last chunk contains all tool calls collected for the turn.

    In non-streaming mode, collects everything with the same logic but then emits a single
    response packet at the end, to be combined with any other choices (for n>1 requests) and
    sent together to the client.

    TODO: Integrate JSON constraint with trigger token (esp. for models without a tool_end token)
    """

    mc = model.container
    full_reasoning = ""
    full_content = ""
    full_tool = ""

    post_reasoning_whitespace = False
    held_whitespace = ""

    in_reasoning = start_in_reasoning_mode
    in_tool = False

    tool_format = mc.tool_format
    t_tool_start, t_tool_end = get_toolcall_tags(tool_format)
    use_tool = params.tool_choice != "none" and bool(t_tool_start)
    t_tool_start = t_tool_start if use_tool else None
    t_tool_end = t_tool_end if use_tool else None

    use_think = mc.reasoning and bool(mc.reasoning_start_token)
    t_think_start = mc.reasoning_start_token if use_think else None
    t_think_end = mc.reasoning_end_token if use_think else None

    # Regex to identify tool/think tags that may or may not arrive with other text
    splits = [re.escape(s) for s in [t_tool_start, t_tool_end, t_think_start, t_think_end] if s]
    split_re = re.compile("|".join(splits)) if splits else None

    # Collect logprobs
    collected_logprobs = []

    try:
        new_generation = mc.stream_generate(
            request_id,
            prompt,
            params,
            disconnect_handler,
            mm_embeddings,
        )
        generation = {}
        async for generation in new_generation:
            generation["index"] = task_idx
            text = generation.get("text", "")
            finish_reason = generation.get("finish_reason")
            delta_reasoning = ""
            delta_content = ""
            delta_tool = ""
            tag = None

            while text:
                # Find + identify tag and split text into before and after parts
                if split_re:
                    match = split_re.search(text)
                    if match:
                        i, j = match.span()
                        sub, text, tag = text[:i], text[j:], match[0]
                    else:
                        sub, text, tag = text, "", None
                else:
                    sub, text, tag = text, "", None

                # Accumulate text up to tag
                if in_tool:
                    delta_tool += sub
                    full_tool += sub
                elif in_reasoning:
                    delta_reasoning += sub
                    full_reasoning += sub
                else:
                    if post_reasoning_whitespace:
                        if not sub.strip():
                            held_whitespace += sub
                            sub = ""
                        else:
                            sub = held_whitespace + sub
                            held_whitespace = ""
                            post_reasoning_whitespace = False
                    delta_content += sub
                    full_content += sub

                # Track output phase. No nesting is expected, except tools may occur in
                # reasoning content
                if tag:
                    if tag == t_tool_end:  # include outer tool tags in output
                        delta_tool += tag
                        full_tool += tag
                    if not in_tool:
                        if tag == t_think_start:
                            post_reasoning_whitespace = False
                            in_reasoning = True
                        elif tag == t_think_end:
                            post_reasoning_whitespace = True
                            in_reasoning = False
                    if tag == t_tool_start:
                        in_tool = True
                        delta_tool += tag  # include outer tool tags in output
                        full_tool += tag
                    elif tag == t_tool_end:
                        in_tool = False

            # Collect logprobs in content span only. Also make sure we're not just coming
            # out of a </think> tag
            if (
                "logprobs_content" in generation
                and tag not in [t_think_end, t_tool_end]
                and not in_reasoning
                and not in_tool
            ):
                collected_logprobs += generation["logprobs_content"]

            # Add the output and emit
            if streaming_mode:
                if delta_content:
                    if len(collected_logprobs):
                        generation["logprob_response"] = ChatCompletionLogprobs(
                            content=collected_logprobs
                        )
                        collected_logprobs = []
                generation["delta_reasoning_content"] = delta_reasoning
                generation["delta_content"] = delta_content
                generation["delta_tool_calls"] = ""
                if finish_reason and full_tool:
                    generation["delta_tool_calls"] = _parse_tool_calls(
                        full_tool, tool_format, request_id
                    )
                    generation["finish_reason"] = "tool_calls"
                await gen_queue.put(generation)

            # End
            if finish_reason:
                break

        # In non-streaming mode, return everything as a single result
        if not streaming_mode:
            has_content = bool(full_content.strip())
            if has_content and len(collected_logprobs):
                generation["logprob_response"] = ChatCompletionLogprobs(content=collected_logprobs)
            generation["reasoning_content"] = full_reasoning
            generation["content"] = full_content if has_content else None
            generation["tool_calls"] = _parse_tool_calls(full_tool, tool_format, request_id)
            if full_tool:
                generation["finish_reason"] = "tool_calls"
            return generation

    except Exception as e:
        if gen_queue:
            await gen_queue.put(e)
        else:
            return e


async def stream_generate_chat_completion(
    prompt: str,
    embeddings: MultimodalEmbeddingWrapper,
    data: ChatCompletionRequest,
    request: Request,
    model_path: pathlib.Path,
    disconnect_handler: DisconnectHandler,
):
    """
    Generator for the generation process.
    """

    gen_queue = asyncio.Queue()
    gen_tasks: List[asyncio.Task] = []
    return_usage = data.stream_options and data.stream_options.include_usage

    try:
        xlogger.info(
            f"Received chat completion streaming request {request.state.id}",
            {
                "prompt": prompt,
                "data": data.model_dump(mode="json"),
                "model_path": str(model_path),
            },
        )

        # Determine if we're streaming content or reasoning_content to start with
        start_in_reasoning_mode = model.container.reasoning and _start_in_reasoning_mode(prompt)

        # For aggregating usage
        usage_stats_list = []

        # Create a stream collector for each choice
        remaining_n = data.n
        for idx in range(0, data.n):
            task_gen_params = data.model_copy(deep=True)
            request_id = _parse_gen_request_id(data.n, request.state.id, idx)
            gen_task = asyncio.create_task(
                _chat_stream_collector(
                    idx,
                    gen_queue,
                    request_id,
                    prompt,
                    task_gen_params,
                    start_in_reasoning_mode,
                    mm_embeddings=embeddings,
                    streaming_mode=True,
                    disconnect_handler=disconnect_handler,
                )
            )
            gen_tasks.append(gen_task)

        # Consumer loop
        while True:
            generation = await gen_queue.get()

            # Stream collector will push an exception to the queue if it fails
            if isinstance(generation, Exception):
                raise generation

            # Create and serialize chunk
            chunk, _, finish_reason, is_empty = _compose_serialize_stream_chunk(
                request.state.id,
                generation,
                model_path.name,
                return_usage and remaining_n == 1,
            )
            if not is_empty:
                yield chunk

            # Send usage chunk on completing last choice
            if finish_reason:
                remaining_n -= 1
                if return_usage:
                    usage_stats_list.append(get_usage_stats(generation))
                    if remaining_n == 0:
                        usage_chunk, usage_chunk_dict = _compose_serialize_stream_usage_chunk(
                            request.state.id,
                            aggregate_usage_stats(usage_stats_list),
                            generation["index"],
                            finish_reason,
                            model_path.name,
                        )
                        yield usage_chunk
                        xlogger.debug(
                            f"Sent UsageStats for request {request.state.id}",
                            usage_chunk_dict,
                        )

            # Check if all tasks are completed
            if all(task.done() for task in gen_tasks) and gen_queue.empty():
                xlogger.info(f"Finished chat completion streaming request {request.state.id}")
                yield "[DONE]"
                break

    except CancelledError:
        raise

    except Exception as e:
        xlogger.error("Error during chat completion", str(e), details=f"\n{str(e)}")
        yield get_generator_error("Chat completion aborted. Please check the server console.")

    finally:
        await disconnect_handler.cleanup()


async def generate_chat_completion(
    prompt: str,
    embeddings: MultimodalEmbeddingWrapper,
    data: ChatCompletionRequest,
    request: Request,
    model_path: pathlib.Path,
    disconnect_handler: DisconnectHandler,
):
    gen_tasks: List[asyncio.Task] = []
    return_usage = data.stream_options and data.stream_options.include_usage

    try:
        xlogger.info(
            f"Received chat completion request {request.state.id}",
            {
                "prompt": prompt,
                "data": data.model_dump(mode="json"),
                "model_path": str(model_path),
            },
        )

        # Determine if we're generating content or reasoning_content to start with
        start_in_reasoning_mode = model.container.reasoning and _start_in_reasoning_mode(prompt)

        # Create a stream collector for each choice
        for idx in range(0, data.n):
            task_gen_params = data.model_copy(deep=True)
            request_id = _parse_gen_request_id(data.n, request.state.id, idx)
            gen_task = asyncio.create_task(
                _chat_stream_collector(
                    idx,
                    None,
                    request_id,
                    prompt,
                    task_gen_params,
                    start_in_reasoning_mode,
                    mm_embeddings=embeddings,
                    streaming_mode=False,
                    disconnect_handler=disconnect_handler,
                )
            )
            gen_tasks.append(gen_task)

        await asyncio.wait([*gen_tasks])

        # Create response
        generations = []
        for task in gen_tasks:
            r = task.result()
            if isinstance(r, Exception):
                raise r
            generations.append(r)
        response = _compose_response(request.state.id, generations, model_path.name, return_usage)

        xlogger.info(f"Finished chat completion request {request.state.id}", {"response": response})
        return response

    except CancelledError:
        raise

    except Exception as exc:
        error_message = handle_request_error(
            f"Chat completion {request.state.id} aborted. Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc

    finally:
        await disconnect_handler.cleanup()
