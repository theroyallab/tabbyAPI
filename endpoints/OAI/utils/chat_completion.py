"""Chat completion utilities for OAI server."""

import asyncio
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
from endpoints.OAI.utils.completion import _parse_gen_request_id, _stream_collector
from endpoints.OAI.utils.tools import ToolCallProcessor, TOOL_CALL_SCHEMA


def _create_response(
    request_id: str, generations: List[dict], model_name: Optional[str]
):
    """Create a chat completion response from the provided text."""

    choices = []
    for index, generation in enumerate(generations):
        message = ChatCompletionMessage(
            role="assistant", content=unwrap(generation.get("text"), "")
        )

        tool_calls = generation["tool_calls"]
        if tool_calls:
            message.tool_calls = ToolCallProcessor.from_json(tool_calls)

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

        # Initialize finish_reason with a default value or from generation data
        finish_reason = generation.get("finish_reason", "stop")

        # If a tool call is present, mark the finish reason as such
        if message.tool_calls:
            finish_reason = "tool_calls"

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
            prompt_time=round(final_generation.get("prompt_time"), 2),
            prompt_tokens_per_sec=round(final_generation.get("prompt_tokens_per_sec"), 2),
            completion_tokens=completion_tokens,
            completion_time=round(final_generation.get("gen_time"), 2),
            completion_tokens_per_sec=round(final_generation.get("gen_tokens_per_sec"), 2),
            total_tokens=prompt_tokens + completion_tokens,
            total_time=round(final_generation.get("total_time"), 2),
        ),
    )

    return response


def _create_stream_chunk(
    request_id: str,
    generation: Optional[dict] = None,
    model_name: Optional[str] = None,
    is_usage_chunk: bool = False,
):
    """Create a chat completion stream chunk from the provided text."""

    index = generation.get("index")
    choices = []
    usage_stats = None

    if is_usage_chunk:
        prompt_tokens = unwrap(generation.get("prompt_tokens"), 0)
        completion_tokens = unwrap(generation.get("gen_tokens"), 0)

        usage_stats = UsageStats(
            prompt_tokens=prompt_tokens,
            prompt_time=round(generation.get("prompt_time"), 2),
            prompt_tokens_per_sec=round(generation.get("prompt_tokens_per_sec"), 2),
            completion_tokens=completion_tokens,
            completion_time=round(generation.get("gen_time"), 2),
            completion_tokens_per_sec=round(generation.get("gen_tokens_per_sec"), 2),
            total_tokens=prompt_tokens + completion_tokens,
            total_time=round(generation.get("total_time"), 2),
        )
    elif "finish_reason" in generation:
        # Get the finish reason from the generation
        finish_reason = generation.get("finish_reason")
        choice = ChatCompletionStreamChoice(index=index, finish_reason=finish_reason)

        # lets check if we have tool calls since we are at the end of the generation
        # Mark finish_reason as tool_calls since this is the last chunk
        if "tool_calls" in generation:
            tool_calls = generation["tool_calls"]
            message = ChatCompletionMessage(
                tool_calls=ToolCallProcessor.from_json(tool_calls)
            )
            choice.delta = message
            choice.finish_reason = "tool_calls"

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
            if disconnect_task.done():
                abort_event.set()
                handle_request_disconnect(
                    f"Chat completion generation {request.state.id} cancelled by user."
                )

            generation = await gen_queue.get()

            # Handle options if a tool model is present
            if tool_start:
                if "stop_str" in generation:
                    generations = await generate_tool_calls(
                        prompt,
                        embeddings,
                        data,
                        [generation],
                        request,
                        current_generation_text=current_generation_text,
                    )

                    # Only one generation present in this case
                    generation = generations[0]
                elif "text" in generation:
                    current_generation_text += generation["text"]

            # Stream collector will push an exception to the queue if it fails
            if isinstance(generation, Exception):
                raise generation

            response = _create_stream_chunk(
                request.state.id, generation, model_path.name
            )
            yield response.model_dump_json()

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
                    yield usage_chunk.model_dump_json()

                logger.info(
                    f"Finished chat completion streaming request {request.state.id}"
                )

                yield "[DONE]"
                break
    except CancelledError:
        # Get out if the request gets disconnected

        if not disconnect_task.done():
            abort_event.set()
            handle_request_disconnect("Chat completion generation cancelled by user.")
    except Exception:
        yield get_generator_error(
            "Chat completion aborted. Please check the server console."
        )


async def generate_chat_completion(
    prompt: str,
    embeddings: MultimodalEmbeddingWrapper,
    data: ChatCompletionRequest,
    request: Request,
    model_path: pathlib.Path,
):
    gen_tasks: List[asyncio.Task] = []
    tool_start = model.container.prompt_template.metadata.tool_start

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

        response = _create_response(request.state.id, generations, model_path.name)

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
    current_generation_text: str = None,
):
    gen_tasks: List[asyncio.Task] = []
    tool_start = model.container.prompt_template.metadata.tool_start
    tool_end = model.container.prompt_template.metadata.tool_end
    if not tool_end:
        tool_end = tool_start

    # Copy to make sure the parent JSON schema doesn't get modified
    tool_data = data.model_copy(deep=True)
    tool_data.json_schema = TOOL_CALL_SCHEMA

    for idx, gen in enumerate(generations):
        # Initialize tool_calls for each generation
        gen["tool_calls"] = []

    processToolCalls = True
    while processToolCalls:
        tool_idx: List[int] = []  # <-- Move inside the loop to reset each iteration
        for idx, gen in enumerate(generations):
            processToolCalls = False
            if gen["stop_str"] != tool_start:
                continue

            logger.info(f"Detected tool call in chat completion request {request.state.id}")

            # Append the existing generation text if present
            precursor_text = current_generation_text or gen.get("text")
            if precursor_text:
                prompt = prompt + precursor_text

            # Append all previous tool_calls, each encapsulated within tool_start and tool_end
            if gen["tool_calls"]:
                for tool_call in gen["tool_calls"]:
                    prompt += f"{tool_start}{tool_call}{tool_end}"
            prompt = f"{prompt}{tool_start}"

            gen_request_id = _parse_gen_request_id(data.n, request.state.id, idx)
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
            gen_tasks.clear()
            logger.info(f"Generated tool calls for chat completion request {tool_calls}")

            # Map tool calls to their appropriate generation
            for gen_idx, tool_call in zip(tool_idx, tool_calls, strict=True):
                # Remove tool_end at the end of tool_call["text"]
                tool_call_text = tool_call["text"].strip()
                if tool_end and tool_call_text.endswith(tool_end):
                    tool_call_text = tool_call_text[: -len(tool_end)]

                generations[gen_idx]["tool_calls"].append(tool_call_text)
                generations[gen_idx]["stop_str"] = tool_call["stop_str"]
                generations[gen_idx]["prompt_tokens"] += tool_call.get("prompt_tokens", 0)
                generations[gen_idx]["prompt_time"] += tool_call.get("prompt_time", 0)
                generations[gen_idx]["prompt_tokens_per_sec"] = round(
                    generations[gen_idx].get("prompt_tokens", 0) / generations[gen_idx].get("prompt_time", 1), 2
                )

                generations[gen_idx]["gen_tokens"] += tool_call.get("gen_tokens", 0)
                generations[gen_idx]["gen_time"] += tool_call.get("gen_time", 0)
                generations[gen_idx]["gen_tokens_per_sec"] = round(
                    generations[gen_idx].get("gen_tokens", 0) / generations[gen_idx].get("gen_time", 1), 2
                )

                generations[gen_idx]["total_time"] += tool_call.get("total_time", 0)
                generations[gen_idx]["queue_time"] += tool_call.get("queue_time", 0)
                generations[gen_idx]["cached_tokens"] = tool_call.get("cached_tokens", 0)
                generations[gen_idx]["generated_tokens"] = tool_call.get("generated_tokens", 0)

                if tool_call["stop_str"] == tool_start:
                    # If the tool call ends with the tool_start, it means another tool call is required
                    processToolCalls = True
        

    for idx, gen in enumerate(generations):
        # Stringify tool calls
        if gen["tool_calls"]:
            gen["tool_calls"] = "[" + ", ".join(str(tool_call) for tool_call in gen["tool_calls"]) + "]"
    
    return generations