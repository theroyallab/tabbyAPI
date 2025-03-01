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
from common.tabby_config import config
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
from endpoints.OAI.utils.completion import _stream_collector
from endpoints.OAI.types.tools import ToolCall


def _extract_think_content(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract content between <think> tags and the remaining content. Only available in none-streaming mode."""
    if config.network.reasoning_start_token not in text and config.network.reasoning_end_token not in text:
        return None, text
    elif config.network.reasoning_start_token in text:
        start_reasoning = text.split(config.network.reasoning_start_token)[1]
        reasoning_content = start_reasoning.split(config.network.reasoning_end_token)[0]
        content = start_reasoning.split(config.network.reasoning_end_token)[1]
        return reasoning_content, content
    else:
        reasoning_content = text.split(config.network.reasoning_end_token)[0]
        content = text.split(config.network.reasoning_end_token)[1]
        return reasoning_content, content

def _create_response(
    request_id: str, generations: List[dict], model_name: Optional[str]
):
    """Create a chat completion response from the provided text."""

    prompt_tokens = unwrap(generations[-1].get("prompt_tokens"), 0)
    completion_tokens = unwrap(generations[-1].get("generated_tokens"), 0)

    choices = []
    for index, generation in enumerate(generations):
        if config.network.reasoning_parser:
            raw_content = unwrap(generation.get("text"), "")
            reasoning_content, content = _extract_think_content(raw_content)
            message = ChatCompletionMessage(
                role="assistant", reasoning_content=reasoning_content, content=content
            )
        else:
            message = ChatCompletionMessage(
                role="assistant", content=unwrap(generation.get("text"), "")
            )

        tool_calls = generation["tool_calls"]
        if tool_calls:
            message.tool_calls = postprocess_tool_call(tool_calls)

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

        choice = ChatCompletionRespChoice(
            index=index,
            finish_reason=generation.get("finish_reason"),
            stop_str=generation.get("stop_str"),
            message=message,
            logprobs=logprob_response,
        )

        choices.append(choice)

    response = ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        choices=choices,
        model=unwrap(model_name, ""),
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return response


def _create_stream_chunk(
    request_id: str,
    generation: Optional[dict] = None,
    model_name: Optional[str] = None,
    is_usage_chunk: bool = False,
    is_reasoning_chunk: bool = False,
):
    """Create a chat completion stream chunk from the provided text."""

    index = generation.get("index")
    choices = []
    usage_stats = None

    if is_usage_chunk:
        prompt_tokens = unwrap(generation.get("prompt_tokens"), 0)
        completion_tokens = unwrap(generation.get("generated_tokens"), 0)

        usage_stats = UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    elif "finish_reason" in generation:
        choice = ChatCompletionStreamChoice(
            index=index,
            finish_reason=generation.get("finish_reason"),
        )

        # lets check if we have tool calls since we are at the end of the generation
        if "tool_calls" in generation:
            tool_calls = generation["tool_calls"]
            message = ChatCompletionMessage(
                tool_calls=postprocess_tool_call(tool_calls)
            )
            choice.delta = message

        choices.append(choice)

    else:
        message = ChatCompletionMessage(
            role="assistant", reasoning_content=unwrap(generation.get("text"), "")
        ) if is_reasoning_chunk else ChatCompletionMessage(
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
        data.stop += template_metadata.stop_strings

    # Tool call start strings
    if template_metadata.tool_starts:
        if data.tool_call_start is None:
            data.tool_call_start = template_metadata.tool_starts

        # Append to stop strings to halt for a tool call generation
        data.stop.extend(template_metadata.tool_starts)


async def format_messages_with_template(
    messages: List[ChatCompletionMessage],
    existing_template_vars: Optional[dict] = None,
    add_bos_token: bool = True,
    ban_eos_token: bool = False,
):
    """Barebones function to format chat completion messages into a prompt."""

    template_vars = unwrap(existing_template_vars, {})
    mm_embeddings = MultimodalEmbeddingWrapper() if model.container.use_vision else None

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

        if message.tool_calls:
            message.tool_calls_json = json.dumps(message.tool_calls, indent=2)

    special_tokens_dict = model.container.get_special_tokens(
        add_bos_token, ban_eos_token
    )

    template_vars.update({"messages": messages, **special_tokens_dict})

    prompt = await model.container.prompt_template.render(template_vars)
    return prompt, mm_embeddings, template_vars


async def apply_chat_template(
    data: ChatCompletionRequest, tool_precursor: Optional[str] = None
):
    """
    Compile the prompt and get any additional stop strings from the template.
    Template stop strings can be overriden by sampler overrides if force is true.
    """

    try:
        data.template_vars.update(
            {
                "add_generation_prompt": data.add_generation_prompt,
                "tools_json": json.dumps(data.model_dump()["tools"], indent=2),
                "functions_json": json.dumps(data.functions, indent=2),
                "tool_precursor": tool_precursor,
            }
        )

        prompt, mm_embeddings, template_vars = await format_messages_with_template(
            data.messages, data.template_vars, data.add_bos_token, data.ban_eos_token
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

        # Removes the starting BOS token if present
        # This is to prevent add_bos_token from adding multiple bos tokens
        bos_token = template_vars.get("bos_token")
        if bos_token and prompt.startswith(bos_token):
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
    disconnect_task = asyncio.create_task(request_disconnect_loop(request))

    try:
        logger.info(f"Received chat completion streaming request {request.state.id}")

        for n in range(0, data.n):
            task_gen_params = data.model_copy(deep=True)

            gen_task = asyncio.create_task(
                _stream_collector(
                    n,
                    gen_queue,
                    prompt,
                    request.state.id,
                    abort_event,
                    embeddings=embeddings,
                    **task_gen_params.model_dump(exclude={"prompt"}),
                )
            )

            gen_tasks.append(gen_task)

        # We need to keep track of the text generated so we can resume the tool calls
        current_generation_text = ""
        
        is_reasoning_chunk = config.network.reasoning_parser

        # Consumer loop
        while True:
            if disconnect_task.done():
                abort_event.set()
                handle_request_disconnect(
                    f"Chat completion generation {request.state.id} cancelled by user."
                )

            generation = await gen_queue.get()
            # lets only append the text if we need it for tool calls later
            if data.tool_call_start and "text" in generation:
                current_generation_text += generation["text"]

            # check if we are running a tool model, and that we are at stop
            if data.tool_call_start and "stop_str" in generation:
                generations = await generate_tool_calls(
                    data,
                    [generation],
                    request,
                    current_generations=current_generation_text,
                )
                generation = generations[0]  # We only have one generation in this case

            # Stream collector will push an exception to the queue if it fails
            if isinstance(generation, Exception):
                raise generation

            if unwrap(generation.get("text"), "") == config.network.reasoning_end_token:
                # Update reasoning chunk flag
                is_reasoning_chunk = False
                # And skip this token
                continue
            
            response = _create_stream_chunk(
                request.state.id, generation, model_path.name, is_reasoning_chunk=is_reasoning_chunk
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

    try:
        for _ in range(0, data.n):
            gen_tasks.append(
                asyncio.create_task(
                    model.container.generate(
                        prompt,
                        request.state.id,
                        embeddings=embeddings,
                        **data.model_dump(exclude={"prompt"}),
                    )
                )
            )

        generations = await asyncio.gather(*gen_tasks)

        # Let's not waste our time if we arn't running a tool model
        if data.tool_call_start:
            generations = await generate_tool_calls(data, generations, request)

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
    data: ChatCompletionRequest,
    generations: List[str],
    request: Request,
    current_generations: str = None,
):
    gen_tasks: List[asyncio.Task] = []
    tool_idx: List[int] = []

    # Copy to make sure the parent JSON schema doesn't get modified
    # FIXME: May not be necessary depending on how the codebase evolves
    tool_data = data.model_copy(deep=True)
    tool_data.json_schema = tool_data.tool_call_schema
    gen_params = tool_data.model_dump()

    for idx, gen in enumerate(generations):
        if gen["stop_str"] in tool_data.tool_call_start:
            if "text" in gen:
                # non streaming, all generations will have the text they generated
                pre_tool_prompt, mm_embeddings = await apply_chat_template(
                    data, gen["text"]
                )
            elif current_generations is not None:
                # streaming, we wont have text in the generation,
                # we'll have to use the current_generations
                pre_tool_prompt, mm_embeddings = await apply_chat_template(
                    data, current_generations
                )

            gen_tasks.append(
                asyncio.create_task(
                    model.container.generate(
                        pre_tool_prompt,
                        request.state.id,
                        embeddings=mm_embeddings,
                        **gen_params,
                    )
                )
            )
            tool_idx.append(idx)

    tool_calls = await asyncio.gather(*gen_tasks)
    for outer_idx in range(0, len(tool_idx)):
        gen_idx = tool_idx[outer_idx]
        generations[gen_idx]["tool_calls"] = tool_calls[outer_idx]["text"]

    return generations


def postprocess_tool_call(call_str: str) -> List[ToolCall]:
    tool_calls = json.loads(call_str)
    for tool_call in tool_calls:
        tool_call["function"]["arguments"] = json.dumps(
            tool_call["function"]["arguments"]
        )
    return [ToolCall(**tool_call) for tool_call in tool_calls]
