"""Chat completion utilities for OAI server."""

import asyncio
import pathlib
from asyncio import CancelledError
from copy import deepcopy
from typing import List, Optional
from uuid import uuid4
import json

from fastapi import HTTPException, Request
from jinja2 import TemplateError
from loguru import logger

from common import model
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


def _create_response(generations: List[dict], model_name: Optional[str]):
    """Create a chat completion response from the provided text."""

    prompt_tokens = unwrap(generations[-1].get("prompt_tokens"), 0)
    completion_tokens = unwrap(generations[-1].get("generated_tokens"), 0)

    choices = []
    for index, generation in enumerate(generations):
        message = ChatCompletionMessage(
            role="assistant", content=unwrap(generation.get("text"), "")
        )

        tool_calls = generation['tool_calls']
        if tool_calls:
            tool_calls_json = json.loads(tool_calls)
            message.tool_calls = tool_calls_json

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
            stop_str=generation.get("stop_str"), # lets check that we are getting the stop str before going forward
            message=message,
            logprobs=logprob_response,
        )

        choices.append(choice)

    response = ChatCompletionResponse(
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
    const_id: str,
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
        id=const_id,
        choices=choices,
        model=unwrap(model_name, ""),
        usage=usage_stats,
    )

    return chunk


def format_prompt_with_template(data: ChatCompletionRequest):
    """
    Compile the prompt and get any additional stop strings from the template.
    Template stop strings can be overriden by sampler overrides if force is true.
    """

    try:
        special_tokens_dict = model.container.get_special_tokens(
            unwrap(data.add_bos_token, True),
            unwrap(data.ban_eos_token, False),
        )

        # Deal with list in messages.content
        # Just replace the content list with the very first text message
        for message in data.messages:
            if message["role"] == "user" and isinstance(message["content"], list):
                message["content"] = next(
                    (
                        content["text"]
                        for content in message["content"]
                        if content["type"] == "text"
                    ),
                    "",
                )

        # Overwrite any protected vars with their values
        data.template_vars.update(
            {
                "messages": data.messages,
                "add_generation_prompt": data.add_generation_prompt,
                "tools": data.tools,
                "functions": data.functions,
                **special_tokens_dict,
            }
        )

        prompt, template_stop_strings = model.container.prompt_template.render(
            data.template_vars
        )

        tool_start, tool_end = model.container.prompt_template.tool_params(
            data.template_vars
        )

        if data.tool_call_start is None and tool_start is not None:
            data.tool_call_start = tool_start

        if data.tool_call_end is None and tool_end is not None:
            data.tool_call_end = tool_end

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
        bos_token = special_tokens_dict.get("bos_token")
        if bos_token and prompt.startswith(bos_token):
            prompt = prompt.removeprefix(bos_token)

        # Append template stop strings
        if isinstance(data.stop, str):
            data.stop = [data.stop] + template_stop_strings
        else:
            data.stop += template_stop_strings

        # Adds the string to stop generation before the model generates a toolcall
        if tool_start:
            data.stop.append(tool_start)
        
        return prompt

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


async def _stream_collector(
    task_idx: int,
    gen_queue: asyncio.Queue,
    prompt: str,
    abort_event: asyncio.Event,
    **kwargs,
):
    """Collects a stream and places results in a common queue"""

    try:
        new_generation = model.container.generate_gen(prompt, abort_event, **kwargs)
        async for generation in new_generation:
            generation["index"] = task_idx

            await gen_queue.put(generation)

            if "finish_reason" in generation:
                break
    except Exception as e:
        await gen_queue.put(e)


async def stream_generate_chat_completion(
    prompt: str, data: ChatCompletionRequest, request: Request, model_path: pathlib.Path
):
    """Generator for the generation process."""
    const_id = f"chatcmpl-{uuid4().hex}"
    abort_event = asyncio.Event()
    gen_queue = asyncio.Queue()
    gen_tasks: List[asyncio.Task] = []
    disconnect_task = asyncio.create_task(request_disconnect_loop(request))

    try:
        gen_params = data.to_gen_params()

        for n in range(0, data.n):
            if n > 0:
                task_gen_params = deepcopy(gen_params)
            else:
                task_gen_params = gen_params

            gen_task = asyncio.create_task(
                _stream_collector(n, gen_queue, prompt, abort_event, **task_gen_params)
            )

            gen_tasks.append(gen_task)

        # Consumer loop
        while True:
            if disconnect_task.done():
                abort_event.set()
                handle_request_disconnect("Completion generation cancelled by user.")

            generation = await gen_queue.get()

            # Stream collector will push an exception to the queue if it fails
            if isinstance(generation, Exception):
                raise generation

            response = _create_stream_chunk(const_id, generation, model_path.name)
            yield response.model_dump_json()

            # Check if all tasks are completed
            if all(task.done() for task in gen_tasks) and gen_queue.empty():
                # Send a usage chunk
                if data.stream_options and data.stream_options.include_usage:
                    usage_chunk = _create_stream_chunk(
                        const_id, generation, model_path.name, is_usage_chunk=True
                    )
                    yield usage_chunk.model_dump_json()

                yield "[DONE]"
                break
    except CancelledError:
        # Get out if the request gets disconnected

        abort_event.set()
        handle_request_disconnect("Chat completion generation cancelled by user.")
    except Exception:
        yield get_generator_error(
            "Chat completion aborted. Please check the server console."
        )

async def generate_chat_completion(
    prompt: str, data: ChatCompletionRequest, model_path: pathlib.Path
):
    gen_tasks: List[asyncio.Task] = []
    gen_params = data.to_gen_params()

    try:
        for n in range(0, data.n):
            # Deepcopy gen params above the first index
            # to ensure nested structures aren't shared
            if n > 0:
                task_gen_params = deepcopy(gen_params)
            else:
                task_gen_params = gen_params

            gen_tasks.append(
                asyncio.create_task(model.container.generate(prompt, **task_gen_params))
            )

        generations = await asyncio.gather(*gen_tasks)
        if data.tool_call_start: # Let's not waste our time if we arn't running a tool model
            generations = await generate_tool_calls(prompt, data, generations)
        response = _create_response(generations, model_path.name)

        return response
    except Exception as exc:
        error_message = handle_request_error(
            "Chat completion aborted. Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc

async def generate_tool_calls(
    prompt: str, data: ChatCompletionRequest, generations: List[str]
):
    gen_tasks: List[asyncio.Task] = []
    tool_idx: List[int] = []
    temp = deepcopy(data) # Do we need to deepcopy this such that we don't 
                          # modify the upstream data when overwriting json_schema with the tool schema??
    temp.json_schema = temp.tool_call_schema
    gen_params = temp.to_gen_params()

    for idx, gen in enumerate(generations):
        if gen['stop_str'] == temp.tool_call_start:
            pre_tool_prompt = prompt + gen['text'] + temp.tool_call_start # Need to add the tool start call here as the engine extacts it
            gen_tasks.append(
                asyncio.create_task(model.container.generate(pre_tool_prompt, **gen_params))
            )
            tool_idx.append(idx)

    tool_calls = await asyncio.gather(*gen_tasks)
    for outer_idx in range(0, len(tool_idx)):
        gen_idx = tool_idx[outer_idx]
        generations[gen_idx]['tool_calls'] = tool_calls[outer_idx]['text']

    return generations




