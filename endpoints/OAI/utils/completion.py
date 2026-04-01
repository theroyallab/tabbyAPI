"""
Completion utilities for OAI server.

Also serves as a common module for completions and chat completions.
"""

import asyncio
import json
import pathlib
from asyncio import CancelledError
from time import time

from fastapi import HTTPException, Request
from common.logger import xlogger
from typing import List, Optional

from common import model
from common.networking import (
    get_generator_error,
    handle_request_error,
    DisconnectHandler,
)
from endpoints.OAI.types.chat_completion import ChatCompletionLogprobs
from endpoints.OAI.types.completion import (
    CompletionRequest,
    CompletionResponse,
    CompletionRespChoice,
    chat_logprobs_to_completion_logprobs,
)
from endpoints.OAI.types.common import UsageStats
from endpoints.OAI.utils.common_ import aggregate_usage_stats, get_usage_stats


def _parse_gen_request_id(n: int, request_id: str, task_idx: int):
    if n > 1:
        return f"{request_id}-{task_idx}"
    else:
        return request_id


def _compose_response(
    request_id: str,
    generations: List[dict],
    model_name: Optional[str],
    return_usage,
) -> CompletionResponse:
    """
    Compose a completion response from generations collected in non-streaming mode.
    """

    choices = []
    for generation in generations:
        # Collected logprobs are in chat completion format, convert them here
        logprobs = generation.get("logprob_response")
        if logprobs:
            logprobs = chat_logprobs_to_completion_logprobs(logprobs)

        choices.append(
            CompletionRespChoice(
                index=generation.get("index"),
                finish_reason=generation.get("finish_reason", "stop"),
                logprobs=logprobs,
                text=generation.get("content"),
            )
        )

    response = CompletionResponse(
        id=f"cmpl-{request_id}",
        choices=choices,
        model=model_name,
        usage=(
            aggregate_usage_stats([get_usage_stats(g) for g in generations])
            if return_usage
            else None
        ),
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
    logprobs = generation.get("logprob_response")

    choice = {
        "index": generation.get("index"),
        "text": delta_content,
        "finish_reason": finish_reason if not suppress_finish else None,
    }

    if logprobs:
        choice["logprobs"] = chat_logprobs_to_completion_logprobs(logprobs).model_dump()

    # Only one choice in a streaming chunk
    choices = [choice]
    data = {
        "id": f"chatcmpl-{request_id}",
        "object": "text_completion",
        "choices": choices,
        "created": int(time()),
    }

    if model_name:
        data["model_name"] = model_name

    # Serialize
    s = json.dumps(data, ensure_ascii=False)  # TODO: Investigate ensure_ascii

    # Check if no data
    is_empty = not delta_content and not (finish_reason and not suppress_finish)
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
    choice = {
        "index": usage_index,
        "text": "",
        "finish_reason": last_finish_reason,
    }
    choices = [choice]
    data = {
        "id": f"chatcmpl-{request_id}",
        "object": "text_completion",
        "choices": choices,
        "created": int(time()),
        "usage": usage_stats.model_dump(mode="json"),
    }

    if model_name:
        data["model_name"] = model_name

    # Serialize
    s = json.dumps(data, ensure_ascii=False)  # TODO: Investigate ensure_ascii
    return s, data


async def _stream_collector(
    task_idx: int,
    gen_queue: asyncio.Queue | None,
    request_id: str,
    prompt: str,
    params: CompletionRequest,
    streaming_mode: bool = True,
    disconnect_handler: DisconnectHandler = None,
):
    """
    Starts a request on the backend and collects generations. Only single phase.

    In streaming mode, emits chunks of text to be emitted as deltas to the client.

    In non-streaming mode, collects everything with the same logic but then emits a single
    response packet at the end, to be combined with any other choices (for n>1 requests) and
    sent together to the client.
    """

    mc = model.container
    full_content = ""
    collected_logprobs = []

    try:
        new_generation = mc.stream_generate(
            request_id,
            prompt,
            params,
            disconnect_handler,
            None,
        )
        generation = {}
        async for generation in new_generation:
            generation["index"] = task_idx
            delta_content = generation.get("text", "")
            full_content += delta_content
            finish_reason = generation.get("finish_reason")

            if "logprobs_content" in generation:
                collected_logprobs += generation["logprobs_content"]

            # Add the output and emit
            if streaming_mode:
                if len(collected_logprobs):
                    generation["logprob_response"] = ChatCompletionLogprobs(
                        content=collected_logprobs
                    )
                    collected_logprobs = []
                generation["delta_content"] = delta_content
                await gen_queue.put(generation)

            # End
            if finish_reason:
                break

        # In non-streaming mode, return everything as a single result
        if not streaming_mode:
            has_content = bool(full_content.strip())
            if len(collected_logprobs):
                generation["logprob_response"] = ChatCompletionLogprobs(content=collected_logprobs)
            generation["content"] = full_content if has_content else ""
            return generation

    except Exception as e:
        if gen_queue:
            await gen_queue.put(e)
        else:
            return e


async def stream_generate_completion(
    prompts: str | list[str],
    data: CompletionRequest,
    request: Request,
    model_path: pathlib.Path,
    disconnect_handler: DisconnectHandler,
):
    """
    Generator for the generation process.
    """

    if isinstance(prompts, str):
        prompts = [prompts]

    gen_queue = asyncio.Queue()
    gen_tasks: List[asyncio.Task] = []
    return_usage = data.stream_options and data.stream_options.include_usage

    try:
        xlogger.info(
            f"Received completion streaming request {request.state.id}",
            {
                "prompts": prompts,
                "data": data.model_dump(mode="json"),
                "model_path": str(model_path),
            },
        )

        # For aggregating usage
        usage_stats_list = []

        # Spec wants us to repeat each batch item n times
        total_n = data.n * len(prompts)
        remaining_n = total_n
        for p_idx, prompt in enumerate(prompts):
            for n_idx in range(0, data.n):
                idx = p_idx * data.n + n_idx

                task_gen_params = data.model_copy(deep=True)
                task_gen_params.max_tokens += idx * 5
                request_id = _parse_gen_request_id(total_n, request.state.id, idx)

                gen_task = asyncio.create_task(
                    _stream_collector(
                        idx,
                        gen_queue,
                        request_id,
                        prompt,
                        task_gen_params,
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
                xlogger.info(f"Finished completion streaming request {request.state.id}")
                yield "[DONE]"
                break

    except CancelledError:
        raise

    except Exception as e:
        xlogger.error("Error during completion", str(e), details=f"\n{str(e)}")
        yield get_generator_error("Completion aborted. Please check the server console.")

    finally:
        await disconnect_handler.cleanup()


async def generate_completion(
    prompts: str | list[str],
    data: CompletionRequest,
    request: Request,
    model_path: pathlib.Path,
    disconnect_handler: DisconnectHandler,
):
    """Non-streaming generate for completions"""

    gen_tasks: List[asyncio.Task] = []
    return_usage = data.stream_options and data.stream_options.include_usage

    if isinstance(prompts, str):
        prompts = [prompts]

    try:
        xlogger.info(
            f"Received completion request {request.state.id}",
            {
                "prompts": prompts,
                "data": data.model_dump(mode="json"),
                "model_path": str(model_path),
            },
        )

        # Spec wants us to repeat each batch item n times
        total_n = data.n * len(prompts)
        for p_idx, prompt in enumerate(prompts):
            for n_idx in range(0, data.n):
                idx = p_idx * data.n + n_idx

                task_gen_params = data.model_copy(deep=True)
                request_id = _parse_gen_request_id(total_n, request.state.id, idx)

                gen_task = asyncio.create_task(
                    _stream_collector(
                        idx,
                        None,
                        request_id,
                        prompt,
                        task_gen_params,
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

        xlogger.info(f"Finished completion request {request.state.id}", {"response": response})
        return response

    except CancelledError:
        raise

    except Exception as exc:
        error_message = handle_request_error(
            f"Completion {request.state.id} aborted. Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc

    finally:
        await disconnect_handler.cleanup()
