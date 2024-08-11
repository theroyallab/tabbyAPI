"""Chat completion utilities for OAI server."""

import asyncio
import pathlib
from asyncio import CancelledError
from copy import deepcopy
from typing import List, Optional
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
from endpoints.OAI.utils.completion import _stream_collector
from endpoints.OAI.types.tools import ToolCall


def _create_response(
    request_id: str, generations: List[dict], model_name: Optional[str]
):
    """Create a chat completion response from the provided text."""

    prompt_tokens = unwrap(generations[-1].get("prompt_tokens"), 0)
    completion_tokens = unwrap(generations[-1].get("generated_tokens"), 0)

    choices = []
    for index, generation in enumerate(generations):
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
            # lets check that we are getting the stop str before going forward
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


def format_prompt_with_template(
        data: ChatCompletionRequest, tool_precursor: Optional[str] = None
    ):
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

            if "tool_calls" in message:
                message["tool_calls_json"] = json.dumps(message["tool_calls"], indent=2)

        # Overwrite any protected vars with their values
        data.template_vars.update(
            {
                "messages": data.messages,
                "add_generation_prompt": data.add_generation_prompt,
                "tools_json": json.dumps(data.model_dump()['tools'], indent=2),
                "functions_json": json.dumps(data.functions, indent=2),
                "tool_precursor": tool_precursor,
                **special_tokens_dict,
            }
        )

        prompt = model.container.prompt_template.render(
            data.template_vars
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
        bos_token = special_tokens_dict.get("bos_token")
        if bos_token and prompt.startswith(bos_token):
            prompt = prompt.removeprefix(bos_token)

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

def update_stop_strings(data: ChatCompletionRequest):
    # Moved out of format_prompt_with_template since this can be called multiple
    # times when a tool call is initiated
    template_stop_strings = model.container.prompt_template.stop_strings(
        data.template_vars
    )
    if isinstance(data.stop, str):
        data.stop = [data.stop] + template_stop_strings
    else:
        data.stop += template_stop_strings

def update_tool_data(data: ChatCompletionRequest):
    # Same as update_stop_strings
    tool_starts = model.container.prompt_template.tool_params(
        data.template_vars
    )

    if data.tool_call_start is None and len(tool_starts) > 0:
        data.tool_call_start = tool_starts

    if len(tool_starts) > 0:
        data.stop.extend(tool_starts)

async def stream_generate_chat_completion(
    prompt: str, data: ChatCompletionRequest, request: Request, model_path: pathlib.Path
):
    """Generator for the generation process."""
    abort_event = asyncio.Event()
    gen_queue = asyncio.Queue()
    gen_tasks: List[asyncio.Task] = []
    disconnect_task = asyncio.create_task(request_disconnect_loop(request))

    try:
        logger.info(f"Received chat completion streaming request {request.state.id}")

        gen_params = data.to_gen_params()

        for n in range(0, data.n):
            if n > 0:
                task_gen_params = deepcopy(gen_params)
            else:
                task_gen_params = gen_params

            gen_task = asyncio.create_task(
                _stream_collector(
                    n,
                    gen_queue,
                    prompt,
                    request.state.id,
                    abort_event,
                    **task_gen_params,
                )
            )

            gen_tasks.append(gen_task)

        # We need to keep track of the text generated so we can resume the tool calls
        current_generation_text = ""
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
    prompt: str, data: ChatCompletionRequest, request: Request, model_path: pathlib.Path
):
    gen_tasks: List[asyncio.Task] = []
    gen_params = data.to_gen_params()

    # save prompt to disk
    with open("prompt.txt", "w") as f:
        f.write(prompt)

    try:
        for n in range(0, data.n):
            # Deepcopy gen params above the first index
            # to ensure nested structures aren't shared
            if n > 0:
                task_gen_params = deepcopy(gen_params)
            else:
                task_gen_params = gen_params

            gen_tasks.append(
                asyncio.create_task(
                    model.container.generate(
                        prompt, request.state.id, **task_gen_params
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
    tool_data = deepcopy(data)  # Do we need to deepcopy this such that we don't
    # modify the upstream data when overwriting json_schema with the tool schema?
    tool_data.json_schema = tool_data.tool_call_schema
    gen_params = tool_data.to_gen_params()

    for idx, gen in enumerate(generations):
        if gen["stop_str"] in tool_data.tool_call_start:
            if (
                "text" in gen
            ):  # non streaming, all generations will have the text they generated
                pre_tool_prompt = format_prompt_with_template(data, gen["text"])
            elif current_generations is not None:
                # streaming, we wont have text in the generation,
                #we'll have to use the current_generations
                pre_tool_prompt = format_prompt_with_template(data, current_generations)
            else:
                raise Exception(
                    "No text found in generation and no current_generations provided"
                )
            
            # save pre_tool_prompt to disk
            with open("pre_tool_prompt.txt", "w") as f:
                f.write(pre_tool_prompt)

            
            gen_tasks.append(
                asyncio.create_task(
                    model.container.generate(
                        pre_tool_prompt,
                        request.state.id,
                        **gen_params
                    )
                )
            )
            tool_idx.append(idx)

    tool_calls = await asyncio.gather(*gen_tasks)
    for outer_idx in range(0, len(tool_idx)):
        gen_idx = tool_idx[outer_idx]
        generations[gen_idx]["tool_calls"] = tool_calls[outer_idx]["text"]

    return generations

def postprocess_tool_call(call_str:str) -> List[ToolCall]:
    tool_calls = json.loads(call_str)
    for tool_call in tool_calls:
        tool_call["function"]["arguments"] = json.dumps(
            tool_call["function"]["arguments"]
        )
    return [ToolCall(**tool_call) for tool_call in tool_calls]
