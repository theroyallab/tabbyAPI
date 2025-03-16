"""Chat completion utilities for OAI server."""

import asyncio
import json
import pathlib
from asyncio import CancelledError
import traceback
from typing import List, Optional
from common.health import HealthManager
from fastapi import HTTPException, Request
from jinja2 import TemplateError
from loguru import logger

from common import model
from common.multimodal import MultimodalEmbeddingWrapper
from common.model_utils import (
    check_model_before_operation,
    track_generation_start,
    track_generation_end,
)
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
from endpoints.OAI.utils.tools import ToolCallProcessor


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

        # If a tool call error occurred, add it to the message content
        if "tool_call_error" in generation:
            error_msg = generation["tool_call_error"]
            message.content += f"\n\nTool call error: {error_msg}"
            finish_reason = "tool_call_error"
        # If a tool call is present, mark the finish reason as such
        elif message.tool_calls:
            finish_reason = "tool_calls"

        choice = ChatCompletionRespChoice(
            index=index,
            finish_reason=finish_reason,
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
        # Get the finish reason from the generation
        finish_reason = generation.get("finish_reason")
        choice = ChatCompletionStreamChoice(index=index, finish_reason=finish_reason)

        # Check if we have a tool call error
        if "tool_call_error" in generation:
            error_msg = generation["tool_call_error"]
            message = ChatCompletionMessage(
                content=f"\n\nTool call error: {error_msg}"
            )
            choice.delta = message
            choice.finish_reason = "tool_call_error"
        # Check if we have tool calls
        elif "tool_calls" in generation:
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
            message.tool_calls_json = ToolCallProcessor.to_json(message.tool_calls)

    # Ensure model container and tokenizer exist
    if model.container is None or model.container.model_is_unloading:
        # If model is being unloaded, raise a clear error
        raise ValueError(
            "Model is currently being unloaded. Please try again in a moment."
        )

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

        try:
            prompt, mm_embeddings, template_vars = await format_messages_with_template(
                data.messages,
                data.template_vars,
                data.add_bos_token,
                data.ban_eos_token,
            )
        except ValueError as e:
            # Handle the case where model is being unloaded
            if "Model is currently being unloaded" in str(e):
                error_message = handle_request_error(
                    str(e),
                    exc_info=False,
                ).error.message
                raise HTTPException(503, error_message) from e
            raise  # Re-raise other ValueError exceptions

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

        # Consumer loop
        while True:
            if disconnect_task.done():
                abort_event.set()
                handle_request_disconnect(
                    f"Chat completion generation {request.state.id} cancelled by user."
                )

            try:
                # Check if model container is still valid before getting from queue
                error_dict = await check_model_before_operation(
                    request.state.id, "chat completion"
                )
                if error_dict:
                    logger.warning("Model was unloaded during generation. Aborting.")
                    yield get_generator_error(
                        f"Chat completion aborted: {error_dict['error']}"
                    )
                    break

                # Use a timeout when getting from the queue to periodically
                # check model state
                try:
                    generation = await asyncio.wait_for(gen_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check model state again after timeout
                    error_dict = await check_model_before_operation(
                        request.state.id, "chat completion"
                    )
                    if error_dict:
                        logger.warning(
                            "Model was unloaded while waiting for generation. Aborting."
                        )
                        yield get_generator_error(
                            f"Chat completion aborted: {error_dict['error']}"
                        )
                        break
                    continue  # Try again

                # Stream collector will push an exception to the queue if it fails
                if isinstance(generation, Exception):
                    raise generation

                # Check for special error indicator
                if isinstance(generation, dict) and "error" in generation:
                    logger.error(f"Generation error: {generation['error']}")
                    yield get_generator_error(
                        f"Chat completion error: {generation['error']}"
                    )
                    break

                # Only try to append text if generation is a dict and has 'text'
                if (
                    data.tool_call_start
                    and isinstance(generation, dict)
                    and "text" in generation
                ):
                    current_generation_text += generation["text"]

                # check if we are running a tool model, and that we are at stop
                if (
                    data.tool_call_start
                    and isinstance(generation, dict)
                    and "stop_str" in generation
                ):
                    try:
                        generations = await generate_tool_calls(
                            data,
                            [generation],
                            request,
                            current_generations=current_generation_text,
                        )
                        generation = generations[
                            0
                        ]  # We only have one generation in this case
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Tool call error: {error_msg}")
                        logger.error(traceback.format_exc())
                        # Add error information to the generation
                        generation["tool_call_error"] = error_msg
                        generation["finish_reason"] = "tool_call_error"

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
                        f"Finished chat completion streaming request "
                        f"{request.state.id}"
                    )

                    yield "[DONE]"
                    break
            except asyncio.CancelledError:
                # Just re-raise cancellation to be handled by outer try/except
                raise
            except Exception as e:
                # Log the error for debugging
                logger.error(f"Error processing generation: {str(e)}")
                logger.error(traceback.format_exc())

                # Push the error to the health manager
                await HealthManager.add_unhealthy_event(e)

                # Yield an error message
                yield get_generator_error(
                    "Chat completion aborted. Please check the server console."
                )
                break

    except CancelledError:
        # Get out if the request gets disconnected
        if not disconnect_task.done():
            abort_event.set()
            handle_request_disconnect("Chat completion generation cancelled by user.")
    except Exception as e:
        # Log the error
        logger.error(f"Stream generation error: {str(e)}")
        logger.error(traceback.format_exc())

        # Add to health manager
        await HealthManager.add_unhealthy_event(e)

        yield get_generator_error(
            "Chat completion aborted. Please check the server console."
        )
    finally:
        # Make sure to clean up any tasks
        for task in gen_tasks:
            if not task.done():
                task.cancel()

        if not disconnect_task.done():
            disconnect_task.cancel()


async def generate_chat_completion(
    prompt: str,
    embeddings: MultimodalEmbeddingWrapper,
    data: ChatCompletionRequest,
    request: Request,
    model_path: pathlib.Path,
):
    gen_tasks: List[asyncio.Task] = []

    try:
        # Check if model container is still valid
        error_dict = await check_model_before_operation(
            request.state.id, "chat completion"
        )
        if error_dict:
            logger.warning("Model was unloaded during generation. Aborting.")
            error_message = handle_request_error(
                f"Chat completion aborted: {error_dict['error']}",
                exc_info=False,
            ).error.message
            raise HTTPException(503, error_message)

        # Track the start of the generation
        await track_generation_start(request.state.id, model=data.model)

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

            # Let's not waste our time if we aren't running a tool model
            if data.tool_call_start:
                generations = await generate_tool_calls(data, generations, request)

            response = _create_response(request.state.id, generations, model_path.name)

            logger.info(f"Finished chat completion request {request.state.id}")

            return response
        finally:
            # Track the end of the generation
            await track_generation_end(request.state.id)
    except asyncio.CancelledError:
        # Make sure to cancel all tasks if one is cancelled
        for task in gen_tasks:
            if not task.done():
                task.cancel()
        raise
    except Exception as exc:
        # Cancel any remaining tasks
        for task in gen_tasks:
            if not task.done():
                task.cancel()

        error_message = handle_request_error(
            f"Chat completion {request.state.id} aborted. "
            "Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Add the issue to the health manager
        await HealthManager.add_unhealthy_event(exc)

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

    try:
        # Check if model container is still valid
        error_dict = await check_model_before_operation(
            request.state.id, "tool call generation"
        )
        if error_dict:
            logger.warning(
                f"Model was unloaded, cannot generate tool calls for "
                f"request {request.state.id}"
            )
            # Return the original generations without tool calls
            return generations
        for idx, gen in enumerate(generations):
            if not isinstance(gen, dict):
                logger.warning(f"Unexpected generation type: {type(gen)}")
                continue

            stop_str = gen.get("stop_str")
            if not stop_str:
                continue

            if stop_str in tool_data.tool_call_start:
                try:
                    if "text" in gen:
                        # non streaming, all generations will have the text
                        # they generated
                        pre_tool_prompt, mm_embeddings = await apply_chat_template(
                            data, gen["text"]
                        )
                    elif current_generations is not None:
                        # streaming, we wont have text in the generation,
                        # we'll have to use the current_generations
                        pre_tool_prompt, mm_embeddings = await apply_chat_template(
                            data, current_generations
                        )
                    else:
                        logger.warning("No text available for tool call generation")
                        continue

                    # Check if model is still available before creating task
                    error_dict = await check_model_before_operation(
                        request.state.id, "tool call generation"
                    )
                    if error_dict:
                        logger.warning(
                            f"Model was unloaded, cannot generate tool calls for "
                            f"request {request.state.id}"
                        )
                        continue

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
                except Exception as e:
                    logger.error(f"Error setting up tool call: {str(e)}")
                    logger.error(traceback.format_exc())
                    await HealthManager.add_unhealthy_event(e)

        if gen_tasks:
            try:
                # Check if model is still available before gathering results
                error_dict = await check_model_before_operation(
                    request.state.id, "tool call generation"
                )
                if error_dict:
                    logger.warning(
                        f"Model was unloaded during tool call generation for "
                        f"request {request.state.id}"
                    )
                    # Cancel any pending tasks
                    for task in gen_tasks:
                        if not task.done():
                            task.cancel()
                    return generations

                tool_calls = await asyncio.gather(*gen_tasks, return_exceptions=True)

                for outer_idx in range(0, len(tool_idx)):
                    gen_idx = tool_idx[outer_idx]

                    # Check if we got an error
                    if isinstance(tool_calls[outer_idx], Exception):
                        error_msg = str(tool_calls[outer_idx])
                        logger.error(f"Tool call generation error: {error_msg}")
                        
                        # Add error information to the generation
                        generations[gen_idx]["tool_call_error"] = error_msg
                        generations[gen_idx]["finish_reason"] = "tool_call_error"
                        continue

                    # Check for model unloading error
                    if (
                        isinstance(tool_calls[outer_idx], dict)
                        and "error" in tool_calls[outer_idx]
                    ):
                        error_msg = tool_calls[outer_idx]["error"]
                        if "model was unloaded" in error_msg.lower():
                            logger.warning(
                                f"Model was unloaded during tool call generation: "
                                f"{error_msg}"
                            )
                        else:
                            logger.error(f"Tool call error: {error_msg}")
                        
                        # Add error information to the generation
                        generations[gen_idx]["tool_call_error"] = error_msg
                        generations[gen_idx]["finish_reason"] = "tool_call_error"
                        continue

                    # Only set tool_calls if we have valid text
                    if (
                        isinstance(tool_calls[outer_idx], dict)
                        and "text" in tool_calls[outer_idx]
                    ):
                        generations[gen_idx]["tool_calls"] = tool_calls[outer_idx][
                            "text"
                        ]
            except asyncio.CancelledError:
                # Re-raise cancellation
                raise
            except Exception as e:
                logger.error(f"Error gathering tool call results: {str(e)}")
                logger.error(traceback.format_exc())
                await HealthManager.add_unhealthy_event(e)
    except Exception as e:
        # Log the error and continue
        logger.error(f"Error in generate_tool_calls: {str(e)}")
        logger.error(traceback.format_exc())
        await HealthManager.add_unhealthy_event(e)

    return generations
