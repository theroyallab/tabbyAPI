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
            message = ChatCompletionMessage(content=f"\n\nTool call error: {error_msg}")
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

        while True:
            if disconnect_task.done():
                abort_event.set()
                handle_request_disconnect(
                    f"Chat completion generation {request.state.id} cancelled by user."
                )

            generation = await gen_queue.get()

            if isinstance(generation, dict) and "error" in generation:
                error_msg = generation["error"]
                logger.error(f"Generation error: {error_msg}")
                yield get_generator_error(error_msg)
                break

            if isinstance(generation, Exception):
                raise generation

            response = _create_stream_chunk(
                request.state.id, generation, model_path.name
            )
            yield response.model_dump_json()

            if all(task.done() for task in gen_tasks) and gen_queue.empty():
                yield "[DONE]"
                logger.info(f"Finished chat completion streaming request {request.state.id}")
                break

    except asyncio.CancelledError:
        if not disconnect_task.done():
            abort_event.set()
            handle_request_disconnect("Chat completion generation cancelled by user.")
    except Exception as e:
        logger.error(f"Stream generation error: {str(e)}")
        logger.error(traceback.format_exc())
        yield get_generator_error("Chat completion aborted. Please check the server console.")
    finally:
        for task in gen_tasks:
            if not task.done():
                task.cancel()
        if not disconnect_task.done():
            disconnect_task.cancel()
# Add this new helper function for periodic model checking
async def _periodic_model_check(request_id: str, abort_event: asyncio.Event):
    """Periodically check if the model is still available."""
    try:
        while not abort_event.is_set():
            # Check model state every second
            error_dict = await check_model_before_operation(request_id, "chat completion")
            if error_dict:
                logger.warning("Model was unloaded during generation. Setting abort event.")
                abort_event.set()
                break
            
            # Sleep briefly to avoid too frequent checks
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        # Task was cancelled, just exit
        pass
    except Exception as e:
        logger.error(f"Error in model check task: {str(e)}")

async def generate_chat_completion(
    prompt: str,
    embeddings: MultimodalEmbeddingWrapper,
    data: ChatCompletionRequest,
    request: Request,
    model_path: pathlib.Path,
):
    gen_tasks: List[asyncio.Task] = []

    try:
        error_dict = await check_model_before_operation(
            request.state.id, "chat completion"
        )
        if error_dict:
            error_message = handle_request_error(
                f"Chat completion aborted: {error_dict['error']}",
                exc_info=False,
            ).error.message
            raise HTTPException(503, error_message)

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

            if data.tool_call_start:
                generations = await generate_tool_calls(data, generations, request)

            response = _create_response(request.state.id, generations, model_path.name)

            logger.info(f"Finished chat completion request {request.state.id}")

            return response

        except ValueError as exc:
            # Handle context length errors explicitly
            if "context length" in str(exc).lower():
                error_message = handle_request_error(
                    f"Chat completion aborted: {str(exc)}",
                    exc_info=False,
                ).error.message
                raise HTTPException(400, error_message) from exc
            else:
                raise

    except HTTPException:
        raise  # re-raise HTTPExceptions directly
    except Exception as exc:
        error_message = handle_request_error(
            f"Chat completion {request.state.id} aborted. Please check the server console.",
            exc_info=True,
        ).error.message
        raise HTTPException(503, error_message) from exc

    finally:
        # Always decrement the active generation counter
        await track_generation_end(request.state.id)

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
