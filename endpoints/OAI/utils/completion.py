"""
Completion utilities for OAI server.

Also serves as a common module for completions and chat completions.
"""

import asyncio
import json
import math
import pathlib
from asyncio import CancelledError
from fastapi import HTTPException, Request, status
from loguru import logger
from typing import List, Optional, Union

from common import model
from common.auth import get_key_permission
from common.multimodal import MultimodalEmbeddingWrapper
from common.networking import (
    get_generator_error,
    handle_request_disconnect,
    handle_request_error,
    request_disconnect_loop,
)
from common.tabby_config import config
from common.utils import unwrap
from endpoints.OAI.types.completion import (
    CompletionRequest,
    CompletionResponse,
    CompletionRespChoice,
    CompletionLogProbs,
)
from endpoints.OAI.types.common import UsageStats
import time
from collections import OrderedDict
from common.sampling import BaseSamplerRequest


def _parse_gen_request_id(n: int, request_id: str, task_idx: int):
    if n > 1:
        return f"{request_id}-{task_idx}"
    else:
        return request_id


def _create_response(
    request_id: str, generations: Union[dict, List[dict]], model_name: str = ""
):
    """Create a completion response from the provided choices."""

    # Convert the single choice object into a list
    if not isinstance(generations, list):
        generations = [generations]

    choices: List[CompletionRespChoice] = []
    for index, generation in enumerate(generations):
        # Either create a logprobs object or set to null based on request
        # Even if logprobs wasn't requested, include the field as null
        logprob_response = None

        # Get list-based token data (preferred) or fall back to dict-based
        tokens = generation.get("tokens", [])
        token_logprobs = generation.get("token_logprobs", [])
        logprobs = unwrap(generation.get("logprobs"), [])
        offset = unwrap(generation.get("offset"), [])

        # Fallback to dict-based approach for backward compatibility
        token_probs = unwrap(generation.get("token_probs"), {})
        if not tokens and token_probs:
            tokens = list(token_probs.keys())
        if not token_logprobs and token_probs:
            token_logprobs = list(token_probs.values())

        # Ensure we have lists
        if tokens is None:
            tokens = []
        if token_logprobs is None:
            token_logprobs = []

        # Ensure text_offset is always a list
        if isinstance(offset, list):
            text_offset = offset
        else:
            text_offset = [offset] if offset is not None else []

        # Ensure top_logprobs is a list of dictionaries
        if isinstance(logprobs, list):
            top_logprobs = logprobs
        elif logprobs and logprobs != {}:  # Non-empty dict
            top_logprobs = [logprobs]
        else:
            top_logprobs = []

        # Only create logprobs response if we have actual data
        has_logprob_data = bool(
            tokens or token_logprobs or (top_logprobs and any(top_logprobs))
        )

        if has_logprob_data:
            # Create the logprobs response object even if arrays are empty
            logprob_response = CompletionLogProbs(
                text_offset=text_offset,
                token_logprobs=token_logprobs,
                tokens=tokens,
                top_logprobs=top_logprobs,
            )

        # Determine the finish reason. Default to "stop" if the backend
        # didn't provide one. This ensures compliance with the OpenAI
        # specification which requires the field to be non-null when the
        # generation is finished.
        finish_reason = unwrap(generation.get("finish_reason"), "stop")

        # The index can be located in the generation itself
        choice = CompletionRespChoice(
            index=unwrap(generation.get("index"), index),
            finish_reason=finish_reason,
            text=unwrap(generation.get("text"), ""),
            logprobs=logprob_response,
        )

        choices.append(choice)

    prompt_tokens = unwrap(generations[-1].get("prompt_tokens"), 0)
    completion_tokens = unwrap(generations[-1].get("generated_tokens"), 0)

    response = CompletionResponse(
        id=f"cmpl-{request_id}",
        choices=choices,
        created=int(time.time()),
        model=model_name,
        object="text_completion",
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return response


async def _stream_collector(
    task_idx: int,
    gen_queue: asyncio.Queue,
    request_id: str,
    prompt: str,
    params: CompletionRequest,
    abort_event: asyncio.Event,
    mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
    echo: bool = False,
):
    """Collects a stream and places results in a common queue"""

    try:
        new_generation = model.container.stream_generate(
            request_id,
            prompt,
            params,
            abort_event,
            mm_embeddings,
        )
        prompt_prefixed = False
        async for generation in new_generation:
            # Handle echo parameter - add prompt at the first chunk
            if echo and not prompt_prefixed:
                # Handle token IDs by decoding them first if needed
                if isinstance(prompt, list):
                    if all(isinstance(item, int) for item in prompt):
                        # Decode token IDs to text
                        prompt_text = model.container.decode_tokens(prompt)
                    else:
                        # Convert list to string (fallback)
                        prompt_text = str(prompt)
                else:
                    prompt_text = prompt

                generation["text"] = f"{prompt_text}{generation.get('text', '')}"
                prompt_prefixed = True

            generation["index"] = task_idx

            await gen_queue.put(generation)

            if "finish_reason" in generation:
                break
    except Exception as e:
        await gen_queue.put(e)


async def load_inline_model(model_name: str, request: Request):
    """Load a model from the data.model parameter"""

    # Return if the model container already exists and the model is fully loaded
    if (
        model.container
        and model.container.model_dir.name == model_name
        and model.container.loaded
    ):
        return

    # Return if inline loading is disabled
    # Also warn if an admin key is used
    if not config.model.inline_model_loading:
        if get_key_permission(request) == "admin":
            logger.warning(
                f"Unable to switch model to {model_name} because "
                '"inline_model_loading" is not True in config.yml.'
            )

        return

    is_dummy_model = (
        config.model.use_dummy_models and model_name in config.model.dummy_model_names
    )

    # Error if an invalid key is passed
    # If a dummy model is provided, don't error
    if get_key_permission(request) != "admin":
        if not is_dummy_model:
            error_message = handle_request_error(
                f"Unable to switch model to {model_name} because "
                + "an admin key isn't provided",
                exc_info=False,
            ).error.message

            raise HTTPException(401, error_message)
        else:
            return

    # Start inline loading
    # Past here, user is assumed to be admin

    # Skip if the model is a dummy
    if is_dummy_model:
        logger.warning(f"Dummy model {model_name} provided. Skipping inline load.")

        return

    model_path = pathlib.Path(config.model.model_dir)
    model_path = model_path / model_name

    # Model path doesn't exist
    if not model_path.exists():
        logger.warning(
            f"Could not find model path {str(model_path)}. Skipping inline model load."
        )

        return

    # Load the model and also add draft dir
    await model.load_model(
        model_path,
        draft_model=config.draft_model.model_dump(include={"draft_model_dir"}),
    )


async def stream_generate_completion(
    data: CompletionRequest, request: Request, model_path: pathlib.Path
):
    """Streaming generation for completions."""

    abort_event = asyncio.Event()
    gen_queue = asyncio.Queue()
    gen_tasks: List[asyncio.Task] = []
    disconnect_task = asyncio.create_task(request_disconnect_loop(request))

    try:
        logger.info(f"Received streaming completion request {request.state.id}")

        # Handle batch prompts
        prompts = (
            data.prompt if isinstance(data.prompt, list) else [data.prompt] * data.n
        )

        for idx in range(0, data.n):
            task_gen_params = data.model_copy(deep=True)
            request_id = _parse_gen_request_id(data.n, request.state.id, idx)
            # Get the corresponding prompt for this task
            task_prompt = prompts[idx] if idx < len(prompts) else prompts[-1]

            gen_task = asyncio.create_task(
                _stream_collector(
                    idx,
                    gen_queue,
                    request_id,
                    task_prompt,
                    task_gen_params,
                    abort_event,
                    echo=data.echo,
                )
            )

            gen_tasks.append(gen_task)

        # Consumer loop
        while True:
            if disconnect_task.done():
                abort_event.set()
                handle_request_disconnect(
                    f"Completion generation {request.state.id} cancelled by user."
                )

            generation = await gen_queue.get()

            # Stream collector will push an exception to the queue if it fails
            if isinstance(generation, Exception):
                raise generation

            response = _create_response(request.state.id, generation, model_path.name)
            
            # Convert to dict for potential modification
            response_dict = response.model_dump()
            
            # Exclude logprobs field when not requested to maintain compatibility
            if not (data.logprobs and data.logprobs > 0):
                for choice in response_dict.get("choices", []):
                    if choice.get("logprobs") is None:
                        choice.pop("logprobs", None)
            
            # Include object as "text_completion.chunk" in streaming responses
            # to match OpenAI behavior
            if isinstance(generation, dict) and not generation.get("finish_reason"):
                response_dict["object"] = "text_completion.chunk"
            
            yield json.dumps(response_dict)

            # Check if all tasks are completed
            if all(task.done() for task in gen_tasks) and gen_queue.empty():
                yield "[DONE]"
                logger.info(f"Finished streaming completion request {request.state.id}")
                break
    except CancelledError:
        # Get out if the request gets disconnected

        if not disconnect_task.done():
            abort_event.set()
            handle_request_disconnect(
                f"Completion generation {request.state.id} cancelled by user."
            )
    except Exception:
        yield get_generator_error(
            f"Completion {request.state.id} aborted. Please check the server console."
        )


async def generate_completion(
    data: CompletionRequest, request: Request, model_path: pathlib.Path
):
    """Non-streaming generate for completions"""

    gen_tasks: List[asyncio.Task] = []

    try:
        logger.info(f"Received completion request {request.state.id}")

        # Handle batch prompts
        prompts = (
            data.prompt if isinstance(data.prompt, list) else [data.prompt] * data.n
        )

        for idx in range(0, data.n):
            task_gen_params = data.model_copy(deep=True)
            request_id = _parse_gen_request_id(data.n, request.state.id, idx)
            # Get the corresponding prompt for this task
            task_prompt = prompts[idx] if idx < len(prompts) else prompts[-1]

            gen_tasks.append(
                asyncio.create_task(
                    model.container.generate(
                        request_id,
                        task_prompt,
                        task_gen_params,
                    )
                )
            )

        generations = await asyncio.gather(*gen_tasks)

        # ------------------------------------------------------------------
        # Populate logprob information for generated tokens if requested and
        # the backend didn't include it directly.
        # ------------------------------------------------------------------
        needs_logprob_computation = unwrap(data.logprobs, 0) > 0 and (
            unwrap(data.max_tokens, 0) > 0 or data.echo
        )

        if needs_logprob_computation:
            data_dict = data.model_dump(exclude={"prompt", "stream", "n"})
            data_dict.update(
                {
                    "max_tokens": 0,
                    "repetition_penalty": 1.0,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "logprobs": data.logprobs,
                    "extract_prompt_logprobs": data.echo,  # Extract prompt logprobs when echo=True
                }
            )
            sampler_params = BaseSamplerRequest(**data_dict)

            # Check if we can use batch processing for multiple prompts
            prompts_to_process = []
            indices_to_process = []
            
            for idx, generation in enumerate(generations):
                if generation.get("token_probs"):
                    continue
                    
                prompt = prompts[idx] if idx < len(prompts) else prompts[-1]
                full_text = f"{prompt}{generation.get('text', '')}"
                prompts_to_process.append(full_text)
                indices_to_process.append(idx)
            
            # Use batch processing if multiple prompts and model supports it
            if len(prompts_to_process) > 1 and hasattr(model.container, 'compute_batch_logprobs'):
                # Process all prompts in a single batch
                batch_results = model.container.compute_batch_logprobs(
                    prompts=prompts_to_process,
                    params=sampler_params,
                )
                
                # Assign results back to generations
                for batch_idx, gen_idx in enumerate(indices_to_process):
                    logprob_data = batch_results[batch_idx]
                    generation = generations[gen_idx]
                    
                    tokens = logprob_data.get("prompt_token_strings", [])
                    token_logprobs = logprob_data.get("prompt_token_logprobs", [])
                    top_logprobs = logprob_data.get("top_logprobs", [])
                    offsets = logprob_data.get("offset", [])
                    
                    ctx_len = len(tokens) - generation.get("generated_tokens", 0)
                    start_idx = 0 if data.echo else ctx_len
                    
                    # Store results
                    generation["tokens"] = tokens[start_idx:]
                    generation["token_logprobs"] = token_logprobs[start_idx:]
                    generation["logprobs"] = top_logprobs[start_idx:]
                    generation["offset"] = offsets[start_idx:]
                    generation["token_probs"] = OrderedDict(
                        zip(tokens[start_idx:], token_logprobs[start_idx:], strict=False)
                    )
            else:
                # Fall back to sequential processing
                for idx, generation in enumerate(generations):
                    if generation.get("token_probs"):
                        continue

                    prompt = prompts[idx] if idx < len(prompts) else prompts[-1]
                    full_text = f"{prompt}{generation.get('text', '')}"

                    logprob_data = model.container.compute_sequence_logprobs(
                        prompt=full_text,
                        params=sampler_params,
                    )

                    tokens = logprob_data.get("prompt_token_strings", [])
                    token_logprobs = logprob_data.get("prompt_token_logprobs", [])
                    top_logprobs = logprob_data.get("top_logprobs", [])
                    offsets = logprob_data.get("offset", [])

                    ctx_len = len(tokens) - generation.get("generated_tokens", 0)

                    # Determine start index based on echo parameter
                    # If echo=True, include prompt tokens (start at 0)
                    # If echo=False, exclude prompt tokens (start at ctx_len)
                    start_idx = 0 if data.echo else ctx_len

                    # Validate array lengths for consistency
                    if not (
                        len(tokens)
                        == len(token_logprobs)
                        == len(top_logprobs)
                        == len(offsets)
                    ):
                        logger.warning(
                            f"Logprob array length mismatch for request {request.state.id}. "
                            f"Tokens: {len(tokens)}, Logprobs: {len(token_logprobs)}, "
                            f"Top_logprobs: {len(top_logprobs)}, Offsets: {len(offsets)}"
                        )

                    # Store both list-based (preferred) and dict-based (compatibility)
                    generation["tokens"] = tokens[start_idx:]
                    generation["token_logprobs"] = token_logprobs[start_idx:]
                    generation["logprobs"] = top_logprobs[start_idx:]
                    generation["offset"] = offsets[start_idx:]

                    # Keep dict-based format for backward compatibility
                    generation["token_probs"] = OrderedDict(
                        zip(tokens[start_idx:], token_logprobs[start_idx:], strict=False)
                    )

        # Handle echo parameter
        if data.echo:
            for idx, generation in enumerate(generations):
                # Get the corresponding prompt for this generation
                prompt = prompts[idx] if idx < len(prompts) else prompts[-1]

                # Handle token IDs by decoding them first if needed
                if isinstance(prompt, list):
                    if all(isinstance(item, int) for item in prompt):
                        # Decode token IDs to text
                        prompt_text = model.container.decode_tokens(prompt)
                    else:
                        # Convert list to string (fallback)
                        prompt_text = str(prompt)
                else:
                    prompt_text = prompt

                generation["text"] = f"{prompt_text}{generation.get('text', '')}"

        response = _create_response(request.state.id, generations, model_path.name)

        logger.info(f"Finished completion request {request.state.id}")

        return response
    except Exception as exc:
        error_message = handle_request_error(
            f"Completion {request.state.id} aborted. Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc


async def generate_prompt_logprobs(
    data: CompletionRequest, request: Request, model_path: pathlib.Path
) -> CompletionResponse:
    """Generate log probabilities for the prompt via the completions endpoint."""

    try:
        # Validate logprobs parameter range (should be 1-5)
        if data.logprobs and not 1 <= data.logprobs <= 5:
            error_message = handle_request_error(
                "logprobs must be between 1 and 5.",
                exc_info=False,
            ).error.message
            raise HTTPException(status.HTTP_400_BAD_REQUEST, error_message)

        # Create a sampler request with parameters for logprob calculation
        # First get the model dump and exclude fields we don't want
        data_dict = data.model_dump(exclude={"prompt", "stream", "echo"})

        # Override specific parameters for logprob calculation
        data_dict.update(
            {
                "max_tokens": 0,
                "repetition_penalty": 1.0,  # No penalties for accurate logprobs
                "temperature": 0.0,  # Deterministic for scoring
                "top_p": 1.0,
            }
        )

        # Create the sampler request
        sampler_params = BaseSamplerRequest(**data_dict)

        generation = model.container.compute_sequence_logprobs(
            prompt=data.prompt,
            params=sampler_params,
        )

        tokens = generation.get("prompt_token_strings", []) or []
        token_logprobs = generation.get("prompt_token_logprobs", []) or []
        top_logprobs = generation.get("top_logprobs", []) or []
        offsets = generation.get("offset", []) or []

        # ------------------------------------------------------------------
        # Make every field JSON-serialisable
        # ------------------------------------------------------------------

        # tokens: ensure strings
        tokens = [str(t) for t in tokens]

        # token-level log-probs: float → finite float or None
        token_logprobs = [
            None if lp is None or not math.isfinite(lp) else float(lp)
            for lp in token_logprobs
        ]

        # top-k dicts: keys to str, values to finite float or None
        top_logprobs = [
            (
                {
                    str(k): None if not math.isfinite(v) else float(v)
                    for k, v in lp.items()
                }
                if lp is not None
                else None
            )
            for lp in top_logprobs
        ]

        # offsets: plain ints
        offsets = [int(o) for o in offsets]

        # Create an ordered dictionary from tokens and logprobs
        token_probs = OrderedDict(zip(tokens, token_logprobs, strict=False))

        # Ensure we have a serialisable text version of the prompt (for echo=True)
        if data.echo:
            if isinstance(data.prompt, list):
                if data.prompt and isinstance(data.prompt[0], int):  # [tok, tok, …]
                    prompt_text = model.container.decode_tokens(data.prompt)
                elif data.prompt and isinstance(data.prompt[0], list):  # [[tok, …]]
                    prompt_text = "\n".join(
                        model.container.decode_tokens(ids) for ids in data.prompt
                    )
                else:
                    prompt_text = str(data.prompt)
            else:
                prompt_text = data.prompt  # already a string
        else:
            prompt_text = ""

        generation_dict = {
            "text": prompt_text,
            "prompt_tokens": len(tokens),
            "generated_tokens": 0,
            "token_probs": token_probs,
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "logprobs": top_logprobs,
            "offset": offsets,
            "finish_reason": "stop",
        }

        return _create_response(request.state.id, generation_dict, model_path.name)
    except HTTPException:
        raise
    except Exception as exc:
        error_message = handle_request_error(
            (
                f"Logprob calculation {request.state.id} failed. "
                "Please check the server console."
            ),
            exc_info=False,
        ).error.message
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, error_message) from exc
