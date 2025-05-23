import asyncio
import time
import traceback
from contextlib import nullcontext
from sys import maxsize

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

from common import model
from common.auth import check_api_key
from common.logger import logger
from common.model import check_embeddings_container, check_model_container
from common.networking import handle_request_error, run_with_request_disconnect
from common.tabby_config import config
from common.utils import unwrap
from common.metrics import get_tracer, record_logprob_metrics

from endpoints.OAI.types.completion import CompletionRequest, CompletionResponse
from endpoints.OAI.types.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from endpoints.OAI.types.embedding import EmbeddingsRequest, EmbeddingsResponse
from endpoints.OAI.types.logprob import LogProbRequest, LogProbResponse

from endpoints.OAI.utils.chat_completion import (
    apply_chat_template,
    generate_chat_completion,
    stream_generate_chat_completion,
)
from endpoints.OAI.utils.completion import (
    generate_completion,
    load_inline_model,
    stream_generate_completion,
    generate_prompt_logprobs,
)
from endpoints.OAI.utils.embeddings import get_embeddings
from endpoints.OAI.utils.logprob import generate_logprobs, stream_generate_logprobs


api_name = "OAI"
router = APIRouter()
urls = {
    "Completions": "http://{host}:{port}/v1/completions",
    "Chat completions": "http://{host}:{port}/v1/chat/completions",
    "LogProb": "http://{host}:{port}/v1/logprob",
}


def _logprobs_requested(data) -> bool:
    """Check if any form of logprobs were requested in the request data."""
    # Check if logprobs is a positive integer (not None, False, or 0)
    logprobs_enabled = unwrap(data.logprobs, False)
    if isinstance(logprobs_enabled, bool):
        logprobs_requested = logprobs_enabled
    else:
        # If it's a number, check if it's > 0
        logprobs_requested = logprobs_enabled and logprobs_enabled > 0
    
    # Check if top_logprobs is a positive integer
    top_logprobs_requested = unwrap(data.top_logprobs, 0) > 0
    
    return logprobs_requested or top_logprobs_requested


def setup():
    return router


# --------------------------------------------------------------------------- #
# Completions (legacy)                                                       #
# --------------------------------------------------------------------------- #
@router.post(
    "/v1/completions",
    dependencies=[Depends(check_api_key)],
    response_model=CompletionResponse,
)
async def completion_request(
    request: Request, data: CompletionRequest, response: Response
) -> CompletionResponse:
    # Add OpenAI-specific headers
    response.headers["OpenAI-Organization"] = "tabby"
    response.headers["OpenAI-Processing-MS"] = str(int(time.time() * 1000))
    response.headers["OpenAI-Version"] = "2025-05-01"
    """
    Generates a completion from a prompt.

    If stream = true, an SSE stream is returned.
    """
    # -------------------------------------------------------------------
    # Echo parameter handling
    # -------------------------------------------------------------------
    # The backend does not natively return the prompt tokens when
    # ``echo`` is requested with normal generation.  Instead of raising an
    # error, gracefully ignore the flag for these requests.  Echoing is
    # still honoured for logprob-only mode and handled in the generation
    # utilities.
    # Validate logprobs parameter range (should be 1-5 if provided as integer)
    if (data.logprobs is not None and 
        not isinstance(data.logprobs, bool) and 
        not 1 <= data.logprobs <= 5):
        error_message = handle_request_error(
            "logprobs must be between 1 and 5.",
            exc_info=False,
        ).error.message
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message,
        )

    # Reject logprobs when streaming is requested since the implementation
    # does not currently populate logprob information in stream chunks.
    if data.stream and _logprobs_requested(data):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming with logprobs is not supported.",
        )

    # Ignore the suffix parameter until insertion is supported
    if data.suffix:
        logger.debug(
            "Suffix parameter provided but insertion is not supported; ignoring"
        )

    if data.model:
        inline_load_task = asyncio.create_task(load_inline_model(data.model, request))
        await run_with_request_disconnect(
            request,
            inline_load_task,
            disconnect_message=f"Model switch for generation {request.state.id} cancelled by user.",
        )
    else:
        await check_model_container()

    model_path = model.container.model_dir

    if isinstance(data.prompt, list):
        if data.prompt and isinstance(data.prompt[0], int):
            # A flat list of token IDs.
            data.prompt = model.container.decode_tokens(data.prompt)
        elif data.prompt and isinstance(data.prompt[0], list):
            # Accept a nested list of token IDs. If a single element is
            # provided, unwrap it; otherwise decode each sub-list and join them
            # together with newlines.
            if (
                len(data.prompt) == 1
                and all(isinstance(tok, int) for tok in data.prompt[0])
            ):
                data.prompt = model.container.decode_tokens(data.prompt[0])
            else:
                decoded_prompts = [
                    model.container.decode_tokens(elem)
                    for elem in data.prompt
                    if isinstance(elem, list)
                ]
                data.prompt = "\n".join(decoded_prompts)
        else:
            data.prompt = "\n".join(data.prompt)

    if data.logit_bias and not model.container.supports_logit_bias():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="logit_bias is currently not supported.",
        )

    # JSON mode ⇒ empty schema placeholder
    if data.response_format.type == "json":
        data.json_schema = {"type": "object"}

    if data.stream and not config.developer.disable_request_streaming:
        return EventSourceResponse(
            stream_generate_completion(data, request, model_path),
            ping=maxsize,
        )

    # OpenAI compatibility: handle logprob-only requests via completions
    if (unwrap(data.max_tokens, 0) == 0) and (unwrap(data.logprobs, 0) > 0):
        if not config.developer.enable_logprob:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="The logprob feature is not enabled on this server.",
            )

        if not model.container.supports_logprob_extraction():
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="The current model does not support logprob extraction.",
            )

        generate_task = asyncio.create_task(
            generate_prompt_logprobs(data, request, model_path)
        )
    else:
        generate_task = asyncio.create_task(
            generate_completion(data, request, model_path)
        )
    return await run_with_request_disconnect(
        request,
        generate_task,
        disconnect_message=f"Completion {request.state.id} cancelled by user.",
    )


# --------------------------------------------------------------------------- #
# Chat completions                                                            #
# --------------------------------------------------------------------------- #
@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(check_api_key)],
    response_model=ChatCompletionResponse,
)
async def chat_completion_request(
    request: Request, data: ChatCompletionRequest
):
    """
    Generates a chat completion from a prompt.

    If stream = true, an SSE stream is returned.
    """
    # -------------------------------------------------------------------
    # Echo parameter handling
    # -------------------------------------------------------------------
    # Chat completions do not currently support the ``echo`` parameter. If a
    # client requests it, return a clear error instead of silently ignoring the
    # flag.
    if data.echo:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Echoing prompt tokens is not supported by this model.",
        )
    
    # Note: Chat completions support logprobs, including streaming logprobs.
    # No rejection needed here unlike the regular completions endpoint.
    
    if data.model:
        await load_inline_model(data.model, request)
    else:
        await check_model_container()

    if model.container.prompt_template is None:
        error_message = handle_request_error(
            "Chat completions are disabled because a prompt template is not set.",
            exc_info=False,
        ).error.message
        raise HTTPException(422, error_message)

    model_path = model.container.model_dir
    prompt, embeddings = await apply_chat_template(data)

    if data.logit_bias and not model.container.supports_logit_bias():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="logit_bias is currently not supported.",
        )

    # JSON mode ⇒ empty schema placeholder
    if data.response_format.type == "json":
        data.json_schema = {"type": "object"}

    if data.stream and not config.developer.disable_request_streaming:
        return EventSourceResponse(
            stream_generate_chat_completion(
                prompt, embeddings, data, request, model_path
            ),
            ping=maxsize,
        )

    generate_task = asyncio.create_task(
        generate_chat_completion(prompt, embeddings, data, request, model_path)
    )
    response = await run_with_request_disconnect(
        request,
        generate_task,
        disconnect_message=f"Chat completion {request.state.id} cancelled by user.",
    )
    
    # Custom serialization to exclude None logprobs when not requested
    response_dict = response.model_dump()
    
    # For each choice, remove logprobs if they are None and weren't explicitly requested
    for choice in response_dict.get("choices", []):
        if choice.get("logprobs") is None:
            # Check if this was a request without logprobs
            if not data.logprobs:
                choice.pop("logprobs", None)
    
    return JSONResponse(content=response_dict)


# --------------------------------------------------------------------------- #
# Embeddings                                                                  #
# --------------------------------------------------------------------------- #
@router.post(
    "/v1/embeddings",
    dependencies=[Depends(check_api_key), Depends(check_embeddings_container)],
)
async def embeddings(request: Request, data: EmbeddingsRequest) -> EmbeddingsResponse:
    embeddings_task = asyncio.create_task(get_embeddings(data, request))
    return await run_with_request_disconnect(
        request,
        embeddings_task,
        f"Embeddings request {request.state.id} cancelled by user.",
    )


# --------------------------------------------------------------------------- #
# Log-probabilities                                                           #
# --------------------------------------------------------------------------- #
@router.post(
    "/v1/logprob",
    dependencies=[Depends(check_api_key)],
    response_model=LogProbResponse,
)
async def logprob_request(
    request: Request, data: LogProbRequest, response: Response
) -> LogProbResponse:
    # Add OpenAI-specific headers
    response.headers["OpenAI-Organization"] = "tabby"
    response.headers["OpenAI-Processing-MS"] = str(int(time.time() * 1000))
    response.headers["OpenAI-Version"] = "2025-05-01"
    """
    Calculates token-level log-probabilities for the supplied prompt.
    """

    # ---------- Feature-flag checks ---------------------------------------- #
    if not config.developer.enable_logprob:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="The logprob endpoint is not enabled on this server.",
        )
    
    if not model.container.supports_logprob_extraction():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="The current model does not support logprob extraction.",
        )

    # ---------- Model preparation ------------------------------------------ #
    if data.model:
        inline_load_task = asyncio.create_task(load_inline_model(data.model, request))
        await run_with_request_disconnect(
            request,
            inline_load_task,
            disconnect_message=f"Model switch for logprob {request.state.id} cancelled by user.",
        )
    else:
        await check_model_container()

    if not model.container.supports_logprob_extraction():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="The current model does not support logprob extraction.",
        )

    # ---------- Configurable timeout & tracing ----------------------------- #
    timeout: float = float(unwrap(config.developer.logprob_timeout_seconds, 60.0))
    tracer = get_tracer()
    span_cm = (
        tracer.start_as_current_span("logprob_request") if tracer else nullcontext()
    )

    # ---------- Metrics timer ---------------------------------------------- #
    start_time = time.monotonic()
    model_path = model.container.model_dir

    # ---------- Execution --------------------------------------------------- #
    try:
        if data.stream and not config.developer.disable_request_streaming:
            return EventSourceResponse(
                stream_generate_logprobs(data, request, model_path),
                ping=maxsize,
            )

        generate_task = asyncio.create_task(
            generate_logprobs(data, request, model_path)
        )

        with span_cm as span:
            response = await asyncio.wait_for(
                run_with_request_disconnect(
                    request,
                    generate_task,
                    disconnect_message=f"LogProb calculation {request.state.id} cancelled by user.",
                ),
                timeout=timeout,
            )

            # ---------- Metrics & tracing --------------------------------- #
            end_time = time.monotonic()
            latency = end_time - start_time
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0

            record_logprob_metrics(prompt_tokens, latency)

            if span is not None:
                span.set_attribute("model", model_path.name)
                span.set_attribute("prompt_tokens", prompt_tokens)
                span.set_attribute("latency_ms", latency * 1000)

            logger.info(
                f"LogProb (ID: {request.state.id}): {prompt_tokens} tokens processed in "
                f"{round(latency * 1000, 2)} ms"
            )

            return response

    # ---------- Timeout guard ---------------------------------------------- #
    except asyncio.TimeoutError:
        logger.error(
            f"LogProb request {request.state.id} timed out after {timeout} seconds"
        )
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=(
                f"LogProb calculation timed out after {timeout} seconds. "
                "Please try with a shorter prompt."
            ),
        )