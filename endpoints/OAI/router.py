import asyncio
from asyncio import CancelledError, InvalidStateError

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette import EventSourceResponse

from common import model, tools_config
from common.auth import check_api_key
from common.debug_requests import (
    log_chat_completion_request,
    write_chat_completion_prompt_log,
)
from common.model import check_embeddings_container, check_model_container
from common.networking import (
    get_sse_ping_interval,
    handle_request_error,
    request_tag,
    DisconnectHandler,
    run_with_request_disconnect,
)
from common.tabby_config import config
from common.logger import xlogger

# Hard ceilings by value: constants that never change across reloads. The
# working set of options is read through the module so that a hot reload,
# which replaces the object, is visible here too.
from common.tools_config import (
    HARD_MAX_PAGE_CHARS,
    HARD_MAX_RESULT_CHARS,
    HARD_MAX_RESULTS,
)
from endpoints.OAI.types.completion import CompletionRequest, CompletionResponse
from endpoints.OAI.types.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from endpoints.OAI.types.embedding import EmbeddingsRequest, EmbeddingsResponse
from endpoints.OAI.types.web_tools import (
    FetchURLRequest,
    FetchURLResponse,
    WebSearchRequest,
    WebSearchResponse,
)
from endpoints.OAI.utils.common_ import load_inline_model
from endpoints.OAI.utils.chat_completion import (
    apply_chat_template,
    generate_chat_completion,
    stream_generate_chat_completion,
)
from endpoints.OAI.utils.completion import (
    generate_completion,
    stream_generate_completion,
)
from endpoints.OAI.utils.embeddings import get_embeddings
from endpoints.OAI.utils.web_tools import fetch_page, searxng_search


api_name = "OAI"
router = APIRouter()
urls = {
    "Completions": "http://{host}:{port}/v1/completions",
    "Chat completions": "http://{host}:{port}/v1/chat/completions",
}

# Block when model is still loading while second inline load request comes in
load_lock: asyncio.Lock = asyncio.Lock()


def setup():
    return router


# Completions endpoint
@router.post(
    "/v1/completions",
    dependencies=[Depends(check_api_key)],
)
async def completion_request(request: Request, data: CompletionRequest) -> CompletionResponse:
    """
    Generates a completion from a prompt.

    If stream = true, this returns an SSE stream.
    """

    raw_json = await request.json()
    xlogger.debug("[ENDPOINT] /v1/completions", {"raw": raw_json})

    async with load_lock:
        if data.model:
            await load_inline_model(data.model, request)
        await check_model_container()
        model_path = model.container.model_dir

    # Prepare raw prompt (will be str or list[str])
    prompt = data.prompt

    # Any JSON object, no particular schema. "json_object" is OpenAI's name for it
    if data.response_format.type in ("json_object", "json"):
        data.json_schema = {"type": "object"}

    # Also accept specific schema from response_format
    if data.response_format.type == "json_schema":
        data.json_schema = data.response_format.json_schema

    try:
        disconnect_handler = DisconnectHandler(request, f"{request_tag(request)} completions")
        await disconnect_handler.poll()

        if data.stream and not config.developer.disable_request_streaming:
            model.check_context_length(prompt, data)
            return EventSourceResponse(
                stream_generate_completion(prompt, data, request, model_path, disconnect_handler),
                ping=get_sse_ping_interval(),
            )
        else:
            response = await generate_completion(
                prompt, data, request, model_path, disconnect_handler
            )
            return response

    except (CancelledError, InvalidStateError) as ex:
        raise HTTPException(422, "/v1/completions request cancelled by user.") from ex


# Chat completions endpoint
@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(check_api_key), Depends(log_chat_completion_request)],
)
async def chat_completion_request(
    request: Request, data: ChatCompletionRequest
) -> ChatCompletionResponse:
    """
    Generates a chat completion from a prompt.

    If stream = true, this returns an SSE stream.
    """

    raw_json = await request.json()
    xlogger.debug("[ENDPOINT] /v1/chat/completions", {"raw": raw_json})

    async with load_lock:
        if data.model:
            await load_inline_model(data.model, request)
        await check_model_container()
        model_path = model.container.model_dir

    # Prepare raw prompt
    if model.container.prompt_template is None:
        error_message = handle_request_error(
            "Chat completions are disabled because a prompt template is not set.",
            exc_info=False,
        ).error.message
        raise HTTPException(422, error_message)
    prompt, mm_embeddings = await apply_chat_template(data)
    await write_chat_completion_prompt_log(request, prompt)

    # Any JSON object, no particular schema. "json_object" is OpenAI's name for it
    if data.response_format.type in ("json_object", "json"):
        data.json_schema = {"type": "object"}

    # Also accept specific schema from response_format
    if data.response_format.type == "json_schema":
        data.json_schema = data.response_format.json_schema

    try:
        disconnect_handler = DisconnectHandler(request, f"{request_tag(request)} chat/completions")
        await disconnect_handler.poll()

        if data.stream and not config.developer.disable_request_streaming:
            model.check_context_length(prompt, data, mm_embeddings)
            return EventSourceResponse(
                stream_generate_chat_completion(
                    prompt, mm_embeddings, data, request, model_path, disconnect_handler
                ),
                ping=get_sse_ping_interval(),
            )
        else:
            response = await generate_chat_completion(
                prompt, mm_embeddings, data, request, model_path, disconnect_handler
            )
            return response

    except (CancelledError, InvalidStateError) as ex:
        raise HTTPException(422, "/v1/chat/completions request cancelled by user.") from ex


# Apply template endpoint (llama-server compatible)
@router.post("/apply-template", dependencies=[Depends(check_api_key)])
@router.post("/v1/apply-template", dependencies=[Depends(check_api_key)])
async def apply_template_request(data: ChatCompletionRequest) -> dict:
    """
    Renders the chat template for the given messages without generating and
    returns the resulting prompt. Clients use this to probe the template, e.g.
    whether it reacts to a thinking toggle.
    """

    await check_model_container()

    if model.container.prompt_template is None:
        error_message = handle_request_error(
            "Cannot apply a template because a prompt template is not set.",
            exc_info=False,
        ).error.message
        raise HTTPException(422, error_message)

    prompt, _ = await apply_chat_template(data)
    return {"prompt": prompt}


# Embeddings endpoint
@router.post(
    "/v1/embeddings",
    dependencies=[Depends(check_api_key), Depends(check_embeddings_container)],
)
async def embeddings(request: Request, data: EmbeddingsRequest) -> EmbeddingsResponse:
    embeddings_task = asyncio.create_task(get_embeddings(data, request))
    response = await run_with_request_disconnect(
        request,
        embeddings_task,
        f"{request_tag(request)} embeddings cancelled by client",
    )

    return response


# Built-in web tools
@router.post("/v1/tools/web_search", dependencies=[Depends(check_api_key)])
async def web_search_request(data: WebSearchRequest) -> WebSearchResponse:
    """Search the web with the configured SearXNG instance."""

    return await searxng_search(data.query, data.max_results, data.max_result_chars)


@router.post("/v1/tools/fetch_url", dependencies=[Depends(check_api_key)])
async def fetch_url_request(data: FetchURLRequest) -> FetchURLResponse:
    """Fetch a web page by URL and return its text."""

    return await fetch_page(data.url, data.max_page_chars)


@router.get("/v1/tools/status", dependencies=[Depends(check_api_key)])
async def tools_status() -> dict:
    """What the built-in web tools are configured to do. Never includes the
    SearXNG URL or token: only booleans and the limits clients may lower."""

    # One reference per call: a watcher reload mid-request cannot mix
    # old and new options into a single status response.
    cfg = tools_config.TOOLS_CONFIG
    searxng = cfg.searxng
    fetch = cfg.fetch_url
    return {
        "web_search": {
            "enabled": searxng.enabled and bool(searxng.url),
            "max_results": searxng.max_results,
            "max_result_chars": searxng.max_result_chars,
        },
        "fetch_url": {
            "enabled": fetch.enabled,
            "max_page_chars": fetch.max_page_chars,
            "allow_private_urls": fetch.allow_private_urls,
        },
        "limits": {
            "max_results": HARD_MAX_RESULTS,
            "max_result_chars": HARD_MAX_RESULT_CHARS,
            "max_page_chars": HARD_MAX_PAGE_CHARS,
        },
    }
