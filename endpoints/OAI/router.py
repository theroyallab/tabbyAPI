import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette import EventSourceResponse
from sys import maxsize

from common import config, model
from common.auth import check_api_key
from common.model import check_embeddings_container, check_model_container
from common.networking import handle_request_error, run_with_request_disconnect
from common.utils import unwrap
from endpoints.OAI.types.completion import CompletionRequest, CompletionResponse
from endpoints.OAI.types.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from endpoints.OAI.types.embedding import EmbeddingsRequest, EmbeddingsResponse
from endpoints.OAI.utils.chat_completion import (
    format_prompt_with_template,
    generate_chat_completion,
    stream_generate_chat_completion,
)
from endpoints.OAI.utils.completion import (
    generate_completion,
    stream_generate_completion,
)
from endpoints.OAI.utils.embeddings import get_embeddings


api_name = "OAI"
router = APIRouter()
urls = {
    "Completions": "http://{host}:{port}/v1/completions",
    "Chat completions": "http://{host}:{port}/v1/chat/completions",
}


def setup():
    return router


# Completions endpoint
@router.post(
    "/v1/completions",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def completion_request(
    request: Request, data: CompletionRequest
) -> CompletionResponse:
    """
    Generates a completion from a prompt.

    If stream = true, this returns an SSE stream.
    """

    model_path = model.container.model_dir

    if isinstance(data.prompt, list):
        data.prompt = "\n".join(data.prompt)

    disable_request_streaming = unwrap(
        config.developer_config().get("disable_request_streaming"), False
    )

    # Set an empty JSON schema if the request wants a JSON response
    if data.response_format.type == "json":
        data.json_schema = {"type": "object"}

    if data.stream and not disable_request_streaming:
        return EventSourceResponse(
            stream_generate_completion(data, request, model_path),
            ping=maxsize,
        )
    else:
        generate_task = asyncio.create_task(
            generate_completion(data, request, model_path)
        )

        response = await run_with_request_disconnect(
            request,
            generate_task,
            disconnect_message=f"Completion {request.state.id} cancelled by user.",
        )
        return response


# Chat completions endpoint
@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def chat_completion_request(
    request: Request, data: ChatCompletionRequest
) -> ChatCompletionResponse:
    """
    Generates a chat completion from a prompt.

    If stream = true, this returns an SSE stream.
    """

    if model.container.prompt_template is None:
        error_message = handle_request_error(
            "Chat completions are disabled because a prompt template is not set.",
            exc_info=False,
        ).error.message

        raise HTTPException(422, error_message)

    model_path = model.container.model_dir

    if isinstance(data.messages, str):
        prompt = data.messages
    else:
        prompt = await format_prompt_with_template(data)

    # Set an empty JSON schema if the request wants a JSON response
    if data.response_format.type == "json":
        data.json_schema = {"type": "object"}

    disable_request_streaming = unwrap(
        config.developer_config().get("disable_request_streaming"), False
    )

    if data.stream and not disable_request_streaming:
        return EventSourceResponse(
            stream_generate_chat_completion(prompt, data, request, model_path),
            ping=maxsize,
        )
    else:
        generate_task = asyncio.create_task(
            generate_chat_completion(prompt, data, request, model_path)
        )

        response = await run_with_request_disconnect(
            request,
            generate_task,
            disconnect_message=f"Chat completion {request.state.id} cancelled by user.",
        )
        return response


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
        f"Embeddings request {request.state.id} cancelled by user.",
    )

    return response
