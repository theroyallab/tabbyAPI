import asyncio
from asyncio import CancelledError, InvalidStateError

from fastapi import APIRouter, Depends, Request
from sse_starlette import EventSourceResponse

from common import model
from common.auth import check_api_key
from common.logger import xlogger
from common.model import check_model_container
from common.networking import DisconnectHandler, get_sse_ping_interval
from common.tabby_config import config
from endpoints.Anthropic.errors import AnthropicRoute, request_error
from endpoints.Anthropic.types.messages import (
    CountTokensRequest,
    CountTokensResponse,
    MessagesRequest,
    MessagesResponse,
)
from endpoints.Anthropic.utils.convert import convert_messages_request
from endpoints.Anthropic.utils.messages import convert_response, count_tokens
from endpoints.Anthropic.utils.stream import stream_generate_message
from endpoints.OAI.utils.chat_completion import apply_chat_template, generate_chat_completion
from endpoints.OAI.utils.common_ import load_inline_model


api_name = "Anthropic"
router = APIRouter(route_class=AnthropicRoute)
urls = {
    "Messages": "http://{host}:{port}/v1/messages",
    "Token counting": "http://{host}:{port}/v1/messages/count_tokens",
}

# Block when model is still loading while second inline load request comes in
load_lock: asyncio.Lock = asyncio.Lock()


def setup():
    return router


async def _resolve_model(model_name: str | None, request: Request):
    """Load an inline model if one was named and return the model directory."""

    async with load_lock:
        if model_name:
            await load_inline_model(model_name, request)
        else:
            await check_model_container()

        return model.container.model_dir


def _check_prompt_template():
    """Reject the request if the loaded model has no prompt template."""

    if model.container.prompt_template is None:
        raise request_error(
            422, "The Anthropic API is disabled because a prompt template is not set."
        )


# Messages endpoint
@router.post(
    "/v1/messages",
    dependencies=[Depends(check_api_key)],
)
async def messages_request(request: Request, data: MessagesRequest) -> MessagesResponse:
    """Generates a message from a conversation."""

    raw_json = await request.json()
    xlogger.debug("[ENDPOINT] /v1/messages", {"raw": raw_json})

    if data.stream and config.developer.disable_request_streaming:
        # Returning a non-streaming body to a client that asked for SSE would
        # fail in the client's parser rather than say what went wrong
        raise request_error(
            400, "Streaming is disabled on this server (developer.disable_request_streaming)."
        )

    model_path = await _resolve_model(data.model, request)
    _check_prompt_template()

    converted = convert_messages_request(data)
    prompt, mm_embeddings = await apply_chat_template(converted)

    try:
        disconnect_handler = DisconnectHandler(request, "/v1/messages")
        await disconnect_handler.poll()

        if data.stream:
            # Checked before the response commits HTTP 200, and reused as the
            # prompt token count message_start has to carry up front
            input_tokens = model.check_context_length(prompt, converted, mm_embeddings)

            return EventSourceResponse(
                stream_generate_message(
                    prompt,
                    mm_embeddings,
                    data,
                    converted,
                    request,
                    model_path,
                    disconnect_handler,
                    input_tokens,
                ),
                ping=get_sse_ping_interval(),
            )

        completion = await generate_chat_completion(
            prompt, mm_embeddings, converted, request, model_path, disconnect_handler
        )

        return convert_response(completion, data, model_path.name)

    except (CancelledError, InvalidStateError) as ex:
        raise request_error(422, "/v1/messages request cancelled by user.") from ex


# Token counting endpoint
@router.post(
    "/v1/messages/count_tokens",
    dependencies=[Depends(check_api_key)],
)
async def count_tokens_request(request: Request, data: CountTokensRequest) -> CountTokensResponse:
    """Counts the tokens an equivalent Messages request would consume."""

    raw_json = await request.json()
    xlogger.debug("[ENDPOINT] /v1/messages/count_tokens", {"raw": raw_json})

    await _resolve_model(data.model, request)
    _check_prompt_template()

    return await count_tokens(data)
