import asyncio
from asyncio import CancelledError
from fastapi import HTTPException, Request
from loguru import logger
from sse_starlette.event import ServerSentEvent

from common import model
from common.networking import (
    get_generator_error,
    handle_request_disconnect,
    handle_request_error,
    request_disconnect_loop,
)
from common.utils import unwrap
from endpoints.Kobold.types.generation import (
    AbortResponse,
    GenerateRequest,
    GenerateResponse,
    GenerateResponseResult,
    StreamGenerateChunk,
)


generation_cache = {}


async def override_request_id(request: Request, data: GenerateRequest):
    """Overrides the request ID with a KAI genkey if present."""

    if data.genkey:
        request.state.id = data.genkey


def _create_response(text: str):
    results = [GenerateResponseResult(text=text)]
    return GenerateResponse(results=results)


def _create_stream_chunk(text: str):
    return StreamGenerateChunk(token=text)


async def _stream_collector(data: GenerateRequest, request: Request):
    """Common async generator for generation streams."""

    abort_event = asyncio.Event()
    disconnect_task = asyncio.create_task(request_disconnect_loop(request))

    # Create a new entry in the cache
    generation_cache[data.genkey] = {"abort": abort_event, "text": ""}

    try:
        logger.info(f"Received Kobold generation request {data.genkey}")

        generator = model.container.stream_generate(
            request_id=data.genkey,
            prompt=data.prompt,
            params=data,
            abort_event=abort_event,
        )

        async for generation in generator:
            if disconnect_task.done():
                raise CancelledError()

            text = generation.get("text")

            # Update the generation cache with the new chunk
            if text:
                generation_cache[data.genkey]["text"] += text
                yield text

            if "finish_reason" in generation:
                logger.info(f"Finished streaming Kobold request {data.genkey}")
                break
    except CancelledError:
        # If the request disconnects, break out
        if not abort_event.is_set():
            abort_event.set()
            handle_request_disconnect(
                f"Kobold generation {data.genkey} cancelled by user."
            )
    finally:
        # Cleanup the cache
        del generation_cache[data.genkey]


async def stream_generation(data: GenerateRequest, request: Request):
    """Wrapper for stream generations."""

    # If the genkey doesn't exist, set it to the request ID
    if not data.genkey:
        data.genkey = request.state.id

    try:
        async for chunk in _stream_collector(data, request):
            response = _create_stream_chunk(chunk)
            yield ServerSentEvent(
                event="message", data=response.model_dump_json(), sep="\n"
            )
    except Exception:
        yield get_generator_error(
            f"Kobold generation {data.genkey} aborted. Please check the server console."
        )


async def get_generation(data: GenerateRequest, request: Request):
    """Wrapper to get a static generation."""

    # If the genkey doesn't exist, set it to the request ID
    if not data.genkey:
        data.genkey = request.state.id

    try:
        full_text = ""
        async for chunk in _stream_collector(data, request):
            full_text += chunk

        response = _create_response(full_text)
        return response
    except Exception as exc:
        error_message = handle_request_error(
            f"Completion {request.state.id} aborted. Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc


async def abort_generation(genkey: str):
    """Aborts a generation from the cache."""

    abort_event = unwrap(generation_cache.get(genkey), {}).get("abort")
    if abort_event:
        abort_event.set()
        handle_request_disconnect(f"Kobold generation {genkey} cancelled by user.")

    return AbortResponse(success=True)


async def generation_status(genkey: str):
    """Fetches the status of a generation from the cache."""

    current_text = unwrap(generation_cache.get(genkey), {}).get("text")
    if current_text:
        response = _create_response(current_text)
    else:
        response = GenerateResponse()

    return response
