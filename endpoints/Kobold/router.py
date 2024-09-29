from sys import maxsize
from fastapi import APIRouter, Depends, Request
from sse_starlette import EventSourceResponse

from common import model
from common.auth import check_api_key
from common.model import check_model_container
from common.utils import unwrap
from endpoints.core.types.tags import Tags
from endpoints.core.utils.model import get_current_model
from endpoints.Kobold.types.generation import (
    AbortRequest,
    AbortResponse,
    CheckGenerateRequest,
    GenerateRequest,
    GenerateResponse,
)
from endpoints.Kobold.types.model import CurrentModelResponse, MaxLengthResponse
from endpoints.Kobold.types.token import TokenCountRequest, TokenCountResponse
from endpoints.Kobold.utils.generation import (
    abort_generation,
    generation_status,
    get_generation,
    stream_generation,
)


api_name = "KoboldAI"
router = APIRouter(prefix="/api", tags=[Tags.Kobold])
urls = {
    "Generation": "http://{host}:{port}/api/v1/generate",
    "Streaming": "http://{host}:{port}/api/extra/generate/stream",
}

kai_router = APIRouter(tags=[Tags.Kobold])
extra_kai_router = APIRouter(tags=[Tags.Kobold])


def setup():
    router.include_router(kai_router, prefix="/v1")
    router.include_router(kai_router, prefix="/latest", include_in_schema=False)
    router.include_router(extra_kai_router, prefix="/extra")

    return router


@kai_router.post(
    "/generate",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
    tags=[Tags.Kobold],
)
async def generate(request: Request, data: GenerateRequest) -> GenerateResponse:
    """Generate a response to a prompt."""
    response = await get_generation(data, request)

    return response


@extra_kai_router.post(
    "/generate/stream",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
    tags=[Tags.Kobold],
)
async def generate_stream(request: Request, data: GenerateRequest) -> GenerateResponse:
    """Stream the chat response to a prompt."""
    response = EventSourceResponse(stream_generation(data, request), ping=maxsize)

    return response


@extra_kai_router.post(
    "/abort",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
    tags=[Tags.Kobold],
)
async def abort_generate(data: AbortRequest) -> AbortResponse:
    """Aborts a generation from the cache."""
    response = await abort_generation(data.genkey)

    return response


@extra_kai_router.get(
    "/generate/check",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
    tags=[Tags.Kobold],
)
@extra_kai_router.post(
    "/generate/check",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
    tags=[Tags.Kobold],
)
async def check_generate(data: CheckGenerateRequest) -> GenerateResponse:
    """Fetches the status of a generation from the cache."""
    response = await generation_status(data.genkey)

    return response


@kai_router.get(
    "/model",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
    tags=[Tags.Kobold],
)
async def current_model() -> CurrentModelResponse:
    """Fetches the current model and who owns it."""

    current_model_card = get_current_model()
    return {"result": f"{current_model_card.owned_by}/{current_model_card.id}"}


@extra_kai_router.post(
    "/tokencount",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
    tags=[Tags.Kobold],
)
async def get_tokencount(data: TokenCountRequest) -> TokenCountResponse:
    """Get the number of tokens in a given prompt."""
    raw_tokens = model.container.encode_tokens(data.prompt)
    tokens = unwrap(raw_tokens, [])
    return TokenCountResponse(value=len(tokens), ids=tokens)


@kai_router.get(
    "/config/max_length",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
    tags=[Tags.Kobold],
)
@kai_router.get(
    "/config/max_context_length",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
    tags=[Tags.Kobold],
)
@extra_kai_router.get(
    "/true_max_context_length",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
    tags=[Tags.Kobold],
)
async def get_max_length() -> MaxLengthResponse:
    """Fetches the max length of the model."""

    max_length = model.container.get_model_parameters().get("max_seq_len")
    return {"value": max_length}


@kai_router.get("/info/version", tags=[Tags.Kobold])
async def get_version():
    """Impersonate KAI United."""

    return {"result": "1.2.5"}


@extra_kai_router.get("/version", tags=[Tags.Kobold])
async def get_extra_version():
    """Impersonate Koboldcpp."""

    return {"result": "KoboldCpp", "version": "1.71"}


@kai_router.get("/config/soft_prompts_list", tags=[Tags.Kobold])
async def get_available_softprompts():
    """Used for KAI compliance."""

    return {"values": []}


@kai_router.get("/config/soft_prompt", tags=[Tags.Kobold])
async def get_current_softprompt():
    """Used for KAI compliance."""

    return {"value": ""}


@kai_router.put("/config/soft_prompt", tags=[Tags.Kobold])
async def set_current_softprompt():
    """Used for KAI compliance."""

    return {}
