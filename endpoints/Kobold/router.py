from sys import maxsize
from fastapi import APIRouter, Depends, Request
from sse_starlette import EventSourceResponse

from common import model
from common.auth import check_api_key
from common.model import check_model_container
from common.utils import unwrap
from endpoints.Kobold.types.generation import (
    AbortRequest,
    CheckGenerateRequest,
    GenerateRequest,
    GenerateResponse,
)
from endpoints.Kobold.types.token import TokenCountRequest, TokenCountResponse
from endpoints.Kobold.utils.generation import (
    abort_generation,
    generation_status,
    get_generation,
    stream_generation,
)
from endpoints.core.utils.model import get_current_model


api_name = "KoboldAI"
router = APIRouter(prefix="/api")
urls = {
    "Generation": "http://{host}:{port}/api/v1/generate",
    "Streaming": "http://{host}:{port}/api/extra/generate/stream",
}

kai_router = APIRouter()
extra_kai_router = APIRouter()


def setup():
    router.include_router(kai_router, prefix="/v1")
    router.include_router(kai_router, prefix="/latest", include_in_schema=False)
    router.include_router(extra_kai_router, prefix="/extra")

    return router


@kai_router.post(
    "/generate",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def generate(request: Request, data: GenerateRequest) -> GenerateResponse:
    response = await get_generation(data, request)

    return response


@extra_kai_router.post(
    "/generate/stream",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def generate_stream(request: Request, data: GenerateRequest) -> GenerateResponse:
    response = EventSourceResponse(stream_generation(data, request), ping=maxsize)

    return response


@extra_kai_router.post(
    "/abort",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def abort_generate(data: AbortRequest):
    response = await abort_generation(data.genkey)

    return response


@extra_kai_router.get(
    "/generate/check",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
@extra_kai_router.post(
    "/generate/check",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def check_generate(data: CheckGenerateRequest) -> GenerateResponse:
    response = await generation_status(data.genkey)

    return response


@kai_router.get(
    "/model", dependencies=[Depends(check_api_key), Depends(check_model_container)]
)
async def current_model():
    """Fetches the current model and who owns it."""

    current_model_card = get_current_model()
    return {"result": f"{current_model_card.owned_by}/{current_model_card.id}"}


@extra_kai_router.post(
    "/tokencount",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def get_tokencount(data: TokenCountRequest):
    raw_tokens = model.container.encode_tokens(data.prompt)
    tokens = unwrap(raw_tokens, [])
    return TokenCountResponse(value=len(tokens), ids=tokens)


@kai_router.get("/info/version")
async def get_version():
    """Impersonate KAI United."""

    return {"result": "1.2.5"}


@extra_kai_router.get("/version")
async def get_extra_version():
    """Impersonate Koboldcpp."""

    return {"result": "KoboldCpp", "version": "1.61"}
