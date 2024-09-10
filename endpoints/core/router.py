import asyncio
import pathlib
from sys import maxsize
from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette import EventSourceResponse

from common import model, sampling
from common.auth import check_admin_key, check_api_key, get_key_permission
from common.downloader import hf_repo_download
from common.model import check_embeddings_container, check_model_container
from common.networking import handle_request_error, run_with_request_disconnect
from common.tabby_config import config
from common.templating import PromptTemplate, get_all_templates
from common.utils import unwrap
from endpoints.core.types.auth import AuthPermissionResponse
from endpoints.core.types.download import DownloadRequest, DownloadResponse
from endpoints.core.types.lora import LoraList, LoraLoadRequest, LoraLoadResponse
from endpoints.core.types.model import (
    EmbeddingModelLoadRequest,
    ModelCard,
    ModelList,
    ModelLoadRequest,
    ModelLoadResponse,
)
from endpoints.core.types.sampler_overrides import (
    SamplerOverrideListResponse,
    SamplerOverrideSwitchRequest,
)
from endpoints.core.types.template import TemplateList, TemplateSwitchRequest
from endpoints.core.types.token import (
    TokenDecodeRequest,
    TokenDecodeResponse,
    TokenEncodeRequest,
    TokenEncodeResponse,
)
from endpoints.core.utils.lora import get_active_loras, get_lora_list
from endpoints.core.utils.model import (
    get_current_model,
    get_current_model_list,
    get_model_list,
    stream_model_load,
)


router = APIRouter()


# Healthcheck endpoint
@router.get("/health")
async def healthcheck():
    """Get the current service health status"""
    return {"status": "healthy"}


# Model list endpoint
@router.get("/v1/models", dependencies=[Depends(check_api_key)])
@router.get("/v1/model/list", dependencies=[Depends(check_api_key)])
async def list_models(request: Request) -> ModelList:
    """
    Lists all models in the model directory.

    Requires an admin key to see all models.
    """

    model_dir = unwrap(config.model.get("model_dir"), "models")
    model_path = pathlib.Path(model_dir)

    draft_model_dir = config.draft_model.get("draft_model_dir")

    if get_key_permission(request) == "admin":
        models = get_model_list(model_path.resolve(), draft_model_dir)
    else:
        models = await get_current_model_list()

    if unwrap(config.model.get("use_dummy_models"), False):
        models.data.insert(0, ModelCard(id="gpt-3.5-turbo"))

    return models


# Currently loaded model endpoint
@router.get(
    "/v1/model",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def current_model() -> ModelCard:
    """Returns the currently loaded model."""

    return get_current_model()


@router.get("/v1/model/draft/list", dependencies=[Depends(check_api_key)])
async def list_draft_models(request: Request) -> ModelList:
    """
    Lists all draft models in the model directory.

    Requires an admin key to see all draft models.
    """

    if get_key_permission(request) == "admin":
        draft_model_dir = unwrap(config.draft_model.get("draft_model_dir"), "models")
        draft_model_path = pathlib.Path(draft_model_dir)

        models = get_model_list(draft_model_path.resolve())
    else:
        models = await get_current_model_list(model_type="draft")

    return models


# Load model endpoint
@router.post("/v1/model/load", dependencies=[Depends(check_admin_key)])
async def load_model(data: ModelLoadRequest) -> ModelLoadResponse:
    """Loads a model into the model container. This returns an SSE stream."""

    # Verify request parameters
    if not data.name:
        error_message = handle_request_error(
            "A model name was not provided for load.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    model_path = pathlib.Path(unwrap(config.model.get("model_dir"), "models"))
    model_path = model_path / data.name

    draft_model_path = None
    if data.draft:
        if not data.draft.draft_model_name:
            error_message = handle_request_error(
                "Could not find the draft model name for model load.",
                exc_info=False,
            ).error.message

            raise HTTPException(400, error_message)

        draft_model_path = unwrap(config.draft_model.get("draft_model_dir"), "models")

    if not model_path.exists():
        error_message = handle_request_error(
            "Could not find the model path for load. Check model name or config.yml?",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    return EventSourceResponse(
        stream_model_load(data, model_path, draft_model_path), ping=maxsize
    )


# Unload model endpoint
@router.post(
    "/v1/model/unload",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def unload_model():
    """Unloads the currently loaded model."""
    await model.unload_model(skip_wait=True)


@router.post("/v1/download", dependencies=[Depends(check_admin_key)])
async def download_model(request: Request, data: DownloadRequest) -> DownloadResponse:
    """Downloads a model from HuggingFace."""

    try:
        download_task = asyncio.create_task(hf_repo_download(**data.model_dump()))

        # For now, the downloader and request data are 1:1
        download_path = await run_with_request_disconnect(
            request,
            download_task,
            "Download request cancelled by user. Files have been cleaned up.",
        )

        return DownloadResponse(download_path=str(download_path))
    except Exception as exc:
        error_message = handle_request_error(str(exc)).error.message

        raise HTTPException(400, error_message) from exc


# Lora list endpoint
@router.get("/v1/loras", dependencies=[Depends(check_api_key)])
@router.get("/v1/lora/list", dependencies=[Depends(check_api_key)])
async def list_all_loras(request: Request) -> LoraList:
    """
    Lists all LoRAs in the lora directory.

    Requires an admin key to see all LoRAs.
    """

    if get_key_permission(request) == "admin":
        lora_path = pathlib.Path(unwrap(config.lora.get("lora_dir"), "loras"))
        loras = get_lora_list(lora_path.resolve())
    else:
        loras = get_active_loras()

    return loras


# Currently loaded loras endpoint
@router.get(
    "/v1/lora",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def active_loras() -> LoraList:
    """Returns the currently loaded loras."""

    return get_active_loras()


# Load lora endpoint
@router.post(
    "/v1/lora/load",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def load_lora(data: LoraLoadRequest) -> LoraLoadResponse:
    """Loads a LoRA into the model container."""

    if not data.loras:
        error_message = handle_request_error(
            "List of loras to load is not found.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    lora_dir = pathlib.Path(unwrap(config.lora.get("lora_dir"), "loras"))
    if not lora_dir.exists():
        error_message = handle_request_error(
            "A parent lora directory does not exist for load. Check your config.yml?",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    load_result = await model.load_loras(
        lora_dir, **data.model_dump(), skip_wait=data.skip_queue
    )

    return LoraLoadResponse(
        success=unwrap(load_result.get("success"), []),
        failure=unwrap(load_result.get("failure"), []),
    )


# Unload lora endpoint
@router.post(
    "/v1/lora/unload",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def unload_loras():
    """Unloads the currently loaded loras."""

    await model.unload_loras()


@router.get("/v1/model/embedding/list", dependencies=[Depends(check_api_key)])
async def list_embedding_models(request: Request) -> ModelList:
    """
    Lists all embedding models in the model directory.

    Requires an admin key to see all embedding models.
    """

    if get_key_permission(request) == "admin":
        embedding_model_dir = unwrap(
            config.embeddings.get("embedding_model_dir"), "models"
        )
        embedding_model_path = pathlib.Path(embedding_model_dir)

        models = get_model_list(embedding_model_path.resolve())
    else:
        models = await get_current_model_list(model_type="embedding")

    return models


@router.get(
    "/v1/model/embedding",
    dependencies=[Depends(check_api_key), Depends(check_embeddings_container)],
)
async def get_embedding_model() -> ModelCard:
    """Returns the currently loaded embedding model."""
    models = await get_current_model_list(model_type="embedding")

    return models.data[0]


@router.post("/v1/model/embedding/load", dependencies=[Depends(check_admin_key)])
async def load_embedding_model(
    request: Request, data: EmbeddingModelLoadRequest
) -> ModelLoadResponse:
    # Verify request parameters
    if not data.name:
        error_message = handle_request_error(
            "A model name was not provided for load.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    embedding_model_dir = pathlib.Path(
        unwrap(config.embeddings.get("embedding_model_dir"), "models")
    )
    embedding_model_path = embedding_model_dir / data.name

    if not embedding_model_path.exists():
        error_message = handle_request_error(
            "Could not find the embedding model path for load. "
            + "Check model name or config.yml?",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    try:
        load_task = asyncio.create_task(
            model.load_embedding_model(embedding_model_path, **data.model_dump())
        )
        await run_with_request_disconnect(
            request, load_task, "Embedding model load request cancelled by user."
        )
    except Exception as exc:
        error_message = handle_request_error(str(exc)).error.message

        raise HTTPException(400, error_message) from exc

    response = ModelLoadResponse(
        model_type="embedding_model", module=1, modules=1, status="finished"
    )

    return response


@router.post(
    "/v1/model/embedding/unload",
    dependencies=[Depends(check_admin_key), Depends(check_embeddings_container)],
)
async def unload_embedding_model():
    """Unloads the current embedding model."""

    await model.unload_embedding_model()


# Encode tokens endpoint
@router.post(
    "/v1/token/encode",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def encode_tokens(data: TokenEncodeRequest) -> TokenEncodeResponse:
    """Encodes a string or chat completion messages into tokens."""

    if isinstance(data.text, str):
        text = data.text
    else:
        special_tokens_dict = model.container.get_special_tokens(
            unwrap(data.add_bos_token, True)
        )

        template_vars = {
            "messages": data.text,
            "add_generation_prompt": False,
            **special_tokens_dict,
        }

        text, _ = model.container.prompt_template.render(template_vars)

    raw_tokens = model.container.encode_tokens(text, **data.get_params())
    tokens = unwrap(raw_tokens, [])
    response = TokenEncodeResponse(tokens=tokens, length=len(tokens))

    return response


# Decode tokens endpoint
@router.post(
    "/v1/token/decode",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def decode_tokens(data: TokenDecodeRequest) -> TokenDecodeResponse:
    """Decodes tokens into a string."""

    message = model.container.decode_tokens(data.tokens, **data.get_params())
    response = TokenDecodeResponse(text=unwrap(message, ""))

    return response


@router.get("/v1/auth/permission", dependencies=[Depends(check_api_key)])
async def key_permission(request: Request) -> AuthPermissionResponse:
    """
    Gets the access level/permission of a provided key in headers.

    Priority:
    - X-admin-key
    - X-api-key
    - Authorization
    """

    try:
        permission = get_key_permission(request)
        return AuthPermissionResponse(permission=permission)
    except ValueError as exc:
        error_message = handle_request_error(str(exc)).error.message

        raise HTTPException(400, error_message) from exc


@router.get("/v1/templates", dependencies=[Depends(check_api_key)])
@router.get("/v1/template/list", dependencies=[Depends(check_api_key)])
async def list_templates(request: Request) -> TemplateList:
    """
    Get a list of all templates.

    Requires an admin key to see all templates.
    """

    template_strings = []
    if get_key_permission(request) == "admin":
        templates = get_all_templates()
        template_strings = [template.stem for template in templates]
    else:
        if model.container and model.container.prompt_template:
            template_strings.append(model.container.prompt_template.name)

    return TemplateList(data=template_strings)


@router.post(
    "/v1/template/switch",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def switch_template(data: TemplateSwitchRequest):
    """Switch the currently loaded template."""

    if not data.name:
        error_message = handle_request_error(
            "New template name not found.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    try:
        template_path = pathlib.Path("templates") / data.name
        model.container.prompt_template = await PromptTemplate.from_file(template_path)
    except FileNotFoundError as e:
        error_message = handle_request_error(
            f"The template name {data.name} doesn't exist. Check the spelling?",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message) from e


@router.post(
    "/v1/template/unload",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def unload_template():
    """Unloads the currently selected template"""

    model.container.prompt_template = None


# Sampler override endpoints
@router.get("/v1/sampling/overrides", dependencies=[Depends(check_api_key)])
@router.get("/v1/sampling/override/list", dependencies=[Depends(check_api_key)])
async def list_sampler_overrides(request: Request) -> SamplerOverrideListResponse:
    """
    List all currently applied sampler overrides.

    Requires an admin key to see all override presets.
    """

    if get_key_permission(request) == "admin":
        presets = sampling.get_all_presets()
    else:
        presets = []

    return SamplerOverrideListResponse(
        presets=presets, **sampling.overrides_container.model_dump()
    )


@router.post(
    "/v1/sampling/override/switch",
    dependencies=[Depends(check_admin_key)],
)
async def switch_sampler_override(data: SamplerOverrideSwitchRequest):
    """Switch the currently loaded override preset"""

    if data.preset:
        try:
            await sampling.overrides_from_file(data.preset)
        except FileNotFoundError as e:
            error_message = handle_request_error(
                f"Sampler override preset with name {data.preset} does not exist. "
                + "Check the spelling?",
                exc_info=False,
            ).error.message

            raise HTTPException(400, error_message) from e
    elif data.overrides:
        sampling.overrides_from_dict(data.overrides)
    else:
        error_message = handle_request_error(
            "A sampler override preset or dictionary wasn't provided.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)


@router.post(
    "/v1/sampling/override/unload",
    dependencies=[Depends(check_admin_key)],
)
async def unload_sampler_override():
    """Unloads the currently selected override preset"""

    sampling.overrides_from_dict({})
