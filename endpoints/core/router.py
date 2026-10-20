import asyncio
import os
import pathlib
from typing import Optional
from common.multimodal import MultimodalEmbeddingWrapper
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from sse_starlette import EventSourceResponse

from common import model, sampling
from common.auth import check_admin_key, check_api_key, get_key_permission
from common.downloader import hf_repo_download
from common.model import check_embeddings_container, check_model_container
from common.networking import (
    get_sse_ping_interval,
    handle_request_error,
    run_with_request_disconnect,
)
from common.tabby_config import config
from common.templating import PromptTemplate, get_all_templates
from common.utils import unwrap
from common.health import HealthManager
from endpoints.OAI.utils.chat_completion import format_messages_with_template
from endpoints.core.types.auth import AuthPermissionResponse
from endpoints.core.types.download import DownloadRequest, DownloadResponse
from endpoints.core.types.lora import LoraList, LoraLoadRequest, LoraLoadResponse
from endpoints.core.types.model import (
    EmbeddingModelLoadRequest,
    LoadOptionsParseRequest,
    LoadOptionsParseResponse,
    ModelCard,
    ModelDefaultGenerationSettings,
    ModelList,
    ModelLoadConfigResponse,
    ModelLoadConfigSaveRequest,
    ModelLoadConfigSaveResponse,
    ModelLoadRequest,
    ModelLoadResponse,
    ModelPropsModalities,
    ModelPropsResponse,
)
from endpoints.core.types.health import HealthCheckResponse
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
    get_dummy_models,
    get_model_list,
    get_model_load_config,
    save_model_load_config,
    stream_model_load,
)


router = APIRouter()


# Healthcheck endpoint
@router.get("/health")
async def healthcheck(response: Response) -> HealthCheckResponse:
    """Get the current service health status"""
    healthy, issues = await HealthManager.is_service_healthy()

    if not healthy:
        response.status_code = 503

    return HealthCheckResponse(status="healthy" if healthy else "unhealthy", issues=issues)


@router.get("/.well-known/serviceinfo")
async def service_info():
    return JSONResponse(
        content={
            "version": 0.1,
            "software": {
                "name": "TabbyAPI",
                "repository": "https://github.com/theroyallab/tabbyAPI",
                "homepage": "https://github.com/theroyallab/tabbyAPI",
            },
            "api": {
                "openai": {
                    "name": "OpenAI API",
                    "relative_url": "/v1",
                    "documentation": "https://theroyallab.github.io/tabbyAPI",
                    "version": 1,
                },
                "koboldai": {
                    "name": "KoboldAI API",
                    "relative_url": "/api",
                    "documentation": "https://theroyallab.github.io/tabbyAPI",
                    "version": 1,
                },
            },
        }
    )


# Model list endpoint
@router.get("/v1/models", dependencies=[Depends(check_api_key)])
@router.get("/v1/model/list", dependencies=[Depends(check_api_key)])
async def list_models(request: Request) -> ModelList:
    """
    Lists all models in the model directory.

    Requires an admin key to see all models.
    """

    model_dir = config.model.model_dir
    model_path = pathlib.Path(model_dir)

    draft_model_dir = config.draft_model.draft_model_dir

    if get_key_permission(request) == "admin":
        models = get_model_list(model_path.resolve(), draft_model_dir)
    else:
        models = await get_current_model_list()

    if config.model.use_dummy_models:
        models.data[:0] = get_dummy_models()

    return models


# Currently loaded model endpoint
@router.get(
    "/v1/model",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def current_model() -> ModelCard:
    """Returns the currently loaded model."""

    return get_current_model()


@router.get("/props", dependencies=[Depends(check_api_key), Depends(check_model_container)])
async def model_props() -> ModelPropsResponse:
    """
    Returns specific properties of a model for clients.

    To get all properties, use /v1/model instead.
    """

    current_model_card = get_current_model()
    resp = ModelPropsResponse(
        total_slots=current_model_card.parameters.max_batch_size,
        model_path=str(model.container.model_dir),
        default_generation_settings=ModelDefaultGenerationSettings(
            n_ctx=current_model_card.parameters.max_seq_len,
        ),
        modalities=ModelPropsModalities(vision=bool(current_model_card.parameters.use_vision)),
    )

    if current_model_card.parameters.prompt_template_content:
        resp.chat_template = current_model_card.parameters.prompt_template_content

    return resp


@router.get("/v1/model/draft/list", dependencies=[Depends(check_api_key)])
async def list_draft_models(request: Request) -> ModelList:
    """
    Lists all draft models in the model directory.

    Requires an admin key to see all draft models.
    """

    if get_key_permission(request) == "admin":
        draft_model_dir = config.draft_model.draft_model_dir
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
    if not data.model_name:
        error_message = handle_request_error(
            "A model name was not provided for load.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    model_path = pathlib.Path(config.model.model_dir)
    model_path = model_path / data.model_name

    if not model_path.exists():
        error_message = handle_request_error(
            "Could not find the model path for load. Check model name or config.yml?",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    return EventSourceResponse(stream_model_load(data, model_path), ping=get_sse_ping_interval())


@router.get("/v1/model/load_config", dependencies=[Depends(check_admin_key)])
async def model_load_config(model_name: str) -> ModelLoadConfigResponse:
    """
    The load options that apply to a model when a load request leaves them
    out: the model folder's tabby_config.yml over config.yml's use_as_default
    keys. Also given as YAML, ready to edit and send back with a load request.
    """

    model_path = pathlib.Path(config.model.model_dir) / model_name
    if not model_path.is_dir():
        error_message = handle_request_error(
            f"Model {model_name} was not found in the model directory.",
            exc_info=False,
        ).error.message

        raise HTTPException(404, error_message)

    try:
        return await get_model_load_config(model_name, model_path)
    except (YAMLError, ValueError) as exc:
        error_message = handle_request_error(
            f"Could not read the tabby_config.yml of {model_name}: {exc}",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message) from exc


def parse_load_options_yaml(text: str) -> dict:
    """Load options written as YAML (or JSON), as a mapping."""

    try:
        settings = YAML(typ="safe").load(text)
    except YAMLError as exc:
        error_message = handle_request_error(
            f"Load options are not valid YAML: {exc}", exc_info=False
        ).error.message

        raise HTTPException(400, error_message) from exc

    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        error_message = handle_request_error(
            "Load options must be a mapping of option names to values.", exc_info=False
        ).error.message

        raise HTTPException(400, error_message)

    return settings


@router.post("/v1/model/load_config/parse", dependencies=[Depends(check_admin_key)])
async def parse_load_options(data: LoadOptionsParseRequest) -> LoadOptionsParseResponse:
    """
    Turn load options written as YAML (or JSON) into a load request body, so
    clients can edit them as YAML without a YAML parser of their own.
    """

    return LoadOptionsParseResponse(settings=parse_load_options_yaml(data.yaml))


@router.post("/v1/model/load_config/save", dependencies=[Depends(check_admin_key)])
async def save_load_options(data: ModelLoadConfigSaveRequest) -> ModelLoadConfigSaveResponse:
    """
    Save load options as the model folder's tabby_config.yml, for loads that
    leave them out. Options that only repeat a use_as_default value are left
    out, unless the file already has them. An existing file is copied to
    tabby_config.bak first, and keeps its comments and any keys a load request
    can't set.
    """

    # Only model folders under the model directory, by name, not by where a
    # symlinked folder points
    model_root = pathlib.Path(os.path.normpath(config.model.model_dir))
    model_path = pathlib.Path(os.path.normpath(model_root / data.model_name))
    if (
        model_path == model_root
        or not model_path.is_relative_to(model_root)
        or not (model_path / "config.json").is_file()
    ):
        error_message = handle_request_error(
            f"Model {data.model_name} was not found in the model directory.",
            exc_info=False,
        ).error.message

        raise HTTPException(404, error_message)

    settings = data.settings if data.yaml is None else parse_load_options_yaml(data.yaml)
    try:
        return await save_model_load_config(data.model_name, model_path, settings)
    except (YAMLError, ValueError) as exc:
        error_message = handle_request_error(
            f"Could not save the load options of {data.model_name}: {exc}",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message) from exc
    except OSError as exc:
        error_message = handle_request_error(
            f"Could not write the tabby_config.yml of {data.model_name}: {exc}",
            exc_info=False,
        ).error.message

        raise HTTPException(500, error_message) from exc


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
        lora_path = pathlib.Path(config.lora.lora_dir)
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

    lora_dir = pathlib.Path(config.lora.lora_dir)
    if not lora_dir.exists():
        error_message = handle_request_error(
            "A parent lora directory does not exist for load. Check your config.yml?",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    load_result = await model.load_loras(lora_dir, **data.model_dump(), skip_wait=data.skip_queue)

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
        embedding_model_dir = config.embeddings.embedding_model_dir
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
    if not data.embedding_model_name:
        error_message = handle_request_error(
            "A model name was not provided for load.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    embedding_model_dir = pathlib.Path(config.embeddings.embedding_model_dir)
    embedding_model_path = embedding_model_dir / data.embedding_model_name

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

    mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None

    if isinstance(data.text, str):
        text = data.text
    elif isinstance(data.text, list):
        if "oai" not in config.network.api_servers:
            error_message = handle_request_error(
                "Enable the OAI server to handle chat completion messages.",
                exc_info=False,
            ).error.message

            raise HTTPException(422, error_message)

        if not model.container.prompt_template:
            error_message = handle_request_error(
                "Cannot tokenize chat completion message because "
                + "a prompt template is not set.",
                exc_info=False,
            ).error.message

            raise HTTPException(422, error_message)

        template_vars = {
            **(data.template_vars or {}),
            "add_generation_prompt": False,
        }

        text, mm_embeddings, rendered_template_vars = await format_messages_with_template(
            data.text, template_vars
        )

        # Let encode_tokens be the sole authority on whether BOS is added.
        bos_token = rendered_template_vars.get("bos_token")
        if bos_token and text.startswith(bos_token):
            text = text.removeprefix(bos_token)
    else:
        error_message = handle_request_error(
            "Unable to tokenize the provided text. Check your formatting?",
            exc_info=False,
        ).error.message

        raise HTTPException(422, error_message)

    raw_tokens = model.container.encode_tokens(text, embeddings=mm_embeddings, **data.get_params())
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

    if not data.prompt_template_name:
        error_message = handle_request_error(
            "New template name not found.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    try:
        template_path = pathlib.Path("templates") / data.prompt_template_name
        model.container.prompt_template = await PromptTemplate.from_file(template_path)
    except FileNotFoundError as e:
        error_message = handle_request_error(
            f"The template name {data.prompt_template_name} doesn't exist. "
            + "Check the spelling?",
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

    return SamplerOverrideListResponse(presets=presets, **sampling.overrides_container.model_dump())


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
