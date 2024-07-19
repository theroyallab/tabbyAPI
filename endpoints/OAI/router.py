import asyncio
import pathlib
from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette import EventSourceResponse
from sys import maxsize

from common import config, model, sampling
from common.auth import check_admin_key, check_api_key, get_key_permission
from common.downloader import hf_repo_download
from common.networking import handle_request_error, run_with_request_disconnect
from common.templating import PromptTemplate, get_all_templates
from common.utils import unwrap
from endpoints.OAI.types.auth import AuthPermissionResponse
from endpoints.OAI.types.completion import CompletionRequest, CompletionResponse
from endpoints.OAI.types.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from endpoints.OAI.types.download import DownloadRequest, DownloadResponse
from endpoints.OAI.types.lora import (
    LoraList,
    LoraLoadRequest,
    LoraLoadResponse,
)
from endpoints.OAI.types.model import (
    ModelCard,
    ModelList,
    ModelLoadRequest,
    ModelLoadResponse,
)
from endpoints.OAI.types.sampler_overrides import (
    SamplerOverrideListResponse,
    SamplerOverrideSwitchRequest,
)
from endpoints.OAI.types.template import TemplateList, TemplateSwitchRequest
from endpoints.OAI.types.token import (
    TokenEncodeRequest,
    TokenEncodeResponse,
    TokenDecodeRequest,
    TokenDecodeResponse,
)
from endpoints.OAI.utils.chat_completion import (
    format_prompt_with_template,
    generate_chat_completion,
    stream_generate_chat_completion,
)
from endpoints.OAI.utils.completion import (
    generate_completion,
    stream_generate_completion,
)
from endpoints.OAI.utils.model import (
    get_current_model,
    get_current_model_list,
    get_model_list,
    stream_model_load,
)
from endpoints.OAI.utils.lora import get_active_loras, get_lora_list


router = APIRouter()


async def check_model_container():
    """FastAPI depends that checks if a model isn't loaded or currently loading."""

    if model.container is None or not (
        model.container.model_is_loading or model.container.model_loaded
    ):
        error_message = handle_request_error(
            "No models are currently loaded.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)


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

    model_path = model.container.get_model_path()

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
        generate_task = asyncio.create_task(generate_completion(data, model_path))

        response = await run_with_request_disconnect(
            request,
            generate_task,
            disconnect_message="Completion generation cancelled by user.",
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

    model_path = model.container.get_model_path()

    if isinstance(data.messages, str):
        prompt = data.messages
    else:
        prompt = format_prompt_with_template(data)

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
            generate_chat_completion(prompt, data, model_path)
        )

        response = await run_with_request_disconnect(
            request,
            generate_task,
            disconnect_message="Chat completion generation cancelled by user.",
        )
        return response


# Model list endpoint
@router.get("/v1/models", dependencies=[Depends(check_api_key)])
@router.get("/v1/model/list", dependencies=[Depends(check_api_key)])
async def list_models(request: Request) -> ModelList:
    """
    Lists all models in the model directory.

    Requires an admin key to see all models.
    """

    model_config = config.model_config()
    model_dir = unwrap(model_config.get("model_dir"), "models")
    model_path = pathlib.Path(model_dir)

    draft_model_dir = config.draft_model_config().get("draft_model_dir")

    if get_key_permission(request) == "admin":
        models = get_model_list(model_path.resolve(), draft_model_dir)
    else:
        models = await get_current_model_list()

    if unwrap(model_config.get("use_dummy_models"), False):
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
        draft_model_dir = unwrap(
            config.draft_model_config().get("draft_model_dir"), "models"
        )
        draft_model_path = pathlib.Path(draft_model_dir)

        models = get_model_list(draft_model_path.resolve())
    else:
        models = await get_current_model_list(is_draft=True)

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

    model_path = pathlib.Path(unwrap(config.model_config().get("model_dir"), "models"))
    model_path = model_path / data.name

    draft_model_path = None
    if data.draft:
        if not data.draft.draft_model_name:
            error_message = handle_request_error(
                "Could not find the draft model name for model load.",
                exc_info=False,
            ).error.message

            raise HTTPException(400, error_message)

        draft_model_path = unwrap(
            config.draft_model_config().get("draft_model_dir"), "models"
        )

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
        lora_path = pathlib.Path(unwrap(config.lora_config().get("lora_dir"), "loras"))
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

    lora_dir = pathlib.Path(unwrap(config.lora_config().get("lora_dir"), "loras"))
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
        model.container.prompt_template = PromptTemplate.from_file(data.name)
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
            sampling.overrides_from_file(data.preset)
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
