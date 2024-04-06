import asyncio
import pathlib
from fastapi import APIRouter, Depends, HTTPException, Header, Request
from functools import partial
from loguru import logger
from sse_starlette import EventSourceResponse
from sys import maxsize
from typing import Optional

from common import config, model, gen_logging, sampling
from common.auth import check_admin_key, check_api_key, validate_key_permission
from common.concurrency import (
    call_with_semaphore,
    generate_with_semaphore,
)
from common.networking import handle_request_error, run_with_request_disconnect
from common.templating import (
    get_all_templates,
    get_template_from_file,
)
from common.utils import coalesce, unwrap
from endpoints.OAI.types.auth import AuthPermissionResponse
from endpoints.OAI.types.completion import CompletionRequest
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.types.lora import (
    LoraCard,
    LoraList,
    LoraLoadRequest,
    LoraLoadResponse,
)
from endpoints.OAI.types.model import (
    ModelCard,
    ModelLoadRequest,
    ModelCardParameters,
)
from endpoints.OAI.types.sampler_overrides import SamplerOverrideSwitchRequest
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
from endpoints.OAI.utils.model import get_model_list, stream_model_load
from endpoints.OAI.utils.lora import get_lora_list


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


# Model list endpoint
@router.get("/v1/models", dependencies=[Depends(check_api_key)])
@router.get("/v1/model/list", dependencies=[Depends(check_api_key)])
async def list_models():
    """Lists all models in the model directory."""
    model_config = config.model_config()
    model_dir = unwrap(model_config.get("model_dir"), "models")
    model_path = pathlib.Path(model_dir)

    draft_model_dir = config.draft_model_config().get("draft_model_dir")

    models = get_model_list(model_path.resolve(), draft_model_dir)
    if unwrap(model_config.get("use_dummy_models"), False):
        models.data.insert(0, ModelCard(id="gpt-3.5-turbo"))

    return models


# Currently loaded model endpoint
@router.get(
    "/v1/model",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def get_current_model():
    """Returns the currently loaded model."""
    model_params = model.container.get_model_parameters()
    draft_model_params = model_params.pop("draft", {})

    if draft_model_params:
        model_params["draft"] = ModelCard(
            id=unwrap(draft_model_params.get("name"), "unknown"),
            parameters=ModelCardParameters.model_validate(draft_model_params),
        )
    else:
        draft_model_params = None

    model_card = ModelCard(
        id=unwrap(model_params.pop("name", None), "unknown"),
        parameters=ModelCardParameters.model_validate(model_params),
        logging=gen_logging.PREFERENCES,
    )

    if draft_model_params:
        draft_card = ModelCard(
            id=unwrap(draft_model_params.pop("name", None), "unknown"),
            parameters=ModelCardParameters.model_validate(draft_model_params),
        )

        model_card.parameters.draft = draft_card

    return model_card


@router.get("/v1/model/draft/list", dependencies=[Depends(check_api_key)])
async def list_draft_models():
    """Lists all draft models in the model directory."""
    draft_model_dir = unwrap(
        config.draft_model_config().get("draft_model_dir"), "models"
    )
    draft_model_path = pathlib.Path(draft_model_dir)

    models = get_model_list(draft_model_path.resolve())

    return models


# Load model endpoint
@router.post("/v1/model/load", dependencies=[Depends(check_admin_key)])
async def load_model(request: Request, data: ModelLoadRequest):
    """Loads a model into the model container."""

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

    load_callback = partial(stream_model_load, data, model_path, draft_model_path)

    # Wrap in a semaphore if the queue isn't being skipped
    if data.skip_queue:
        logger.warning(
            "Model load request is skipping the completions queue. "
            "Unexpected results may occur."
        )
    else:
        load_callback = partial(generate_with_semaphore, load_callback)

    return EventSourceResponse(load_callback(), ping=maxsize)


# Unload model endpoint
@router.post(
    "/v1/model/unload",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def unload_model():
    """Unloads the currently loaded model."""
    await model.unload_model()


@router.get("/v1/templates", dependencies=[Depends(check_api_key)])
@router.get("/v1/template/list", dependencies=[Depends(check_api_key)])
async def get_templates():
    templates = get_all_templates()
    template_strings = list(map(lambda template: template.stem, templates))
    return TemplateList(data=template_strings)


@router.post(
    "/v1/template/switch",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def switch_template(data: TemplateSwitchRequest):
    """Switch the currently loaded template"""
    if not data.name:
        error_message = handle_request_error(
            "New template name not found.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

    try:
        template = get_template_from_file(data.name)
        model.container.prompt_template = template
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
async def list_sampler_overrides():
    """API wrapper to list all currently applied sampler overrides"""

    return sampling.overrides


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


# Lora list endpoint
@router.get("/v1/loras", dependencies=[Depends(check_api_key)])
@router.get("/v1/lora/list", dependencies=[Depends(check_api_key)])
async def get_all_loras():
    """Lists all LoRAs in the lora directory."""
    lora_path = pathlib.Path(unwrap(config.lora_config().get("lora_dir"), "loras"))
    loras = get_lora_list(lora_path.resolve())

    return loras


# Currently loaded loras endpoint
@router.get(
    "/v1/lora",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def get_active_loras():
    """Returns the currently loaded loras."""
    active_loras = LoraList(
        data=list(
            map(
                lambda lora: LoraCard(
                    id=pathlib.Path(lora.lora_path).parent.name,
                    scaling=lora.lora_scaling * lora.lora_r / lora.lora_alpha,
                ),
                model.container.active_loras,
            )
        )
    )

    return active_loras


# Load lora endpoint
@router.post(
    "/v1/lora/load",
    dependencies=[Depends(check_admin_key), Depends(check_model_container)],
)
async def load_lora(data: LoraLoadRequest):
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

    load_callback = partial(model.load_loras, lora_dir, **data.model_dump())

    # Wrap in a semaphore if the queue isn't being skipped
    if data.skip_queue:
        logger.warning(
            "Lora load request is skipping the completions queue. "
            "Unexpected results may occur."
        )
    else:
        load_callback = partial(call_with_semaphore, load_callback)

    load_result = await load_callback()

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

    model.unload_loras()


# Encode tokens endpoint
@router.post(
    "/v1/token/encode",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def encode_tokens(data: TokenEncodeRequest):
    """Encodes a string into tokens."""
    raw_tokens = model.container.encode_tokens(data.text, **data.get_params())
    tokens = unwrap(raw_tokens, [])
    response = TokenEncodeResponse(tokens=tokens, length=len(tokens))

    return response


# Decode tokens endpoint
@router.post(
    "/v1/token/decode",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def decode_tokens(data: TokenDecodeRequest):
    """Decodes tokens into a string."""
    message = model.container.decode_tokens(data.tokens, **data.get_params())
    response = TokenDecodeResponse(text=unwrap(message, ""))

    return response


@router.get("/v1/auth/permission", dependencies=[Depends(check_api_key)])
async def get_key_permission(
    x_admin_key: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """
    Gets the access level/permission of a provided key in headers.

    Priority:
    - X-api-key
    - X-admin-key
    - Authorization
    """

    test_key = coalesce(x_admin_key, x_api_key, authorization)

    try:
        permission = await validate_key_permission(test_key)
        return AuthPermissionResponse(permission=permission)
    except ValueError as exc:
        error_message = handle_request_error(str(exc)).error.message

        raise HTTPException(400, error_message) from exc


# Completions endpoint
@router.post(
    "/v1/completions",
    dependencies=[Depends(check_api_key), Depends(check_model_container)],
)
async def completion_request(request: Request, data: CompletionRequest):
    """Generates a completion from a prompt."""
    model_path = model.container.get_model_path()

    if isinstance(data.prompt, list):
        data.prompt = "\n".join(data.prompt)

    disable_request_streaming = unwrap(
        config.developer_config().get("disable_request_streaming"), False
    )

    if data.stream and not disable_request_streaming:
        generator_callback = partial(stream_generate_completion, data, model_path)

        return EventSourceResponse(
            generate_with_semaphore(generator_callback),
            ping=maxsize,
        )
    else:
        generate_task = asyncio.create_task(
            call_with_semaphore(partial(generate_completion, data, model_path))
        )

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
async def chat_completion_request(request: Request, data: ChatCompletionRequest):
    """Generates a chat completion from a prompt."""

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

    disable_request_streaming = unwrap(
        config.developer_config().get("disable_request_streaming"), False
    )

    if data.stream and not disable_request_streaming:
        generator_callback = partial(
            stream_generate_chat_completion, prompt, data, model_path
        )

        return EventSourceResponse(
            generate_with_semaphore(generator_callback),
            ping=maxsize,
        )
    else:
        generate_task = asyncio.create_task(
            call_with_semaphore(
                partial(generate_chat_completion, prompt, data, model_path)
            )
        )

        response = await run_with_request_disconnect(
            request,
            generate_task,
            disconnect_message="Chat completion generation cancelled by user.",
        )
        return response
