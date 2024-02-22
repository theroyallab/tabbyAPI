"""The main tabbyAPI module. Contains the FastAPI server and endpoints."""
import os
import pathlib
import uvicorn
from asyncio import CancelledError
from typing import Optional
from uuid import uuid4
from jinja2 import TemplateError
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from functools import partial
from progress.bar import IncrementalBar

import common.gen_logging as gen_logging
from backends.exllamav2.model import ExllamaV2Container
from backends.exllamav2.utils import check_exllama_version
from common.args import convert_args_to_dict, init_argparser
from common.auth import check_admin_key, check_api_key, load_auth_keys
from common.config import (
    get_developer_config,
    get_sampling_config,
    override_config_from_args,
    read_config_from_file,
    get_gen_logging_config,
    get_model_config,
    get_draft_model_config,
    get_lora_config,
    get_network_config,
)
from common.generators import call_with_semaphore, generate_with_semaphore
from common.sampling import (
    get_sampler_overrides,
    set_overrides_from_file,
    set_overrides_from_dict,
)
from common.templating import (
    get_all_templates,
    get_prompt_from_template,
    get_template_from_file,
)
from common.utils import get_generator_error, get_sse_packet, load_progress, unwrap
from common.logger import init_logger
from OAI.types.completion import CompletionRequest
from OAI.types.chat_completion import ChatCompletionRequest
from OAI.types.lora import LoraCard, LoraList, LoraLoadRequest, LoraLoadResponse
from OAI.types.model import (
    ModelCard,
    ModelLoadRequest,
    ModelLoadResponse,
    ModelCardParameters,
)
from OAI.types.sampler_overrides import SamplerOverrideSwitchRequest
from OAI.types.template import TemplateList, TemplateSwitchRequest
from OAI.types.token import (
    TokenEncodeRequest,
    TokenEncodeResponse,
    TokenDecodeRequest,
    TokenDecodeResponse,
)
from OAI.utils.completion import (
    create_completion_response,
    create_chat_completion_response,
    create_chat_completion_stream_chunk,
)
from OAI.utils.model import get_model_list
from OAI.utils.lora import get_lora_list

logger = init_logger(__name__)

app = FastAPI(
    title="TabbyAPI",
    summary="An OAI compatible exllamav2 API that's both lightweight and fast",
    description=(
        "This docs page is not meant to send requests! Please use a service "
        "like Postman or a frontend UI."
    ),
)

# Globally scoped variables. Undefined until initalized in main
MODEL_CONTAINER: Optional[ExllamaV2Container] = None


def _check_model_container():
    if MODEL_CONTAINER is None or MODEL_CONTAINER.model is None:
        raise HTTPException(400, "No models are loaded.")


# ALlow CORS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model list endpoint
@app.get("/v1/models", dependencies=[Depends(check_api_key)])
@app.get("/v1/model/list", dependencies=[Depends(check_api_key)])
async def list_models():
    """Lists all models in the model directory."""
    model_config = get_model_config()
    model_dir = unwrap(model_config.get("model_dir"), "models")
    model_path = pathlib.Path(model_dir)

    draft_model_dir = get_draft_model_config().get("draft_model_dir")

    models = get_model_list(model_path.resolve(), draft_model_dir)
    if unwrap(model_config.get("use_dummy_models"), False):
        models.data.insert(0, ModelCard(id="gpt-3.5-turbo"))

    return models


# Currently loaded model endpoint
@app.get(
    "/v1/model",
    dependencies=[Depends(check_api_key), Depends(_check_model_container)],
)
@app.get(
    "/v1/internal/model/info",
    dependencies=[Depends(check_api_key), Depends(_check_model_container)],
)
async def get_current_model():
    """Returns the currently loaded model."""
    model_name = MODEL_CONTAINER.get_model_path().name
    prompt_template = MODEL_CONTAINER.prompt_template
    model_card = ModelCard(
        id=model_name,
        parameters=ModelCardParameters(
            rope_scale=MODEL_CONTAINER.config.scale_pos_emb,
            rope_alpha=MODEL_CONTAINER.config.scale_alpha_value,
            max_seq_len=MODEL_CONTAINER.config.max_seq_len,
            cache_mode="FP8" if MODEL_CONTAINER.cache_fp8 else "FP16",
            prompt_template=prompt_template.name if prompt_template else None,
            num_experts_per_token=MODEL_CONTAINER.config.num_experts_per_token,
            use_cfg=MODEL_CONTAINER.use_cfg,
        ),
        logging=gen_logging.PREFERENCES,
    )

    if MODEL_CONTAINER.draft_config:
        draft_card = ModelCard(
            id=MODEL_CONTAINER.get_model_path(True).name,
            parameters=ModelCardParameters(
                rope_scale=MODEL_CONTAINER.draft_config.scale_pos_emb,
                rope_alpha=MODEL_CONTAINER.draft_config.scale_alpha_value,
                max_seq_len=MODEL_CONTAINER.draft_config.max_seq_len,
            ),
        )
        model_card.parameters.draft = draft_card

    return model_card


@app.get("/v1/model/draft/list", dependencies=[Depends(check_api_key)])
async def list_draft_models():
    """Lists all draft models in the model directory."""
    draft_model_dir = unwrap(get_draft_model_config().get("draft_model_dir"), "models")
    draft_model_path = pathlib.Path(draft_model_dir)

    models = get_model_list(draft_model_path.resolve())

    return models


# Load model endpoint
@app.post("/v1/model/load", dependencies=[Depends(check_admin_key)])
async def load_model(request: Request, data: ModelLoadRequest):
    """Loads a model into the model container."""
    global MODEL_CONTAINER

    if not data.name:
        raise HTTPException(400, "A model name was not provided.")

    # Unload the existing model
    if MODEL_CONTAINER and MODEL_CONTAINER.model:
        loaded_model_name = MODEL_CONTAINER.get_model_path().name

        if loaded_model_name == data.name:
            raise HTTPException(
                400, f"Model \"{loaded_model_name}\"is already loaded! Aborting."
            )
        else:
            MODEL_CONTAINER.unload()

    model_path = pathlib.Path(unwrap(get_model_config().get("model_dir"), "models"))
    model_path = model_path / data.name

    load_data = data.model_dump()

    if data.draft:
        if not data.draft.draft_model_name:
            raise HTTPException(
                400, "draft_model_name was not found inside the draft object."
            )

        load_data["draft"]["draft_model_dir"] = unwrap(
            get_draft_model_config().get("draft_model_dir"), "models"
        )

    if not model_path.exists():
        raise HTTPException(400, "model_path does not exist. Check model_name?")

    MODEL_CONTAINER = ExllamaV2Container(model_path.resolve(), False, **load_data)

    async def generator():
        """Generator for the loading process."""

        model_type = "draft" if MODEL_CONTAINER.draft_config else "model"
        load_status = MODEL_CONTAINER.load_gen(load_progress)

        try:
            for module, modules in load_status:
                if await request.is_disconnected():
                    break

                if module == 0:
                    loading_bar: IncrementalBar = IncrementalBar("Modules", max=modules)
                elif module == modules:
                    loading_bar.next()
                    loading_bar.finish()

                    response = ModelLoadResponse(
                        model_type=model_type,
                        module=module,
                        modules=modules,
                        status="finished",
                    )

                    yield get_sse_packet(response.model_dump_json())

                    # Switch to model progress if the draft model is loaded
                    if MODEL_CONTAINER.draft_config:
                        model_type = "model"
                else:
                    loading_bar.next()

                    response = ModelLoadResponse(
                        model_type=model_type,
                        module=module,
                        modules=modules,
                        status="processing",
                    )

                    yield get_sse_packet(response.model_dump_json())
        except CancelledError:
            logger.error(
                "Model load cancelled by user. "
                "Please make sure to run unload to free up resources."
            )
        except Exception as exc:
            yield get_generator_error(str(exc))

    return StreamingResponse(generator(), media_type="text/event-stream")


# Unload model endpoint
@app.post(
    "/v1/model/unload",
    dependencies=[Depends(check_admin_key), Depends(_check_model_container)],
)
async def unload_model():
    """Unloads the currently loaded model."""
    global MODEL_CONTAINER

    MODEL_CONTAINER.unload()
    MODEL_CONTAINER = None


@app.get("/v1/templates", dependencies=[Depends(check_api_key)])
@app.get("/v1/template/list", dependencies=[Depends(check_api_key)])
async def get_templates():
    templates = get_all_templates()
    template_strings = list(map(lambda template: template.stem, templates))
    return TemplateList(data=template_strings)


@app.post(
    "/v1/template/switch",
    dependencies=[Depends(check_admin_key), Depends(_check_model_container)],
)
async def switch_template(data: TemplateSwitchRequest):
    """Switch the currently loaded template"""
    if not data.name:
        raise HTTPException(400, "New template name not found.")

    try:
        template = get_template_from_file(data.name)
        MODEL_CONTAINER.prompt_template = template
    except FileNotFoundError as e:
        raise HTTPException(400, "Template does not exist. Check the name?") from e


@app.post(
    "/v1/template/unload",
    dependencies=[Depends(check_admin_key), Depends(_check_model_container)],
)
async def unload_template():
    """Unloads the currently selected template"""

    MODEL_CONTAINER.prompt_template = None


# Sampler override endpoints
@app.get("/v1/sampling/overrides", dependencies=[Depends(check_api_key)])
@app.get("/v1/sampling/override/list", dependencies=[Depends(check_api_key)])
async def list_sampler_overrides():
    """API wrapper to list all currently applied sampler overrides"""

    return get_sampler_overrides()


@app.post(
    "/v1/sampling/override/switch",
    dependencies=[Depends(check_admin_key)],
)
async def switch_sampler_override(data: SamplerOverrideSwitchRequest):
    """Switch the currently loaded override preset"""

    if data.preset:
        try:
            set_overrides_from_file(data.preset)
        except FileNotFoundError as e:
            raise HTTPException(
                400, "Sampler override preset does not exist. Check the name?"
            ) from e
    elif data.overrides:
        set_overrides_from_dict(data.overrides)
    else:
        raise HTTPException(
            400, "A sampler override preset or dictionary wasn't provided."
        )


@app.post(
    "/v1/sampling/override/unload",
    dependencies=[Depends(check_admin_key)],
)
async def unload_sampler_override():
    """Unloads the currently selected override preset"""

    set_overrides_from_dict({})


# Lora list endpoint
@app.get("/v1/loras", dependencies=[Depends(check_api_key)])
@app.get("/v1/lora/list", dependencies=[Depends(check_api_key)])
async def get_all_loras():
    """Lists all LoRAs in the lora directory."""
    lora_path = pathlib.Path(unwrap(get_lora_config().get("lora_dir"), "loras"))
    loras = get_lora_list(lora_path.resolve())

    return loras


# Currently loaded loras endpoint
@app.get(
    "/v1/lora",
    dependencies=[Depends(check_api_key), Depends(_check_model_container)],
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
                MODEL_CONTAINER.active_loras,
            )
        )
    )

    return active_loras


# Load lora endpoint
@app.post(
    "/v1/lora/load",
    dependencies=[Depends(check_admin_key), Depends(_check_model_container)],
)
async def load_lora(data: LoraLoadRequest):
    """Loads a LoRA into the model container."""
    if not data.loras:
        raise HTTPException(400, "List of loras to load is not found.")

    lora_dir = pathlib.Path(unwrap(get_lora_config().get("lora_dir"), "loras"))
    if not lora_dir.exists():
        raise HTTPException(
            400,
            "A parent lora directory does not exist. Check your config.yml?",
        )

    # Clean-up existing loras if present
    if len(MODEL_CONTAINER.active_loras) > 0:
        MODEL_CONTAINER.unload(True)

    result = MODEL_CONTAINER.load_loras(lora_dir, **data.model_dump())
    return LoraLoadResponse(
        success=unwrap(result.get("success"), []),
        failure=unwrap(result.get("failure"), []),
    )


# Unload lora endpoint
@app.post(
    "/v1/lora/unload",
    dependencies=[Depends(check_admin_key), Depends(_check_model_container)],
)
async def unload_loras():
    """Unloads the currently loaded loras."""
    MODEL_CONTAINER.unload(True)


# Encode tokens endpoint
@app.post(
    "/v1/token/encode",
    dependencies=[Depends(check_api_key), Depends(_check_model_container)],
)
async def encode_tokens(data: TokenEncodeRequest):
    """Encodes a string into tokens."""
    raw_tokens = MODEL_CONTAINER.encode_tokens(data.text, **data.get_params())
    tokens = unwrap(raw_tokens, [])
    response = TokenEncodeResponse(tokens=tokens, length=len(tokens))

    return response


# Decode tokens endpoint
@app.post(
    "/v1/token/decode",
    dependencies=[Depends(check_api_key), Depends(_check_model_container)],
)
async def decode_tokens(data: TokenDecodeRequest):
    """Decodes tokens into a string."""
    message = MODEL_CONTAINER.decode_tokens(data.tokens, **data.get_params())
    response = TokenDecodeResponse(text=unwrap(message, ""))

    return response


# Completions endpoint
@app.post(
    "/v1/completions",
    dependencies=[Depends(check_api_key), Depends(_check_model_container)],
)
async def generate_completion(request: Request, data: CompletionRequest):
    """Generates a completion from a prompt."""
    model_path = MODEL_CONTAINER.get_model_path()

    if isinstance(data.prompt, list):
        data.prompt = "\n".join(data.prompt)

    disable_request_streaming = unwrap(
        get_developer_config().get("disable_request_streaming"), False
    )

    if data.stream and not disable_request_streaming:

        async def generator():
            """Generator for the generation process."""
            try:
                new_generation = MODEL_CONTAINER.generate_gen(
                    data.prompt, **data.to_gen_params()
                )
                for generation in new_generation:
                    if await request.is_disconnected():
                        break

                    response = create_completion_response(generation, model_path.name)

                    yield get_sse_packet(response.model_dump_json())

                # Yield a finish response on successful generation
                yield get_sse_packet("[DONE]")
            except CancelledError:
                logger.error("Completion request cancelled by user.")
            except Exception as exc:
                yield get_generator_error(str(exc))

        return StreamingResponse(
            generate_with_semaphore(generator), media_type="text/event-stream"
        )

    generation = await call_with_semaphore(
        partial(MODEL_CONTAINER.generate, data.prompt, **data.to_gen_params())
    )
    response = create_completion_response(generation, model_path.name)

    return response


# Chat completions endpoint
@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(check_api_key), Depends(_check_model_container)],
)
async def generate_chat_completion(request: Request, data: ChatCompletionRequest):
    """Generates a chat completion from a prompt."""
    if MODEL_CONTAINER.prompt_template is None:
        raise HTTPException(
            422,
            "This endpoint is disabled because a prompt template is not set.",
        )

    model_path = MODEL_CONTAINER.get_model_path()

    if isinstance(data.messages, str):
        prompt = data.messages
    else:
        try:
            special_tokens_dict = MODEL_CONTAINER.get_special_tokens(
                unwrap(data.add_bos_token, True),
                unwrap(data.ban_eos_token, False),
            )
            prompt = get_prompt_from_template(
                data.messages,
                MODEL_CONTAINER.prompt_template,
                data.add_generation_prompt,
                special_tokens_dict,
            )
        except KeyError as exc:
            raise HTTPException(
                400,
                "Could not find a Conversation from prompt template "
                f"'{MODEL_CONTAINER.prompt_template.name}'. "
                "Check your spelling?",
            ) from exc
        except TemplateError as exc:
            raise HTTPException(
                400,
                f"TemplateError: {str(exc)}",
            ) from exc

    disable_request_streaming = unwrap(
        get_developer_config().get("disable_request_streaming"), False
    )

    if data.stream and not disable_request_streaming:
        const_id = f"chatcmpl-{uuid4().hex}"

        async def generator():
            """Generator for the generation process."""
            try:
                new_generation = MODEL_CONTAINER.generate_gen(
                    prompt, **data.to_gen_params()
                )
                for generation in new_generation:
                    if await request.is_disconnected():
                        break

                    response = create_chat_completion_stream_chunk(
                        const_id, generation, model_path.name
                    )

                    yield get_sse_packet(response.model_dump_json())

                # Yield a finish response on successful generation
                finish_response = create_chat_completion_stream_chunk(
                    const_id, finish_reason="stop"
                )

                yield get_sse_packet(finish_response.model_dump_json())
            except CancelledError:
                logger.error("Chat completion cancelled by user.")
            except Exception as exc:
                yield get_generator_error(str(exc))

        return StreamingResponse(
            generate_with_semaphore(generator), media_type="text/event-stream"
        )

    generation = await call_with_semaphore(
        partial(MODEL_CONTAINER.generate, prompt, **data.to_gen_params())
    )
    response = create_chat_completion_response(generation, model_path.name)

    return response


def entrypoint(args: Optional[dict] = None):
    """Entry function for program startup"""
    global MODEL_CONTAINER

    # Load from YAML config
    read_config_from_file(pathlib.Path("config.yml"))

    # Parse and override config from args
    if args is None:
        parser = init_argparser()
        args = convert_args_to_dict(parser.parse_args(), parser)

    override_config_from_args(args)

    developer_config = get_developer_config()

    # Check exllamav2 version and give a descriptive error if it's too old
    # Skip if launching unsafely

    if unwrap(developer_config.get("unsafe_launch"), False):
        logger.warning(
            "UNSAFE: Skipping ExllamaV2 version check.\n"
            "If you aren't a developer, please keep this off!"
        )
    else:
        check_exllama_version()

    # Enable CUDA malloc backend
    if unwrap(developer_config.get("cuda_malloc_backend"), False):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
        logger.warning("Enabled the experimental CUDA malloc backend.")

    network_config = get_network_config()

    # Initialize auth keys
    load_auth_keys(unwrap(network_config.get("disable_auth"), False))

    # Override the generation log options if given
    log_config = get_gen_logging_config()
    if log_config:
        gen_logging.update_from_dict(log_config)

    gen_logging.broadcast_status()

    # Set sampler parameter overrides if provided
    sampling_config = get_sampling_config()
    sampling_override_preset = sampling_config.get("override_preset")
    if sampling_override_preset:
        try:
            set_overrides_from_file(sampling_override_preset)
        except FileNotFoundError as e:
            logger.warning(str(e))

    # If an initial model name is specified, create a container
    # and load the model
    model_config = get_model_config()
    model_name = model_config.get("model_name")
    if model_name:
        model_path = pathlib.Path(unwrap(model_config.get("model_dir"), "models"))
        model_path = model_path / model_name

        MODEL_CONTAINER = ExllamaV2Container(
            model_path.resolve(), False, **model_config
        )
        load_status = MODEL_CONTAINER.load_gen(load_progress)
        for module, modules in load_status:
            if module == 0:
                loading_bar: IncrementalBar = IncrementalBar("Modules", max=modules)
            elif module == modules:
                loading_bar.next()
                loading_bar.finish()
            else:
                loading_bar.next()

        # Load loras after loading the model
        lora_config = get_lora_config()
        if lora_config.get("loras"):
            lora_dir = pathlib.Path(unwrap(lora_config.get("lora_dir"), "loras"))
            MODEL_CONTAINER.load_loras(lora_dir.resolve(), **lora_config)

    host = unwrap(network_config.get("host"), "127.0.0.1")
    port = unwrap(network_config.get("port"), 5000)

    # TODO: Move OAI API to a separate folder
    logger.info(f"Developer documentation: http://{host}:{port}/docs")
    logger.info(f"Completions: http://{host}:{port}/v1/completions")
    logger.info(f"Chat completions: http://{host}:{port}/v1/chat/completions")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug",
    )


if __name__ == "__main__":
    entrypoint()
