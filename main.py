"""The main tabbyAPI module. Contains the FastAPI server and endpoints."""
import pathlib
from asyncio import CancelledError
from typing import Optional
from uuid import uuid4

import uvicorn
import yaml
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from progress.bar import IncrementalBar

import gen_logging
from auth import check_admin_key, check_api_key, load_auth_keys
from generators import generate_with_semaphore
from model import ModelContainer
from OAI.types.completion import CompletionRequest
from OAI.types.chat_completion import ChatCompletionRequest
from OAI.types.lora import (LoraCard, LoraList, LoraLoadRequest,
                            LoraLoadResponse)
from OAI.types.model import (ModelCard, ModelLoadRequest, ModelLoadResponse,
                             ModelCardParameters)
from OAI.types.token import (TokenEncodeRequest, TokenEncodeResponse,
                             TokenDecodeRequest, TokenDecodeResponse)
from OAI.utils_oai import (create_completion_response, get_model_list,
                       get_lora_list, create_chat_completion_response,
                       create_chat_completion_stream_chunk)
from templating import get_prompt_from_template
from utils import get_generator_error, get_sse_packet, load_progress, unwrap

app = FastAPI()

# Globally scoped variables. Undefined until initalized in main
MODEL_CONTAINER: Optional[ModelContainer] = None
config: dict = {}


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
    model_config = unwrap(config.get("model"), {})  # pylint: disable=redefined-outer-name
    model_dir = unwrap(model_config.get("model_dir"), "models")
    model_path = pathlib.Path(model_dir)  # pylint: disable=redefined-outer-name

    draft_config = unwrap(model_config.get("draft"), {})
    draft_model_dir = draft_config.get("draft_model_dir")

    models = get_model_list(model_path.resolve(), draft_model_dir)
    if unwrap(model_config.get("use_dummy_models"), False):
        # pylint: disable=no-member
        models.data.insert(0, ModelCard(id="gpt-3.5-turbo"))

    return models


# Currently loaded model endpoint
@app.get(
    "/v1/model",
    dependencies=[Depends(check_api_key),
                  Depends(_check_model_container)])
@app.get(
    "/v1/internal/model/info",
    dependencies=[Depends(check_api_key),
                  Depends(_check_model_container)])
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
            prompt_template=prompt_template.name if prompt_template else None),
        logging=gen_logging.config)  # pylint: disable=no-member

    if MODEL_CONTAINER.draft_config:
        draft_card = ModelCard(
            id=MODEL_CONTAINER.get_model_path(True).name,
            parameters=ModelCardParameters(
                rope_scale=MODEL_CONTAINER.draft_config.scale_pos_emb,
                rope_alpha=MODEL_CONTAINER.draft_config.scale_alpha_value,
                max_seq_len=MODEL_CONTAINER.draft_config.max_seq_len))
        model_card.parameters.draft = draft_card

    return model_card


@app.get("/v1/model/draft/list")
async def list_draft_models():
    """Lists all draft models in the model directory."""
    model_config = unwrap(config.get("model"), {})  # pylint: disable=redefined-outer-name
    draft_config = unwrap(model_config.get("draft"), {})
    draft_model_dir = unwrap(draft_config.get("draft_model_dir"), "models")
    draft_model_path = pathlib.Path(draft_model_dir)

    models = get_model_list(draft_model_path.resolve())
    print(models)

    return models


# Load model endpoint
@app.post("/v1/model/load", dependencies=[Depends(check_admin_key)])
async def load_model(request: Request, data: ModelLoadRequest):
    """Loads a model into the model container."""
    global MODEL_CONTAINER  # pylint: disable=global-statement

    if MODEL_CONTAINER and MODEL_CONTAINER.model:
        raise HTTPException(
            400, "A model is already loaded! Please unload it first.")

    if not data.name:
        raise HTTPException(400, "model_name not found.")

    model_config = unwrap(config.get("model"), {})  # pylint: disable=redefined-outer-name
    model_path = pathlib.Path(unwrap(model_config.get("model_dir"), "models"))  # pylint: disable=redefined-outer-name
    model_path = model_path / data.name

    load_data = data.model_dump()

    draft_config = unwrap(model_config.get("draft"), {})
    if data.draft:
        if not data.draft.draft_model_name:
            raise HTTPException(
                400, "draft_model_name was not found inside the draft object.")

        load_data["draft"]["draft_model_dir"] = unwrap(
            draft_config.get("draft_model_dir"), "models")

    if not model_path.exists():
        raise HTTPException(400,
                            "model_path does not exist. Check model_name?")

    MODEL_CONTAINER = ModelContainer(model_path.resolve(), False, **load_data)

    async def generator():
        """Generator for the loading process."""
        global MODEL_CONTAINER  # pylint: disable=global-statement

        model_type = "draft" if MODEL_CONTAINER.draft_config else "model"
        load_status = MODEL_CONTAINER.load_gen(load_progress)  # pylint: disable=redefined-outer-name

        try:
            for (module, modules) in load_status:  # pylint: disable=redefined-outer-name
                if await request.is_disconnected():
                    break

                if module == 0:
                    loading_bar: IncrementalBar = IncrementalBar(  # pylint: disable=redefined-outer-name
                        "Modules",
                        max=modules)
                elif module == modules:
                    loading_bar.next()
                    loading_bar.finish()

                    response = ModelLoadResponse(model_type=model_type,
                                                 module=module,
                                                 modules=modules,
                                                 status="finished")

                    yield get_sse_packet(response.model_dump_json())

                    # Switch to model progress if the draft model is loaded
                    if MODEL_CONTAINER.draft_config:
                        model_type = "model"
                else:
                    loading_bar.next()

                    response = ModelLoadResponse(model_type=model_type,
                                                 module=module,
                                                 modules=modules,
                                                 status="processing")

                    yield get_sse_packet(response.model_dump_json())
        except CancelledError:
            print("\nError: Model load cancelled by user. "
                  "Please make sure to run unload to free up resources.")
        except Exception as exc:  # pylint: disable=broad-except
            yield get_generator_error(str(exc))

    return StreamingResponse(generator(), media_type="text/event-stream")


# Unload model endpoint
@app.get(
    "/v1/model/unload",
    dependencies=[Depends(check_admin_key),
                  Depends(_check_model_container)])
async def unload_model():
    """Unloads the currently loaded model."""
    global MODEL_CONTAINER  # pylint: disable=global-statement

    MODEL_CONTAINER.unload()
    MODEL_CONTAINER = None


# Lora list endpoint
@app.get("/v1/loras", dependencies=[Depends(check_api_key)])
@app.get("/v1/lora/list", dependencies=[Depends(check_api_key)])
async def get_all_loras():
    """Lists all LoRAs in the lora directory."""
    model_config = unwrap(config.get("model"), {})  # pylint: disable=redefined-outer-name
    lora_config = unwrap(model_config.get("lora"), {})  # pylint: disable=redefined-outer-name
    lora_path = pathlib.Path(unwrap(lora_config.get("lora_dir"), "loras"))

    loras = get_lora_list(lora_path.resolve())

    return loras


# Currently loaded loras endpoint
@app.get(
    "/v1/lora",
    dependencies=[Depends(check_api_key),
                  Depends(_check_model_container)])
async def get_active_loras():
    """Returns the currently loaded loras."""
    active_loras = LoraList(data=list(
        map(
            lambda lora: LoraCard(id=pathlib.Path(lora.lora_path).parent.name,
                                  scaling=lora.lora_scaling * lora.lora_r /
                                  lora.lora_alpha),
            MODEL_CONTAINER.active_loras)))

    return active_loras


# Load lora endpoint
@app.post(
    "/v1/lora/load",
    dependencies=[Depends(check_admin_key),
                  Depends(_check_model_container)])
async def load_lora(data: LoraLoadRequest):
    """Loads a LoRA into the model container."""
    if not data.loras:
        raise HTTPException(400, "List of loras to load is not found.")

    model_config = unwrap(config.get("model"), {})  # pylint: disable=redefined-outer-name
    lora_config = unwrap(model_config.get("lora"), {})  # pylint: disable=redefined-outer-name
    lora_dir = pathlib.Path(unwrap(lora_config.get("lora_dir"), "loras"))  # pylint: disable=redefined-outer-name
    if not lora_dir.exists():
        raise HTTPException(
            400,
            "A parent lora directory does not exist. Check your config.yml?")

    # Clean-up existing loras if present
    if len(MODEL_CONTAINER.active_loras) > 0:
        MODEL_CONTAINER.unload(True)

    result = MODEL_CONTAINER.load_loras(lora_dir, **data.model_dump())
    return LoraLoadResponse(success=unwrap(result.get("success"), []),
                            failure=unwrap(result.get("failure"), []))


# Unload lora endpoint
@app.get(
    "/v1/lora/unload",
    dependencies=[Depends(check_admin_key),
                  Depends(_check_model_container)])
async def unload_loras():
    """Unloads the currently loaded loras."""
    MODEL_CONTAINER.unload(True)


# Encode tokens endpoint
@app.post(
    "/v1/token/encode",
    dependencies=[Depends(check_api_key),
                  Depends(_check_model_container)])
async def encode_tokens(data: TokenEncodeRequest):
    """Encodes a string into tokens."""
    raw_tokens = MODEL_CONTAINER.get_tokens(data.text, None,
                                            **data.get_params())

    # Have to use this if check otherwise Torch's tensors error out with a boolean issue
    tokens = raw_tokens[0].tolist() if raw_tokens is not None else []
    response = TokenEncodeResponse(tokens=tokens, length=len(tokens))

    return response


# Decode tokens endpoint
@app.post(
    "/v1/token/decode",
    dependencies=[Depends(check_api_key),
                  Depends(_check_model_container)])
async def decode_tokens(data: TokenDecodeRequest):
    """Decodes tokens into a string."""
    message = MODEL_CONTAINER.get_tokens(None, data.tokens,
                                         **data.get_params())
    response = TokenDecodeResponse(text=unwrap(message, ""))

    return response


# Completions endpoint
@app.post(
    "/v1/completions",
    dependencies=[Depends(check_api_key),
                  Depends(_check_model_container)])
async def generate_completion(request: Request, data: CompletionRequest):
    """Generates a completion from a prompt."""
    model_path = MODEL_CONTAINER.get_model_path()  # pylint: disable=redefined-outer-name

    if isinstance(data.prompt, list):
        data.prompt = "\n".join(data.prompt)

    if data.stream:

        async def generator():
            """Generator for the generation process."""
            try:
                new_generation = MODEL_CONTAINER.generate_gen(
                    data.prompt, **data.to_gen_params())
                for (part, prompt_tokens, completion_tokens) in new_generation:
                    if await request.is_disconnected():
                        break

                    response = create_completion_response(
                        part, prompt_tokens, completion_tokens,
                        model_path.name)

                    yield get_sse_packet(response.model_dump_json())
            except CancelledError:
                print("Error: Completion request cancelled by user.")
            except Exception as exc:  # pylint: disable=broad-except
                yield get_generator_error(str(exc))

        return StreamingResponse(generate_with_semaphore(generator),
                                 media_type="text/event-stream")

    response_text, prompt_tokens, completion_tokens = MODEL_CONTAINER.generate(
        data.prompt, **data.to_gen_params())
    response = create_completion_response(response_text, prompt_tokens,
                                          completion_tokens, model_path.name)

    return response


# Chat completions endpoint
@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(check_api_key),
                  Depends(_check_model_container)])
async def generate_chat_completion(request: Request,
                                   data: ChatCompletionRequest):
    """Generates a chat completion from a prompt."""
    if MODEL_CONTAINER.prompt_template is None:
        return HTTPException(
            422,
            "This endpoint is disabled because a prompt template is not set.")

    model_path = MODEL_CONTAINER.get_model_path()  # pylint: disable=redefined-outer-name

    if isinstance(data.messages, str):
        prompt = data.messages
    else:
        try:
            prompt = get_prompt_from_template(data.messages,
                                              MODEL_CONTAINER.prompt_template,
                                              data.add_generation_prompt)
        except KeyError:
            return HTTPException(
                400, "Could not find a Conversation from prompt template "
                f"'{MODEL_CONTAINER.prompt_template.name}'. "
                "Check your spelling?")

    if data.stream:
        const_id = f"chatcmpl-{uuid4().hex}"

        async def generator():
            """Generator for the generation process."""
            try:
                new_generation = MODEL_CONTAINER.generate_gen(
                    prompt, **data.to_gen_params())
                for (part, _, _) in new_generation:
                    if await request.is_disconnected():
                        break

                    response = create_chat_completion_stream_chunk(
                        const_id, part, model_path.name)

                    yield get_sse_packet(response.model_dump_json())

                # Yield a finish response on successful generation
                finish_response = create_chat_completion_stream_chunk(
                    const_id, finish_reason="stop")

                yield get_sse_packet(finish_response.model_dump_json())
            except CancelledError:
                print("Error: Chat completion cancelled by user.")
            except Exception as exc:  # pylint: disable=broad-except
                yield get_generator_error(str(exc))

        return StreamingResponse(generate_with_semaphore(generator),
                                 media_type="text/event-stream")

    response_text, prompt_tokens, completion_tokens = MODEL_CONTAINER.generate(
        prompt, **data.to_gen_params())
    response = create_chat_completion_response(response_text, prompt_tokens,
                                               completion_tokens,
                                               model_path.name)

    return response


if __name__ == "__main__":
    # Initialize auth keys
    load_auth_keys()

    # Load from YAML config. Possibly add a config -> kwargs conversion function
    try:
        with open('config.yml', 'r', encoding="utf8") as config_file:
            config = unwrap(yaml.safe_load(config_file), {})
    except Exception as exc:  # pylint: disable=broad-except
        print(
            "The YAML config couldn't load because of the following error:",
            f"\n\n{exc}",
            "\n\nTabbyAPI will start anyway and not parse this config file.")
        config = {}

    # Override the generation log options if given
    log_config = unwrap(config.get("logging"), {})
    if log_config:
        gen_logging.update_from_dict(log_config)

    gen_logging.broadcast_status()

    # If an initial model name is specified, create a container and load the model
    model_config = unwrap(config.get("model"), {})
    if "model_name" in model_config:
        model_path = pathlib.Path(
            unwrap(model_config.get("model_dir"), "models"))
        model_path = model_path / model_config.get("model_name")

        MODEL_CONTAINER = ModelContainer(model_path.resolve(), False,
                                         **model_config)
        load_status = MODEL_CONTAINER.load_gen(load_progress)
        for (module, modules) in load_status:
            if module == 0:
                loading_bar: IncrementalBar = IncrementalBar("Modules",
                                                             max=modules)
            elif module == modules:
                loading_bar.next()
                loading_bar.finish()
            else:
                loading_bar.next()

    # Load loras
    lora_config = unwrap(model_config.get("lora"), {})
    if "loras" in lora_config:
        lora_dir = pathlib.Path(unwrap(lora_config.get("lora_dir"), "loras"))
        MODEL_CONTAINER.load_loras(lora_dir.resolve(), **lora_config)

    network_config = unwrap(config.get("network"), {})
    uvicorn.run(app,
                host=network_config.get("host", "127.0.0.1"),
                port=network_config.get("port", 5000),
                log_level="debug")
