import uvicorn
import yaml
import pathlib, os
from auth import check_admin_key, check_api_key, load_auth_keys
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from model import ModelContainer
from progress.bar import IncrementalBar
from sse_starlette import EventSourceResponse
from OAI.types.completion import CompletionRequest
from OAI.types.chat_completion import ChatCompletionRequest
from OAI.types.model import ModelCard, ModelLoadRequest, ModelLoadResponse
from OAI.types.token import (
    TokenEncodeRequest,
    TokenEncodeResponse,
    TokenDecodeRequest,
    TokenDecodeResponse
)
from OAI.utils import (
    create_completion_response,
    get_model_list,
    get_chat_completion_prompt,
    create_chat_completion_response, 
    create_chat_completion_stream_chunk
)
from typing import Optional
from utils import load_progress
from uuid import uuid4

app = FastAPI()

# Globally scoped variables. Undefined until initalized in main
model_container: Optional[ModelContainer] = None
config: dict = {}

def _check_model_container():
    if model_container is None or model_container.model is None:
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
    model_config = config.get("model") or {}
    if "model_dir" in model_config:
        model_path = pathlib.Path(model_config["model_dir"])
    else:
        model_path = pathlib.Path("models")

    draft_config = model_config.get("draft") or {}
    draft_model_dir = draft_config.get("draft_model_dir")

    models = get_model_list(model_path.resolve(), draft_model_dir)

    return models

# Currently loaded model endpoint
@app.get("/v1/model", dependencies=[Depends(check_api_key), Depends(_check_model_container)])
@app.get("/v1/internal/model/info", dependencies=[Depends(check_api_key), Depends(_check_model_container)])
async def get_current_model():
    model_name = model_container.get_model_path().name
    model_card = ModelCard(id = model_name)

    return model_card

# Load model endpoint
@app.post("/v1/model/load", dependencies=[Depends(check_admin_key)])
async def load_model(data: ModelLoadRequest):
    global model_container

    if model_container and model_container.model:
        raise HTTPException(400, "A model is already loaded! Please unload it first.")

    if not data.name:
        raise HTTPException(400, "model_name not found.")

    model_config = config.get("model") or {}
    model_path = pathlib.Path(model_config.get("model_dir") or "models")
    model_path = model_path / data.name

    load_data = data.dict()
    if data.draft and "draft" in model_config:
        draft_config = model_config.get("draft") or {}

        if not data.draft.draft_model_name:
            raise HTTPException(400, "draft_model_name was not found inside the draft object.")

        load_data["draft_model_dir"] = draft_config.get("draft_model_dir") or "models"

    if not model_path.exists():
        raise HTTPException(400, "model_path does not exist. Check model_name?")

    model_container = ModelContainer(model_path.resolve(), False, **load_data)

    def generator():
        model_type = "draft" if model_container.draft_enabled else "model"
        load_status = model_container.load_gen(load_progress)

        for (module, modules) in load_status:
            if module == 0:
                loading_bar: IncrementalBar = IncrementalBar("Modules", max = modules)        
            elif module == modules:
                loading_bar.next()
                loading_bar.finish()

                response = ModelLoadResponse(
                    model_type=model_type,
                    module=module,
                    modules=modules,
                    status="finished"
                )

                yield response.json(ensure_ascii=False)

                if model_container.draft_enabled:
                    model_type = "model"
            else:
                loading_bar.next()
 
                response = ModelLoadResponse(
                    model_type=model_type,
                    module=module,
                    modules=modules,
                    status="processing"
                )

                yield response.json(ensure_ascii=False)

    return EventSourceResponse(generator())

# Unload model endpoint
@app.get("/v1/model/unload", dependencies=[Depends(check_admin_key), Depends(_check_model_container)])
async def unload_model():
    global model_container

    model_container.unload()
    model_container = None

# Encode tokens endpoint
@app.post("/v1/token/encode", dependencies=[Depends(check_api_key), Depends(_check_model_container)])
async def encode_tokens(data: TokenEncodeRequest):
    raw_tokens = model_container.get_tokens(data.text, None, **data.get_params())

    # Have to use this if check otherwise Torch's tensors error out with a boolean issue
    tokens = raw_tokens[0].tolist() if raw_tokens is not None else []
    response = TokenEncodeResponse(tokens=tokens, length=len(tokens))

    return response

# Decode tokens endpoint
@app.post("/v1/token/decode", dependencies=[Depends(check_api_key), Depends(_check_model_container)])
async def decode_tokens(data: TokenDecodeRequest):
    message = model_container.get_tokens(None, data.tokens, **data.get_params())
    response = TokenDecodeResponse(text = message or "")

    return response

# Completions endpoint
@app.post("/v1/completions", dependencies=[Depends(check_api_key), Depends(_check_model_container)])
async def generate_completion(request: Request, data: CompletionRequest):
    model_path = model_container.get_model_path()

    if isinstance(data.prompt, list):
        data.prompt = "\n".join(data.prompt)

    if data.stream:
        async def generator():
            new_generation = model_container.generate_gen(data.prompt, **data.to_gen_params())
            for part in new_generation:
                if await request.is_disconnected():
                    break

                response = create_completion_response(part, model_path.name)

                yield response.json(ensure_ascii=False)

        return EventSourceResponse(generator())
    else:
        response_text = model_container.generate(data.prompt, **data.to_gen_params())
        response = create_completion_response(response_text, model_path.name)

        return response

# Chat completions endpoint
@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key), Depends(_check_model_container)])
async def generate_chat_completion(request: Request, data: ChatCompletionRequest):
    model_path = model_container.get_model_path()

    if isinstance(data.messages, str):
        prompt = data.messages
    else:
        prompt = get_chat_completion_prompt(model_path.name, data.messages)

    if data.stream:
        const_id = f"chatcmpl-{uuid4().hex}"
        async def generator():
            new_generation = model_container.generate_gen(prompt, **data.to_gen_params())
            for part in new_generation:
                if await request.is_disconnected():
                    break

                response = create_chat_completion_stream_chunk(
                    const_id,
                    part,
                    model_path.name
                )

                yield response.json(ensure_ascii=False)

        return EventSourceResponse(generator())
    else:
        response_text = model_container.generate(prompt, **data.to_gen_params())
        response = create_chat_completion_response(response_text, model_path.name)

        return response

if __name__ == "__main__":
    # Initialize auth keys
    load_auth_keys()

    # Load from YAML config. Possibly add a config -> kwargs conversion function
    try:
        with open('config.yml', 'r') as config_file:
            config = yaml.safe_load(config_file) or {}
    except Exception as e:
        print(
            "The YAML config couldn't load because of the following error:",
            f"\n\n{e}",
            "\n\nTabbyAPI will start anyway and not parse this config file."
        )
        config = {}

    # If an initial model name is specified, create a container and load the model

    model_config = config.get("model") or {}
    if "model_name" in model_config:
        model_path = pathlib.Path(model_config.get("model_dir") or "models")
        model_path = model_path / model_config.get("model_name")
    
        model_container = ModelContainer(model_path.resolve(), False, **model_config)
        load_status = model_container.load_gen(load_progress)
        for (module, modules) in load_status:
            if module == 0:
                loading_bar: IncrementalBar = IncrementalBar("Modules", max = modules)        
            elif module == modules:
                loading_bar.next()
                loading_bar.finish()
            else:
                loading_bar.next()

    network_config = config.get("network") or {}
    uvicorn.run(
        app,
        host=network_config.get("host", "127.0.0.1"),
        port=network_config.get("port", 5000),
        log_level="debug"
    )
