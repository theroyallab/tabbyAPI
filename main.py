import uvicorn
import yaml
import pathlib
from auth import check_admin_key, check_api_key, load_auth_keys
from fastapi import FastAPI, Request, HTTPException, Depends
from model import ModelContainer
from progress.bar import IncrementalBar
from sse_starlette import EventSourceResponse
from OAI.types.completion import CompletionRequest
from OAI.types.chat_completion import ChatCompletionRequest
from OAI.types.model import ModelCard, ModelLoadRequest, ModelLoadResponse
from OAI.types.token import TokenEncodeRequest, TokenEncodeResponse, TokenDecodeRequest, TokenDecodeResponse
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
config: Optional[dict] = None

@app.get("/v1/models", dependencies=[Depends(check_api_key)])
@app.get("/v1/model/list", dependencies=[Depends(check_api_key)])
async def list_models():
    model_config = config["model"]
    models = get_model_list(pathlib.Path(model_config["model_dir"] or "models"))

    return models.json()

@app.get("/v1/model", dependencies=[Depends(check_api_key)])
async def get_current_model():
    if model_container is None or model_container.model is None:
        return HTTPException(400, "No models are loaded.")

    model_card = ModelCard(id=model_container.get_model_path().name)
    return model_card.json()

@app.post("/v1/model/load", dependencies=[Depends(check_admin_key)])
async def load_model(data: ModelLoadRequest):
    if model_container and model_container.model:
        raise HTTPException(400, "A model is already loaded! Please unload it first.")

    def generator():
        global model_container
        model_config = config["model"]
        model_path = pathlib.Path(model_config["model_dir"] or "models")
        model_path = model_path / data.name

        model_container = ModelContainer(model_path, False, **data.dict())
        load_status = model_container.load_gen(load_progress)
        for (module, modules) in load_status:
            if module == 0:
                loading_bar: IncrementalBar = IncrementalBar("Modules", max = modules)        
            elif module == modules:
                loading_bar.next()
                loading_bar.finish()
            else:
                loading_bar.next()
 
                yield ModelLoadResponse(
                    module=module,
                    modules=modules,
                    status="processing"
                ).json()

        yield ModelLoadResponse(
            module=module,
            modules=modules,
            status="finished"
        ).json()

    return EventSourceResponse(generator())

@app.get("/v1/model/unload", dependencies=[Depends(check_admin_key)])
async def unload_model():
    global model_container

    if model_container is None:
        raise HTTPException(400, "No models are loaded.")

    model_container.unload()
    model_container = None

@app.post("/v1/token/encode", dependencies=[Depends(check_api_key)])
async def encode_tokens(data: TokenEncodeRequest):
    tokens = model_container.get_tokens(data.text, None, **data.get_params())[0].tolist()
    response = TokenEncodeResponse(tokens=tokens, length=len(tokens))

    return response.json()

@app.post("/v1/token/decode", dependencies=[Depends(check_api_key)])
async def decode_tokens(data: TokenDecodeRequest):
    message = model_container.get_tokens(None, data.tokens, **data.get_params())
    response = TokenDecodeResponse(text=message)

    return response.json()

@app.post("/v1/completions", dependencies=[Depends(check_api_key)])
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

                yield response.json()

        return EventSourceResponse(generator())
    else:
        response_text = model_container.generate(data.prompt, **data.to_gen_params())
        response = create_completion_response(response_text, model_path.name)

        return response.json()

@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
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

                yield response.json()

        return EventSourceResponse(generator())
    else:
        response_text = model_container.generate(prompt, **data.to_gen_params())
        response = create_chat_completion_response(response_text, model_path.name)

        return response.json()

if __name__ == "__main__":
    # Initialize auth keys
    load_auth_keys()

    # Load from YAML config. Possibly add a config -> kwargs conversion function
    with open('config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # If an initial model name is specified, create a container and load the model
    model_config = config["model"]
    if model_config["model_name"]:
        model_path = pathlib.Path(model_config["model_dir"] or "models")
        model_path = model_path / model_config["model_name"]
    
        model_container = ModelContainer(model_path, False, **model_config)
        load_status = model_container.load_gen(load_progress)
        for (module, modules) in load_status:
            if module == 0:
                loading_bar: IncrementalBar = IncrementalBar("Modules", max = modules)        
            elif module == modules:
                loading_bar.next()
                loading_bar.finish()
            else:
                loading_bar.next()

    network_config = config["network"]
    uvicorn.run(app, host=network_config["host"] or "127.0.0.1", port=network_config["port"] or 8012, log_level="debug")
