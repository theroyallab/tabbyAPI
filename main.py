import uvicorn
import yaml
import pathlib
from auth import check_admin_key, check_api_key, load_auth_keys
from fastapi import FastAPI, Request, HTTPException, Depends
from model import ModelContainer
from progress.bar import IncrementalBar
from sse_starlette import EventSourceResponse
from OAI.types.completions import CompletionRequest, CompletionResponse
from OAI.types.models import ModelCard, ModelList, ModelLoadRequest, ModelLoadResponse
from OAI.utils import create_completion_response, get_model_list
from typing import Optional
from utils import load_progress

app = FastAPI()

# Globally scoped variables. Undefined until initalized in main
model_container: Optional[ModelContainer] = None
config: Optional[dict] = None

@app.get("/v1/models", dependencies=[Depends(check_api_key)])
@app.get("/v1/model/list", dependencies=[Depends(check_api_key)])
async def list_models():
    model_config = config["model"]
    models = get_model_list(pathlib.Path(model_config["model_dir"] or "models"))

    return models.model_dump_json()

@app.get("/v1/model", dependencies=[Depends(check_api_key)])
async def get_current_model():
    if model_container is None or model_container.model is None:
        return HTTPException(400, "No models are loaded.")

    model_card = ModelCard(id=model_container.get_model_path().name)
    return model_card.model_dump_json()

@app.post("/v1/model/load", response_class=ModelLoadResponse, dependencies=[Depends(check_admin_key)])
async def load_model(data: ModelLoadRequest):
    if model_container and model_container.model:
        raise HTTPException(400, "A model is already loaded! Please unload it first.")

    def generator():
        global model_container
        model_config = config["model"]
        model_path = pathlib.Path(model_config["model_dir"] or "models")
        model_path = model_path / data.name
    
        model_container = ModelContainer(model_path, False, **data.model_dump())
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
                ).model_dump_json()

        yield ModelLoadResponse(
            module=module,
            modules=modules,
            status="finished"
        ).model_dump_json()

    return EventSourceResponse(generator())

@app.get("/v1/model/unload", dependencies=[Depends(check_admin_key)])
async def unload_model():
    global model_container

    if model_container is None:
        raise HTTPException(400, "No models are loaded.")

    model_container.unload()
    model_container = None

@app.post("/v1/completions", response_class=CompletionResponse, dependencies=[Depends(check_api_key)])
async def generate_completion(request: Request, data: CompletionRequest):
    if data.stream:
        async def generator():
            new_generation = model_container.generate_gen(**data.to_gen_params())
            for index, part in enumerate(new_generation):
                if await request.is_disconnected():
                    break

                response = create_completion_response(part, index, model_container.get_model_path().name)

                yield response.model_dump_json()

        return EventSourceResponse(generator())
    else:
        response_text = model_container.generate(**data.to_gen_params())
        response = create_completion_response(response_text, 0, model_container.get_model_path().name)

        return response.model_dump_json()

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
    
        print("Model successfully loaded.")

    network_config = config["network"]
    uvicorn.run(app, host=network_config["host"] or "127.0.0.1", port=network_config["port"] or 8012, log_level="debug")
