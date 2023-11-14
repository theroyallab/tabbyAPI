import uvicorn
import yaml
from fastapi import FastAPI, Request
from model import ModelContainer
from progress.bar import IncrementalBar
from sse_starlette import EventSourceResponse
from OAI.models.completions import CompletionRequest, CompletionResponse
from OAI.models.models import ModelCard, ModelList
from OAI.utils import create_completion_response, get_model_list

app = FastAPI()

# Initialize a model container. This can be undefined at any period of time
model_container: ModelContainer = None

@app.get("/v1/models")
@app.get("/v1/model/list")
async def list_models():
    models = get_model_list(model_container.get_model_path())

    return models.model_dump_json()

@app.get("/v1/model")
async def get_current_model():
    return ModelCard(id = model_container.get_model_path().name)

@app.post("/v1/completions", response_class=CompletionResponse)
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


# Wrapper callback for load progress
def load_progress(module, modules):
    yield module, modules

if __name__ == "__main__":
    # Load from YAML config. Possibly add a config -> kwargs conversion function
    with open('config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # If an initial model name is specified, create a container and load the model
    if config["model_name"]:
        model_path = f"{config['model_dir']}/{config['model_name']}" if config['model_dir'] else f"models/{config['model_name']}"

        model_container = ModelContainer(model_path, False, **config)
        load_status = model_container.load_gen(load_progress)
        for (module, modules) in load_status:
            if module == 0:
                loading_bar: IncrementalBar = IncrementalBar("Modules", max = modules)        
            else:
                loading_bar.next()

                if module == modules:
                    loading_bar.finish()

        print("Model successfully loaded.")

    uvicorn.run(app, host="0.0.0.0", port=8012, log_level="debug")
