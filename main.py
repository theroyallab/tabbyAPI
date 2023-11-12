import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import ModelContainer
from progress.bar import IncrementalBar

app = FastAPI()

# Initialize a model container. This can be undefined at any period of time
model_container: ModelContainer = None

class TextRequest(BaseModel):
    model: str = None  # Make the "model" field optional with a default value of None
    prompt: str
    max_tokens: int = 200
    temperature: float = 1
    top_p: float = 0.9
    seed: int = 10
    stream: bool = False
    token_repetition_penalty: float = 1.0
    stop: list = None

class TextResponse(BaseModel):
    response: str
    generation_time: str

# TODO: Currently broken
@app.post("/generate-text", response_model=TextResponse)
def generate_text(request: TextRequest):
    global modelManager
    try:
        prompt = request.prompt  # Get the prompt from the request
        user_message = prompt  # Assuming that prompt is equivalent to the user's message
        output, generation_time = modelManager.generate_text(prompt=user_message)
        return {"response": output, "generation_time": generation_time}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

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

    # Reload is for dev purposes ONLY!
    uvicorn.run("main:app", host="0.0.0.0", port=8012, log_level="debug", reload=True)
