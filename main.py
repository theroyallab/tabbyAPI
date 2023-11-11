import os
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import ModelContainer
from utils import add_args

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

# Debug progress check
def progress(module, modules):
    print(f"Loaded {module}/{modules} modules")
    yield

if __name__ == "__main__":
    # Convert this parser to use a YAML config
    parser = argparse.ArgumentParser(description = "TabbyAPI - An API server for exllamav2")
    add_args(parser)
    args = parser.parse_args()

    # If an initial model dir is specified, create a container and load the model
    if args.model_dir:
        model_container = ModelContainer(args.model_dir, False, **vars(args))
        print("Loading an initial model...")
        model_container.load(progress)
        print("Model successfully loaded.")

    # Reload is for dev purposes ONLY!
    uvicorn.run("main:app", host="0.0.0.0", port=8012, log_level="debug", reload=True)
