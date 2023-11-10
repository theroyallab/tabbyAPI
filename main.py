import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm import ModelManager
from uvicorn import run

app = FastAPI()

# Initialize the modelManager with a default model path
default_model_path = "/home/david/Models/SynthIA-7B-v2.0-5.0bpw-h6-exl2"
modelManager = ModelManager(default_model_path)
print(output)
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

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8012, reload=True)
