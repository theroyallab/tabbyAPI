# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm import ModelManager
from uvicorn import run

app = FastAPI()

# Example: Using a different model directory
modelManager = ModelManager("/home/david/Models/SynthIA-7B-v2.0-5.0bpw-h6-exl2")

class TextRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float

class TextResponse(BaseModel):
    response: str
    generation_time: str

@app.post("/generate-text", response_model=TextResponse)
def generate_text(request: TextRequest):
    try:
        #model_path = request.model  # You can use this path to load a specific model if needed
        messages = request.messages
        #temperature = request.temperature

        # Assuming you need to extract the user's message from the messages list
        user_message = next(msg["content"] for msg in messages if msg["role"] == "user")

        # You can then use user_message as the prompt for generation
        output, generation_time = modelManager.generate_text(user_message)
        return {"response": output, "generation_time": generation_time}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8012, reload=True)
