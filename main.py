import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm import ModelManager
from uvicorn import run

app = FastAPI()

# Initialize the modelManager with a default model path
default_model_path = "~/Models/SynthIA-7B-v2.0-5.0bpw-h6-exl2"
modelManager = ModelManager(default_model_path)

class TextRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float

class TextResponse(BaseModel):
    response: str
    generation_time: str

@app.post("/generate-text", response_model=TextResponse)
def generate_text(request: TextRequest):
    global modelManager
    try:
        model_path = request.model

        if model_path and model_path != modelManager.config.model_path:
            # Check if the specified model path exists
            if not os.path.exists(model_path):
                raise HTTPException(status_code=400, detail="Model path does not exist")

            # Reinitialize the modelManager with the new model path
            modelManager = ModelManager(model_path)

        messages = request.messages
        user_message = next(msg["content"] for msg in messages if msg["role"] == "user")

        output, generation_time = modelManager.generate_text(user_message)
        return {"response": output, "generation_time": generation_time}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8012, reload=True)
