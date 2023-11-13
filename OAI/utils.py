from OAI.models.completions import CompletionResponse, CompletionRespChoice
from OAI.models.common import UsageStats
from typing import Optional

def create_completion_response(text: str, index: int, model_name: Optional[str]):
    # TODO: Add method to get token amounts in model for UsageStats

    choice = CompletionRespChoice(
        finish_reason="Generated",
        index = index,
        text = text
    )

    response = CompletionResponse(
        choices = [choice],
        model = model_name or ""
    )

    return response
