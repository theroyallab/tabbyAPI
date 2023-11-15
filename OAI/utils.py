import pathlib
from OAI.types.completions import CompletionResponse, CompletionRespChoice
from OAI.types.common import UsageStats
from OAI.types.model import ModelList, ModelCard
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

def get_model_list(model_path: pathlib.Path):
    model_card_list = ModelList()
    for path in model_path.parent.iterdir():
        if path.is_dir():
            model_card = ModelCard(id = path.name)
            model_card_list.data.append(model_card)

    return model_card_list
