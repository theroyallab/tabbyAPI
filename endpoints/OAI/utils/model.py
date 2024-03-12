import pathlib
from typing import Optional

from endpoints.OAI.types.model import ModelCard, ModelList


def get_model_list(model_path: pathlib.Path, draft_model_path: Optional[str] = None):
    """Get the list of models from the provided path."""

    # Convert the provided draft model path to a pathlib path for
    # equality comparisons
    if draft_model_path:
        draft_model_path = pathlib.Path(draft_model_path).resolve()

    model_card_list = ModelList()
    for path in model_path.iterdir():
        # Don't include the draft models path
        if path.is_dir() and path != draft_model_path:
            model_card = ModelCard(id=path.name)
            model_card_list.data.append(model_card)  # pylint: disable=no-member

    return model_card_list
