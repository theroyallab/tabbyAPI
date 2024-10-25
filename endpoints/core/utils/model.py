import pathlib
from asyncio import CancelledError
from typing import Optional

from common import model
from common.networking import get_generator_error, handle_request_disconnect
from common.tabby_config import config
from common.utils import unwrap
from endpoints.core.types.model import (
    ModelCard,
    ModelCardParameters,
    ModelList,
    ModelLoadRequest,
    ModelLoadResponse,
)


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


async def get_current_model_list(model_type: str = "model"):
    """
    Gets the current model in list format and with path only.

    Unified for fetching both models and embedding models.
    """

    current_models = []
    model_path = None

    # Make sure the model container exists
    match model_type:
        case "model":
            if model.container:
                model_path = model.container.model_dir
        case "draft":
            if model.container:
                model_path = model.container.draft_model_dir
        case "embedding":
            if model.embeddings_container:
                model_path = model.embeddings_container.model_dir

    if model_path:
        current_models.append(ModelCard(id=model_path.name))

    return ModelList(data=current_models)


def get_current_model():
    """Gets the current model with all parameters."""

    model_params = model.container.get_model_parameters()
    draft_model_params = model_params.pop("draft", {})

    if draft_model_params:
        model_params["draft"] = ModelCard(
            id=unwrap(draft_model_params.get("name"), "unknown"),
            parameters=ModelCardParameters.model_validate(draft_model_params),
        )
    else:
        draft_model_params = None

    model_card = ModelCard(
        id=unwrap(model_params.pop("name", None), "unknown"),
        parameters=ModelCardParameters.model_validate(model_params),
        logging=config.logging,
    )

    if draft_model_params:
        draft_card = ModelCard(
            id=unwrap(draft_model_params.pop("name", None), "unknown"),
            parameters=ModelCardParameters.model_validate(draft_model_params),
        )

        model_card.parameters.draft = draft_card

    return model_card


async def stream_model_load(
    data: ModelLoadRequest,
    model_path: pathlib.Path,
    draft_model_path: str,
):
    """Request generation wrapper for the loading process."""

    # Get trimmed load data
    load_data = data.model_dump(exclude_none=True)

    # Set the draft model path if it exists
    if draft_model_path:
        load_data["draft_model"]["draft_model_dir"] = draft_model_path

    load_status = model.load_model_gen(
        model_path, skip_wait=data.skip_queue, **load_data
    )
    try:
        async for module, modules, model_type in load_status:
            if module != 0:
                response = ModelLoadResponse(
                    model_type=model_type,
                    module=module,
                    modules=modules,
                    status="processing",
                )

                yield response.model_dump_json()

            if module == modules:
                response = ModelLoadResponse(
                    model_type=model_type,
                    module=module,
                    modules=modules,
                    status="finished",
                )

                yield response.model_dump_json()
    except CancelledError:
        # Get out if the request gets disconnected

        handle_request_disconnect(
            "Model load cancelled by user. "
            "Please make sure to run unload to free up resources."
        )
    except Exception as exc:
        yield get_generator_error(str(exc))
