import pathlib
from asyncio import CancelledError
from typing import Optional

from common import model
from common.networking import get_generator_error, handle_request_disconnect

from endpoints.OAI.types.model import (
    ModelCard,
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


async def stream_model_load(
    data: ModelLoadRequest,
    model_path: pathlib.Path,
    draft_model_path: str,
):
    """Request generation wrapper for the loading process."""

    # Set the draft model path if it exists
    load_data = data.model_dump()
    if draft_model_path:
        load_data["draft"]["draft_model_dir"] = draft_model_path

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
