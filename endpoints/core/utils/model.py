import asyncio
import pathlib
from asyncio import CancelledError
from typing import Optional

from common import model
from common.networking import get_generator_error, handle_request_disconnect
from common.model_meta import read_model_meta
from common.tabby_config import config
from endpoints.core.types.model import (
    ModelCard,
    ModelList,
    ModelLoadRequest,
    ModelLoadResponse,
)


# How deep below `model_dir` to look for a model. Guards against following
# pointless trees (dataset dumps, backup folders) instead of checkpoints.
MAX_MODEL_SEARCH_DEPTH = 4


def relative_model_id(directory: pathlib.Path, root: pathlib.Path) -> str:
    """
    The id a model is advertised with: its path relative to `model_dir`.

    This is the same string `/v1/model/load` and inline model loading take, so
    every advertised model can actually be loaded. For a model directly inside
    `model_dir` the result is the directory name, exactly as before.
    """

    try:
        return directory.relative_to(root).as_posix()
    except ValueError:
        return directory.name


def get_model_list(model_path: pathlib.Path, draft_model_path: Optional[str] = None):
    """Get the list of models from the provided path.

    Nested directories are searched, so quant-style layouts such as
    `<model>/exl3/<bpw>` are advertised and not just the parent folder. A
    directory that holds a `config.json` is treated as a checkpoint and is not
    searched any deeper, which keeps a checkpoint's own subfolders from being
    listed as models.
    """

    # Convert the provided draft model path to a pathlib path for
    # equality comparisons
    if draft_model_path:
        draft_model_path = pathlib.Path(draft_model_path).resolve()

    root = model_path.resolve()
    model_card_list = ModelList()

    def in_draft_tree(directory: pathlib.Path) -> bool:
        """True for the draft model directory and anything inside it."""

        if draft_model_path is None:
            return False
        resolved = directory.resolve()
        return resolved == draft_model_path or draft_model_path in resolved.parents

    def search(directory: pathlib.Path, depth: int):
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name.casefold())
        except OSError:
            return

        for path in entries:
            # Hidden directories are never checkpoints. HuggingFace keeps its
            # download cache in `.cache`, which holds snapshot `config.json`
            # files that would otherwise be advertised as models.
            if not path.is_dir() or path.name.startswith(".") or in_draft_tree(path):
                continue

            if (path / "config.json").exists():
                model_card = ModelCard(id=relative_model_id(path, root), meta=read_model_meta(path))
                model_card_list.data.append(model_card)  # pylint: disable=no-member
                continue

            if depth < MAX_MODEL_SEARCH_DEPTH:
                search(path, depth + 1)

    search(model_path, 1)

    return model_card_list


async def get_current_model_list(model_type: str = "model"):
    """
    Gets the current model in list format and with path only.

    Unified for fetching both models and embedding models.
    """

    current_models = []
    model_path = None
    n_ctx = None

    # Make sure the model container exists
    match model_type:
        case "model":
            if model.container:
                model_path = model.container.model_dir
                n_ctx = model.container.max_seq_len
        case "draft":
            if model.container:
                model_path = model.container.draft_model_dir
        case "embedding":
            if model.embeddings_container:
                model_path = model.embeddings_container.model_dir

    if model_path:
        meta = read_model_meta(model_path, n_ctx=n_ctx, include_size=True)
        root = pathlib.Path(config.model.model_dir).resolve()
        current_models.append(
            ModelCard(id=relative_model_id(model_path.resolve(), root), meta=meta)
        )

    return ModelList(data=current_models)


def get_current_model():
    """Gets the current model with all parameters."""

    model_card = model.container.model_info()

    return model_card


def get_dummy_models():
    if config.model.dummy_model_names:
        return [ModelCard(id=dummy_id) for dummy_id in config.model.dummy_model_names]
    else:
        return [ModelCard(id="gpt-3.5-turbo")]


# Keep strong references to detached load tasks; asyncio only holds weak ones
_load_tasks: set = set()


async def stream_model_load(
    data: ModelLoadRequest,
    model_path: pathlib.Path,
):
    """Request generation wrapper for the loading process."""

    # Get trimmed load data
    load_data = data.model_dump(exclude_none=True)

    # Set the draft model directory
    load_data.setdefault("draft_model", {})["draft_model_dir"] = config.draft_model.draft_model_dir

    # Drive the load in a detached task and observe it through a queue,
    # so a client disconnect doesn't cancel a load in progress
    progress_queue: asyncio.Queue = asyncio.Queue()

    async def run_load():
        try:
            load_status = model.load_model_gen(model_path, skip_wait=data.skip_queue, **load_data)
            async for progress in load_status:
                progress_queue.put_nowait(progress)

            progress_queue.put_nowait(None)
        except Exception as exc:
            progress_queue.put_nowait(exc)

    load_task = asyncio.create_task(run_load())
    _load_tasks.add(load_task)
    load_task.add_done_callback(_load_tasks.discard)

    try:
        while True:
            progress = await progress_queue.get()

            if progress is None:
                break

            if isinstance(progress, Exception):
                yield get_generator_error(str(progress))
                break

            module, modules, model_type = progress
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
        # The client disconnected, but the load task keeps running.
        # A repeated request for the same model returns once this load finishes.

        handle_request_disconnect("Model load request disconnected. The load will continue.")
