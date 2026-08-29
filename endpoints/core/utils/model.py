import asyncio
import json
import pathlib
from asyncio import CancelledError
from typing import Optional

from common import model
from common.networking import get_generator_error, handle_request_disconnect
from common.tabby_config import config
from endpoints.core.types.model import (
    ModelCard,
    ModelCardMeta,
    ModelCardParameters,
    ModelEntry,
    ModelList,
    ModelLoadRequest,
    ModelLoadResponse,
)


def read_model_meta(model_dir: pathlib.Path, n_ctx: Optional[int] = None) -> ModelCardMeta:
    """llama.cpp-style `meta` block, read from the model's config.json."""
    try:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf8"))
    except Exception:
        cfg = {}
    # Multimodal configs nest text settings under text_config; let those win.
    cfg = {**cfg, **cfg.get("text_config", {})}
    train = cfg.get("max_position_embeddings") or 0
    try:
        size = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
    except Exception:
        size = 0
    return ModelCardMeta(
        n_vocab=cfg.get("vocab_size") or 0,
        n_ctx=n_ctx or train,
        n_ctx_train=train,
        n_embd=cfg.get("hidden_size") or 0,
        size=size,
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
            meta = read_model_meta(path)
            model_card = ModelCard(id=path.name, meta=meta)
            if meta.n_ctx_train:
                model_card.parameters = ModelCardParameters(max_seq_len=meta.n_ctx_train)
            model_card_list.data.append(model_card)  # pylint: disable=no-member

    return model_card_list


async def get_current_model_list(model_type: str = "model"):
    """
    Gets the current model in list format and with path only.

    Unified for fetching both models and embedding models.
    """

    current_models = []
    model_path = None
    context_length = None

    # Make sure the model container exists
    match model_type:
        case "model":
            if model.container:
                model_path = model.container.model_dir
                context_length = model.container.max_seq_len
        case "draft":
            if model.container:
                model_path = model.container.draft_model_dir
        case "embedding":
            if model.embeddings_container:
                model_path = model.embeddings_container.model_dir

    if model_path:
        model_card = ModelCard(id=model_path.name, meta=read_model_meta(model_path, context_length))
        if context_length is not None:
            model_card.parameters = ModelCardParameters(max_seq_len=context_length)
        current_models.append(model_card)

    return ModelList(data=current_models)


def get_current_model():
    """Gets the current model with all parameters."""

    model_card = model.container.model_info()

    return model_card


def apply_llama_compat(model_list: ModelList) -> ModelList:
    """Mirror `data` into an Ollama-style `models` array (llama.cpp compat)."""

    model_list.models = []
    for card in model_list.data:
        card.aliases = [card.id]
        vision = bool(card.parameters and card.parameters.use_vision)
        model_list.models.append(
            ModelEntry(
                name=card.id,
                model=card.id,
                size=str(card.meta.size) if card.meta else "",
                capabilities=["completion"] + (["multimodal"] if vision else []),
            )
        )

    return model_list


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
