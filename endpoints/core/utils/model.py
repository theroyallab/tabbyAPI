import asyncio
import io
import pathlib
from asyncio import CancelledError
from typing import Optional

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from common import model
from common.networking import get_generator_error, handle_request_disconnect
from common.model_meta import is_model_dir, read_model_meta
from common.tabby_config import config
from common.utils import deep_merge_dict
from endpoints.core.types.model import (
    DraftModelLoadRequest,
    ModelCard,
    ModelList,
    ModelLoadConfigResponse,
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
    directory whose `config.json` names the model architecture is a checkpoint
    and is not searched any deeper, which keeps a checkpoint's own subfolders
    from being listed as models. Folders without one are never listed.
    """

    # Convert the provided draft model path to a pathlib path for
    # equality comparisons
    if draft_model_path:
        draft_model_path = pathlib.Path(draft_model_path).resolve()

    root = model_path.resolve()
    model_card_list = ModelList()

    # Draft models are only kept out when they have their own folder inside
    # model_dir. A draft_model_dir that is model_dir (or holds it) would
    # otherwise hide every model.
    if draft_model_path is not None and (
        draft_model_path == root or draft_model_path in root.parents
    ):
        draft_model_path = None

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

            if is_model_dir(path):
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

    # Make sure the model container exists. Each kind of model is listed from
    # its own directory, so its id is relative to that directory.
    match model_type:
        case "model":
            model_root = config.model.model_dir
            if model.container:
                model_path = model.container.model_dir
                n_ctx = model.container.max_seq_len
        case "draft":
            model_root = config.draft_model.draft_model_dir
            if model.container:
                model_path = model.container.draft_model_dir
        case "embedding":
            model_root = config.embeddings.embedding_model_dir
            if model.embeddings_container:
                model_path = model.embeddings_container.model_dir

    if model_path:
        meta = read_model_meta(model_path, n_ctx=n_ctx, include_size=True)
        root = pathlib.Path(model_root).resolve()
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

    # Only the options the client set: they override the model folder's
    # tabby_config.yml, so a field's default must not stand in for one
    load_data = data.model_dump(exclude_unset=True, exclude_none=True)

    # Set the draft model directory
    load_data.setdefault("draft_model", {})["draft_model_dir"] = config.draft_model.draft_model_dir

    # Drive the load in a detached task and observe it through a queue,
    # so a client disconnect doesn't cancel a load in progress
    progress_queue: asyncio.Queue = asyncio.Queue()

    async def run_load():
        try:
            load_status = model.load_model_gen(
                model_path, request_wins=True, skip_wait=data.skip_queue, **load_data
            )
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
                yield get_generator_error(str(progress), exception=progress)
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


# Where a load setting comes from, as reported by get_model_load_config
SOURCE_MODEL_CONFIG = model.MODEL_CONFIG_FILE
SOURCE_USE_AS_DEFAULT = "use_as_default"

# Load request fields that aren't settings of the model itself
NON_SETTING_FIELDS = {"model_name", "draft_model", "skip_queue"}


async def get_model_load_config(model_name: str, model_path: pathlib.Path):
    """
    The load options that apply to a model when a request leaves them out: the
    use_as_default keys of config.yml, overridden by the model folder's
    tabby_config.yml. Only fields a load request can set are included, so the
    result can be edited and sent back as one.
    """

    file_config = await model.read_model_config_file(model_path)
    model_fields = set(ModelLoadRequest.model_fields) - NON_SETTING_FIELDS
    draft_fields = set(DraftModelLoadRequest.model_fields)

    settings, sources, ignored = {}, {}, []
    draft_settings, draft_sources = {}, {}

    def merge(target: dict, target_sources: dict, key, value, source: str):
        if value is None:
            return
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            # Nested options (e.g. sampling) merge the same way as in a load
            value = deep_merge_dict(target[key], value, copy=True)
        target[key] = value
        target_sources[key] = source

    def add_layer(model_options: dict, draft_options: dict, source: str):
        for key, value in model_options.items():
            if key in model_fields:
                merge(settings, sources, key, value, source)
            elif source == SOURCE_MODEL_CONFIG and key != "draft_model":
                ignored.append(key)
        for key, value in draft_options.items():
            if key in draft_fields:
                merge(draft_settings, draft_sources, key, value, source)
            elif source == SOURCE_MODEL_CONFIG:
                ignored.append(f"draft_model.{key}")

    defaults = model.use_as_default_settings()
    add_layer(defaults, defaults.pop("draft_model"), SOURCE_USE_AS_DEFAULT)
    if file_config is not None:
        add_layer(file_config["model"], file_config["draft_model"], SOURCE_MODEL_CONFIG)

    if draft_settings:
        settings["draft_model"] = draft_settings
        sources["draft_model"] = draft_sources

    return ModelLoadConfigResponse(
        model_name=model_name,
        has_tabby_config=file_config is not None,
        settings=settings,
        sources=sources,
        ignored=ignored,
        yaml=load_settings_yaml(model_name, settings, sources, ignored),
    )


def load_settings_yaml(model_name: str, settings: dict, sources: dict, ignored: list) -> str:
    """Load settings as YAML in the shape of a load request, each line noting its source."""

    def node(value):
        if isinstance(value, dict):
            return CommentedMap((key, node(item)) for key, item in value.items())
        if isinstance(value, list):
            # Lists such as gpu_split read best on one line
            seq = CommentedSeq(node(item) for item in value)
            seq.fa.set_flow_style()
            return seq
        return value

    header = [
        f"# Load options for {model_name}",
        f"# From the model's {SOURCE_MODEL_CONFIG} and config.yml (use_as_default).",
        "# Anything not listed here uses the backend default.",
    ]
    if ignored:
        header.append(
            f"# Also in {SOURCE_MODEL_CONFIG}, but not settable per request: " + ", ".join(ignored)
        )
    if not settings:
        header.append("# No settings apply, so this model loads with defaults.")
        return "\n".join(header) + "\n"

    yaml = YAML()
    yaml.width = 4096
    stream = io.StringIO()
    yaml.dump(node(settings), stream)
    lines = stream.getvalue().splitlines()

    # Note each setting's source after its key line, in one aligned column.
    # Settings are top-level keys, or keys one level into draft_model.
    notes = {}
    section = None
    for index, line in enumerate(lines):
        indent = len(line) - len(line.lstrip(" "))
        key = line.strip().split(":", 1)[0]
        if indent == 0:
            section = key
            source = sources.get(key)
        elif section == "draft_model" and indent == 2:
            source = sources["draft_model"].get(key)
        else:
            continue
        if isinstance(source, str):
            notes[index] = source

    column = max(len(lines[index]) for index in notes) + 2 if notes else 0
    for index, source in notes.items():
        lines[index] = lines[index].ljust(column) + f"# {source}"

    return "\n".join(header + lines) + "\n"
