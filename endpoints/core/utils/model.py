import asyncio
import io
import os
import pathlib
import shutil
from asyncio import CancelledError
from typing import Optional

import aiofiles
from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.tokens import CommentToken

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
    ModelLoadConfigSaveResponse,
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

    # The loaded model's entry also says how it is loaded, as the current-model
    # list does, so a client reading the full list can size its context window
    loaded_path = None
    loaded_n_ctx = None
    if model.container:
        loaded_path = pathlib.Path(model.container.model_dir).resolve()
        loaded_n_ctx = model.container.max_seq_len

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
                if loaded_path is not None and path.resolve() == loaded_path:
                    model_card = ModelCard(
                        id=relative_model_id(path, root),
                        meta=read_model_meta(path, n_ctx=loaded_n_ctx, include_size=True),
                        max_model_len=loaded_n_ctx,
                    )
                else:
                    model_card = ModelCard(
                        id=relative_model_id(path, root), meta=read_model_meta(path)
                    )
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
            ModelCard(
                id=relative_model_id(model_path.resolve(), root), meta=meta, max_model_len=n_ctx
            )
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


def yaml_node(value):
    """A value for ruamel to write, with lists such as gpu_split on one line."""

    if isinstance(value, dict):
        return CommentedMap((key, yaml_node(item)) for key, item in value.items())
    if isinstance(value, list):
        seq = CommentedSeq(yaml_node(item) for item in value)
        seq.fa.set_flow_style()
        return seq
    return value


def load_settings_yaml(model_name: str, settings: dict, sources: dict, ignored: list) -> str:
    """Load settings as YAML in the shape of a load request, each line noting its source."""

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
    yaml.dump(yaml_node(settings), stream)
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


MODEL_CONFIG_BACKUP_FILE = "tabby_config.bak"


def explicit_settings(settings: dict, defaults: dict, existing: dict) -> dict:
    """
    The part of some load settings that use_as_default doesn't already supply:
    values that differ from the default or have none, and anything the model's
    tabby_config.yml already sets, even to the default value. Nested options
    (sampling, say) are compared key by key. Unset (null) values are dropped.
    """

    result = {}
    for key, value in settings.items():
        if value is None:
            continue

        default = defaults.get(key)
        if isinstance(value, dict) and isinstance(default, dict):
            in_file = existing.get(key)
            value = explicit_settings(value, default, in_file if isinstance(in_file, dict) else {})
            if value:
                result[key] = value
        elif key in existing or key not in defaults or value != default:
            result[key] = value

    return result


def move_trailing_blank_lines(section: CommentedMap, old_last, new_last):
    """
    Keep the blank lines that ended a section of a YAML file at its end once
    keys are added. ruamel keeps them with the last key, so the new keys would
    otherwise come after them. Comment lines stay with the key they follow.
    """

    if old_last is None or new_last == old_last:
        return

    comment = section.ca.items.get(old_last)
    token = comment[2] if comment and len(comment) > 2 else None
    if token is None:
        return

    text = token.value.rstrip("\n")
    blank_lines = len(token.value) - len(text) - 1
    if blank_lines > 0:
        token.value = text + "\n"
        trailing = CommentToken("\n" * (blank_lines + 1), token.start_mark)
        section.ca.items[new_last] = [None, None, trailing, None]


def yaml_indent(text: str) -> int:
    """The indentation of a YAML file, taken from its first indented line."""

    for line in text.splitlines():
        stripped = line.lstrip(" ")
        if stripped and not stripped.startswith("#") and len(stripped) < len(line):
            return len(line) - len(stripped)

    return 2


async def save_model_load_config(model_name: str, model_path: pathlib.Path, settings: dict):
    """
    Save load settings, in the shape of a load request, as the model folder's
    tabby_config.yml. Settings that only repeat a use_as_default value are left
    out, unless the file already has them. An existing file is copied to
    tabby_config.bak first, and keeps its comments, other sections and the
    keys a load request can't set.
    """

    settings = {
        key: value
        for key, value in settings.items()
        if key not in NON_SETTING_FIELDS or key == "draft_model"
    }
    draft_settings = settings.get("draft_model")
    draft_settings = {} if draft_settings is None else draft_settings
    if not isinstance(draft_settings, dict):
        raise ValueError("draft_model must be a mapping of draft model options.")

    model_fields = set(ModelLoadRequest.model_fields) - NON_SETTING_FIELDS
    draft_fields = set(DraftModelLoadRequest.model_fields)
    unknown = [key for key in settings if key not in model_fields and key != "draft_model"]
    unknown += [f"draft_model.{key}" for key in draft_settings if key not in draft_fields]
    if unknown:
        raise ValueError("Not load options: " + ", ".join(unknown))

    # The same checks as a load request with these settings
    try:
        ModelLoadRequest.model_validate(
            {**settings, "model_name": model_name, "draft_model": draft_settings}
        )
    except ValidationError as exc:
        problems = [
            ".".join(str(part) for part in error["loc"]) + ": " + error["msg"]
            for error in exc.errors()
        ]
        raise ValueError("; ".join(problems)) from exc
    settings.pop("draft_model", None)

    config_path = model_path / model.MODEL_CONFIG_FILE
    yaml = YAML()
    yaml.width = 4096

    exists = config_path.exists()
    if exists:
        async with aiofiles.open(config_path, "r", encoding="utf8") as config_file:
            text = await config_file.read()
        document = yaml.load(text)
        if document is None:
            document = CommentedMap()
        if not isinstance(document, dict) or not all(
            isinstance(document.get(section), (dict, type(None)))
            for section in ("model", "draft_model")
        ):
            raise ValueError(
                f"The existing {model.MODEL_CONFIG_FILE} must hold mappings of options "
                "under model: and draft_model:, so it can't be updated."
            )
        indent = yaml_indent(text)
        yaml.indent(mapping=indent, sequence=indent + 2, offset=indent)
    else:
        document = CommentedMap()
        document.yaml_set_start_comment(
            f"Load options for {model_name}. Options that only repeat config.yml's\n"
            "use_as_default values are left out."
        )

    defaults = model.use_as_default_settings()
    draft_defaults = defaults.pop("draft_model")

    def update(section: str, new: dict, fields: set):
        current = document.get(section)
        if current is None:
            current = CommentedMap()
            if section in document or section != "model":
                document[section] = current
            else:
                document.insert(0, section, current)

        # Keys a load request can't set are left alone
        for key in [key for key in current if key in fields and key not in new]:
            del current[key]
        last_key = next(reversed(current), None)
        for key, value in new.items():
            if key not in current or current[key] != value:
                current[key] = yaml_node(value)
        move_trailing_blank_lines(current, last_key, next(reversed(current), None))

        if not current:
            del document[section]

    saved = explicit_settings(settings, defaults, document.get("model") or {})
    saved_draft = explicit_settings(
        draft_settings, draft_defaults, document.get("draft_model") or {}
    )
    update("model", saved, model_fields)
    update("draft_model", saved_draft, draft_fields)
    if saved_draft:
        saved["draft_model"] = saved_draft

    backup = None
    if exists:
        shutil.copy2(config_path, model_path / MODEL_CONFIG_BACKUP_FILE)
        backup = MODEL_CONFIG_BACKUP_FILE

    # Nothing left to say: no file at all reads the same, without the warning
    # an empty one gets at load
    if not document:
        if exists:
            config_path.unlink()
        return ModelLoadConfigSaveResponse(
            model_name=model_name, saved=saved, backup=backup, removed=exists
        )

    stream = io.StringIO()
    yaml.dump(document, stream)
    temp_path = config_path.with_name(config_path.name + ".tmp")
    async with aiofiles.open(temp_path, "w", encoding="utf8") as config_file:
        await config_file.write(stream.getvalue())
    os.replace(temp_path, config_path)

    return ModelLoadConfigSaveResponse(model_name=model_name, saved=saved, backup=backup)
