"""A request naming a model it can't get must fail, not run on the loaded model."""

import asyncio
import pathlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from common import model
from common.tabby_config import config
from endpoints.OAI.utils import common_


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Model directory with one loadable model, the container holding another."""

    (tmp_path / "present").mkdir()
    loaded_dir = tmp_path / "loaded"
    loaded_dir.mkdir()

    monkeypatch.setattr(config.model, "model_dir", str(tmp_path))
    monkeypatch.setattr(config.model, "inline_model_loading", True)
    monkeypatch.setattr(config.model, "use_dummy_models", False)
    monkeypatch.setattr(config.model, "dummy_model_names", ["gpt-3.5-turbo"])
    monkeypatch.setattr(common_, "get_key_permission", lambda request: "admin")
    monkeypatch.setattr(model, "container", SimpleNamespace(model_dir=loaded_dir, loaded=True))

    loads = []

    async def fake_load(path, **kwargs):
        loads.append(pathlib.Path(path).name)
        if path.name == "present" and env.fail:
            raise RuntimeError("out of VRAM")

    monkeypatch.setattr(model, "load_model", fake_load)
    env = SimpleNamespace(root=tmp_path, loads=loads, fail=False)
    return env


def load(name):
    return asyncio.run(common_.load_inline_model(name, request=None))


def test_missing_model_is_rejected(env):
    with pytest.raises(HTTPException) as info:
        load("missing")

    assert info.value.status_code == 404
    assert "missing" in info.value.detail
    assert env.loads == []


def test_load_failure_is_reported_not_swallowed(env):
    env.fail = True
    with pytest.raises(HTTPException) as info:
        load("present")

    assert info.value.status_code == 503
    assert "out of VRAM" in info.value.detail


def test_existing_model_loads(env):
    load("present")
    assert env.loads == ["present"]


def test_loaded_model_is_a_no_op(env):
    load("loaded")
    assert env.loads == []


def test_dummy_model_runs_on_loaded_model(env, monkeypatch):
    monkeypatch.setattr(config.model, "use_dummy_models", True)
    load("gpt-3.5-turbo")
    assert env.loads == []


def test_inline_loading_disabled_ignores_model_field(env, monkeypatch):
    # Pre-existing behaviour: with inline loading off the model field is
    # advisory, since many clients send a name without meaning a specific model
    monkeypatch.setattr(config.model, "inline_model_loading", False)
    load("missing")
    assert env.loads == []
