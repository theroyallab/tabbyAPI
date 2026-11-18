"""Tests for LoRA loading and unloading on the ExLlamaV3 container."""

import asyncio
import pathlib
from types import SimpleNamespace

import pytest

pytest.importorskip("exllamav3")

import backends.exllamav3.model as model_module  # noqa: E402
from backends.exllamav3.model import ExllamaV3Container, LoadedLora  # noqa: E402
from common import model as common_model  # noqa: E402
from endpoints.core.utils.lora import get_active_loras  # noqa: E402


class FakeAdapter:
    def __init__(self, name, scaling, target_modules):
        self.name = name
        # The effective factor differs from the requested scaling in real adapters
        self.lora_scaling = scaling * 0.5
        self.target_modules = target_modules
        self.unloads = 0

    def unload(self):
        self.unloads += 1


class FakeGenerator:
    def __init__(self, **kwargs):
        self.closed = False
        self.error = None
        self.generator = SimpleNamespace(max_batch_size=4, recurrent_cache=None)

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_lora(monkeypatch):
    class FakeLoRA:
        missing = set()
        no_match = set()
        calls = []
        adapters = {}

        @staticmethod
        def from_directory(model, directory, lora_scaling):
            name = pathlib.PurePath(directory).name
            FakeLoRA.calls.append((model, directory, lora_scaling))
            if name in FakeLoRA.missing:
                raise FileNotFoundError(directory)

            target_modules = [] if name in FakeLoRA.no_match else ["q_proj"]
            adapter = FakeAdapter(name, lora_scaling, target_modules)
            FakeLoRA.adapters[name] = adapter
            return adapter

    monkeypatch.setattr(model_module, "LoRA", FakeLoRA)
    return FakeLoRA


@pytest.fixture
def built(monkeypatch):
    """The generators constructed by create_generator, in order."""

    generators = []

    class RecordingGenerator(FakeGenerator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            generators.append(self)

    monkeypatch.setattr(model_module, "AsyncGenerator", RecordingGenerator)
    return generators


def make_container():
    container = ExllamaV3Container.__new__(ExllamaV3Container)
    container.loaded = True
    container.load_lock = asyncio.Lock()
    container.load_condition = asyncio.Condition()
    container.active_job_ids = {}
    container.generator = FakeGenerator()
    container.model = SimpleNamespace(unloads=0)
    container.model.unload = lambda: setattr(
        container.model, "unloads", container.model.unloads + 1
    )
    container.cache = None
    container.draft_model = None
    container.draft_cache = None
    container.tokenizer = None
    container.max_batch_size = 4
    container.chunk_size = 1024
    container.draft_num_tokens = None
    container.dynamic_draft = False
    container.ngram_match_min = 0
    container.hybrid_draft_tokens = 0
    container.draft_max_rows = 0
    container.sparse_draft_head = 0
    container.recurrent_checkpoint_interval = None
    container.recurrent_checkpoint_interval_pp = None
    container.use_tp = False
    container.disk_cache_kwargs = {}
    container.loras = []
    return container


def test_load_two_loras_rebuilds_the_generator_once(tmp_path, fake_lora, built, monkeypatch):
    container = make_container()
    old = container.generator
    monkeypatch.setattr(common_model, "container", container)

    result = asyncio.run(
        container.load_loras(
            tmp_path,
            loras=[{"name": "alpha", "scaling": 0.8}, {"name": "beta"}],
        )
    )

    assert result == {"success": ["alpha", "beta"], "failure": []}

    # The adapters are loaded from their own folders at the requested scaling
    assert fake_lora.calls == [
        (container.model, str(tmp_path / "alpha"), 0.8),
        (container.model, str(tmp_path / "beta"), 1.0),
    ]

    # The API reports what was requested, not the adapter's effective factor
    cards = get_active_loras().data
    assert [(card.id, card.scaling) for card in cards] == [("alpha", 0.8), ("beta", 1.0)]

    # One rebuild for the whole batch, with the old generator retired
    assert len(built) == 1
    assert container.generator is built[0]
    assert old.closed
    assert not built[0].closed
    assert not container.load_lock.locked()


def test_partial_failure_keeps_only_the_good_adapter(tmp_path, fake_lora, built):
    fake_lora.missing = {"gone"}
    fake_lora.no_match = {"other-base"}
    container = make_container()

    result = asyncio.run(
        container.load_loras(
            tmp_path,
            loras=[
                {"name": "good", "scaling": 1.0},
                {"name": "gone"},
                {"scaling": 1.0},
                {"name": "other-base"},
            ],
        )
    )

    # The nameless entry is skipped, not reported as a failed None
    assert result == {"success": ["good"], "failure": ["gone", "other-base"]}
    assert [lora.name for lora in container.get_loras()] == ["good"]

    # An adapter that matched no layers is detached from the model again
    assert fake_lora.adapters["other-base"].unloads == 1
    assert fake_lora.adapters["good"].unloads == 0
    assert len(built) == 1
    assert not container.load_lock.locked()


def test_all_failed_leaves_the_generator_and_its_cache_alone(tmp_path, fake_lora, built):
    fake_lora.missing = {"gone"}
    container = make_container()
    old = container.generator

    result = asyncio.run(container.load_loras(tmp_path, loras=[{"name": "gone"}]))

    assert result == {"success": [], "failure": ["gone"]}
    assert container.generator is old
    assert not old.closed
    assert built == []
    assert not container.load_lock.locked()


@pytest.mark.parametrize(
    "attribute, value",
    [("use_tp", True), ("disk_cache_kwargs", {"disk_cache_dir": "/tmp/cache"})],
)
def test_unsupported_configurations_fail_every_lora(tmp_path, fake_lora, built, attribute, value):
    container = make_container()
    old = container.generator
    setattr(container, attribute, value)

    result = asyncio.run(
        container.load_loras(
            tmp_path,
            loras=[{"name": "alpha"}, {"scaling": 1.0}, {"name": "beta", "scaling": 0.5}],
        )
    )

    assert result == {"success": [], "failure": ["alpha", "beta"]}
    assert fake_lora.calls == []
    assert container.get_loras() == []
    assert container.generator is old
    assert not old.closed
    assert built == []
    assert not container.load_lock.locked()


def test_unload_loras_detaches_every_adapter_and_keeps_the_model(fake_lora, built):
    container = make_container()
    old = container.generator
    adapters = [FakeAdapter("alpha", 1.0, ["q_proj"]), FakeAdapter("beta", 1.0, ["q_proj"])]
    container.loras = [LoadedLora(name=a.name, scaling=1.0, lora=a) for a in adapters]

    asyncio.run(container.unload(loras_only=True))

    assert [adapter.unloads for adapter in adapters] == [1, 1]
    assert container.get_loras() == []
    assert container.model is not None
    assert container.model.unloads == 0
    assert container.loaded

    # The prompt cache goes with the adapters
    assert len(built) == 1
    assert container.generator is built[0]
    assert old.closed
    assert not container.load_lock.locked()

    # Nothing left to unload: no adapter is touched twice, no needless rebuild
    asyncio.run(container.unload(loras_only=True))

    assert [adapter.unloads for adapter in adapters] == [1, 1]
    assert len(built) == 1
    assert container.generator is built[0]
    assert not built[0].closed
    assert not container.load_lock.locked()


def test_waiting_requests_are_released_once_loras_are_loaded(tmp_path, fake_lora, built):
    container = make_container()

    async def scenario():
        # A running request keeps the load waiting for it, with the lock held
        container.active_job_ids["req"] = None
        loading = asyncio.create_task(container.load_loras(tmp_path, loras=[{"name": "alpha"}]))
        await asyncio.sleep(0.05)
        assert container.load_lock.locked()

        # What stream_generate does before it accepts a request
        async def request():
            async with container.load_condition:
                await container.load_condition.wait_for(lambda: not container.load_lock.locked())

        waiting = asyncio.create_task(request())
        await asyncio.sleep(0.05)
        assert not waiting.done()

        container.active_job_ids.clear()
        result = await loading
        await asyncio.wait_for(waiting, timeout=1)
        return result

    result = asyncio.run(scenario())

    assert result == {"success": ["alpha"], "failure": []}
    assert not container.load_lock.locked()
