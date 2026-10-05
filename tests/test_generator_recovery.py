"""Tests for generator recovery after a generation error (PR #461)."""

import asyncio
import weakref
from types import SimpleNamespace

import pytest

pytest.importorskip("exllamav3")

import backends.exllamav3.model as model_module  # noqa: E402
from backends.exllamav3.model import ExllamaV3Container  # noqa: E402
from common.health import HealthManager  # noqa: E402


class FakeJob:
    def __init__(self, cancelled=False):
        self.cancelled = cancelled
        self.cancel_calls = 0

    async def cancel(self):
        self.cancel_calls += 1
        self.cancelled = True


def make_container(generator, recreation_takes=0):
    container = ExllamaV3Container.__new__(ExllamaV3Container)
    container.generator = generator
    container.recreations = 0

    async def create_generator():
        container.recreations += 1
        for _ in range(recreation_takes):
            await asyncio.sleep(0)

    container.create_generator = create_generator
    return container


async def recover(container, job, ex=RuntimeError("boom")):
    await container._recover_from_generation_error(ex, job)
    # Let a scheduled create_generator() run
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def clean_health():
    HealthManager.issues.clear()
    yield
    HealthManager.issues.clear()


def test_latched_detection():
    assert make_container(None)._generator_latched()
    assert make_container(SimpleNamespace())._generator_latched()
    assert make_container(SimpleNamespace(error=RuntimeError("x")))._generator_latched()
    assert not make_container(SimpleNamespace(error=None))._generator_latched()


def test_contained_error_keeps_generator_and_cancels_only_this_job():
    container = make_container(SimpleNamespace(error=None))
    job = FakeJob()

    asyncio.run(recover(container, job))

    assert container.recreations == 0
    assert job.cancel_calls == 1
    assert HealthManager.issues.maxlen == 100 and len(HealthManager.issues) == 0


def test_contained_error_does_not_cancel_twice():
    container = make_container(SimpleNamespace(error=None))
    job = FakeJob(cancelled=True)

    asyncio.run(recover(container, job))

    assert job.cancel_calls == 0
    assert container.recreations == 0


def test_latched_error_recreates_and_records_health_event():
    container = make_container(SimpleNamespace(error=RuntimeError("engine died")))
    job = FakeJob()

    asyncio.run(recover(container, job, ex=RuntimeError("engine died")))

    assert container.recreations == 1
    # Recreation cancels every job itself; the consumer must not touch the dead generator
    assert job.cancel_calls == 0
    assert [issue.description for issue in HealthManager.issues] == ["RuntimeError: engine died"]


def test_wrapper_without_latch_attribute_recreates():
    container = make_container(SimpleNamespace())
    job = FakeJob()

    asyncio.run(recover(container, job))

    assert container.recreations == 1
    assert len(HealthManager.issues) == 1


class LatchedAsyncGenerator:
    """Stand-in for the old exllamav3 AsyncGenerator after its iteration task died."""

    def __init__(self, closed):
        self.error = RuntimeError("engine died")
        self.closed = closed

    async def close(self):
        self.closed.append(True)


def make_loaded_container(generator):
    container = ExllamaV3Container.__new__(ExllamaV3Container)
    container.loaded = True
    container.load_lock = asyncio.Lock()
    container.load_condition = asyncio.Condition()
    container.active_job_ids = {}
    container.generator = generator
    container.model = None
    container.cache = None
    container.draft_model = None
    container.draft_cache = None
    container.tokenizer = None
    container.max_batch_size = 4
    container.chunk_size = 1024
    container.draft_num_tokens = None
    container.dynamic_draft = False
    container.ngram_match_min = 0
    container.recurrent_checkpoint_interval = None
    container.recurrent_checkpoint_interval_pp = None
    return container


def test_recreation_releases_old_generator_before_constructing_the_new_one(monkeypatch):
    closed = []
    old = LatchedAsyncGenerator(closed)
    old_ref = weakref.ref(old)
    container = make_loaded_container(old)
    del old

    observed = {}

    class NewAsyncGenerator:
        def __init__(self, **kwargs):
            # The old generator and its host caches must be gone by the time the
            # replacement allocates its own
            observed["closed"] = list(closed)
            observed["reference_dropped"] = container.generator is None
            observed["old_collected"] = old_ref() is None
            self.error = None
            self.generator = SimpleNamespace(max_batch_size=4, recurrent_cache=None)

    monkeypatch.setattr(model_module, "AsyncGenerator", NewAsyncGenerator)

    asyncio.run(container.create_generator())

    assert observed == {"closed": [True], "reference_dropped": True, "old_collected": True}
    assert isinstance(container.generator, NewAsyncGenerator)
    assert not container.load_lock.locked()


def test_concurrent_failures_share_one_recreation():
    """Every in-flight request fails on a latch; only the first schedules a recreation."""

    container = make_container(SimpleNamespace(error=RuntimeError("x")), recreation_takes=10)

    async def run():
        jobs = [FakeJob() for _ in range(3)]
        for job in jobs:
            await container._recover_from_generation_error(RuntimeError("boom"), job)
        await asyncio.sleep(0)  # let the scheduled recreation start
        assert container.recreations == 1
        assert not container.recreate_task.done()
        await container.recreate_task
        assert container.recreations == 1

        # A later latch, after that recreation finished, is handled afresh
        await container._recover_from_generation_error(RuntimeError("again"), FakeJob())
        await container.recreate_task
        assert container.recreations == 2

    asyncio.run(run())
    assert len(HealthManager.issues) == 4
