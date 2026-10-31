"""The server side of the chat client's raw text sessions: stepping through a
generation, the stop token on the completion stream, and prompts with nothing
to generate from."""

import asyncio
import json
import math
import pathlib
import time
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException

pytest.importorskip("exllamav3")

from backends.exllamav3 import model as model_module  # noqa: E402
from backends.exllamav3.model import ExllamaV3Container  # noqa: E402
from backends.exllamav3.sampler import (  # noqa: E402
    ExllamaV3SamplerBuilder,
    SS_Candidates,
    SS_OnlyPath,
)
from common import model  # noqa: E402
from common.errors import EmptyPromptError  # noqa: E402
from common.sampling import BaseSamplerRequest  # noqa: E402
from endpoints.OAI.types.chat_completion import ChatCompletionRequest  # noqa: E402
from endpoints.OAI.types.completion import (  # noqa: E402
    CompletionRequest,
    ContinuationsRequest,
)
from endpoints.OAI.utils import completion  # noqa: E402

FINISH = {
    "finish_reason": "stop",
    "eos_reason": "stop_token",
    "stop_str": "<|im_end|>",
    "prompt_tokens": 4,
    "gen_tokens": 1,
    "prompt_time": 0.1,
    "gen_time": 0.1,
    "cached_tokens": 0,
}


class FakeContainer:
    async def stream_generate(self, request_id, prompt, params, handler, mm, label=None):
        yield {"text": "Hi"}
        yield dict(FINISH)


class FakeDisconnectHandler:
    async def cleanup(self):
        pass


def test_stop_after_output_defaults_off():
    assert CompletionRequest(prompt="x").stop_after_output is False
    assert CompletionRequest(prompt="x", stop_after_output=True).stop_after_output is True
    assert ChatCompletionRequest(messages=[]).stop_after_output is False


def test_stream_chunk_names_the_stop_token():
    generation = {"index": 0, **FINISH}
    _, data, _, _ = completion._compose_serialize_stream_chunk("req", generation, "model-x")
    assert data["choices"][0]["eos_reason"] == "stop_token"
    assert data["choices"][0]["stop_str"] == "<|im_end|>"

    # Nothing on a chunk that ended for another reason
    generation = {"index": 0, "finish_reason": "length", "eos_reason": "max_new_tokens"}
    _, data, _, _ = completion._compose_serialize_stream_chunk("req", generation, "model-x")
    assert "stop_str" not in data["choices"][0]


def test_response_names_the_stop_token():
    response = completion._compose_response(
        "req", [{"index": 0, "content": "Hi", **FINISH}], "model-x", True
    )
    assert response.choices[0].stop_str == "<|im_end|>"


def test_stream_runs_without_max_tokens(monkeypatch):
    """A notebook left on "auto" sends no max_tokens, which the server fills in."""

    monkeypatch.setattr(model, "container", FakeContainer())
    data = CompletionRequest(prompt="x", stream=True)
    assert data.max_tokens is None

    async def collect():
        chunks = []
        stream = completion.stream_generate_completion(
            "x",
            data,
            SimpleNamespace(state=SimpleNamespace(id="req")),
            pathlib.Path("model-x"),
            FakeDisconnectHandler(),
        )
        async for chunk in stream:
            chunks.append(chunk)
            # As the response does between events, so the collector task finishes
            await asyncio.sleep(0)
        return chunks

    chunks = asyncio.run(asyncio.wait_for(collect(), 5))
    assert chunks[-1] == "[DONE]"
    choices = [json.loads(chunk)["choices"][0] for chunk in chunks[:-1]]
    assert [choice["text"] for choice in choices] == ["Hi", ""]
    assert choices[-1]["stop_str"] == "<|im_end|>"


def test_job_times_cover_a_cancelled_job():
    start = time.time() - 10
    job = SimpleNamespace(
        job=SimpleNamespace(
            time_enqueue=start,
            time_first_prefill=start + 1,
            time_first_token=start + 4,
            # Spent before a requeue
            time_enqueued=0.5,
            time_prefill=0.25,
            time_generate=2.0,
        )
    )

    times = ExllamaV3Container._job_times(job)
    assert times["time_enqueued"] == pytest.approx(1.5)
    assert times["time_prefill"] == pytest.approx(3.25)
    assert times["time_generate"] == pytest.approx(8.0, abs=1)


def test_job_times_of_a_job_that_never_started():
    job = SimpleNamespace(job=SimpleNamespace(time_enqueue=time.time()))
    assert ExllamaV3Container._job_times(job) == {
        "time_enqueued": 0.0,
        "time_prefill": 0.0,
        "time_generate": 0.0,
    }


def test_stopped_at_output_finishes_as_length():
    container = ExllamaV3Container.__new__(ExllamaV3Container)
    chunk = container.handle_finish_chunk(
        {
            "eos_reason": "max_new_tokens",
            "prompt_tokens": 4,
            "cached_tokens": 0,
            "new_tokens": 1,
            **ExllamaV3Container._job_times(SimpleNamespace(job=None)),
        },
        "req",
        "Hi",
    )
    assert chunk["finish_reason"] == "length"
    assert chunk["gen_tokens"] == 1


def fake_encoder(tokens):
    container = ExllamaV3Container.__new__(ExllamaV3Container)
    container.hf_model = SimpleNamespace(add_bos_token=lambda: False)
    container.bos_token_id = None
    container.encode_prompt = lambda *args, **kwargs: SimpleNamespace(numel=lambda: tokens)
    return container


def distribution(logit_values):
    """A sampler state as the explorer steps get it: sorted, normalized."""

    logits = torch.tensor([logit_values], dtype=torch.float)
    probs, indices = torch.sort(torch.softmax(logits, dim=-1), dim=-1, descending=True)
    return probs, indices


def explore(step, logit_values, past=(), **params):
    """Runs a real sampler stack, as a job would, with an explorer step at its end."""

    params = {"temperature": 1.0, "top_p": 1.0, "min_p": 0.0, "top_k": 0, **params}
    request = BaseSamplerRequest(**params)
    tokenizer = SimpleNamespace(actual_vocab_size=len(logit_values))
    builder = ExllamaV3SamplerBuilder.from_params(request, tokenizer, 4096)
    sampler = builder.build_explorer(request.temperature == 0, step)
    logits = torch.tensor([[logit_values]], dtype=torch.float)
    past_ids = torch.tensor([list(past)], dtype=torch.long)
    return sampler.forward(logits, past_ids, tokenizer=tokenizer).view(-1).tolist()[0]


LOGITS = [0.0, 5.0, 4.5, 1.0, -3.0]


def test_candidates_are_what_passes_truncation():
    step = SS_Candidates(10)
    assert explore(step, LOGITS, min_p=0.01) == 1
    assert [token for token, _ in step.candidates] == [1, 2, 3]
    assert sum(prob for _, prob in step.candidates) == pytest.approx(1.0)
    assert step.count == 3


def test_candidates_are_limited_to_the_likeliest():
    step = SS_Candidates(2)
    explore(step, LOGITS, top_k=4)
    assert [token for token, _ in step.candidates] == [1, 2]
    assert step.count == 4


def test_candidates_leave_out_banned_tokens():
    step = SS_Candidates(10)
    assert explore(step, LOGITS, min_p=0.01, banned_tokens=[1]) == 2
    assert [token for token, _ in step.candidates] == [2, 3, 0]


def test_candidates_of_a_greedy_request_are_one_token():
    step = SS_Candidates(10)
    explore(step, LOGITS, temperature=0, banned_tokens=[1])
    assert step.candidates == [(2, pytest.approx(1.0))]


def test_candidates_stay_with_the_truncation_under_adaptive_p():
    step = SS_Candidates(10)
    explore(step, LOGITS, min_p=0.01, adaptive_target=0.5)
    assert [token for token, _ in step.candidates] == [1, 2, 3]


def only_path(limit=20, text="text"):
    return SS_OnlyPath(0, limit, 99, lambda ids: text)


def test_path_takes_the_only_token_that_passes():
    step = only_path()
    assert explore(step, [0.0, 9.0, 0.0], past=[5], top_k=1) == 1
    assert step.span == [5, 1] and step.end is None


def test_path_ends_at_a_branch_point():
    step = only_path()
    assert explore(step, LOGITS, past=[5], min_p=0.05) == 99
    assert step.span == [5] and step.end == "branch"


def test_path_ends_at_its_limit():
    step = only_path(limit=2)
    assert explore(step, [0.0, 9.0, 0.0], past=[5, 6], top_k=1) == 99
    assert step.span == [5, 6] and step.end == "limit"


def test_path_finishes_a_character_at_a_branch_point_or_limit():
    step = only_path(limit=1, text="\ufffd")
    assert explore(step, LOGITS, past=[5], min_p=0.05) == 1
    assert step.span == [5, 1] and step.end is None


def test_explorer_stacks_stay_off_the_fused_kernel():
    sampler = ExllamaV3SamplerBuilder().build_explorer(False, SS_Candidates(10))
    assert sampler.reqs_past_ids
    assert not any(type(step).__name__ == "SS_Fused" for step in sampler.steps)


def test_continuations_request_limits():
    request = ContinuationsRequest(prompt="x")
    assert (request.max_options, request.max_branch_tokens) == (10, 20)
    with pytest.raises(ValueError):
        ContinuationsRequest(prompt="x", max_options=0)


class FakeAsyncJob:
    """Stands in for the engine: runs the sampler step on a script of what comes next."""

    script = {}

    def __init__(self, generator, sampler, input_ids, **kwargs):
        self.step = sampler.steps[-1]
        self.ids = input_ids[0].tolist()
        self.stop = kwargs["stop_conditions"]
        self.healing = kwargs.get("token_healing", False)
        self.max_new_tokens = kwargs["max_new_tokens"]
        self.cancelled = False

    async def cancel(self):
        self.cancelled = True

    async def __aiter__(self):
        while True:
            healing = self.healing
            if healing:
                # The last token is sampled again, so past it nothing has been decided
                logits = self.script[(*self.ids, "healing")]
                self.ids.pop()
                self.healing = False
            else:
                logits = self.script[tuple(self.ids)]
            probs, indices = distribution(logits)
            kept = probs[0] >= 0.05
            state = SimpleNamespace(
                probs=probs * kept,
                indices=indices,
                past_ids=torch.tensor([self.ids], dtype=torch.long),
                bsz=1,
                in_logits=probs,
                sample=None,
                state=None,
            )
            self.step.run(state)
            token = state.sample.item()
            self.ids.append(token)
            # As in the engine, a healed token doesn't count towards the limit
            self.max_new_tokens -= not healing
            if token in self.stop:
                yield {"eos": True, "eos_reason": "stop_token", "cached_tokens": 3}
                return
            if self.max_new_tokens == 0:
                yield {"eos": True, "eos_reason": "max_new_tokens", "cached_tokens": 3}
                return
            yield {"eos": False, "stage": "streaming"}


class FakeHandler:
    async def add_cleanup_task(self, key, task, args):
        pass

    async def finish(self, key):
        pass

    async def poll(self):
        pass


def explorer(monkeypatch):
    """A container that decodes token 1 as a, 2 as b, and so on, ending on token 7."""

    monkeypatch.setattr(model_module, "AsyncJob", FakeAsyncJob)
    container = ExllamaV3Container.__new__(ExllamaV3Container)
    container.load_condition = asyncio.Condition()
    container.load_lock = asyncio.Lock()
    container.loaded = True
    container.max_seq_len = 4096
    container.max_rq_tokens = None
    container.active_job_ids = {}
    container.bos_token_id = None
    container.hf_model = SimpleNamespace(add_bos_token=lambda: False, eos_tokens=lambda: [7])
    container.config = SimpleNamespace(eos_token_id_list=[7])
    container.cache = SimpleNamespace(max_num_tokens=4096)
    container.generator = SimpleNamespace(generator=SimpleNamespace(max_draft_tokens=0))
    container.tokenizer = SimpleNamespace(actual_vocab_size=10, eos_token_id=7)
    container.encode_prompt = lambda *a, **k: torch.tensor([[1, 2]], dtype=torch.long)
    container.decode_tokens = lambda ids, **k: "".join(" abcdefghij"[t] for t in ids)
    return container


def test_each_candidate_is_followed_to_its_end(monkeypatch):
    container = explorer(monkeypatch)
    uniform = [0.0] * 10
    FakeAsyncJob.script = {
        # Three tokens pass: 3, 4 and 7, the stop token
        (1, 2): [0.0, -9, -9, 3.0, 2.0, -9, -9, 1.0, -9, -9],
        # 3 goes on with 5 alone, and from there with 7, the stop token
        (1, 2, 3): [-9, -9, -9, -9, -9, 9.0, -9, -9, -9, -9],
        (1, 2, 3, 5): [-9, -9, -9, -9, -9, -9, -9, 9.0, -9, -9],
        # 4 branches right away
        (1, 2, 4): uniform,
    }

    result = asyncio.run(
        container.explore_continuations(
            "req", "x", BaseSamplerRequest(min_p=0.01), 10, 20, FakeHandler()
        )
    )

    options = result["options"]
    assert [option["token_ids"] for option in options] == [[3, 5, 7], [4], [7]]
    assert [option["end"] for option in options] == ["stop", "branch", "stop"]
    assert [option["text"] for option in options] == ["ceg", "d", "g"]
    assert options[0]["first"] == "c"
    assert result["candidates"] == 3
    assert result["truncated"] is True
    assert container.active_job_ids == {}


def mostly(token_probs):
    """Logits for tokens with these probabilities, and practically nothing for the rest."""

    return [math.log(token_probs[token]) if token in token_probs else -30.0 for token in range(10)]


def explore_healing(monkeypatch, script, **params):
    FakeAsyncJob.script = script
    container = explorer(monkeypatch)
    # Tokens 3 and 4 start with token 2, as a token needs to for healing
    pieces = {1: "a", 2: "b", 3: "bc", 4: "bd", 5: "e", 6: "f", 7: "g", 8: "h", 9: "i"}
    container.decode_tokens = lambda ids, **k: "".join(pieces[t] for t in ids)
    request = BaseSamplerRequest(token_healing=True, min_p=0.01, **params)
    return asyncio.run(container.explore_continuations("req", "x", request, 10, 20, FakeHandler()))


def test_healing_mixes_longer_tokens_with_what_follows_the_last_one(monkeypatch):
    result = explore_healing(
        monkeypatch,
        {
            # Token 2 can stay, or grow into 3 or 4
            (1, 2, "healing"): mostly({2: 0.5, 3: 0.35, 4: 0.15}),
            # Staying leads to 5 or 6
            (1, 2): mostly({5: 0.6, 6: 0.4}),
            # Each of the rest stops at a branch point right away
            (1, 3): mostly({8: 0.5, 9: 0.5}),
            (1, 4): mostly({8: 0.5, 9: 0.5}),
            (1, 2, 5): mostly({8: 0.5, 9: 0.5}),
            (1, 2, 6): mostly({8: 0.5, 9: 0.5}),
        },
    )

    options = result["options"]
    assert [(o["token_ids"], o["healed"]) for o in options] == [
        ([3], True),
        ([5], False),
        ([6], False),
        ([4], True),
    ]
    assert [o["prob"] for o in options] == pytest.approx([0.35, 0.3, 0.2, 0.15], abs=0.01)
    # The prompt decodes as "ab", and a healed token adds to its last letter
    assert [o["text"] for o in options] == ["c", "e", "f", "d"]
    assert [o["first"] for o in options] == ["c", "e", "f", "d"]
    assert result["candidates"] == 4


def test_healing_leaves_out_what_follows_a_token_that_must_grow(monkeypatch):
    result = explore_healing(
        monkeypatch,
        {
            (1, 2, "healing"): mostly({3: 0.9, 4: 0.1}),
            (1, 3): mostly({8: 0.5, 9: 0.5}),
            (1, 4): mostly({8: 0.5, 9: 0.5}),
        },
    )
    assert [o["token_ids"] for o in result["options"]] == [[3], [4]]
    assert all(o["healed"] for o in result["options"])


def test_stop_token_is_not_healed():
    ids = torch.tensor([[1, 7]])
    assert not ExllamaV3Container._can_heal(ids, [7, "stop string"])
    assert ExllamaV3Container._can_heal(ids, [8, "stop string"])


def test_continuations_leave_a_final_stop_token_alone(monkeypatch):
    container = explorer(monkeypatch)
    container.encode_prompt = lambda *a, **k: torch.tensor([[1, 7]], dtype=torch.long)
    # No healing entry: asking for one would fail
    FakeAsyncJob.script = {
        (1, 7): mostly({3: 1.0}),
        (1, 7, 3): mostly({7: 1.0}),
    }
    request = BaseSamplerRequest(token_healing=True, min_p=0.01)
    result = asyncio.run(
        container.explore_continuations("req", "x", request, 10, 20, FakeHandler())
    )
    assert [o["token_ids"] for o in result["options"]] == [[3, 7]]


def test_healed_candidates_keep_the_last_token_past_the_limit(monkeypatch):
    # The last token is the least likely of the three, with room for two
    FakeAsyncJob.script = {
        (1, 2, "healing"): mostly({3: 0.5, 4: 0.4, 2: 0.1}),
        (1, 2): mostly({5: 1.0}),
        (1, 3): mostly({8: 0.5, 9: 0.5}),
        (1, 2, 5): mostly({8: 0.5, 9: 0.5}),
    }
    container = explorer(monkeypatch)
    request = BaseSamplerRequest(token_healing=True, min_p=0.01)
    result = asyncio.run(container.explore_continuations("req", "x", request, 2, 20, FakeHandler()))
    assert [o["token_ids"] for o in result["options"]] == [[3], [5]]
    assert [o["prob"] for o in result["options"]] == pytest.approx([0.5, 0.1], abs=0.01)


def test_prompt_with_no_tokens_is_refused(monkeypatch):
    container = fake_encoder(0)
    with pytest.raises(EmptyPromptError):
        container.validate_context_length("", CompletionRequest(prompt=""))

    monkeypatch.setattr(model, "container", container)
    with pytest.raises(HTTPException) as caught:
        model.check_context_length("", CompletionRequest(prompt=""))
    assert caught.value.status_code == 400
    assert "empty" in caught.value.detail
