import pathlib
from common import model
from endpoints.OAI.types.common import (
    CompletionTokensDetails,
    PromptTokensDetails,
    Timings,
    UsageStats,
)
from common.tabby_config import config
from common.auth import get_key_permission
from common.logger import xlogger
from common.networking import handle_request_error
from fastapi import HTTPException, Request


def get_usage_stats(
    generation: dict,
) -> UsageStats | None:
    """
    Collect usage stats from generation if it is a finish chunk
    """
    if "finish_reason" not in generation:
        return None

    prompt_tokens = generation.get("prompt_tokens", 0)
    completion_tokens = generation.get("gen_tokens", 0)
    usage_stats = UsageStats(
        prompt_tokens=prompt_tokens,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=round(generation.get("cached_tokens") or 0)
        ),
        prompt_time=generation.get("prompt_time"),
        prompt_tokens_per_sec=generation.get("prompt_tokens_per_sec"),
        completion_tokens=completion_tokens,
        completion_tokens_details=CompletionTokensDetails(
            accepted_prediction_tokens=generation.get("draft_accept") or 0,
            rejected_prediction_tokens=generation.get("draft_reject") or 0,
        ),
        completion_time=generation.get("gen_time"),
        completion_tokens_per_sec=generation.get("gen_tokens_per_sec"),
        total_tokens=prompt_tokens + completion_tokens,
        total_time=generation.get("total_time"),
    )
    return usage_stats


def get_timings(
    generation: dict,
) -> Timings | None:
    """
    Collect llama-server compatible timings from generation if it is a finish chunk

    Key mapping follows llama.cpp's server_slot_stats::to_json
    (tools/server/server-common.cpp). Rates are computed from the times, never
    from the backend's *_tokens_per_sec fields, which carry the string
    "Indeterminate" when a time is zero; a zero time yields 0.0 like llama.cpp.
    """
    if "finish_reason" not in generation:
        return None

    cache_n = round(generation.get("cached_tokens") or 0)
    prompt_n = max((generation.get("prompt_tokens") or 0) - cache_n, 0)
    prompt_ms = (generation.get("prompt_time") or 0) * 1000
    predicted_n = generation.get("gen_tokens") or 0
    predicted_ms = (generation.get("gen_time") or 0) * 1000

    # llama.cpp divides by n_gen - 1 because its first token comes from the
    # prompt batch's logits, outside the generation time. In exllamav3 the
    # prompt is prefilled up to its last token (Job.is_prefill_done), and
    # time_first_token is stamped before the decode pass that produces the
    # first token, so gen_time covers all gen_tokens tokens. Dividing by
    # gen_tokens gives the same meaning as llama.cpp's figure on this backend.

    # llama.cpp sets the draft keys only when draft tokens were produced
    # (n_draft_tokens > 0), so they stay absent otherwise
    draft = {}
    draft_accept = generation.get("draft_accept") or 0
    draft_reject = generation.get("draft_reject") or 0
    if draft_accept + draft_reject > 0:
        draft["draft_n"] = draft_accept + draft_reject
        draft["draft_n_accepted"] = draft_accept

    return Timings(
        cache_n=cache_n,
        prompt_n=prompt_n,
        prompt_ms=prompt_ms,
        prompt_per_token_ms=prompt_ms / prompt_n if prompt_n > 0 else 0.0,
        prompt_per_second=1e3 / prompt_ms * prompt_n if prompt_ms > 0 else 0.0,
        predicted_n=predicted_n,
        predicted_ms=predicted_ms,
        predicted_per_token_ms=predicted_ms / predicted_n if predicted_n > 0 else 0.0,
        predicted_per_second=1e3 / predicted_ms * predicted_n if predicted_ms > 0 else 0.0,
        **draft,
    )


def aggregate_usage_stats(usage_stats_list: list[UsageStats]) -> UsageStats:
    if len(usage_stats_list) == 1:
        return usage_stats_list[0]

    usl = usage_stats_list
    prompt_tokens = usl[0].prompt_tokens
    prompt_time = usl[0].prompt_time
    prompt_tokens_per_sec = usl[0].prompt_tokens_per_sec
    completion_tokens = sum(us.completion_tokens for us in usl)
    completion_time = max(us.completion_time for us in usl)
    completion_tokens_per_sec = completion_tokens / (completion_time + 1e-20)
    total_tokens = prompt_tokens + completion_tokens
    total_time = prompt_time + completion_time

    # n > 1 generations share one prompt, so prompt-side details come from the
    # first entry while generation-side counters accumulate
    usage_stats = UsageStats(
        prompt_tokens=prompt_tokens,
        prompt_tokens_details=usl[0].prompt_tokens_details,
        prompt_time=prompt_time,
        prompt_tokens_per_sec=prompt_tokens_per_sec,
        completion_tokens=completion_tokens,
        completion_tokens_details=CompletionTokensDetails(
            accepted_prediction_tokens=sum(
                us.completion_tokens_details.accepted_prediction_tokens for us in usl
            ),
            rejected_prediction_tokens=sum(
                us.completion_tokens_details.rejected_prediction_tokens for us in usl
            ),
        ),
        completion_time=completion_time,
        completion_tokens_per_sec=completion_tokens_per_sec,
        total_tokens=total_tokens,
        total_time=total_time,
    )
    return usage_stats


def _is_loaded_model(model_name: str) -> bool:
    """
    True if model_name refers to the currently loaded model, either by its
    advertised id (the model directory's basename) or as a path. A basename
    alone is ambiguous in quant-style layouts (<model>/exl3/<bpw>), so paths
    are compared fully resolved against the model directory.
    """

    if not (model.container and model.container.loaded):
        return False

    loaded_model_dir = model.container.model_dir
    if loaded_model_dir.name == model_name:
        return True

    requested_path = pathlib.Path(config.model.model_dir) / model_name
    try:
        return requested_path.resolve() == loaded_model_dir.resolve()
    except OSError:
        return False


async def load_inline_model(model_name: str, request: Request):
    """Load a model from the data.model parameter"""

    # Return if the model container already exists and the model is fully loaded
    if _is_loaded_model(model_name):
        return

    # Return if inline loading is disabled
    # Also warn if an admin key is used
    if not config.model.inline_model_loading:
        if get_key_permission(request) == "admin":
            xlogger.warning(
                f"Unable to switch model to {model_name} because "
                '"inline_model_loading" is not True in config.yml.'
            )

        return

    is_dummy_model = config.model.use_dummy_models and model_name in config.model.dummy_model_names

    # Error if an invalid key is passed
    # If a dummy model is provided, don't error
    if get_key_permission(request) != "admin":
        if not is_dummy_model:
            error_message = handle_request_error(
                f"Unable to switch model to {model_name} because " + "an admin key isn't provided",
                exc_info=False,
            ).error.message

            raise HTTPException(401, error_message)
        else:
            return

    # Start inline loading
    # Past here, user is assumed to be admin

    # Skip if the model is a dummy
    if is_dummy_model:
        xlogger.warning(f"Dummy model {str(model_name)} provided. Skipping inline load.")
        return

    model_path = pathlib.Path(config.model.model_dir)
    model_path = model_path / model_name

    # A request that names a model it can't get must fail rather than run on
    # whatever happens to be loaded: the client asked for a specific model, and
    # an answer from a different one is wrong in a way it cannot detect
    if not model_path.exists():
        error_message = handle_request_error(
            f"Model {model_name} was not found in the model directory.",
            exc_info=False,
        ).error.message

        raise HTTPException(404, error_message)

    # Load the model and also add draft dir
    try:
        await model.load_model(
            model_path,
            draft_model=config.draft_model.model_dump(include={"draft_model_dir"}),
        )
    except HTTPException:
        raise
    except Exception as exc:
        error_message = handle_request_error(
            f"Model {model_name} failed to load: {exc}"
        ).error.message

        raise HTTPException(503, error_message) from exc
