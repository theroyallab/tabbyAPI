from typing import List, Optional
from fastapi import HTTPException, Request, status
import pathlib
import asyncio
import time

from common import model
from common.networking import handle_request_error
from common.sampling import BaseSamplerRequest
from endpoints.OAI.types.logprob import (
    LogProbRequest,
    LogProbResponse,
    LogProbChoice,
    TokenLogProbs,
)
from endpoints.OAI.types.common import UsageStats
from loguru import logger


async def generate_logprobs(
    data: LogProbRequest, request: Request, model_path: pathlib.Path
) -> LogProbResponse:
    """Generate log probabilities for the given prompt."""

    # Pre-validate prompt length against model context
    max_context = model.container.model_info().parameters.max_seq_len
    prompt_tokens = model.container.encode_tokens(data.prompt)

    if len(prompt_tokens) > max_context:
        error_message = handle_request_error(
            f"Prompt length {len(prompt_tokens)} exceeds max_seq_len {max_context}.",
            exc_info=False,
        ).error.message

        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            error_message,
        )
    try:
        # Ensure n is exactly 1 to avoid unnecessary computation
        # Note: The LogProbRequest pydantic model already ensures n=1
        n = data.n or 1
        
        # Create a sampler request with parameters for logprob calculation
        # First get the model dump and exclude fields we don't want
        data_dict = data.model_dump(exclude={"prompt", "stream", "n"})
        
        # Override specific parameters for logprob calculation
        data_dict.update({
            "repetition_penalty": 1.0,  # No penalties for accurate logprobs
            "temperature": 0.0,  # Deterministic for scoring
            "top_p": 1.0
        })
        
        # Create the sampler request
        sampler_params = BaseSamplerRequest(**data_dict)

        try:
            # Directly call the dedicated logprob calculation method
            # This avoids the roundabout way of using flags and going through the generate method
            generation = model.container.compute_sequence_logprobs(
                prompt=data.prompt,
                params=sampler_params
            )
        except asyncio.TimeoutError:
            # Handle timeout explicitly
            error_message = handle_request_error(
                f"LogProb calculation for prompt timed out.",
                exc_info=False,
            ).error.message
            
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=error_message,
            )

        tokens: List[str] = generation.get("prompt_token_strings", [])
        token_logprobs: List[Optional[float]] = generation.get("prompt_token_logprobs", [])
        top_logprobs: List[Optional[Dict[str, float]]] = generation.get("top_logprobs", [])
        text_offset: List[int] = generation.get("offset", [])

        # Calculate the sum of token logprobs, ignoring None values (first token)
        logprobs_sum = sum(lp for lp in token_logprobs if lp is not None) if token_logprobs else 0.0

        # Create the choice object with logprobs info
        choice = LogProbChoice(
            index=0,
            logprobs=TokenLogProbs(
                tokens=tokens,
                token_logprobs=token_logprobs,
                top_logprobs=top_logprobs,
                text_offset=text_offset,
                sum=logprobs_sum,
            ),
        )

        # While we always calculate once, we replicate the results to match n
        # This is simply to conform to the API schema which expects n choices
        # In practice for the logprob endpoint n is constrained to 1 by the schema
        choices = [choice]
        
        # Create the response
        prompt_tokens = len(tokens) if tokens else 0
        response = LogProbResponse(
            choices=choices,
            model=model_path.name,
            usage=UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens,
            ),
        )

        return response
    except HTTPException:
        raise
    except (ValueError, MemoryError) as exc:
        error_message = handle_request_error(str(exc), exc_info=False).error.message
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, error_message) from exc
    except Exception as exc:
        error_response = handle_request_error(
            f"LogProb calculation {request.state.id} failed. Please check the server console."
        )
        status_code = (
            error_response.error.code if hasattr(error_response.error, "code") else 503
        )

        raise HTTPException(
            status_code=status_code, detail=error_response.error.message
        ) from exc


async def stream_generate_logprobs(
    data: LogProbRequest, request: Request, model_path: pathlib.Path
):
    """
    Stream generate log probabilities for the given prompt.

    Not fully implemented in v1, but included for future compatibility.
    """
    response = await generate_logprobs(data, request, model_path)
    yield f"data: {response.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
