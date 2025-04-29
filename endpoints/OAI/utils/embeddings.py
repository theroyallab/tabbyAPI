"""
This file is derived from
[text-generation-webui openai extension embeddings](https://github.com/oobabooga/text-generation-webui/blob/1a7c027386f43b84f3ca3b0ff04ca48d861c2d7a/extensions/openai/embeddings.py)
and modified.
The changes introduced are: Suppression of progress bar,
typing/pydantic classes moved into this file,
embeddings function declared async.
"""

import base64
from fastapi import Request
import numpy as np
from loguru import logger

from common import model
from endpoints.OAI.types.embedding import (
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    UsageInfo,
)


def float_list_to_base64(float_array: np.ndarray) -> str:
    """
    Converts the provided list to a float32 array for OpenAI
    Ex. float_array = np.array(float_list, dtype="float32")
    """

    # Encode raw bytes into base64
    encoded_bytes = base64.b64encode(float_array.tobytes())

    # Turn raw base64 encoded bytes into ASCII
    ascii_string = encoded_bytes.decode("ascii")
    return ascii_string


async def get_embeddings(data: EmbeddingsRequest, request: Request) -> dict:
    model_path = model.embeddings_container.model_dir

    logger.info(f"Received embeddings request {request.state.id}")

    if not isinstance(data.input, list):
        data.input = [data.input]

    embedding_data = await model.embeddings_container.generate(data.input)

    # OAI expects a return of base64 if the input is base64
    embedding_object = [
        EmbeddingObject(
            embedding=float_list_to_base64(emb)
            if data.encoding_format == "base64"
            else emb.tolist(),
            index=n,
        )
        for n, emb in enumerate(embedding_data.get("embeddings"))
    ]

    usage = embedding_data.get("usage")
    response = EmbeddingsResponse(
        data=embedding_object,
        model=model_path.name,
        usage=UsageInfo(prompt_tokens=usage, total_tokens=usage),
    )

    logger.info(f"Finished embeddings request {request.state.id}")

    return response
