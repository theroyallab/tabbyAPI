"""
This file is derived from
[text-generation-webui openai extension embeddings](https://github.com/oobabooga/text-generation-webui/blob/1a7c027386f43b84f3ca3b0ff04ca48d861c2d7a/extensions/openai/embeddings.py)
and modified.
The changes introduced are: Suppression of progress bar,
typing/pydantic classes moved into this file,
embeddings function declared async.
"""

import asyncio
import os
import base64
import pathlib
from loguru import logger
import numpy as np
from transformers import AutoModel

from common import config
from common.utils import unwrap
from endpoints.OAI.types.embedding import (
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
)


embeddings_model = None


def load_embedding_model(model_path: pathlib.Path, device: str):
    try:
        from infinity_emb import EngineArgs, AsyncEmbeddingEngine
    except ModuleNotFoundError:
        logger.error(
            "Skipping embeddings because infinity-emb is not installed.\n"
            "Please run the following command in your environment "
            "to install extra packages:\n"
            "pip install -U .[extras]"
        )
        raise ModuleNotFoundError from None

    global embeddings_model
    try:
        engine_args = EngineArgs(
            model_name_or_path=str(model_path.resolve()),
            engine="torch",
            device="cpu",
            bettertransformer=False,
            model_warmup=False,
        )
        embeddings_model = AsyncEmbeddingEngine.from_args(engine_args)
        logger.info(f"Trying to load embeddings model: {model_path.name} on {device}")
    except Exception as e:
        embeddings_model = None
        raise e


async def embeddings(data: EmbeddingsRequest) -> dict:
    embeddings_config = config.embeddings_config()

    # Use CPU by default
    device = embeddings_config.get("embeddings_device", "cpu")
    if device == "auto":
        device = None

    model_path = pathlib.Path(embeddings_config.get("embeddings_model_dir"))
    model_path: pathlib.Path = model_path / embeddings_config.get(
        "embeddings_model_name"
    )
    if not model_path:
        logger.info("Embeddings model path not found")

    load_embedding_model(model_path, device)

    async with embeddings_model:
        embeddings, usage = await embeddings_model.embed(data.input)

        # OAI expects a return of base64 if the input is base64
        if data.encoding_format == "base64":
            embedding_data = [
                {
                    "object": "embedding",
                    "embedding": float_list_to_base64(emb),
                    "index": n,
                }
                for n, emb in enumerate(embeddings)
            ]
        else:
            embedding_data = [
                {"object": "embedding", "embedding": emb.tolist(), "index": n}
                for n, emb in enumerate(embeddings)
            ]

        response = {
            "object": "list",
            "data": embedding_data,
            "model": model_path.name,
            "usage": {
                "prompt_tokens": usage,
                "total_tokens": usage,
            },
        }
        return response


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
