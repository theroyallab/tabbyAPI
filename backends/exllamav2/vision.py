"""Vision utilities for ExLlamaV2."""

import io
import base64
import re
from PIL import Image
from common import model
import aiohttp
from common.networking import (
    handle_request_error,
)
from fastapi import HTTPException
from exllamav2.generator import ExLlamaV2MMEmbedding
from async_lru import alru_cache


async def get_image(url: str) -> Image:
    if url.startswith("data:image"):
        # Handle base64 image
        match = re.match(r"^data:image\/[a-zA-Z0-9]+;base64,(.*)$", url)
        if match:
            base64_image = match.group(1)
            bytes_image = base64.b64decode(base64_image)
        else:
            error_message = handle_request_error(
                "Failed to read base64 image input.",
                exc_info=False,
            ).error.message

            raise HTTPException(400, error_message)

    else:
        # Handle image URL
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    bytes_image = await response.read()
                else:
                    error_message = handle_request_error(
                        f"Failed to fetch image from {url}.",
                        exc_info=False,
                    ).error.message

                    raise HTTPException(400, error_message)

    return Image.open(io.BytesIO(bytes_image))


@alru_cache(20)
async def get_image_embedding(url: str) -> ExLlamaV2MMEmbedding:
    image = await get_image(url)
    return model.container.vision_model.get_image_embeddings(
        model=model.container.model,
        tokenizer=model.container.tokenizer,
        image=image,
        text_alias=None,
    )


def clear_image_embedding_cache():
    get_image_embedding.cache_clear()
