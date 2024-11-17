"""Vision utilities for ExLlamaV2."""

import io
import base64
import re
from PIL import Image
import aiohttp
from common.networking import (
    handle_request_error,
)
from fastapi import HTTPException
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Tokenizer,
    ExLlamaV2VisionTower,
    ExLlamaV2MMEmbedding,
)
from functools import lru_cache


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


@lru_cache(20)
async def get_image_embedding(
    model: ExLlamaV2,
    tokenizer: ExLlamaV2Tokenizer,
    vision_model: ExLlamaV2VisionTower,
    url: str,
) -> ExLlamaV2MMEmbedding:
    image = await get_image(url)
    return vision_model.get_image_embeddings(
        model=model, tokenizer=tokenizer, image=image
    )
