import aiohttp
import base64
import io
import re

from fastapi import HTTPException
from PIL import Image

from common.networking import (
    handle_request_error,
)
from common.tabby_config import config


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
        if config.network.disable_fetch_requests:
            error_message = handle_request_error(
                f"Failed to fetch image from {url} as fetch requests are disabled.",
                exc_info=False,
            ).error.message

            raise HTTPException(400, error_message)

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
