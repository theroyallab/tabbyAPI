"""Vision utilities for ExLlamaV3."""

from async_lru import alru_cache

from common import model
from common.optional_dependencies import dependencies
from common.image_util import get_image
from hashlib import blake2b

# Since this is used outside the Exl3 backend, the dependency
# may be optional
if dependencies.exllamav3:
    from exllamav3.tokenizer import MMEmbedding

_embedding_cache: dict[bytes, tuple[str, "MMEmbedding"]] = {}

def _image_key_128(s: str) -> bytes:
    return blake2b(s.encode("utf-8"), digest_size=16).digest()

async def get_image_embedding_exl3(url: str) -> "MMEmbedding":
    key = _image_key_128(url)

    cached = _embedding_cache.get(key)
    if cached is not None:
        cached_url, embedding = cached
        if cached_url == url:   # safe collision check
            return embedding

    image = await get_image(url)
    embedding = model.container.vision_model.get_image_embeddings(
        tokenizer=model.container.tokenizer,
        image=image,
        text_alias=None,
    )
    _embedding_cache[key] = (url, embedding)
    return embedding

def clear_image_embedding_cache():
    _embedding_cache.clear()
