"""Vision utilities for ExLlamaV3."""

from common import model
from common.optional_dependencies import dependencies
from common.image_util import get_image
from common.logger import xlogger

# Since this is used outside the Exl3 backend, the dependency
# may be optional
if dependencies.exllamav3:
    from exllamav3.tokenizer import MMEmbedding

from collections import OrderedDict
from hashlib import blake2b
from typing import OrderedDict as OrderedDictType

_EMBEDDING_CACHE_CAPACITY = 32
_embedding_cache: OrderedDictType[bytes, tuple[str, "MMEmbedding"]] = OrderedDict()


def _image_key_128(s: str) -> bytes:
    return blake2b(s.encode("utf-8"), digest_size=16).digest()


async def get_image_embedding_exl3(url: str) -> "MMEmbedding":
    key = _image_key_128(url)

    cached = _embedding_cache.get(key)
    if cached is not None:
        cached_url, embedding = cached
        if cached_url == url:
            _embedding_cache.move_to_end(key)
            return embedding

    image = await get_image(url)
    embedding = model.container.vision_model.get_image_embeddings(
        tokenizer=model.container.tokenizer,
        image=image,
        text_alias=None,
    )

    _embedding_cache[key] = (url, embedding)
    _embedding_cache.move_to_end(key)

    if len(_embedding_cache) > _EMBEDDING_CACHE_CAPACITY:
        _embedding_cache.popitem(last=False)

    xlogger.debug(
        "Stored MMEmbedding",
        {
            "text_alias": embedding.text_alias,
            "metadata": embedding.metadata,
            "token_length": embedding.mm_length,
            "cache_size": len(_embedding_cache),
        },
    )

    return embedding


def clear_image_embedding_cache():
    _embedding_cache.clear()
