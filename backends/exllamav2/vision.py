"""Vision utilities for ExLlamaV2."""

from async_lru import alru_cache

from common import model
from common.optional_dependencies import dependencies
from common.image_util import get_image

# Since this is used outside the Exl2 backend, the dependency
# may be optional
if dependencies.exllamav2:
    from exllamav2.generator import ExLlamaV2MMEmbedding


# Fetch the return type on runtime
@alru_cache(20)
async def get_image_embedding(url: str) -> "ExLlamaV2MMEmbedding":
    image = await get_image(url)
    return model.container.vision_model.get_image_embeddings(
        model=model.container.model,
        tokenizer=model.container.tokenizer,
        image=image,
        text_alias=None,
    )


def clear_image_embedding_cache():
    get_image_embedding.cache_clear()
