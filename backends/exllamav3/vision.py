"""Vision utilities for ExLlamaV2."""

from async_lru import alru_cache

from common import model
from common.optional_dependencies import dependencies
from common.image_util import get_image

# Since this is used outside the Exl3 backend, the dependency
# may be optional
if dependencies.exllamav3:
    from exllamav3.tokenizer import MMEmbedding


# Fetch the return type on runtime
@alru_cache(20)
async def get_image_embedding_exl3(url: str) -> "MMEmbedding":
    image = await get_image(url)
    return model.container.vision_model.get_image_embeddings(
        tokenizer=model.container.tokenizer,
        image=image,
        text_alias=None,
    )


def clear_image_embedding_cache():
    get_image_embedding_exl3.cache_clear()
