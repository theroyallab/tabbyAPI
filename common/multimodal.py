from typing import List
from backends.exllamav2.vision import get_image_embedding
from common import model
from pydantic import BaseModel
from loguru import logger

from common.optional_dependencies import dependencies

if dependencies.exllamav2:
    from exllamav2 import ExLlamaV2VisionTower


class MultimodalEmbeddingWrapper(BaseModel):
    """Common multimodal embedding wrapper"""

    type: str = None
    content: List = []
    text_alias: List[str] = []


async def add_image_embedding(
    embeddings: MultimodalEmbeddingWrapper, url: str
) -> MultimodalEmbeddingWrapper:
    # Determine the type of vision embedding to use
    if not embeddings.type:
        if isinstance(model.container.vision_model, ExLlamaV2VisionTower):
            embeddings.type = "ExLlamaV2MMEmbedding"

    if embeddings.type == "ExLlamaV2MMEmbedding":
        embedding = await get_image_embedding(url)
        embeddings.content.append(embedding)
        embeddings.text_alias.append(embedding.text_alias)
    else:
        logger.error("No valid vision model to create embedding")

    return embeddings
