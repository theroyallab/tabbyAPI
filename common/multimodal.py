from typing import List
from backends.exllamav2.vision import get_image_embedding
from common import model
from loguru import logger

from common.optional_dependencies import dependencies

if dependencies.exllamav2:
    from exllamav2 import ExLlamaV2VisionTower


class MultimodalEmbeddingWrapper:
    """Common multimodal embedding wrapper"""

    type: str = None
    content: List = []
    text_alias: List[str] = []

    async def add(self, url: str):
        # Determine the type of vision embedding to use
        if not self.type:
            if isinstance(model.container.vision_model, ExLlamaV2VisionTower):
                self.type = "ExLlamaV2MMEmbedding"

        if self.type == "ExLlamaV2MMEmbedding":
            embedding = await get_image_embedding(url)
            self.content.append(embedding)
            self.text_alias.append(embedding.text_alias)
        else:
            logger.error("No valid vision model to create embedding")
