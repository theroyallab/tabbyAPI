from common import model
from loguru import logger
from pydantic import BaseModel, Field
from typing import List

from common.optional_dependencies import dependencies


class MultimodalEmbeddingWrapper(BaseModel):
    """Common multimodal embedding wrapper"""

    type: str = None
    content: list = Field(default_factory=list)
    text_alias: List[str] = Field(default_factory=list)

    async def add(self, url: str):
        # Determine the type of vision embedding to use
        if not self.type:
            container = model.container
            if container and getattr(container, "vision_model", None):
                container_name = container.__class__.__name__
                if dependencies.exllamav2 and container_name == "ExllamaV2Container":
                    self.type = "ExLlamaV2MMEmbedding"
                elif dependencies.exllamav3 and container_name == "ExllamaV3Container":
                    self.type = "MMEmbedding"

        # Create the embedding
        if self.type == "ExLlamaV2MMEmbedding":
            from backends.exllamav2.vision import get_image_embedding_exl2

            embedding = await get_image_embedding_exl2(url)
            self.content.append(embedding)
            self.text_alias.append(embedding.text_alias)
        elif self.type == "MMEmbedding":
            from backends.exllamav3.vision import get_image_embedding_exl3

            embedding = await get_image_embedding_exl3(url)
            self.content.append(embedding)
            self.text_alias.append(embedding.text_alias)
        else:
            logger.error("No valid vision model to create embedding")
