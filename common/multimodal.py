from backends.exllamav3.vision import get_image_embedding_exl3
from common import model
from common.logger import xlogger
from pydantic import BaseModel, Field
from typing import List

from common.optional_dependencies import dependencies

if dependencies.exllamav3:
    from exllamav3 import Model


class MultimodalEmbeddingWrapper(BaseModel):
    """Common multimodal embedding wrapper"""

    type: str = None
    content: list = Field(default_factory=list)
    text_alias: List[str] = Field(default_factory=list)

    async def add(self, url: str):
        # Determine the type of vision embedding to use
        if not self.type:
            if dependencies.exllamav3 and isinstance(model.container.vision_model, Model):
                self.type = "MMEmbedding"

        # Create the embedding
        if self.type == "MMEmbedding":
            embedding = await get_image_embedding_exl3(url)
            self.content.append(embedding)
            self.text_alias.append(embedding.text_alias)
        else:
            xlogger.error("No valid vision model to create embedding")
