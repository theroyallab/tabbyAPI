from typing import List, TYPE_CHECKING

from pydantic import BaseModel, Field

from backends.exllamav3.vision import get_image_embedding_exl3

if TYPE_CHECKING:
    from backends.exllamav3.model import ExllamaV3Container


class MultimodalEmbeddingWrapper(BaseModel):
    """Common multimodal embedding wrapper"""

    content: list = Field(default_factory=list)
    text_alias: List[str] = Field(default_factory=list)

    async def add(self, container: "ExllamaV3Container", url: str):
        embedding = await get_image_embedding_exl3(container, url)
        self.content.append(embedding)
        self.text_alias.append(embedding.text_alias)
