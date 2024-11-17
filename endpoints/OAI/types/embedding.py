from typing import List, Optional, Union

from pydantic import BaseModel, Field


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]] = Field(
        ..., description="List of input texts to generate embeddings for."
    )
    encoding_format: str = Field(
        "float",
        description="Encoding format for the embeddings. "
        "Can be 'float' or 'base64'.",
    )
    model: Optional[str] = Field(
        None,
        description="Name of the embedding model to use. "
        "If not provided, the default model will be used.",
    )


class EmbeddingObject(BaseModel):
    object: str = Field("embedding", description="Type of the object.")
    embedding: List[float] = Field(
        ..., description="Embedding values as a list of floats."
    )
    index: int = Field(
        ..., description="Index of the input text corresponding to " "the embedding."
    )


class EmbeddingsResponse(BaseModel):
    object: str = Field("list", description="Type of the response object.")
    data: List[EmbeddingObject] = Field(..., description="List of embedding objects.")
    model: str = Field(..., description="Name of the embedding model used.")
    usage: UsageInfo = Field(..., description="Information about token usage.")
