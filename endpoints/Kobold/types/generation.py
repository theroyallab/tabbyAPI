from typing import List, Optional

from pydantic import BaseModel, Field
from common.sampling import BaseSamplerRequest


class GenerateRequest(BaseSamplerRequest):
    prompt: str
    genkey: Optional[str] = None
    use_default_badwordsids: Optional[bool] = False

    def to_gen_params(self, **kwargs):
        # Exl2 uses -1 to include all tokens in repetition penalty
        if self.penalty_range == 0:
            self.penalty_range = -1

        return super().to_gen_params(**kwargs)


class GenerateResponseResult(BaseModel):
    text: str


class GenerateResponse(BaseModel):
    results: List[GenerateResponseResult] = Field(default_factory=list)


class StreamGenerateChunk(BaseModel):
    token: str


class AbortRequest(BaseModel):
    genkey: str


class AbortResponse(BaseModel):
    success: bool


class CheckGenerateRequest(BaseModel):
    genkey: str
