from typing import List, Optional

from pydantic import BaseModel, Field
from common.sampling import BaseSamplerRequest, get_default_sampler_value


class GenerateRequest(BaseSamplerRequest):
    prompt: str
    use_default_badwordsids: Optional[bool] = False
    genkey: Optional[str] = None

    max_length: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("max_tokens"),
        examples=[150],
    )
    rep_pen_range: Optional[int] = Field(
        default_factory=lambda: get_default_sampler_value("penalty_range", -1),
    )
    rep_pen: Optional[float] = Field(
        default_factory=lambda: get_default_sampler_value("repetition_penalty", 1.0),
    )

    def to_gen_params(self, **kwargs):
        # Swap kobold generation params to OAI/Exl2 ones
        self.max_tokens = self.max_length
        self.repetition_penalty = self.rep_pen
        self.penalty_range = -1 if self.rep_pen_range == 0 else self.rep_pen_range

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
