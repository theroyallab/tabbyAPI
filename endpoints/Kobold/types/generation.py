from typing import List, Optional

from pydantic import BaseModel, Field
from common import model
from common.sampling import BaseSamplerRequest, get_default_sampler_value
from common.utils import flat_map, unwrap


class GenerateRequest(BaseSamplerRequest):
    prompt: str
    genkey: Optional[str] = None
    use_default_badwordsids: Optional[bool] = False
    dynatemp_range: Optional[float] = Field(
        default_factory=get_default_sampler_value("dynatemp_range")
    )

    def to_gen_params(self, **kwargs):
        # Exl2 uses -1 to include all tokens in repetition penalty
        if self.penalty_range == 0:
            self.penalty_range = -1

        if self.dynatemp_range:
            self.min_temp = self.temperature - self.dynatemp_range
            self.max_temp = self.temperature + self.dynatemp_range

        # Move badwordsids into banned tokens for generation
        if self.use_default_badwordsids:
            bad_words_ids = unwrap(
                model.container.generation_config.bad_words_ids,
                model.container.hf_config.get_badwordsids(),
            )

            if bad_words_ids:
                self.banned_tokens += flat_map(bad_words_ids)

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
