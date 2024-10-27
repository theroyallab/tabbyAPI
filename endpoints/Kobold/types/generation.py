from functools import partial
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

from common import model
from common.sampling import BaseSamplerRequest, get_default_sampler_value
from common.utils import flat_map, unwrap


class GenerateRequest(BaseSamplerRequest):
    prompt: str
    genkey: Optional[str] = None
    use_default_badwordsids: Optional[bool] = False
    dynatemp_range: Optional[float] = Field(
        default_factory=partial(get_default_sampler_value, "dynatemp_range")
    )

    # Validate on the parent class's fields
    @field_validator("penalty_range", mode="before")
    def validate_penalty_range(cls, v):
        return -1 if v == 0 else v

    @field_validator("dynatemp_range", mode="before")
    def validate_temp_range(cls, v, field_info):
        if v > 0:
            # A default temperature is always 1
            temperature = unwrap(field_info.data.get("temperature"), 1)

            field_info.data["min_temp"] = temperature - v
            field_info.data["max_temp"] = temperature + v

        return v

    @field_validator("use_default_badwordsids", mode="before")
    def validate_badwordsids(cls, v, field_info):
        if v:
            bad_words_ids = []

            # Try fetching badwordsids from generation config and hf config
            if model.container.generation_config:
                bad_words_ids += model.container.generation_config.bad_words_ids

            if model.container.hf_config:
                bad_words_ids += model.container.hf_config.get_badwordsids()

            # Add badwordsids to existing banned_tokens and ban the EOS token
            banned_tokens = unwrap(field_info.data.get("banned_tokens"), [])

            if bad_words_ids:
                field_info.data["banned_tokens"] = banned_tokens + flat_map(
                    bad_words_ids
                )
                field_info.data["ban_eos_token"] = True

        return v


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
