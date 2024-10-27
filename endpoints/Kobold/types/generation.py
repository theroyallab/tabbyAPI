from functools import partial
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
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

    # Validate on the parent class's values
    @field_validator("penalty_range")
    def validate_penalty_range(cls, v):
        return -1 if v == 0 else v

    @field_validator("min_temp", "max_temp")
    def validate_temp_range(cls, v, field_info):
        if (
            "dynatemp_range" in field_info.data
            and field_info.data["dynatemp_range"] is not None
        ):
            temperature = unwrap(field_info.data.get("temperature"), 0)

            if field_info.field_name == "min_temp":
                return temperature - field_info.data["dynatemp_range"]
            elif field_info.field_name == "max_temp":
                return temperature + field_info.data["dynatemp_range"]

        return v

    @field_validator("banned_tokens")
    def validate_badwordsids(cls, v, field_info):
        if field_info.data.get("use_default_badwordsids"):
            bad_words_ids = unwrap(
                model.container.generation_config.bad_words_ids,
                model.container.hf_config.get_badwordsids(),
            )

            if bad_words_ids:
                return v + flat_map(bad_words_ids)

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
