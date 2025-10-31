from functools import partial
from pydantic import BaseModel, Field, field_validator

from common.sampling import BaseSamplerRequest, get_default_sampler_value
from common.utils import unwrap


class GenerateRequest(BaseSamplerRequest):
    prompt: str
    genkey: str | None = None
    use_default_badwordsids: bool | None = False
    dynatemp_range: float | None = Field(
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

    # Currently only serves to ban EOS token, but can change
    @field_validator("use_default_badwordsids", mode="before")
    def validate_badwordsids(cls, v, field_info):
        field_info.data["ban_eos_token"] = v

        return v


class GenerateResponseResult(BaseModel):
    text: str


class GenerateResponse(BaseModel):
    results: list[GenerateResponseResult] = Field(default_factory=list)


class StreamGenerateChunk(BaseModel):
    token: str


class AbortRequest(BaseModel):
    genkey: str


class AbortResponse(BaseModel):
    success: bool


class CheckGenerateRequest(BaseModel):
    genkey: str
