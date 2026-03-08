from pydantic import BaseModel, Field

from common.sampling import SamplerOverridesContainer


class SamplerOverrideListResponse(SamplerOverridesContainer):
    """Sampler override list response"""

    presets: list[str] | None


class SamplerOverrideSwitchRequest(BaseModel):
    """Sampler override switch request"""

    preset: str | None = Field(
        default=None, description="Pass a sampler override preset name"
    )

    overrides: dict | None = Field(
        default=None,
        description=(
            "Sampling override parent takes in individual keys and overrides. "
            + "Ignored if preset is provided."
        ),
        examples=[
            {
                "top_p": {
                    "override": 1.5,
                    "force": False,
                }
            }
        ],
    )
