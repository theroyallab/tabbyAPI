from pydantic import BaseModel, Field
from typing import Optional


class SamplerOverrideSwitchRequest(BaseModel):
    """Sampler override switch request"""

    preset: Optional[str] = Field(
        default=None, description="Pass a sampler override preset name"
    )

    overrides: Optional[dict] = Field(
        default=None,
        description=(
            "Sampling override parent takes in individual keys and overrides."
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
