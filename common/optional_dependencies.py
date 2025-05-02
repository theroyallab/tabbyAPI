"""Construct a model of all optional dependencies"""

import importlib.util
from pydantic import BaseModel, computed_field


# Declare the exported parts of this module
__all__ = ["dependencies"]


class DependenciesModel(BaseModel):
    """Model of which optional dependencies are installed."""

    torch: bool
    exllamav2: bool
    exllamav3: bool
    flash_attn: bool
    infinity_emb: bool
    sentence_transformers: bool

    @computed_field
    @property
    def extras(self) -> bool:
        return self.infinity_emb and self.sentence_transformers

    @computed_field
    @property
    def inference(self) -> bool:
        return self.torch and (self.exllamav2 or self.exllamav3) and self.flash_attn


def is_installed(package_name: str) -> bool:
    """Utility function to check if a package is installed."""

    spec = importlib.util.find_spec(package_name)
    return spec is not None


def get_installed_deps() -> DependenciesModel:
    """Check if optional dependencies are installed by looping over the fields."""

    fields = DependenciesModel.model_fields

    installed_deps = {}

    for field_name in fields.keys():
        installed_deps[field_name] = is_installed(field_name)

    return DependenciesModel(**installed_deps)


dependencies = get_installed_deps()
