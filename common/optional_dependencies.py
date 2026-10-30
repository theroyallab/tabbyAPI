"""Construct a model of all optional dependencies"""

import functools
import importlib.util
import pathlib
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Optional
from common.logger import xlogger
from packaging import version
from pydantic import BaseModel, computed_field


# Declare the exported parts of this module
__all__ = ["dependencies"]


class DependenciesModel(BaseModel):
    """Model of which optional dependencies are installed."""

    torch: bool
    exllamav3: bool
    infinity_emb: bool
    sentence_transformers: bool

    @computed_field
    @property
    def extras(self) -> bool:
        return self.infinity_emb and self.sentence_transformers

    @computed_field
    @property
    def inference(self) -> bool:
        return self.torch and self.exllamav3


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


def check_package_version(package_name: str, required_version_str: str):
    """
    Fetches and verifies a given package version.

    This assumes that the required package is installed.
    """

    required_version = version.parse(required_version_str)

    # Prefer the version reported by the package itself. Installed metadata
    # can be stale when running a source checkout from sys.path.
    try:
        version_str = importlib.import_module(f"{package_name}.version").__version__
    except (ImportError, AttributeError):
        version_str = package_version(package_name)

    current_version = version.parse(version_str.split("+")[0])

    unsupported_message = (
        f"ERROR: TabbyAPI requires {package_name} {required_version} "
        f"or greater. Your current version is {current_version}. "
        "Please update your dependencies."
    )

    if current_version < required_version:
        raise RuntimeError(unsupported_message)
    else:
        xlogger.info(f"{package_name} version: {current_version}")


dependencies = get_installed_deps()


@functools.cache
def package_version_info(package_name: str) -> Optional[dict]:
    """
    An installed package's version, read without importing it, and the git
    commit when it runs from a source checkout, whose version file can lag
    behind. None when the package isn't installed.

    Cached, so the first call fixes the commit: call it at startup, before a
    checkout can move on from the code that is running.
    """

    spec = importlib.util.find_spec(package_name)
    if spec is None or not spec.submodule_search_locations:
        return None
    package_dir = pathlib.Path(next(iter(spec.submodule_search_locations)))

    version_str = None
    try:
        text = (package_dir / "version.py").read_text(encoding="utf8")
        match = re.search(r"__version__\s*=\s*[\"']([^\"']+)[\"']", text)
        version_str = match.group(1) if match else None
    except OSError:
        pass
    if version_str is None:
        try:
            version_str = package_version(package_name)
        except PackageNotFoundError:
            pass

    commit = None
    if (package_dir.parent / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=package_dir.parent,
                capture_output=True,
                text=True,
                timeout=5,
            )
            commit = result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass

    return {"version": version_str, "commit": commit}
