"""Small replication of AutoTokenizer's chat template system for efficiency"""
import json
import pathlib
from functools import lru_cache
from importlib.metadata import version as package_version

from jinja2.sandbox import ImmutableSandboxedEnvironment
from packaging import version
from pydantic import BaseModel


class PromptTemplate(BaseModel):
    """A template for chat completion prompts."""

    name: str
    template: str


def get_prompt_from_template(
    messages, prompt_template: PromptTemplate, add_generation_prompt: bool
):
    """Get a prompt from a template and a list of messages."""
    if version.parse(package_version("jinja2")) < version.parse("3.0.0"):
        raise ImportError(
            "Parsing these chat completion messages requires jinja2 3.0.0 "
            f"or greater. Current version: {package_version('jinja2')}\n"
            "Please upgrade jinja by running the following command: "
            "pip install --upgrade jinja2"
        )

    compiled_template = _compile_template(prompt_template.template)
    return compiled_template.render(
        messages=messages, add_generation_prompt=add_generation_prompt
    )


# Inspired from
# https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1761
@lru_cache
def _compile_template(template: str):
    jinja_env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    )
    jinja_template = jinja_env.from_string(template)
    return jinja_template


def find_template_from_model(model_path: pathlib.Path):
    """Find a matching template name from a model path."""
    model_name = model_path.name
    template_directory = pathlib.Path("templates")
    for filepath in template_directory.glob("*.jinja"):
        template_name = filepath.stem.lower()

        # Check if the template name is present in the model name
        if template_name in model_name.lower():
            return template_name

    return None


def get_template_from_file(prompt_template_name: str):
    """Get a template from a jinja file."""
    with open(
        pathlib.Path(f"templates/{prompt_template_name}.jinja"),
        "r",
        encoding="utf8",
    ) as raw_template:
        return PromptTemplate(
            name=prompt_template_name, template=raw_template.read()
        )


def get_template_from_config(model_config_path: pathlib.Path):
    """Get a template from model config."""
    with open(model_config_path, "r", encoding="utf8") as model_config_file:
        model_config = json.load(model_config_file)
        chat_template = model_config.get("chat_template")
        if chat_template:
            return PromptTemplate(
                name="from_model_config", template=chat_template
            )

    return None
