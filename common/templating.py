"""Small replication of AutoTokenizer's chat template system for efficiency"""

import json
import pathlib
from functools import lru_cache
from importlib.metadata import version as package_version
from jinja2 import Template, TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from loguru import logger
from packaging import version
from pydantic import BaseModel
from typing import Optional, Dict


class PromptTemplate(BaseModel):
    """A template for chat completion prompts."""

    name: str
    template: str


def get_prompt_from_template(
    messages,
    prompt_template: PromptTemplate,
    add_generation_prompt: bool,
    special_tokens: Optional[Dict[str, str]] = None,
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
    rendered_template = compiled_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        **special_tokens,
    )
    template_stop_strings = _get_template_stop_strings(compiled_template)

    return rendered_template, template_stop_strings


# Inspired from
# https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1761
# TODO: Migrate to compile when template is loaded (removes the need for an lru_cache)
@lru_cache
def _compile_template(template: str):
    """Compiles a Jinja2 template"""

    # Exception handler
    def raise_exception(message):
        raise TemplateError(message)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals["raise_exception"] = raise_exception

    jinja_template = jinja_env.from_string(template)
    return jinja_template


# TODO: Migrate to run during template load
def _get_template_stop_strings(prompt_template: Template):
    """Appends extra stop strings if present in a chat template."""

    extra_stop_strings = []

    if hasattr(prompt_template.module, "stop_strings"):
        if isinstance(prompt_template.module.stop_strings, list):
            extra_stop_strings += prompt_template.module.stop_strings
        else:
            logger.warning(
                "Skipping append of stopping strings from chat template "
                "because stop_strings isn't a list."
            )

    return extra_stop_strings


def get_all_templates():
    """Fetches all templates from the templates directory"""

    template_directory = pathlib.Path("templates")
    return template_directory.glob("*.jinja")


def find_template_from_model(model_path: pathlib.Path):
    """Find a matching template name from a model path."""
    model_name = model_path.name
    template_files = get_all_templates()

    for filepath in template_files:
        template_name = filepath.stem.lower()

        # Check if the template name is present in the model name
        if template_name in model_name.lower():
            return template_name
        else:
            raise LookupError("Could not find template from model name.")


def get_template_from_file(prompt_template_name: str):
    """Get a template from a jinja file."""

    template_path = pathlib.Path(f"templates/{prompt_template_name}.jinja")
    if template_path.exists():
        with open(template_path, "r", encoding="utf8") as raw_template:
            return PromptTemplate(
                name=prompt_template_name, template=raw_template.read()
            )
    else:
        # Let the user know if the template file isn't found
        raise FileNotFoundError(f'Template "{prompt_template_name}" not found.')


# Get a template from a JSON file
# Requires a key and template name
def get_template_from_model_json(json_path: pathlib.Path, key: str, name: str):
    """Get a template from a JSON file. Requires a key and template name"""
    if json_path.exists():
        with open(json_path, "r", encoding="utf8") as config_file:
            model_config = json.load(config_file)
            chat_template = model_config.get(key)
            if chat_template:
                return PromptTemplate(name=name, template=chat_template)
    else:
        raise FileNotFoundError(f'Model JSON path "{json_path}" not found.')
