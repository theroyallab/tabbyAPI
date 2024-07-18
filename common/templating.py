"""Small replication of AutoTokenizer's chat template system for efficiency"""

import json
import pathlib
from importlib.metadata import version as package_version
from typing import Optional
from jinja2 import Template, TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from loguru import logger
from packaging import version

from common.utils import unwrap


class TemplateLoadError(Exception):
    """Raised on prompt template load"""

    pass


class PromptTemplate:
    """A template for chat completion prompts."""

    name: str
    raw_template: str
    template: Template
    environment: ImmutableSandboxedEnvironment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    )

    def stop_strings(self, template_vars: dict):
        """Appends extra stop strings if present in a chat template."""

        extra_stop_strings = []
        template_module = self.template.make_module(template_vars)

        if hasattr(template_module, "stop_strings"):
            if isinstance(template_module.stop_strings, list):
                extra_stop_strings += template_module.stop_strings
            else:
                logger.warning(
                    "Skipping append of stopping strings from chat template "
                    "because stop_strings isn't a list."
                )

        return extra_stop_strings
    
    def tool_params(self, template_vars: dict):
        """grabs tool params from the template"""

        tool_start = None
        tool_end = None
        template_module = self.template.make_module(template_vars)

        if hasattr(template_module, "tool_start"):
            if isinstance(template_module.tool_start, str):
                tool_start = template_module.tool_start

        if hasattr(template_module, "tool_end"):
            if isinstance(template_module.tool_end, str):
                tool_end = template_module.tool_end

        return tool_start, tool_end

    def render(self, template_vars: dict):
        """Get a prompt from a template and a list of messages."""
        if version.parse(package_version("jinja2")) < version.parse("3.0.0"):
            raise ImportError(
                "Parsing these chat completion messages requires jinja2 3.0.0 "
                f"or greater. Current version: {package_version('jinja2')}\n"
                "Please upgrade jinja by running the following command: "
                "pip install --upgrade jinja2"
            )

        rendered_template = self.template.render(**template_vars)
        template_stop_strings = self.stop_strings(template_vars)

        return rendered_template, template_stop_strings

    def compile(self, template_str: str):
        """Compiles and stores a jinja2 template"""

        # Exception handler
        def raise_exception(message):
            raise TemplateError(message)

        self.environment.globals["raise_exception"] = raise_exception

        return self.environment.from_string(template_str)

    def __init__(self, name: str, raw_template: str):
        """Initializer for the PromptTemplate class."""

        self.name = name
        self.raw_template = raw_template
        self.template = self.compile(raw_template)

    @classmethod
    def from_file(self, prompt_template_name: str):
        """Get a template from a jinja file."""

        template_path = pathlib.Path(f"templates/{prompt_template_name}.jinja")
        if template_path.exists():
            with open(template_path, "r", encoding="utf8") as raw_template_stream:
                return PromptTemplate(
                    name=prompt_template_name,
                    raw_template=raw_template_stream.read(),
                )
        else:
            # Let the user know if the template file isn't found
            raise TemplateLoadError(
                f'Chat template "{prompt_template_name}" not found in files.'
            )

    @classmethod
    def from_model_json(
        self, json_path: pathlib.Path, key: str, name: Optional[str] = None
    ):
        """Get a template from a JSON file. Requires a key and template name"""
        if not json_path.exists():
            raise TemplateLoadError(f'Model JSON path "{json_path}" not found.')

        with open(json_path, "r", encoding="utf8") as config_file:
            model_config = json.load(config_file)
            chat_template = model_config.get(key)

            if not chat_template:
                raise TemplateLoadError(
                    "Could not find a value from chat_template key in the passed JSON. "
                    "Check the tokenizer config?"
                )

            if isinstance(chat_template, list):
                # Handles the new list style of chat templates
                if name:
                    wrapped_template = next(
                        (x for x in chat_template if x.get("name") == name),
                        {},
                    )
                else:
                    wrapped_template = chat_template[0]
                    name = unwrap(wrapped_template.get("name"), "from_tokenizer_config")

                selected_template = wrapped_template.get("template")

                if selected_template:
                    return PromptTemplate(name=name, raw_template=selected_template)
                else:
                    raise TemplateLoadError(
                        f'Chat template with name "{name}" not found '
                        "in model templates list."
                    )
            else:
                # Can safely assume the chat template is the old style
                return PromptTemplate(
                    name="from_tokenizer_config",
                    raw_template=chat_template,
                )


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
            raise TemplateLoadError("Could not find template from model name.")
