"""Small replication of AutoTokenizer's chat template system for efficiency"""

import aiofiles
import json
import pathlib
from importlib.metadata import version as package_version
from typing import List, Optional
from jinja2 import Template, TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from loguru import logger
from packaging import version

from common.utils import unwrap


class TemplateLoadError(Exception):
    """Raised on prompt template load"""

    pass


class TemplateMetadata:
    """Represents the parsed metadata from a template."""

    stop_strings: List[str] = []
    tool_starts: List[str] = []


class PromptTemplate:
    """A template for chat completion prompts."""

    name: str
    raw_template: str
    template: Template
    environment: ImmutableSandboxedEnvironment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        enable_async=True,
        extensions=[loopcontrols],
    )
    metadata: Optional[TemplateMetadata] = None

    async def extract_metadata(self, template_vars: dict):
        """
        Returns deserialized template metadata from a chat template.

        NOTE: Requires all template vars to be passed in since the template
        is run once to make a module and errors can result.
        """

        # No need to extract new metadata if it already exists
        # This might be removed if stored metadata becomes arbitrary
        if self.metadata:
            return self.metadata

        template_metadata = TemplateMetadata()

        template_module = await self.template.make_module_async(template_vars)

        if hasattr(template_module, "stop_strings"):
            if isinstance(template_module.stop_strings, list):
                template_metadata.stop_strings += template_module.stop_strings
            else:
                logger.warning(
                    "Skipping append of stopping strings from chat template "
                    "because stop_strings isn't a list."
                )

        if hasattr(template_module, "tool_start"):
            if isinstance(template_module.tool_start, str):
                template_metadata.tool_starts.append(template_module.tool_start)

        if hasattr(template_module, "tool_start_token"):
            if isinstance(template_module.tool_start_token, int):
                template_metadata.tool_starts.append(template_module.tool_start_token)

        self.metadata = template_metadata
        return template_metadata

    async def render(self, template_vars: dict):
        """Get a prompt from a template and a list of messages."""
        if version.parse(package_version("jinja2")) < version.parse("3.0.0"):
            raise ImportError(
                "Parsing these chat completion messages requires jinja2 3.0.0 "
                f"or greater. Current version: {package_version('jinja2')}\n"
                "Please update jinja by running the following command: "
                "pip install --upgrade jinja2"
            )

        rendered_template = await self.template.render_async(**template_vars)

        return rendered_template

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
    async def from_file(cls, template_path: pathlib.Path):
        """Get a template from a jinja file."""

        # Add the jinja extension if it isn't provided
        if template_path.suffix.endswith(".jinja"):
            template_name = template_path.name.split(".jinja")[0]
        else:
            template_name = template_path.name
            template_path = template_path.with_suffix(".jinja")

        if template_path.exists():
            async with aiofiles.open(
                template_path, "r", encoding="utf8"
            ) as raw_template_stream:
                contents = await raw_template_stream.read()
                return cls(
                    name=template_name,
                    raw_template=contents,
                )
        else:
            # Let the user know if the template file isn't found
            raise TemplateLoadError(
                f'Chat template "{template_name}" not found in files.'
            )

    @classmethod
    async def from_model_json(
        cls, json_path: pathlib.Path, key: str, name: Optional[str] = None
    ):
        """Get a template from a JSON file. Requires a key and template name"""
        if not json_path.exists():
            raise TemplateLoadError(f'Model JSON path "{json_path}" not found.')

        async with aiofiles.open(json_path, "r", encoding="utf8") as config_file:
            contents = await config_file.read()
            model_config = json.loads(contents)
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
                return cls(
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
