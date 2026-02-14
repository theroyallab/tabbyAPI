"""Small replication of AutoTokenizer's chat template system for efficiency"""

import traceback
import aiofiles
import json
import pathlib
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import version as package_version
from typing import List, Optional
from jinja2 import Template, TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from loguru import logger
from markupsafe import Markup
from packaging import version


from common.utils import unwrap


class TemplateLoadError(Exception):
    """Raised on prompt template load"""

    pass


VALID_TOOL_CALL_FORMATS = {"json", "xml", "auto"}


@dataclass
class TemplateMetadata:
    """Represents the parsed metadata from a template."""

    stop_strings: List[str] = field(default_factory=list)
    tool_start: Optional[str] = None
    tool_end: Optional[str] = None
    tool_call_format: str = "json"


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

    @staticmethod
    def _tojson_compat(value, indent=None, ensure_ascii=True):
        """Compatibility JSON filter for chat templates.

        Some model templates call ``tojson(ensure_ascii=False)`` while the
        bundled Jinja filter may not accept that keyword in sandboxed mode.
        """
        return Markup(
            json.dumps(
                value,
                indent=indent,
                ensure_ascii=ensure_ascii,
                separators=(",", ": "),
            )
        )

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
                template_metadata.tool_start = template_module.tool_start

        if hasattr(template_module, "tool_end"):
            if isinstance(template_module.tool_end, str):
                template_metadata.tool_end = template_module.tool_end

        if hasattr(template_module, "tool_call_format"):
            fmt = template_module.tool_call_format
            if isinstance(fmt, str) and fmt in VALID_TOOL_CALL_FORMATS:
                template_metadata.tool_call_format = fmt
                logger.debug(f"Template tool_call_format: {fmt}")
            else:
                logger.warning(
                    f"Invalid tool_call_format '{fmt}' in template, "
                    f"defaulting to 'json'. "
                    f"Valid values: {VALID_TOOL_CALL_FORMATS}"
                )

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

        # Some models require strftime_now, e.g. Granite3
        def strftime_now(format):
            current_time = datetime.now()
            return current_time.strftime(format)

        # Exception handler
        def raise_exception(message):
            raise TemplateError(message)

        self.environment.globals["strftime_now"] = strftime_now
        self.environment.globals["raise_exception"] = raise_exception
        self.environment.filters["tojson"] = self._tojson_compat

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


async def find_prompt_template(template_name, model_dir: pathlib.Path):
    """Tries to find a prompt template using various methods."""

    logger.info("Attempting to load a prompt template if present.")

    find_template_functions = [
        lambda: PromptTemplate.from_file(model_dir / "chat_template.jinja"),
        lambda: PromptTemplate.from_model_json(
            model_dir / "chat_template.json",
            key="chat_template",
        ),
        lambda: PromptTemplate.from_model_json(
            model_dir / "tokenizer_config.json",
            key="chat_template",
        ),
        lambda: PromptTemplate.from_file(find_template_from_model(model_dir)),
    ]

    # Find the template in the model directory if it exists
    model_dir_template_path = model_dir / "tabby_template.jinja"
    if model_dir_template_path.exists():
        find_template_functions[:0] = [
            lambda: PromptTemplate.from_file(model_dir_template_path)
        ]

    # Add lookup from prompt template name if provided
    # TODO: Possibly link to the TokenizerConfig class
    if template_name:
        find_template_functions[:0] = [
            lambda: PromptTemplate.from_file(pathlib.Path("templates") / template_name),
            lambda: PromptTemplate.from_model_json(
                model_dir / "tokenizer_config.json",
                key="chat_template",
                name=template_name,
            ),
        ]

    # Continue on exception since functions are tried as they fail
    for template_func in find_template_functions:
        try:
            prompt_template = await template_func()
            if prompt_template is not None:
                return prompt_template
        except TemplateLoadError as e:
            logger.warning(f"TemplateLoadError: {str(e)}")
            continue
        except Exception:
            logger.error(traceback.format_exc())
            logger.warning(
                "An unexpected error happened when trying to load the template. "
                "Trying other methods."
            )
            continue
