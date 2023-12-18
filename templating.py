import pathlib
from functools import lru_cache
from importlib.metadata import version as package_version
from jinja2.sandbox import ImmutableSandboxedEnvironment
from packaging import version
from pydantic import BaseModel

# Small replication of AutoTokenizer's chat template system for efficiency

class PromptTemplate(BaseModel):
    name: str
    template: str

def get_prompt_from_template(messages, prompt_template: PromptTemplate, add_generation_prompt: bool):
    if version.parse(package_version("jinja2")) < version.parse("3.0.0"):
        raise ImportError(
            "Parsing these chat completion messages requires fastchat 0.2.23 or greater. "
            f"Current version: {version('jinja2')}\n"
            "Please upgrade fastchat by running the following command: "
            "pip install -U fschat[model_worker]"
        )

    compiled_template = _compile_template(prompt_template.template)
    return compiled_template.render(
        messages = messages,
        add_generation_prompt = add_generation_prompt
    )

# Inspired from https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1761
@lru_cache
def _compile_template(template: str):
    jinja_env = ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
    jinja_template = jinja_env.from_string(template)
    return jinja_template

def get_template_from_file(prompt_template_name: str):
    with open(pathlib.Path(f"templates/{prompt_template_name}.jinja"), "r") as raw_template:
        return PromptTemplate(
            name = prompt_template_name,
            template = raw_template.read()
        )
