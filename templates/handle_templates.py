# Acknowledgements:
# The file 'available_templates.yaml' corresponds to the 'config.yml' file found in the path 'text-generation-webui/models/config.yaml'.
# This file is credited to the developer 'oobabooga'.

import yaml
import re

from fastchat.conversation import (
    get_conv_template,
    conv_templates,
    register_conv_template,
    Conversation,
    SeparatorStyle,
)

TEMPLATES = yaml.safe_load(open("./templates/available_templates.yaml", "r").read())

memo = {}


def get_conversatiom_template_name(model_name: str, template: str, system_message: str = None):
    if template == "ChatML":
        # todo: load stop_token_ids from config
        conversation_settings= {
            "system_template": "<|im_start|>system\n{system_message}",
            "system_message": (
                system_message
                or "You are Tiago, a Large Language Model. Write out your reasoning step-by-step to be sure you get the right answers!"
            ),
            "roles": ("<|im_start|>user", "<|im_start|>assistant"),
            "sep_style": SeparatorStyle.CHATML,
            "sep": "<|im_end|>",
            "stop_token_ids": [32000, 32001],
        }
        register_conv_template(
            Conversation(
                name=model_name,
                **conversation_settings,
            )
        )
        return model_name
    if template == 'Mistral':
        return 'mistral'
    if template == 'Alpaca':
        return 'alpaca'
    if template == 'Llama-v2':
        return 'llama-2'
    
    # Todo: add more templates


def get_instructions_template(template: str) -> dict[str, str]:
    return yaml.safe_load(open(f"./templates/instructions/{template}.yaml", "r").read())


def get_conversation_template(model_path: str):
    # return memoized value if available
    if model_path in memo:
        conversation_template_name = memo[model_path]
        return get_conv_template(conversation_template_name)

    # return default template if available
    if model_path in conv_templates:
        return get_conv_template(model_path)

    # -- attempt to register new template
    # metadata
    metadata = {
        k: v
        for pat in TEMPLATES
        if re.match(pat.lower(), model_path.lower())
        for k, v in TEMPLATES[pat].items()
    }

    # return default template if no metadata is available
    if "instruction_template" not in metadata:
        return get_conv_template("one_shot")

    # registration
    template_name: str = metadata["instruction_template"]
    conversation_template_name = get_conversatiom_template_name(model_path, template_name)
    
    if conversation_template_name:
        memo[model_path] = conversation_template_name
        return get_conv_template(conversation_template_name)

    # return default template if registration fails
    return get_conv_template("one_shot")
