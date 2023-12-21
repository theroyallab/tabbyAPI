from typing import Dict
from pydantic import BaseModel


# Logging preference config
class LogConfig(BaseModel):
    prompt: bool = False
    generation_params: bool = False


# Global reference to logging preferences
config = LogConfig()


# Wrapper to set the logging config for generations
def update_from_dict(options_dict: Dict[str, bool]):
    global config

    # Force bools on the dict
    for value in options_dict.values():
        if value is None:
            value = False

    config = LogConfig.model_validate(options_dict)


def broadcast_status():
    enabled = []
    if config.prompt:
        enabled.append("prompts")

    if config.generation_params:
        enabled.append("generation params")

    if len(enabled) > 0:
        print("Generation logging is enabled for: " + ", ".join(enabled))
    else:
        print("Generation logging is disabled")


# Logs generation parameters to console
def log_generation_params(**kwargs):
    if config.generation_params:
        print(f"Generation options: {kwargs}\n")


def log_prompt(prompt: str):
    if config.prompt:
        print(f"Prompt: {prompt if prompt else 'Empty'}\n")


def log_response(response: str):
    if config.prompt:
        print(f"Response: {response if response else 'Empty'}\n")
