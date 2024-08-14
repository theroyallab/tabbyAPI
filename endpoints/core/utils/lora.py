import pathlib

from common import model
from endpoints.core.types.lora import LoraCard, LoraList


def get_lora_list(lora_path: pathlib.Path):
    """Get the list of Lora cards from the provided path."""
    lora_list = LoraList()
    for path in lora_path.iterdir():
        if path.is_dir():
            lora_card = LoraCard(id=path.name)
            lora_list.data.append(lora_card)

    return lora_list


def get_active_loras():
    if model.container:
        active_loras = [
            LoraCard(
                id=pathlib.Path(lora.lora_path).parent.name,
                scaling=lora.lora_scaling * lora.lora_r / lora.lora_alpha,
            )
            for lora in model.container.get_loras()
        ]
    else:
        active_loras = []

    return LoraList(data=active_loras)
