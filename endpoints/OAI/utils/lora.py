import pathlib

from endpoints.OAI.types.lora import LoraCard, LoraList


def get_lora_list(lora_path: pathlib.Path):
    """Get the list of Lora cards from the provided path."""
    lora_list = LoraList()
    for path in lora_path.iterdir():
        if path.is_dir():
            lora_card = LoraCard(id=path.name)
            lora_list.data.append(lora_card)

    return lora_list
