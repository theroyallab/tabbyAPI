"""Argparser for overriding config values"""

import argparse
from pydantic import BaseModel

from common.config_models import TabbyConfigModel
from common.utils import is_list_type


def add_field_to_group(group, field_name, field_type, field) -> None:
    """
    Adds a Pydantic field to an argparse argument group.
    """

    kwargs = {
        "help": field.description if field.description else "No description available",
    }

    # If the inner type contains a list, specify argparse as such
    if is_list_type(field_type):
        kwargs["nargs"] = "+"

    group.add_argument(f"--{field_name}", **kwargs)


def init_argparser() -> argparse.ArgumentParser:
    """
    Initializes an argparse parser based on a Pydantic config schema.
    """

    parser = argparse.ArgumentParser(description="TabbyAPI server")

    # Loop through each top-level field in the config
    for field_name, field_info in TabbyConfigModel.model_fields.items():
        field_type = field_info.annotation
        group = parser.add_argument_group(
            field_name, description=f"Arguments for {field_name}"
        )

        # Check if the field_type is a Pydantic model
        if issubclass(field_type, BaseModel):
            for sub_field_name, sub_field_info in field_type.model_fields.items():
                sub_field_name = sub_field_name.replace("_", "-")
                sub_field_type = sub_field_info.annotation
                add_field_to_group(
                    group, sub_field_name, sub_field_type, sub_field_info
                )
        else:
            field_name = field_name.replace("_", "-")
            group.add_argument(f"--{field_name}", help=f"Argument for {field_name}")

    return parser


def convert_args_to_dict(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict:
    """Broad conversion of surface level arg groups to dictionaries"""

    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {}
        for arg in group._group_actions:
            value = getattr(args, arg.dest, None)
            if value is not None:
                group_dict[arg.dest] = value

            arg_groups[group.title] = group_dict

    return arg_groups
