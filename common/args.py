"""Argparser for overriding config values"""

import argparse
from typing import get_origin, get_args, Optional, Union, List
from pydantic import BaseModel
from common.tabby_config import config


def str_to_bool(value):
    """Converts a string into a boolean value"""

    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def argument_with_auto(value):
    """
    Argparse type wrapper for any argument that has an automatic option.

    Ex. rope_alpha
    """

    if value == "auto":
        return "auto"

    try:
        return float(value)
    except ValueError as ex:
        raise argparse.ArgumentTypeError(
            'This argument only takes a type of float or "auto"'
        ) from ex


def init_argparser():
    parser = argparse.ArgumentParser(description="TabbyAPI server")

    for field_name, field_type in config.__annotations__.items():
        group = parser.add_argument_group(
            field_name, description=f"Arguments for {field_name}"
        )

        # Loop through each field in the sub-model
        for sub_field_name, sub_field_type in field_type.__annotations__.items():
            field = field_type.__fields__[sub_field_name]
            help_text = (
                field.description if field.description else "No description available"
            )

            origin = get_origin(sub_field_type)
            if origin is Union:
                sub_field_type = next(
                    t for t in get_args(sub_field_type) if t is not type(None)
                )
            elif origin is List:
                sub_field_type = get_args(sub_field_type)[0]

            # Map Pydantic types to argparse types
            if isinstance(sub_field_type, type) and issubclass(
                sub_field_type, (int, float, str, bool)
            ):
                arg_type = sub_field_type
            else:
                arg_type = str  # Default to string for unknown types

            group.add_argument(f"--{sub_field_name}", type=arg_type, help=help_text)

    return parser


def convert_args_to_dict(args: argparse.Namespace, parser: argparse.ArgumentParser):
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
