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


def map_pydantic_type_to_argparse(pydantic_type):
    """
    Maps Pydantic types to argparse compatible types.
    Handles special cases like Union and List.
    """
    origin = get_origin(pydantic_type)

    # Handle optional types
    if origin is Union:
        # Filter out NoneType
        pydantic_type = next(t for t in get_args(pydantic_type) if t is not type(None))

    elif origin is List:
        pydantic_type = get_args(pydantic_type)[0]  # Get the list item type

    # Map basic types (int, float, str, bool)
    if isinstance(pydantic_type, type) and issubclass(
        pydantic_type, (int, float, str, bool)
    ):
        return pydantic_type

    return str


def add_field_to_group(group, field_name, field_type, field):
    """
    Adds a Pydantic field to an argparse argument group.
    """
    arg_type = map_pydantic_type_to_argparse(field_type)
    help_text = field.description if field.description else "No description available"

    group.add_argument(f"--{field_name}", type=arg_type, help=help_text)


def init_argparser():
    """
    Initializes an argparse parser based on a Pydantic config schema.
    """
    parser = argparse.ArgumentParser(description="TabbyAPI server")

    # Loop through each top-level field in the config
    for field_name, field_type in config.__annotations__.items():
        group = parser.add_argument_group(
            field_name, description=f"Arguments for {field_name}"
        )

        # Check if the field_type is a Pydantic model
        if hasattr(field_type, "__annotations__"):
            for sub_field_name, sub_field_type in field_type.__annotations__.items():
                field = field_type.__fields__[sub_field_name]
                add_field_to_group(group, sub_field_name, sub_field_type, field)
        else:
            # Handle cases where the field_type is not a Pydantic mode
            arg_type = map_pydantic_type_to_argparse(field_type)
            group.add_argument(
                f"--{field_name}", type=arg_type, help=f"Argument for {field_name}"
            )

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
