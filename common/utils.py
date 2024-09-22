"""Common utility functions"""

from types import NoneType
from typing import Dict, Optional, Type, Union, get_args, get_origin, TypeVar

T = TypeVar("T")


def unwrap(wrapped: Optional[T], default: T = None) -> T:
    """Unwrap function for Optionals."""
    if wrapped is None:
        return default

    return wrapped


def coalesce(*args):
    """Coalesce function for multiple unwraps."""
    return next((arg for arg in args if arg is not None), None)


def prune_nonetype_values(inp: Union[dict, list]) -> Dict:
    """Delete None values recursively"""
    if isinstance(inp, dict):
        for key, value in list(inp.items()):
            if isinstance(value, dict):
                prune_nonetype_values(value)
            elif value is None:
                del inp[key]
            elif isinstance(value, list):
                for v_i in value:
                    if isinstance(v_i, dict):
                        prune_nonetype_values(v_i)

        return inp

    elif isinstance(inp, list):
        out = []
        for value in inp:
            out.append(prune_nonetype_values(value))

        return out

    else:
        raise ValueError(f"input should be list or dict, got {type(inp)} {inp}")


def merge_dict(dict1: Dict, dict2: Dict) -> Dict:
    """Merge 2 dictionaries"""
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            merge_dict(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge an arbitrary amount of dictionaries"""
    result = {}
    for dictionary in dicts:
        result = merge_dict(result, dictionary)

    return result


def flat_map(input_list):
    """Flattens a list of lists into a single list."""

    return [item for sublist in input_list for item in sublist]


def is_list_type(type_hint) -> bool:
    """Checks if a type contains a list."""

    if get_origin(type_hint) is list:
        return True

    # Recursively check for lists inside type arguments
    type_args = get_args(type_hint)
    if type_args:
        return any(is_list_type(arg) for arg in type_args)

    return False


def unwrap_optional_type(type_hint) -> Type:
    """
    Unwrap Optional[type] annotations.
    This is not the same as unwrap.
    """

    if get_origin(type_hint) is Union:
        args = get_args(type_hint)
        if NoneType in args:
            for arg in args:
                if arg is not NoneType:
                    return arg

    return type_hint
