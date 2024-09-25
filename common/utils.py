"""Common utility functions"""

from types import NoneType
from typing import Dict, Type, Union, get_args, get_origin, TypeVar
from pydantic import BaseModel

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

def unwrap(wrapped: Type[T], default: Type[T]) -> T:
    """Unwrap function for Optionals."""
    if wrapped is None:
        return default

    return wrapped


def coalesce(*args):
    """Coalesce function for multiple unwraps."""
    return next((arg for arg in args if arg is not None), None)


def filter_none_values(collection: Union[dict, list]) -> Union[dict, list]:
    """Remove None values from a collection."""

    if isinstance(collection, dict):
        return {
            k: filter_none_values(v) for k, v in collection.items() if v is not None
        }
    elif isinstance(collection, list):
        return [filter_none_values(i) for i in collection if i is not None]
    else:
        return collection


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

def cast_model(model: BaseModel, new: Type[M]) -> M:
    return new(**model.model_dump())