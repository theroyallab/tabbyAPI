"""Common utility functions"""

import inspect
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


def with_defer(func):
    """
    Decorator for a go-style defer
    """

    def wrapper(*args, **kwargs):
        deferred_calls = []

        # This 'defer' function is what you'll call inside your decorated function
        def defer(fn, *fn_args, **fn_kwargs):
            deferred_calls.append((fn, fn_args, fn_kwargs))

        try:
            # Inject 'defer' into the kwargs of the original function
            return func(*args, defer=defer, **kwargs)
        finally:
            # After the original function finishes (or raises), run deferred calls
            for fn, fn_args, fn_kwargs in reversed(deferred_calls):
                fn(*fn_args, **fn_kwargs)

    return wrapper


def with_defer_async(func):
    """
    Decorator for running async functions in go-style defer blocks
    """

    async def wrapper(*args, **kwargs):
        deferred_calls = []

        # This 'defer' function is what you'll call inside your decorated function
        def defer(fn, *fn_args, **fn_kwargs):
            deferred_calls.append((fn, fn_args, fn_kwargs))

        try:
            # Inject 'defer' into the kwargs of the original function
            return await func(*args, defer=defer, **kwargs)
        finally:
            # After the original function finishes (or raises), run deferred calls
            for fn, fn_args, fn_kwargs in reversed(deferred_calls):
                if inspect.iscoroutinefunction(fn):
                    await fn(*fn_args, **fn_kwargs)
                elif inspect.iscoroutine(fn):
                    await fn
                else:
                    fn(*fn_args, **fn_kwargs)

    return wrapper
