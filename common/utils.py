"""Common utility functions"""


def unwrap(wrapped, default=None):
    """Unwrap function for Optionals."""
    if wrapped is None:
        return default

    return wrapped


def coalesce(*args):
    """Coalesce function for multiple unwraps."""
    return next((arg for arg in args if arg is not None), None)


def prune_dict(input_dict):
    """Trim out instances of None from a dictionary."""

    return {k: v for k, v in input_dict.items() if v is not None}


def merge_dict(dict1, dict2):
    """Merge 2 dictionaries"""
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            merge_dict(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def merge_dicts(*dicts):
    """Merge an arbitrary amount of dictionaries"""
    result = {}
    for dictionary in dicts:
        result = merge_dict(result, dictionary)

    return result


def flat_map(input_list):
    """Flattens a list of lists into a single list."""

    return [item for sublist in input_list for item in sublist]
