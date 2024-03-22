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
    """Trim out instances of None from a dictionary"""

    return {k: v for k, v in input_dict.items() if v is not None}
