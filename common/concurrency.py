"""Concurrency handling"""

import asyncio
from fastapi.concurrency import run_in_threadpool  # noqa
from typing import AsyncGenerator, Generator


# Originally from https://github.com/encode/starlette/blob/master/starlette/concurrency.py
# Uses generators instead of generics
class _StopIteration(Exception):
    """Wrapper for StopIteration because it doesn't send across threads."""

    pass


def gen_next(generator: Generator):
    """Threaded function to get the next value in an iterator."""

    try:
        return next(generator)
    except StopIteration as e:
        raise _StopIteration from e


async def iterate_in_threadpool(generator: Generator) -> AsyncGenerator:
    """Iterates a generator within a threadpool."""

    while True:
        try:
            yield await asyncio.to_thread(gen_next, generator)
        except _StopIteration:
            break
