"""Concurrency handling"""

import asyncio
import inspect
from fastapi.concurrency import run_in_threadpool  # noqa
from functools import partialmethod
from typing import AsyncGenerator, Generator, Union

generate_semaphore = asyncio.Semaphore(1)


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


def release_semaphore():
    generate_semaphore.release()


async def generate_with_semaphore(generator: Union[AsyncGenerator, Generator]):
    """Generate with a semaphore."""

    async with generate_semaphore:
        if not inspect.isasyncgenfunction:
            generator = iterate_in_threadpool(generator())

        async for result in generator():
            yield result


async def call_with_semaphore(callback: partialmethod):
    """Call with a semaphore."""

    async with generate_semaphore:
        if not inspect.iscoroutinefunction:
            callback = run_in_threadpool(callback)

        return await callback()
