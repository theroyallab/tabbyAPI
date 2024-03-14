"""Concurrency handling"""

import asyncio
import inspect
from functools import partialmethod
from typing import AsyncGenerator, Generator, Union

generate_semaphore = asyncio.Semaphore(1)


def release_semaphore():
    generate_semaphore.release()


async def generate_with_semaphore(generator: Union[AsyncGenerator, Generator]):
    """Generate with a semaphore."""

    async with generate_semaphore:
        if inspect.isasyncgenfunction:
            async for result in generator():
                yield result
        else:
            for result in generator():
                yield result


async def call_with_semaphore(callback: partialmethod):
    """Call with a semaphore."""

    async with generate_semaphore:
        if inspect.iscoroutinefunction(callback):
            return await callback()
        else:
            return callback()
