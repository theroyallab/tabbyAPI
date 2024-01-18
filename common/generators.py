"""Generator functions for the tabbyAPI."""
import inspect
from asyncio import Semaphore
from functools import partialmethod
from typing import AsyncGenerator

generate_semaphore = Semaphore(1)


# Async generation that blocks on a semaphore
async def generate_with_semaphore(generator: AsyncGenerator):
    """Generate with a semaphore."""
    async with generate_semaphore:
        if inspect.isasyncgenfunction:
            async for result in generator():
                yield result
        else:
            for result in generator():
                yield result


# Block a function with semaphore
async def call_with_semaphore(callback: partialmethod):
    if inspect.iscoroutinefunction(callback):
        return await callback()
    async with generate_semaphore:
        return callback()
