from asyncio import Semaphore
from typing import AsyncGenerator

generate_semaphore = Semaphore(1)


# Async generation that blocks on a semaphore
async def generate_with_semaphore(generator: AsyncGenerator):
    async with generate_semaphore:
        async for result in generator():
            yield result
