from __future__ import annotations

import gc
import pathlib
import torch
from loguru import logger

from common.utils import unwrap
from common.optional_dependencies import dependencies

# Conditionally import infinity to sidestep its logger
if dependencies.extras:
    from infinity_emb import EngineArgs, AsyncEmbeddingEngine


class InfinityContainer:
    model_dir: pathlib.Path
    loaded: bool = False

    # Use a runtime type hint here
    engine: AsyncEmbeddingEngine | None = None

    def __init__(self, model_directory: pathlib.Path):
        self.model_dir = model_directory

    async def load(self, **kwargs):
        # Use cpu by default
        device = unwrap(kwargs.get("embeddings_device"), "cpu")

        engine_args = EngineArgs(
            model_name_or_path=str(self.model_dir),
            engine="torch",
            device=device,
            bettertransformer=False,
            model_warmup=False,
        )

        self.engine = AsyncEmbeddingEngine.from_args(engine_args)
        await self.engine.astart()

        self.loaded = True
        logger.info("Embedding model successfully loaded.")

    async def unload(self):
        await self.engine.astop()
        self.engine = None

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Embedding model unloaded.")

    async def generate(self, sentence_input: list[str]):
        result_embeddings, usage = await self.engine.embed(sentence_input)

        return {"embeddings": result_embeddings, "usage": usage}
