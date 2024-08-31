import gc
import pathlib
import torch
from loguru import logger
from typing import List, Optional

from common.utils import unwrap

# Conditionally import infinity to sidestep its logger
has_infinity_emb: bool = False
try:
    from infinity_emb import EngineArgs, AsyncEmbeddingEngine
    has_infinity_emb = True
    logger.debug("Successfully imported infinity.")
except ImportError:
    logger.debug("Failed to import infinity.")


class InfinityContainer:
    model_dir: pathlib.Path
    model_is_loading: bool = False
    model_loaded: bool = False

    # Conditionally set the type hint based on importablity
    # TODO: Clean this up
    if has_infinity_emb:
        engine: Optional[AsyncEmbeddingEngine] = None
    else:
        engine = None

    def __init__(self, model_directory: pathlib.Path):
        self.model_dir = model_directory

    async def load(self, **kwargs):
        self.model_is_loading = True

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

        self.model_loaded = True
        logger.info("Embedding model successfully loaded.")

    async def unload(self):
        await self.engine.astop()
        self.engine = None

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Embedding model unloaded.")

    async def generate(self, sentence_input: List[str]):
        result_embeddings, usage = await self.engine.embed(sentence_input)

        return {"embeddings": result_embeddings, "usage": usage}
