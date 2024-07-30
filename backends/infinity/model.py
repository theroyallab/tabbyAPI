import gc
import pathlib
import torch
from typing import List, Optional

from common.utils import unwrap

# Conditionally import infinity to sidestep its logger
# TODO: Make this prettier
try:
    from infinity_emb import EngineArgs, AsyncEmbeddingEngine

    has_infinity_emb = True
except ImportError:
    has_infinity_emb = False


class InfinityContainer:
    model_dir: pathlib.Path

    # Conditionally set the type hint based on importablity
    # TODO: Clean this up
    if has_infinity_emb:
        engine: Optional[AsyncEmbeddingEngine] = None
    else:
        engine = None

    def __init__(self, model_directory: pathlib.Path):
        self.model_dir = model_directory

    async def load(self, **kwargs):
        # Use cpu by default
        device = unwrap(kwargs.get("device"), "cpu")

        engine_args = EngineArgs(
            model_name_or_path=str(self.model_dir),
            engine="torch",
            device=device,
            bettertransformer=False,
            model_warmup=False,
        )

        self.engine = AsyncEmbeddingEngine.from_args(engine_args)
        await self.engine.astart()

    async def unload(self):
        await self.engine.astop()
        self.engine = None

        gc.collect()
        torch.cuda.empty_cache()

    async def generate(self, sentence_input: List[str]):
        result_embeddings, usage = await self.engine.embed(sentence_input)

        return {"embeddings": result_embeddings, "usage": usage}
