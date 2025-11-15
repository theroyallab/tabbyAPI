import gc
import pathlib
import torch
from loguru import logger
from typing import List, Optional

from common.utils import unwrap
from common.optional_dependencies import dependencies

# Conditionally import infinity to sidestep its logger
if dependencies.extras:
    from infinity_emb import EngineArgs, AsyncEmbeddingEngine


class InfinityContainer:
    model_dir: pathlib.Path
    loaded: bool = False

    # Use a runtime type hint here
    engine: Optional["AsyncEmbeddingEngine"] = None

    def __init__(self, model_directory: pathlib.Path):
        self.model_dir = model_directory

    async def load(self, **kwargs):
        # Use cpu by default
        device = unwrap(kwargs.get("embeddings_device"), "cpu")
        
        # Extract device ID if specified
        device_id = kwargs.get("embeddings_device_id", [])
        
        # Validate device ID if using CUDA
        if device == "cuda" and device_id:
            if not isinstance(device_id, list):
                device_id = [device_id]
            
            # Validate GPU exists
            available_gpus = torch.cuda.device_count()
            for gpu_id in device_id:
                if gpu_id >= available_gpus:
                    logger.error(f"GPU {gpu_id} not found. Available GPUs: 0-{available_gpus-1}")
                    device_id = []  # Fallback to auto-select
                    break

        engine_args = EngineArgs(
            model_name_or_path=str(self.model_dir),
            engine="torch",
            device=device,
            device_id=device_id,  # Pass device ID to infinity_emb
            bettertransformer=False,
            model_warmup=False,
        )

        self.engine = AsyncEmbeddingEngine.from_args(engine_args)
        await self.engine.astart()

        self.loaded = True
        gpu_info = f" on GPU {device_id}" if device_id else ""
        logger.info(f"Embedding model successfully loaded{gpu_info}.")

    async def unload(self):
        await self.engine.astop()
        self.engine = None

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Embedding model unloaded.")

    async def generate(self, sentence_input: List[str]):
        result_embeddings, usage = await self.engine.embed(sentence_input)

        return {"embeddings": result_embeddings, "usage": usage}
