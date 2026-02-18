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
        
        # Handle mixed device types (CPU/CUDA conflict)
        if device == "cpu" and device_id:
            logger.warning("embeddings_device is set to 'cpu' but embeddings_device_id is specified. Ignoring device_id and using CPU.")
            device_id = []
        
        # Validate device ID if using CUDA
        if device == "cuda" and device_id:
            if not isinstance(device_id, list):
                device_id = [device_id]
            
            # Validate GPU exists
            available_gpus = torch.cuda.device_count()
            valid_device_ids = []
            
            for gpu_id in device_id:
                if gpu_id >= available_gpus:
                    logger.error(f"GPU {gpu_id} not found. Available GPUs: 0-{available_gpus-1}")
                    continue  # Skip invalid GPU but continue checking others
                else:
                    valid_device_ids.append(gpu_id)
            
            # Use only valid device IDs
            device_id = valid_device_ids
            
            # Handle multiple device IDs with infinity_emb compatibility
            if len(device_id) > 1:
                logger.warning("infinity_emb may not support multiple GPU IDs. Using first valid GPU: {device_id[0]}")
                device_id = [device_id[0]]  # Use only first GPU

        try:
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
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"GPU {device_id} has insufficient memory for embedding model. Error: {str(e)}")
                logger.error("Try using a different GPU or loading the model on CPU.")
                raise
            elif "cuda" in str(e).lower() or "device" in str(e).lower():
                logger.error(f"Failed to load embedding model on GPU {device_id}. Error: {str(e)}")
                logger.error("The GPU may be busy or unavailable. Try using a different GPU or CPU.")
                raise
            else:
                logger.error(f"Unexpected error loading embedding model: {str(e)}")
                raise

    async def unload(self):
        await self.engine.astop()
        self.engine = None

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Embedding model unloaded.")

    async def generate(self, sentence_input: List[str]):
        result_embeddings, usage = await self.engine.embed(sentence_input)

        return {"embeddings": result_embeddings, "usage": usage}
