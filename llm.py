# exllama.py
import random
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)
import time
class ModelManager:
    def __init__(self, model_directory: str = None):
        if model_directory is None:
            model_directory = "/mnt/str/models/_exl2/mistral-7b-instruct-exl2/4.0bpw/"
        self.config = ExLlamaV2Config()
        self.config.model_dir = model_directory
        self.config.prepare()
        self.model = ExLlamaV2(self.config)
        print("Loading model: " + model_directory)
        self.cache = ExLlamaV2Cache(self.model, lazy=True)
        self.model.load_autosplit(self.cache)
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)
    def generate_text(self, prompt: str, max_new_tokens: int = 150,seed: int = random.randint(0,999999) ):
        try:
            self.generator.warmup()
            time_begin = time.time()
            output = self.generator.generate_simple(
                prompt, ExLlamaV2Sampler.Settings(), max_new_tokens, seed=seed
            )
            time_end = time.time()
            time_total = time_end - time_begin
            return output, f"{time_total:.2f} seconds"
        except Exception as e:
            raise RuntimeError(f"Error generating text: {str(e)}")
