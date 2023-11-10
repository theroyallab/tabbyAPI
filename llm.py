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

    def generate_text(self,
                      prompt: str,
                      max_tokens: int = 150,
                      temperature=0.5,
                      seed: int = random.randint(0, 999999),
                      token_repetition_penalty: float = 1.0,
                      stop: list = None):
        try:
            self.generator.warmup()
            time_begin = time.time()
            settings = ExLlamaV2Sampler.Settings()
            settings.token_repetition_penalty = token_repetition_penalty

            if stop:
                settings.stop_sequence = stop

            output = self.generator.generate_simple(
                prompt, settings, max_tokens, seed=seed
            )
            time_end = time.time()
            time_total = time_end - time_begin
            return output, f"{time_total:.2f} seconds"
        except Exception as e:
            raise RuntimeError(f"Error generating text: {str(e)}")
