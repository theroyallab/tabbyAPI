import gc, time, pathlib
import torch
from datetime import datetime
from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import(
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)
from typing import List, Optional, Union

# Bytes to reserve on first device when loading with auto split
auto_split_reserve_bytes = 96 * 1024**2

class ModelContainer:

    config: Optional[ExLlamaV2Config] = None
    draft_config: Optional[ExLlamaV2Config] = None
    model: Optional[ExLlamaV2] = None
    draft_model: Optional[ExLlamaV2] = None
    cache: Optional[ExLlamaV2Cache] = None
    draft_cache: Optional[ExLlamaV2Cache] = None
    tokenizer: Optional[ExLlamaV2Tokenizer] = None
    generator: Optional[ExLlamaV2StreamingGenerator] = None

    cache_fp8: bool = False
    draft_enabled: bool = False
    gpu_split_auto: bool = True
    gpu_split: list or None = None

    def __init__(self, model_directory: pathlib.Path, quiet = False, **kwargs):
        """
        Create model container

        Args:
            model_dir (int): Model directory containing config.json, tokenizer.model etc.
            quiet (bool): Suppress console output
            load_progress_callback (function, optional): A function to call for each module loaded. Prototype:
                def progress(loaded_modules: int, total_modules: int, loading_draft: bool)
            **kwargs:
                `cache_mode` (str): Sets cache mode, "FP16" or "FP8" (defaulf: "FP16")
                'max_seq_len' (int): Override model's default max sequence length
                'rope_scale' (float): Set RoPE scaling factor for model (default: 1.0)
                'rope_alpha' (float): Set RoPE alpha (NTK) factor for model (default: 1.0)
                'chunk_size' (int): Sets the maximum chunk size for the model (default: 2048)
                    Inferencing in chunks reduces overall VRAM overhead by processing very long sequences in smaller
                    batches. This limits the size of temporary buffers needed for the hidden state and attention
                    weights.
                'draft_model_dir' (str): Draft model directory
                'draft_rope_alpha' (float): RoPE alpha (NTK) factor for draft model.
                    By default, the draft model's alpha value is calculated automatically to scale to the size of the
                    full model.
                'gpu_split_auto' (bool): Automatically split model across available devices (default: True)
                'gpu_split' (list[float]): Allocation for weights and (some) tensors, per device
                'no_flash_attn' (bool): Turns off flash attention (increases vram usage)
        """

        self.quiet = quiet

        self.cache_fp8 = "cache_mode" in kwargs and kwargs["cache_mode"] == "FP8"
        self.gpu_split = kwargs.get("gpu_split", None)
        self.gpu_split_auto = kwargs.get("gpu_split_auto", True)

        self.config = ExLlamaV2Config()
        self.config.model_dir = str(model_directory.resolve())
        self.config.prepare()

        # Grab the base model's sequence length before overrides for rope calculations
        base_seq_len = self.config.max_seq_len

        # Then override the max_seq_len if present
        if "max_seq_len" in kwargs: self.config.max_seq_len = kwargs["max_seq_len"]
        if "rope_scale" in kwargs: self.config.scale_pos_emb = kwargs["rope_scale"]

        # Automatically calculate rope alpha
        self.config.scale_alpha_value = kwargs.get("rope_alpha") or self.calculate_rope_alpha(base_seq_len)

        if "no_flash_attn" in kwargs: self.config.no_flash_attn = kwargs["no_flash_attn"]

        # low_mem is currently broken in exllamav2. Don't use it until it's fixed.
        """
        if "low_mem" in kwargs and kwargs["low_mem"]:
            self.config.set_low_mem()
        """

        chunk_size = min(kwargs.get("chunk_size", 2048), self.config.max_seq_len)
        self.config.max_input_len = chunk_size
        self.config.max_attn_size = chunk_size ** 2

        draft_config = kwargs.get("draft") or {}
        draft_model_name = draft_config.get("draft_model_name")
        enable_draft = bool(draft_config) and draft_model_name is not None

        if bool(draft_config) and draft_model_name is None:
            print("A draft config was found but a model name was not given. Please check your config.yml! Skipping draft load.")
            self.draft_enabled = False
        else:
            self.draft_enabled = enable_draft

        if self.draft_enabled:

            self.draft_config = ExLlamaV2Config()
            draft_model_path = pathlib.Path(draft_config.get("draft_model_dir") or "models")
            draft_model_path = draft_model_path / draft_model_name

            self.draft_config.model_dir = str(draft_model_path.resolve())
            self.draft_config.prepare()

            self.draft_config.scale_pos_emb = draft_config.get("draft_rope_scale") or 1.0
            self.draft_config.scale_alpha_value = draft_config.get("draft_rope_alpha") or self.calculate_rope_alpha(self.draft_config.max_seq_len)
            self.draft_config.max_seq_len = self.config.max_seq_len            

            if "chunk_size" in kwargs:
                self.draft_config.max_input_len = kwargs["chunk_size"]
                self.draft_config.max_attn_size = kwargs["chunk_size"] ** 2


    def calculate_rope_alpha(self, base_seq_len):
        ratio = self.config.max_seq_len / base_seq_len

        # Default to a 1 alpha if the sequence length is ever less than or equal to 1
        alpha = 1 if ratio <= 1.0 else -0.13436 + 0.80541 * ratio + 0.28833 * ratio ** 2
        return alpha

    def get_model_path(self):
        model_path = pathlib.Path(self.config.model_dir)
        return model_path


    def load(self, progress_callback = None):
        """
        Load model

        Args:
            progress_callback (function, optional): A function to call for each module loaded. Prototype:
                def progress(loaded_modules: int, total_modules: int)
        """
        for _ in self.load_gen(progress_callback): pass


    def load_gen(self, progress_callback = None):
        """
        Load model, generator function

        Args:
            progress_callback (function, optional): A function to call for each module loaded. Prototype:
                def progress(loaded_modules: int, total_modules: int)
        """

        # Load tokenizer

        self.tokenizer = ExLlamaV2Tokenizer(self.config)

        # Load draft model

        if self.draft_enabled:

            self.draft_model = ExLlamaV2(self.draft_config)
            if not self.quiet:
                print("Loading draft model: " + self.draft_config.model_dir)

            self.draft_cache = ExLlamaV2Cache(self.draft_model, lazy = True)
            reserve = [auto_split_reserve_bytes] + [0] * 16
            yield from self.draft_model.load_autosplit_gen(self.draft_cache, reserve_vram = reserve, last_id_only = True, callback_gen = progress_callback)

            # Test VRAM allocation with a full-length forward pass

            input_ids = torch.zeros((1, self.config.max_input_len), dtype = torch.long)
            self.draft_model.forward(input_ids, cache = self.cache, preprocess_only = True)

        # Load model

        self.model = ExLlamaV2(self.config)
        if not self.quiet:
            print("Loading model: " + self.config.model_dir)

        if not self.gpu_split_auto:
            for value in self.model.load_gen(self.gpu_split, callback_gen = progress_callback):
                if isinstance(value, str):
                    yield value

        if self.cache_fp8:
            self.cache = ExLlamaV2Cache_8bit(self.model, lazy = self.gpu_split_auto)
        else:
            self.cache = ExLlamaV2Cache(self.model, lazy = self.gpu_split_auto)

        if self.gpu_split_auto:
            reserve = [auto_split_reserve_bytes] + [0] * 16
            yield from self.model.load_autosplit_gen(self.cache, reserve_vram = reserve, last_id_only = True, callback_gen = progress_callback)

        # Test VRAM allocation with a full-length forward pass

        input_ids = torch.zeros((1, self.config.max_input_len), dtype = torch.long)
        self.model.forward(input_ids, cache = self.cache, preprocess_only = True)

        # Create generator

        self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer, self.draft_model, self.draft_cache)

        print("Model successfully loaded.")


    def unload(self):
        """
        Free all VRAM resources used by this model
        """

        if self.model: self.model.unload()
        self.model = None
        if self.draft_model: self.draft_model.unload()
        self.draft_model = None
        self.config = None
        self.cache = None
        self.tokenizer = None
        self.generator = None
        gc.collect()
        torch.cuda.empty_cache()


    # Common function for token operations
    def get_tokens(self, text: Optional[str], ids: Optional[List[int]], **kwargs):
        if text:
            # Assume token encoding
            return self.tokenizer.encode(
                text,
                add_bos = kwargs.get("add_bos_token", True),
                encode_special_tokens = kwargs.get("encode_special_tokens", True)
            )
        if ids:
            # Assume token decoding
            ids = torch.tensor([ids])
            return self.tokenizer.decode(ids, decode_special_tokens = kwargs.get("decode_special_tokens", True))[0]


    def generate(self, prompt: str, **kwargs):
        gen = list(self.generate_gen(prompt, **kwargs))
        reponse = "".join(map(lambda o: o[0], gen))
        return reponse, gen[-1][1], gen[-1][2]

    def generate_gen(self, prompt: str, **kwargs):
        """
        Create generator function for prompt completion

        Args:
            prompt (str): Input prompt
            **kwargs:
                'token_healing' (bool): Use token healing (default: False)
                'temperature' (float): Sampling temperature (default: 1.0)
                'temperature_last' (bool): Apply temperature after all other samplers (default: False)
                'top_k' (int): Sampling top-K (default: 0)
                'top_p' (float): Sampling top-P (default: 1.0)
                'min_p' (float): Sampling min-P (default: 0.0)
                'tfs' (float): Tail-free sampling (default: 0.0)
                'typical' (float): Sampling typical (default: 0.0)
                'mirostat' (bool): Use Mirostat (default: False)
                'mirostat_tau' (float) Mirostat tau parameter (default: 1.5)
                'mirostat_eta' (float) Mirostat eta parameter (default: 0.1)
                'repetition_penalty' (float): Token repetition/presence penalty (default: 1.15)
                'repetition_range' (int): Repetition penalty range (default: whole context)
                'repetition_decay' (int): Repetition penalty range (default: same as range)
                'stop' (List[Union[str, int]]): List of stop strings/tokens to end response (default: [EOS])
                'max_tokens' (int): Max no. tokens in response (default: 150)
                'add_bos_token' (bool): Adds the BOS token to the start of the prompt (default: True)
                'ban_eos_token' (bool): Bans the EOS token from generation (default: False)
                'stream_interval' (float): Interval in seconds between each output chunk (default: immediate)
                'generate_window' (int): Space to reserve at the end of the model's context when generating.
                    Rolls context window by the same amount if context length is exceeded to allow generating past
                    the models max_seq_len.

        """

        token_healing = kwargs.get("token_healing", False)
        max_tokens = kwargs.get("max_tokens", 150)
        stream_interval = kwargs.get("stream_interval", 0)
        generate_window = min(kwargs.get("generate_window", 512), max_tokens)

        # Sampler settings

        gen_settings = ExLlamaV2Sampler.Settings()

        # Warn of unsupported settings if the setting is enabled

        if kwargs.get("mirostat", False) and not hasattr(gen_settings, "mirostat"):
            print(" !! Warning: Currently installed ExLlamaV2 does not support Mirostat sampling")

        if kwargs.get("min_p", 0.0) not in [0.0, 1.0] and not hasattr(gen_settings, "min_p"):
            print(" !! Warning: Currently installed ExLlamaV2 does not support min-P sampling")

        if kwargs.get("tfs", 0.0) not in [0.0, 1.0] and not hasattr(gen_settings, "tfs"):
            print(" !! Warning: Currently installed ExLlamaV2 does not support tail-free sampling (TFS)")

        if kwargs.get("temperature_last", False) and not hasattr(gen_settings, "temperature_last"):
            print(" !! Warning: Currently installed ExLlamaV2 does not support temperature_last")

        #Apply settings

        gen_settings.temperature = kwargs.get("temperature", 1.0)
        gen_settings.temperature_last = kwargs.get("temperature_last", False)
        gen_settings.top_k = kwargs.get("top_k", 1)
        gen_settings.top_p = kwargs.get("top_p", 1.0)
        gen_settings.min_p = kwargs.get("min_p", 0.0)
        gen_settings.tfs = kwargs.get("tfs", 0.0)
        gen_settings.typical = kwargs.get("typical", 0.0)
        gen_settings.mirostat = kwargs.get("mirostat", False)

        # Default tau and eta fallbacks don't matter if mirostat is off
        gen_settings.mirostat_tau = kwargs.get("mirostat_tau", 1.5)
        gen_settings.mirostat_eta = kwargs.get("mirostat_eta", 0.1)
        gen_settings.token_repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        gen_settings.token_repetition_range = kwargs.get("repetition_range", self.config.max_seq_len)


        # Always make sure the fallback is 0 if range < 0
        # It's technically fine to use -1, but this just validates the passed fallback
        fallback_decay = 0 if gen_settings.token_repetition_penalty <= 0 else gen_settings.token_repetition_range
        gen_settings.token_repetition_decay = kwargs.get("repetition_decay", fallback_decay or 0)

        stop_conditions: List[Union[str, int]] = kwargs.get("stop", [])
        ban_eos_token = kwargs.get("ban_eos_token", False)


        # Ban the EOS token if specified. If not, append to stop conditions as well.

        if ban_eos_token:
            gen_settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])
        else:
            stop_conditions.append(self.tokenizer.eos_token_id)

        # Override sampler settings for temp = 0

        if gen_settings.temperature == 0:
            gen_settings.temperature = 1.0
            gen_settings.top_k = 1
            gen_settings.top_p = 0
            gen_settings.typical = 0

        # Stop conditions           

        self.generator.set_stop_conditions(stop_conditions)

        # Tokenized context

        ids = self.tokenizer.encode(
            prompt,
            add_bos=kwargs.get("add_bos_token", True),
            encode_special_tokens = True
        )
        context_len = len(ids[0])

        if context_len > self.config.max_seq_len:
            print(
                f"WARNING: The context length {context_len} is greater than the max_seq_len {self.config.max_seq_len}.",
                "Generation is truncated and metrics may not be accurate."
            )

        prompt_tokens = ids.shape[-1]

        # Begin

        generated_tokens = 0
        full_response = ""
        start_time = time.time()
        last_chunk_time = start_time

        save_tokens = torch.empty((1, 0), dtype = torch.bool)
        chunk_buffer = ""
        chunk_tokens = 0

        while True:

            # Ingest prompt

            if chunk_tokens == 0:

                ids = torch.cat((ids, save_tokens), dim = - 1)
                save_tokens = torch.empty((1, 0), dtype = torch.bool)
                overflow = ids.shape[-1] + generate_window - self.config.max_seq_len
                active_ids = ids[:, max(0, overflow):]
                chunk_tokens = self.config.max_seq_len - active_ids.shape[-1]

                self.generator.begin_stream(active_ids, gen_settings, token_healing = token_healing)

            # Generate

            chunk, eos, tokens = self.generator.stream()

            if token_healing:
                ids[:, -1] = self.generator.sequence_ids[:, -2]  # Extract healed token
                token_healing = False

            save_tokens = torch.cat((save_tokens, tokens), dim=-1)
            chunk_buffer += chunk

            generated_tokens += 1
            chunk_tokens -= 1

            # Yield output

            now = time.time()
            elapsed = now - last_chunk_time

            if chunk_buffer != "" and (elapsed > stream_interval or eos or generated_tokens == max_tokens):
                yield chunk_buffer, prompt_tokens, generated_tokens
                full_response += chunk_buffer
                chunk_buffer = ""
                last_chunk_time = now

            if eos or generated_tokens == max_tokens: break

        elapsed_time = last_chunk_time - start_time

        initial_response = f"Response: {generated_tokens} tokens generated in {round(elapsed_time, 2)} seconds"
        itemization = []
        extra_parts = []

        # Add tokens per second
        itemization.append(f"{'Indeterminate' if elapsed_time == 0 else round(generated_tokens / elapsed_time, 2)} T/s")

        # Add context (original token count)
        if ids is not None:
            itemization.append(f"context {context_len} tokens")

        if context_len > self.config.max_seq_len:
            extra_parts.append("<-- Not accurate (truncated)")

        # Print output
        print(initial_response + " (" + ", ".join(itemization) + ") " + " ".join(extra_parts))
