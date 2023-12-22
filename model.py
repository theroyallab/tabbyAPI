"""The model container class for ExLlamaV2 models."""
import gc
import pathlib
import time
from typing import List, Optional, Union

import torch
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler

from gen_logging import log_generation_params, log_prompt, log_response
from templating import (
    PromptTemplate,
    find_template_from_model,
    get_template_from_config,
    get_template_from_file,
)
from utils import coalesce, unwrap

# Bytes to reserve on first device when loading with auto split
AUTO_SPLIT_RESERVE_BYTES = 96 * 1024**2


class ModelContainer:
    """The model container class for ExLlamaV2 models."""

    config: Optional[ExLlamaV2Config] = None
    draft_config: Optional[ExLlamaV2Config] = None
    model: Optional[ExLlamaV2] = None
    draft_model: Optional[ExLlamaV2] = None
    cache: Optional[ExLlamaV2Cache] = None
    draft_cache: Optional[ExLlamaV2Cache] = None
    tokenizer: Optional[ExLlamaV2Tokenizer] = None
    generator: Optional[ExLlamaV2StreamingGenerator] = None
    prompt_template: Optional[PromptTemplate] = None

    cache_fp8: bool = False
    gpu_split_auto: bool = True
    gpu_split: Optional[list] = None

    active_loras: List[ExLlamaV2Lora] = []

    def __init__(self, model_directory: pathlib.Path, quiet=False, **kwargs):
        """
        Create model container

        Args:
            model_dir (int): Model directory containing config.json,
                tokenizer.model etc.
            quiet (bool): Suppress console output
            load_progress_callback (function, optional): A function to call for
                each module loaded. Prototype:
                def progress(loaded_modules: int, total_modules: int,
                             loading_draft: bool)
            **kwargs:
                `cache_mode` (str): Sets cache mode, "FP16" or "FP8"
                    (defaulf: "FP16")
                'max_seq_len' (int): Override model's default max sequence
                    length (default: 4096)
                'rope_scale' (float): Set RoPE scaling factor for model
                    (default: 1.0)
                'rope_alpha' (float): Set RoPE alpha (NTK) factor for model
                    (default: 1.0)
                'prompt_template' (str): Manually sets the prompt template for
                    this model (default: None)
                'chunk_size' (int): Sets the maximum chunk size for the model
                    (default: 2048)
                    Inferencing in chunks reduces overall VRAM overhead by
                    processing very long sequences in smaller batches. This
                    limits the size of temporary buffers needed for the hidden
                    state and attention weights.
                'draft_model_dir' (str): Draft model directory
                'draft_rope_scale' (float): Set RoPE scaling factor for draft
                    model (default: 1.0)
                'draft_rope_alpha' (float): RoPE alpha (NTK) factor for draft
                    model. By default, the draft model's alpha value is
                    calculated automatically to scale to the size of the
                    full model.
                'lora_dir' (str): LoRA directory
                'loras' (list[dict]): List of loras to be loaded, consisting of
                    'name' and 'scaling'
                'gpu_split_auto' (bool): Automatically split model across
                    available devices (default: True)
                'gpu_split' (list[float]): Allocation for weights and (some)
                    tensors, per device
                'no_flash_attn' (bool): Turns off flash attention
                    (increases vram usage) (default: False)
        """

        self.quiet = quiet

        self.cache_fp8 = (
            "cache_mode" in kwargs and kwargs["cache_mode"] == "FP8"
        )
        self.gpu_split = kwargs.get("gpu_split")
        self.gpu_split_auto = unwrap(kwargs.get("gpu_split_auto"), True)

        self.config = ExLlamaV2Config()
        self.config.model_dir = str(model_directory.resolve())

        # Make the max seq len 4096 before preparing the config
        # This is a better default than 2038
        self.config.max_seq_len = 4096
        self.config.prepare()

        # Then override the base_seq_len if present
        override_base_seq_len = kwargs.get("override_base_seq_len")
        if override_base_seq_len:
            self.config.max_seq_len = override_base_seq_len

        # Grab the base model's sequence length before overrides for
        # rope calculations
        base_seq_len = self.config.max_seq_len

        # Set the target seq len if present
        target_max_seq_len = kwargs.get("max_seq_len")
        if target_max_seq_len:
            self.config.max_seq_len = target_max_seq_len

        # Set the rope scale
        self.config.scale_pos_emb = unwrap(kwargs.get("rope_scale"), 1.0)

        # Automatically calculate rope alpha
        self.config.scale_alpha_value = unwrap(
            kwargs.get("rope_alpha"), self.calculate_rope_alpha(base_seq_len)
        )

        # Turn off flash attention?
        self.config.no_flash_attn = unwrap(
            kwargs.get("no_flash_attention"), False
        )

        # low_mem is currently broken in exllamav2. Don't use it until it's
        # fixed.
        """
        if "low_mem" in kwargs and kwargs["low_mem"]:
            self.config.set_low_mem()
        """

        # Set prompt template override if provided
        prompt_template_name = kwargs.get("prompt_template")
        try:
            if prompt_template_name:
                # Read the template
                self.prompt_template = get_template_from_file(
                    prompt_template_name
                )
            else:
                # Try finding the chat template from the model's config.json
                self.prompt_template = get_template_from_config(
                    pathlib.Path(self.config.model_config)
                )

                # If that fails, attempt fetching from model name
                if self.prompt_template is None:
                    template_match = find_template_from_model(model_directory)
                    if template_match:
                        self.prompt_template = get_template_from_file(
                            template_match
                        )
        except OSError:
            # The template or config.json couldn't be found in the user's
            # filesystem
            print(
                "Could not find template file with name "
                f"{prompt_template_name}.jinja"
            )
            self.prompt_template = None

        # Catch all for template lookup errors
        if self.prompt_template:
            print(
                f"Using template {self.prompt_template.name} for chat "
                "completions."
            )
        else:
            print(
                "Chat completions are disabled because a prompt template "
                "wasn't provided or auto-detected."
            )

        # Set num of experts per token if provided
        num_experts_override = kwargs.get("num_experts_per_token")
        if num_experts_override:
            if hasattr(self.config, "num_experts_per_token"):
                self.config.num_experts_per_token = num_experts_override
            else:
                print(
                    " !! Warning: Currently installed ExLlamaV2 does not "
                    "support overriding MoE experts"
                )

        chunk_size = min(
            unwrap(kwargs.get("chunk_size"), 2048), self.config.max_seq_len
        )
        self.config.max_input_len = chunk_size
        self.config.max_attn_size = chunk_size**2

        draft_args = unwrap(kwargs.get("draft"), {})
        draft_model_name = draft_args.get("draft_model_name")
        enable_draft = draft_args and draft_model_name

        # Always disable draft if params are incorrectly configured
        if draft_args and draft_model_name is None:
            print(
                "A draft config was found but a model name was not given. "
                "Please check your config.yml! Skipping draft load."
            )
            enable_draft = False

        if enable_draft:
            self.draft_config = ExLlamaV2Config()
            draft_model_path = pathlib.Path(
                unwrap(draft_args.get("draft_model_dir"), "models")
            )
            draft_model_path = draft_model_path / draft_model_name

            self.draft_config.model_dir = str(draft_model_path.resolve())
            self.draft_config.prepare()

            self.draft_config.scale_pos_emb = unwrap(
                draft_args.get("draft_rope_scale"), 1.0
            )

            # Automatically calculate draft rope alpha
            self.draft_config.scale_alpha_value = unwrap(
                draft_args.get("draft_rope_alpha"),
                self.calculate_rope_alpha(self.draft_config.max_seq_len),
            )
            self.draft_config.max_seq_len = self.config.max_seq_len

            if "chunk_size" in kwargs:
                self.draft_config.max_input_len = kwargs["chunk_size"]
                self.draft_config.max_attn_size = kwargs["chunk_size"] ** 2

    def calculate_rope_alpha(self, base_seq_len):
        """Calculate the rope alpha value for a given sequence length."""
        ratio = self.config.max_seq_len / base_seq_len

        # Default to a 1 alpha if the sequence length is ever less
        # than or equal to 1
        if ratio <= 1.0:
            alpha = 1
        else:
            alpha = -0.13436 + 0.80541 * ratio + 0.28833 * ratio**2
        return alpha

    def get_model_path(self, is_draft: bool = False):
        """Get the path for this model."""
        model_path = pathlib.Path(
            self.draft_config.model_dir if is_draft else self.config.model_dir
        )
        return model_path

    def load(self, progress_callback=None):
        """
        Load model

        Args:
            progress_callback (function, optional): A function to call for each
                module loaded. Prototype:
                def progress(loaded_modules: int, total_modules: int)
        """
        for _ in self.load_gen(progress_callback):
            pass

    def load_loras(self, lora_directory: pathlib.Path, **kwargs):
        """
        Load loras
        """

        loras = unwrap(kwargs.get("loras"), [])
        success: List[str] = []
        failure: List[str] = []

        for lora in loras:
            lora_name = lora.get("name")
            lora_scaling = unwrap(lora.get("scaling"), 1.0)

            if lora_name is None:
                print(
                    "One of your loras does not have a name. Please check your "
                    "config.yml! Skipping lora load."
                )
                failure.append(lora_name)
                continue

            print(f"Loading lora: {lora_name} at scaling {lora_scaling}")
            lora_path = lora_directory / lora_name
            # FIXME(alpin): Does self.model need to be passed here?
            self.active_loras.append(
                ExLlamaV2Lora.from_directory(
                    self.model, lora_path, lora_scaling
                )
            )
            print("Lora successfully loaded.")
            success.append(lora_name)

        # Return success and failure names
        return {"success": success, "failure": failure}

    def load_gen(self, progress_callback=None):
        """
        Load model, generator function

        Args:
            progress_callback (function, optional): A function to call for each
                module loaded. Prototype:
                def progress(loaded_modules: int, total_modules: int)
        """

        # Load tokenizer
        self.tokenizer = ExLlamaV2Tokenizer(self.config)

        # Load draft model if a config is present
        if self.draft_config:
            self.draft_model = ExLlamaV2(self.draft_config)
            if not self.quiet:
                print("Loading draft model: " + self.draft_config.model_dir)

            self.draft_cache = ExLlamaV2Cache(self.draft_model, lazy=True)
            reserve = [AUTO_SPLIT_RESERVE_BYTES] + [0] * 16
            yield from self.draft_model.load_autosplit_gen(
                self.draft_cache,
                reserve_vram=reserve,
                last_id_only=True,
                callback_gen=progress_callback,
            )

            # Test VRAM allocation with a full-length forward pass
            input_ids = torch.zeros(
                (1, self.config.max_input_len), dtype=torch.long
            )
            self.draft_model.forward(
                input_ids, cache=self.cache, preprocess_only=True
            )

        # Load model
        self.model = ExLlamaV2(self.config)
        if not self.quiet:
            print("Loading model: " + self.config.model_dir)

        if not self.gpu_split_auto:
            for value in self.model.load_gen(
                self.gpu_split, callback_gen=progress_callback
            ):
                if isinstance(value, str):
                    yield value

        if self.cache_fp8:
            self.cache = ExLlamaV2Cache_8bit(
                self.model, lazy=self.gpu_split_auto
            )
        else:
            self.cache = ExLlamaV2Cache(self.model, lazy=self.gpu_split_auto)

        if self.gpu_split_auto:
            reserve = [AUTO_SPLIT_RESERVE_BYTES] + [0] * 16
            yield from self.model.load_autosplit_gen(
                self.cache,
                reserve_vram=reserve,
                last_id_only=True,
                callback_gen=progress_callback,
            )

        # Test VRAM allocation with a full-length forward pass
        input_ids = torch.zeros(
            (1, self.config.max_input_len), dtype=torch.long
        )
        self.model.forward(input_ids, cache=self.cache, preprocess_only=True)

        # Create generator
        self.generator = ExLlamaV2StreamingGenerator(
            self.model,
            self.cache,
            self.tokenizer,
            self.draft_model,
            self.draft_cache,
        )

        print("Model successfully loaded.")

    def unload(self, loras_only: bool = False):
        """
        Free all VRAM resources used by this model
        """

        for lora in self.active_loras:
            lora.unload()

        self.active_loras = []

        # Unload the entire model if not just unloading loras
        if not loras_only:
            if self.model:
                self.model.unload()
            self.model = None

            if self.draft_model:
                self.draft_model.unload()
            self.draft_model = None

            self.config = None
            self.cache = None
            self.tokenizer = None
            self.generator = None

        gc.collect()
        torch.cuda.empty_cache()

    def get_tokens(
        self, text: Optional[str], ids: Optional[List[int]], **kwargs
    ):
        """Common function for token operations"""
        if text:
            # Assume token encoding
            return self.tokenizer.encode(
                text,
                add_bos=unwrap(kwargs.get("add_bos_token"), True),
                encode_special_tokens=unwrap(
                    kwargs.get("encode_special_tokens"), True
                ),
            )
        if ids:
            # Assume token decoding
            ids = torch.tensor([ids])
            return self.tokenizer.decode(
                ids,
                decode_special_tokens=unwrap(
                    kwargs.get("decode_special_tokens"), True
                ),
            )[0]

        return None

    def generate(self, prompt: str, **kwargs):
        """Generate a response to a prompt"""
        generation = list(self.generate_gen(prompt, **kwargs))
        if generation:
            response = "".join(map(lambda chunk: chunk[0], generation))
            return response, generation[-1][1], generation[-1][2]

        return "", 0, 0

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def generate_gen(self, prompt: str, **kwargs):
        """
        Create generator function for prompt completion

        Args:
            prompt (str): Input prompt
            **kwargs:
                'token_healing' (bool): Use token healing (default: False)
                'temperature' (float): Sampling temperature (default: 1.0)
                'temperature_last' (bool): Apply temperature after all other
                    samplers (default: False)
                'top_k' (int): Sampling top-K (default: 0)
                'top_p' (float): Sampling top-P (default: 1.0)
                'min_p' (float): Sampling min-P (default: 0.0)
                'tfs' (float): Tail-free sampling (default: 0.0)
                'typical' (float): Sampling typical (default: 0.0)
                'mirostat' (bool): Use Mirostat (default: False)
                'mirostat_tau' (float) Mirostat tau parameter (default: 1.5)
                'mirostat_eta' (float) Mirostat eta parameter (default: 0.1)
                'repetition_penalty' (float): Token repetition/presence penalty
                    (default: 1.15)
                'repetition_range' (int): Repetition penalty range
                    (default: whole context)
                'repetition_decay' (int): Repetition penalty range
                    (default: same as range)
                'stop' (List[Union[str, int]]): List of stop strings/tokens to
                    end response (default: [EOS])
                'max_tokens' (int): Max no. tokens in response (default: 150)
                'add_bos_token' (bool): Adds the BOS token to the start of the
                    prompt (default: True)
                'ban_eos_token' (bool): Bans the EOS token from generation
                    (default: False)
                'logit_bias' (Dict[int, float]): Biases specific tokens to
                    either show up more or less (default: None)
                'stream_interval' (float): Interval in seconds between each
                    output chunk (default: immediate)
                'generate_window' (int): Space to reserve at the end of the
                    model's context when generating. Rolls context window by
                    the same amount if context length is exceeded to allow
                    generating pastthe models max_seq_len.
        """

        token_healing = unwrap(kwargs.get("token_healing"), False)
        max_tokens = unwrap(kwargs.get("max_tokens"), 150)
        stream_interval = unwrap(kwargs.get("stream_interval"), 0)
        generate_window = min(
            unwrap(kwargs.get("generate_window"), 512), max_tokens
        )

        # Sampler settings
        gen_settings = ExLlamaV2Sampler.Settings()

        # Warn of unsupported settings if the setting is enabled
        if (unwrap(kwargs.get("mirostat"), False)) and not hasattr(
            gen_settings, "mirostat"
        ):
            print(
                " !! Warning: Currently installed ExLlamaV2 does not support "
                "Mirostat sampling"
            )

        if (unwrap(kwargs.get("min_p"), 0.0)) not in [0.0, 1.0] and not hasattr(
            gen_settings, "min_p"
        ):
            print(
                " !! Warning: Currently installed ExLlamaV2 does not "
                "support min-P sampling"
            )

        if (unwrap(kwargs.get("tfs"), 0.0)) not in [0.0, 1.0] and not hasattr(
            gen_settings, "tfs"
        ):
            print(
                " !! Warning: Currently installed ExLlamaV2 does not support "
                "tail-free sampling (TFS)"
            )

        if (unwrap(kwargs.get("temperature_last"), False)) and not hasattr(
            gen_settings, "temperature_last"
        ):
            print(
                " !! Warning: Currently installed ExLlamaV2 does not support "
                "temperature_last"
            )

        # Apply settings
        gen_settings.temperature = unwrap(kwargs.get("temperature"), 1.0)
        gen_settings.temperature_last = unwrap(
            kwargs.get("temperature_last"), False
        )
        gen_settings.top_k = unwrap(kwargs.get("top_k"), 0)
        gen_settings.top_p = unwrap(kwargs.get("top_p"), 1.0)
        gen_settings.min_p = unwrap(kwargs.get("min_p"), 0.0)
        gen_settings.tfs = unwrap(kwargs.get("tfs"), 1.0)
        gen_settings.typical = unwrap(kwargs.get("typical"), 1.0)
        gen_settings.mirostat = unwrap(kwargs.get("mirostat"), False)

        # Default tau and eta fallbacks don't matter if mirostat is off
        gen_settings.mirostat_tau = unwrap(kwargs.get("mirostat_tau"), 1.5)
        gen_settings.mirostat_eta = unwrap(kwargs.get("mirostat_eta"), 0.1)
        gen_settings.token_repetition_penalty = unwrap(
            kwargs.get("repetition_penalty"), 1.0
        )
        gen_settings.token_repetition_range = unwrap(
            kwargs.get("repetition_range"), self.config.max_seq_len
        )

        # Always make sure the fallback is 0 if range < 0
        # It's technically fine to use -1, but this just validates the passed
        # fallback
        # Always default to 0 if something goes wrong
        if gen_settings.token_repetition_range <= 0:
            fallback_decay = 0
        else:
            fallback_decay = gen_settings.token_repetition_range
        gen_settings.token_repetition_decay = coalesce(
            kwargs.get("repetition_decay"), fallback_decay, 0
        )

        stop_conditions: List[Union[str, int]] = unwrap(kwargs.get("stop"), [])
        add_bos_token = unwrap(kwargs.get("add_bos_token"), True)
        ban_eos_token = unwrap(kwargs.get("ban_eos_token"), False)
        logit_bias = kwargs.get("logit_bias")

        # Override sampler settings for temp = 0
        if gen_settings.temperature == 0:
            gen_settings.temperature = 1.0
            gen_settings.top_k = 1
            gen_settings.top_p = 0
            gen_settings.typical = 0

        # Log generation options to console
        # Some options are too large, so log the args instead
        log_generation_params(
            max_tokens=max_tokens,
            **vars(gen_settings),
            token_healing=token_healing,
            add_bos_token=add_bos_token,
            ban_eos_token=ban_eos_token,
            stop_conditions=stop_conditions,
            logit_bias=logit_bias,
        )

        # Log prompt to console
        log_prompt(prompt)

        # Set logit bias
        if logit_bias:
            # Create a vocab tensor if it doesn't exist for token biasing
            if gen_settings.token_bias is None:
                padding = -self.tokenizer.config.vocab_size % 32
                gen_settings.token_bias = torch.zeros(
                    (self.tokenizer.config.vocab_size + padding,),
                    dtype=torch.float,
                )

            # Map logits to the tensor with their biases
            for token, bias in logit_bias.items():
                gen_settings.token_bias[token] = bias

        # Ban the EOS token if specified. If not, append to stop conditions
        # as well.
        # Set this below logging to avoid polluting the stop strings array
        if ban_eos_token:
            gen_settings.disallow_tokens(
                self.tokenizer, [self.tokenizer.eos_token_id]
            )
        else:
            stop_conditions.append(self.tokenizer.eos_token_id)

        # Stop conditions
        self.generator.set_stop_conditions(stop_conditions)

        # Tokenized context
        ids = self.tokenizer.encode(
            prompt, add_bos=add_bos_token, encode_special_tokens=True
        )
        context_len = len(ids[0])

        if context_len > self.config.max_seq_len:
            print(
                f"WARNING: The context length {context_len} is greater than "
                f"the max_seq_len {self.config.max_seq_len}.",
                "Generation is truncated and metrics may not be accurate.",
            )

        prompt_tokens = ids.shape[-1]

        # Begin
        generated_tokens = 0
        full_response = ""
        start_time = time.time()
        last_chunk_time = start_time

        save_tokens = torch.empty((1, 0), dtype=torch.bool)
        chunk_buffer = ""
        chunk_tokens = 0

        while True:
            # Ingest prompt
            if chunk_tokens == 0:
                ids = torch.cat((ids, save_tokens), dim=-1)
                save_tokens = torch.empty((1, 0), dtype=torch.bool)
                overflow = (
                    ids.shape[-1] + generate_window - self.config.max_seq_len
                )
                active_ids = ids[:, max(0, overflow) :]
                chunk_tokens = self.config.max_seq_len - active_ids.shape[-1]

                self.generator.begin_stream(
                    active_ids,
                    gen_settings,
                    token_healing=token_healing,
                    loras=self.active_loras,
                )

            # Generate
            chunk, eos, tokens = self.generator.stream()

            if token_healing:
                # Extract healed token
                ids[:, -1] = self.generator.sequence_ids[:, -2]
                token_healing = False

            save_tokens = torch.cat((save_tokens, tokens), dim=-1)
            chunk_buffer += chunk

            generated_tokens += 1
            chunk_tokens -= 1

            # Yield output
            now = time.time()
            elapsed = now - last_chunk_time

            if chunk_buffer != "" and (
                elapsed > stream_interval
                or eos
                or generated_tokens == max_tokens
            ):
                yield chunk_buffer, prompt_tokens, generated_tokens
                full_response += chunk_buffer
                chunk_buffer = ""
                last_chunk_time = now

            if eos or generated_tokens == max_tokens:
                break

        # Print response
        log_response(full_response)

        elapsed_time = last_chunk_time - start_time

        initial_response = (
            f"Metrics: {generated_tokens} tokens generated in "
            f"{round(elapsed_time, 2)} seconds"
        )
        itemization = []
        extra_parts = []

        # Add tokens per second
        tokens_per_second = (
            "Indeterminate"
            if elapsed_time == 0
            else round(generated_tokens / elapsed_time, 2)
        )
        itemization.append(f"{tokens_per_second} T/s")

        # Add context (original token count)
        if ids is not None:
            itemization.append(f"context {context_len} tokens")

        if context_len > self.config.max_seq_len:
            extra_parts.append("<-- Not accurate (truncated)")

        # Print output
        print(
            initial_response
            + " ("
            + ", ".join(itemization)
            + ") "
            + " ".join(extra_parts)
        )
