"""The model container class for ExLlamaV2 models."""
import gc
from itertools import zip_longest
import pathlib
import time

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
from typing import List, Optional, Union

from common.gen_logging import log_generation_params, log_prompt, log_response
from common.templating import (
    PromptTemplate,
    find_template_from_model,
    get_template_from_model_json,
    get_template_from_file,
)
from common.utils import coalesce, unwrap
from common.logger import init_logger

logger = init_logger(__name__)

# Bytes to reserve on first device when loading with auto split
AUTO_SPLIT_RESERVE_BYTES = 96 * 1024**2


class ExllamaV2Container:
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
    use_cfg: bool = False

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
                'use_cfg" (bool): Enables CFG support. Disables flash attention
                    (default: False)
        """

        self.quiet = quiet

        self.cache_fp8 = "cache_mode" in kwargs and kwargs["cache_mode"] == "FP8"

        # Turn off GPU split if the user is using 1 GPU
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            self.gpu_split = kwargs.get("gpu_split")
            self.gpu_split_auto = unwrap(kwargs.get("gpu_split_auto"), True)
        else:
            self.gpu_split_auto = False
            logger.info("Disabling GPU split because one GPU is in use.")

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
        self.config.scale_pos_emb = unwrap(
            kwargs.get("rope_scale"), self.config.scale_pos_emb
        )

        # Automatically calculate rope alpha
        self.config.scale_alpha_value = unwrap(
            kwargs.get("rope_alpha"), self.calculate_rope_alpha(base_seq_len)
        )

        # Enable CFG if present
        self.use_cfg = unwrap(kwargs.get("use_cfg"), False)

        # Enable fasttensors loading if present
        self.config.fasttensors = unwrap(kwargs.get("fasttensors"), False)

        # Turn off flash attention if CFG is on
        # Workaround until batched FA2 is fixed in exllamav2 upstream
        self.config.no_flash_attn = (
            True if self.use_cfg else unwrap(kwargs.get("no_flash_attention"), False)
        )

        # low_mem is currently broken in exllamav2. Don't use it until it's
        # fixed.
        """
        if "low_mem" in kwargs and kwargs["low_mem"]:
            self.config.set_low_mem()
        """

        # Try to set prompt template
        self.prompt_template = self.find_prompt_template(
            kwargs.get("prompt_template"), model_directory
        )

        # Catch all for template lookup errors
        if self.prompt_template:
            logger.info(
                f"Using template {self.prompt_template.name} " "for chat completions."
            )
        else:
            logger.warning(
                "Chat completions are disabled because a prompt "
                "template wasn't provided or auto-detected."
            )

        # Set num of experts per token if provided
        num_experts_override = kwargs.get("num_experts_per_token")
        if num_experts_override:
            self.config.num_experts_per_token = kwargs.get("num_experts_per_token")

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
            logger.warning(
                "Draft model is disabled because a model name "
                "wasn't provided. Please check your config.yml!"
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

    def find_prompt_template(self, prompt_template_name, model_directory):
        """Tries to find a prompt template using various methods"""

        logger.info("Attempting to load a prompt template if present.")

        find_template_functions = [
            lambda: get_template_from_model_json(
                pathlib.Path(self.config.model_dir) / "tokenizer_config.json",
                "chat_template",
                "from_tokenizer_config",
            ),
            lambda: get_template_from_file(find_template_from_model(model_directory)),
        ]

        # Add lookup from prompt template name if provided
        if prompt_template_name:
            find_template_functions.insert(
                0, lambda: get_template_from_file(prompt_template_name)
            )

        for func in find_template_functions:
            try:
                prompt_template = func()
                if prompt_template is not None:
                    return prompt_template
            except (FileNotFoundError, LookupError):
                continue

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
                logger.warning(
                    "One of your loras does not have a name. Please check your "
                    "config.yml! Skipping lora load."
                )
                failure.append(lora_name)
                continue

            logger.info(f"Loading lora: {lora_name} at scaling {lora_scaling}")
            lora_path = lora_directory / lora_name
            # FIXME(alpin): Does self.model need to be passed here?
            self.active_loras.append(
                ExLlamaV2Lora.from_directory(self.model, lora_path, lora_scaling)
            )
            logger.info(f"Lora successfully loaded: {lora_name}")
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
                logger.info("Loading draft model: " + self.draft_config.model_dir)

            self.draft_cache = ExLlamaV2Cache(self.draft_model, lazy=True)
            reserve = [AUTO_SPLIT_RESERVE_BYTES] + [0] * 16
            yield from self.draft_model.load_autosplit_gen(
                self.draft_cache,
                reserve_vram=reserve,
                last_id_only=True,
                callback_gen=progress_callback,
            )

            # Test VRAM allocation with a full-length forward pass
            input_ids = torch.zeros((1, self.config.max_input_len), dtype=torch.long)
            self.draft_model.forward(input_ids, cache=self.cache, preprocess_only=True)

        self.model = ExLlamaV2(self.config)
        if not self.quiet:
            logger.info("Loading model: " + self.config.model_dir)

        # Load model with manual split
        # Entrypoint for single GPU users
        if not self.gpu_split_auto:
            logger.info("Loading with a manual GPU split (or a one GPU setup)")

            for value in self.model.load_gen(
                self.gpu_split,
                callback_gen=progress_callback,
            ):
                if value:
                    yield value

        batch_size = 2 if self.use_cfg else 1
        if self.cache_fp8:
            self.cache = ExLlamaV2Cache_8bit(
                self.model, lazy=self.gpu_split_auto, batch_size=batch_size
            )
        else:
            self.cache = ExLlamaV2Cache(
                self.model, lazy=self.gpu_split_auto, batch_size=batch_size
            )

        # Load model with autosplit
        if self.gpu_split_auto:
            logger.info("Loading with autosplit")

            reserve = [AUTO_SPLIT_RESERVE_BYTES] + [0] * 16
            for value in self.model.load_autosplit_gen(
                self.cache,
                reserve_vram=reserve,
                last_id_only=True,
                callback_gen=progress_callback,
            ):
                if value:
                    yield value

        # Test VRAM allocation with a full-length forward pass
        input_ids = torch.zeros((1, self.config.max_input_len), dtype=torch.long)
        self.model.forward(input_ids, cache=self.cache, preprocess_only=True)

        # Create generator
        self.generator = ExLlamaV2StreamingGenerator(
            self.model,
            self.cache,
            self.tokenizer,
            self.draft_model,
            self.draft_cache,
        )

        # Always return logprobs and logits
        self.generator.return_probabilities = True
        self.generator.return_logits = True

        # Clean up any extra vram usage from torch and cuda
        # (Helps reduce VRAM bottlenecking on Windows)
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Model successfully loaded.")

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

    def encode_tokens(self, text: str, **kwargs):
        """Wrapper to encode tokens from a text string"""

        return self.tokenizer.encode(
            text,
            add_bos=unwrap(kwargs.get("add_bos_token"), True),
            encode_special_tokens=unwrap(kwargs.get("encode_special_tokens"), True),
        )[0].tolist()

    def decode_tokens(self, ids: List[int], **kwargs):
        """Wrapper to decode tokens from a list of IDs"""

        ids = torch.tensor([ids])
        return self.tokenizer.decode(
            ids,
            decode_special_tokens=unwrap(kwargs.get("decode_special_tokens"), True),
        )[0]

    def get_special_tokens(self, add_bos_token: bool, ban_eos_token: bool):
        return {
            "bos_token": self.tokenizer.bos_token if add_bos_token else "",
            "eos_token": self.tokenizer.eos_token if not ban_eos_token else "",
            "pad_token": self.tokenizer.pad_token,
            "unk_token": self.tokenizer.unk_token,
        }

    def get_logprobs(self, logits: torch.Tensor, max_logprobs: int):
        normalized_logits = torch.log_softmax(logits, dim=-1)
        top_values, top_ids = torch.topk(normalized_logits, max_logprobs, dim=-1)

        top_tokens = list(
            map(
                lambda index: self.tokenizer.extended_id_to_piece.get(
                    index, self.tokenizer.id_to_piece[index]
                ),
                top_ids[0].tolist(),
            )
        )
        top_values = top_values[0].tolist()

        return dict(zip_longest(top_tokens, top_values))

    def get_token_probs(self, token_ids: torch.tensor, token_probs: torch.Tensor):
        normalized_probs = torch.log(token_probs)

        tokens = list(
            map(
                lambda index: self.tokenizer.extended_id_to_piece.get(
                    index, self.tokenizer.id_to_piece[index]
                ),
                token_ids[0].tolist(),
            )
        )

        return dict(zip_longest(tokens, normalized_probs[0].tolist()))

    def generate(self, prompt: str, **kwargs):
        """Generate a response to a prompt"""
        generations = list(self.generate_gen(prompt, **kwargs))

        joined_generation = {
            "text": "",
            "prompt_tokens": 0,
            "generation_tokens": 0,
            "offset": [],
            "token_probs": {},
            "logprobs": [],
        }

        if generations:
            for generation in generations:
                joined_generation["text"] += unwrap(generation.get("text"), "")
                joined_generation["offset"].append(unwrap(generation.get("offset"), []))
                joined_generation["token_probs"].update(
                    unwrap(generation.get("token_probs"), {})
                )

                # Include empty logprob dicts for index preservation
                joined_generation["logprobs"].append(
                    unwrap(generation.get("logprobs"), {})
                )

            joined_generation["prompt_tokens"] = unwrap(
                generations[-1].get("prompt_tokens"), 0
            )
            joined_generation["generation_tokens"] = unwrap(
                generations[-1].get("generated_tokens"), 0
            )

        return joined_generation

    def check_unsupported_settings(self, **kwargs):
        """Check and warn the user if a sampler is unsupported. Meant for dev wheels!"""

        pass

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
                'frequency_penalty' (float): Token frequency penalty (default: 0.0)
                'presence_penalty' (float): Token presence penalty (default: 0.0)
                'repetition_penalty' (float): Token repetition penalty
                    (default: 1.15)
                'penalty_range' (int): Penalty range
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
        generate_window = max(
            unwrap(kwargs.get("generate_window"), 512), self.config.max_seq_len // 8
        )

        # Sampler settings
        gen_settings = ExLlamaV2Sampler.Settings()

        # Check unsupported settings for dev wheels
        self.check_unsupported_settings(**kwargs)

        # Apply settings
        gen_settings.temperature = unwrap(kwargs.get("temperature"), 1.0)
        gen_settings.temperature_last = unwrap(kwargs.get("temperature_last"), False)
        gen_settings.smoothing_factor = unwrap(kwargs.get("smoothing_factor"), 0.0)
        gen_settings.top_k = unwrap(kwargs.get("top_k"), 0)
        gen_settings.top_p = unwrap(kwargs.get("top_p"), 1.0)
        gen_settings.top_a = unwrap(kwargs.get("top_a"), 0.0)
        gen_settings.min_p = unwrap(kwargs.get("min_p"), 0.0)
        gen_settings.tfs = unwrap(kwargs.get("tfs"), 1.0)
        gen_settings.typical = unwrap(kwargs.get("typical"), 1.0)
        gen_settings.mirostat = unwrap(kwargs.get("mirostat"), False)

        # DynaTemp settings
        max_temp = unwrap(kwargs.get("max_temp"), 1.0)
        min_temp = unwrap(kwargs.get("min_temp"), 1.0)

        if max_temp > min_temp:
            gen_settings.max_temp = max_temp
            gen_settings.min_temp = min_temp
            gen_settings.temp_exponent = unwrap(kwargs.get("temp_exponent"), 1.0)
        else:
            # Force to default values
            gen_settings.max_temp = 0.0
            gen_settings.min_temp = 0.0
            gen_settings.temp_exponent = 1.0

        # Warn if max/min temp values are > 0
        # and if they're less than or equal to each other
        if max_temp < min_temp or (
            1 not in {min_temp, max_temp} and max_temp == min_temp
        ):
            logger.warning(
                "Max temp is less than or equal to min temp, skipping DynaTemp."
            )

        # Default tau and eta fallbacks don't matter if mirostat is off
        gen_settings.mirostat_tau = unwrap(kwargs.get("mirostat_tau"), 1.5)
        gen_settings.mirostat_eta = unwrap(kwargs.get("mirostat_eta"), 0.1)

        # Set CFG scale and negative prompt
        cfg_scale = unwrap(kwargs.get("cfg_scale"), 1.0)
        negative_prompt = None
        if cfg_scale not in [None, 1.0]:
            if self.use_cfg:
                gen_settings.cfg_scale = cfg_scale

                # If the negative prompt is empty, use the BOS token
                negative_prompt = unwrap(
                    kwargs.get("negative_prompt"), self.tokenizer.bos_token
                )
            else:
                logger.warn(
                    "CFG is currently disabled. "
                    "Please reload your model with use_cfg = True.",
                )

        gen_settings.token_repetition_penalty = unwrap(
            kwargs.get("repetition_penalty"), 1.0
        )
        gen_settings.token_frequency_penalty = unwrap(
            kwargs.get("frequency_penalty"), 0.0
        )
        gen_settings.token_presence_penalty = unwrap(
            kwargs.get("presence_penalty"), 0.0
        )

        # Applies for all penalties despite being called token_repetition_range
        gen_settings.token_repetition_range = unwrap(
            kwargs.get("penalty_range"), self.config.max_seq_len
        )

        # Dynamically scale penalty range to output tokens
        # Only do this if freq/pres pen is enabled
        # and the repetition range is -1
        auto_scale_penalty_range = (
            gen_settings.token_frequency_penalty != 0
            or gen_settings.token_presence_penalty != 0
        ) and gen_settings.token_repetition_range == -1

        # Always make sure the fallback is 0 if range < 0
        # It's technically fine to use -1, but this just validates the passed
        # fallback
        # Always default to 0 if something goes wrong
        if gen_settings.token_repetition_range < 0:
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
        request_logprobs = unwrap(kwargs.get("logprobs"), 0)

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
            auto_scale_penalty_range=auto_scale_penalty_range,
            generate_window=generate_window,
            add_bos_token=add_bos_token,
            ban_eos_token=ban_eos_token,
            logprobs=request_logprobs,
            stop_conditions=stop_conditions,
            logit_bias=logit_bias,
        )

        # Log prompt to console
        log_prompt(prompt, negative_prompt)

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
            gen_settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])
        else:
            stop_conditions.append(self.tokenizer.eos_token_id)

        # Stop conditions
        self.generator.set_stop_conditions(stop_conditions)

        # Tokenized context
        ids, offsets = self.tokenizer.encode(
            [prompt, negative_prompt]
            if negative_prompt and gen_settings.cfg_scale not in [None, 1.0]
            else prompt,
            add_bos=add_bos_token,
            encode_special_tokens=True,
            return_offsets=True,
        )
        mask = (
            self.tokenizer.padding_mask(ids)
            if self.use_cfg and gen_settings.cfg_scale not in [None, 1.0]
            else None
        )
        context_len = len(ids[0])

        if context_len > self.config.max_seq_len:
            logger.warning(
                f"Context length {context_len} is greater than max_seq_len "
                f"{self.config.max_seq_len}. Generation is truncated and "
                "metrics may not be accurate."
            )

        prompt_tokens = ids.shape[-1]

        # Begin
        generated_tokens = 0
        full_response = ""
        start_time = time.time()
        last_chunk_time = start_time

        save_tokens = torch.empty((ids.shape[0], 0), dtype=torch.bool)
        chunk_buffer = ""
        chunk_tokens = 0

        while True:
            # Ingest prompt
            if chunk_tokens == 0:
                ids = torch.cat((ids, save_tokens), dim=-1)
                save_tokens = torch.empty((ids.shape[0], 0), dtype=torch.bool)
                overflow = ids.shape[-1] + generate_window - self.config.max_seq_len
                active_ids = ids[:, max(0, overflow) :]
                chunk_tokens = self.config.max_seq_len - active_ids.shape[-1]

                # Split for exllama versions that have CFG
                if self.use_cfg:
                    self.generator.begin_stream(
                        active_ids,
                        gen_settings,
                        token_healing=token_healing,
                        loras=self.active_loras,
                        input_mask=mask,
                        position_offsets=offsets,
                    )
                else:
                    self.generator.begin_stream(
                        active_ids,
                        gen_settings,
                        token_healing=token_healing,
                        loras=self.active_loras,
                    )

                # Reset offsets for subsequent passes if the context is truncated
                offsets = None

            if auto_scale_penalty_range:
                gen_settings.token_repetition_range = generated_tokens

            # Generate
            chunk, eos, tokens, token_probs, logits = self.generator.stream()

            if token_healing:
                # Extract healed token
                ids[:, -1] = self.generator.sequence_ids[:, -2]
                token_healing = False

            save_tokens = torch.cat(
                (save_tokens, tokens.expand(save_tokens.shape[0], -1)), dim=-1
            )
            chunk_buffer += chunk

            generated_tokens += 1
            chunk_tokens -= 1

            # Yield output
            now = time.time()
            elapsed = now - last_chunk_time

            if chunk_buffer != "" and (
                elapsed > stream_interval or eos or generated_tokens == max_tokens
            ):
                generation = {
                    "text": chunk_buffer,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "offset": len(full_response),
                }

                if request_logprobs > 0:
                    # Get sampled token probs
                    if token_probs.numel() > 0 and tokens.numel() > 0:
                        generation["token_probs"] = self.get_token_probs(
                            tokens, token_probs
                        )

                    # Get logprob choices
                    if logits.numel() > 0:
                        generation["logprobs"] = self.get_logprobs(
                            logits, request_logprobs
                        )

                yield generation
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
        logger.info(
            initial_response
            + " ("
            + ", ".join(itemization)
            + ") "
            + " ".join(extra_parts)
        )
