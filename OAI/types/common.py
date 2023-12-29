""" Common types for OAI. """
from pydantic import BaseModel, Field, AliasChoices
from typing import List, Dict, Optional, Union


class LogProbs(BaseModel):
    """Represents log probabilities."""

    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[float] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Dict[str, float]] = Field(default_factory=list)


class UsageStats(BaseModel):
    """Represents usage stats."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CommonCompletionRequest(BaseModel):
    """Represents a common completion request."""

    # Model information
    # This parameter is not used, the loaded model is used instead
    model: Optional[str] = None

    # Extra OAI request stuff
    best_of: Optional[int] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )
    echo: Optional[bool] = Field(
        description="Not parsed. Only used for OAI compliance.", default=False
    )
    logprobs: Optional[int] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )
    n: Optional[int] = Field(
        description="Not parsed. Only used for OAI compliance.", default=1
    )
    suffix: Optional[str] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )
    user: Optional[str] = Field(
        description="Not parsed. Only used for OAI compliance.", default=None
    )

    # Generation info
    # seed: Optional[int] = -1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = []

    # Default to 150 as 16 makes no sense as a default
    max_tokens: Optional[int] = 150

    # Sampling params
    token_healing: Optional[bool] = False
    temperature: Optional[float] = 1.0
    temperature_last: Optional[bool] = False
    top_k: Optional[int] = 0
    top_p: Optional[float] = 1.0
    top_a: Optional[float] = 0.0
    typical: Optional[float] = 1.0
    min_p: Optional[float] = 0.0
    tfs: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    repetition_decay: Optional[int] = 0
    mirostat_mode: Optional[int] = 0
    mirostat_tau: Optional[float] = 1.5
    mirostat_eta: Optional[float] = 0.1
    add_bos_token: Optional[bool] = True
    ban_eos_token: Optional[bool] = False
    logit_bias: Optional[Dict[int, float]] = Field(default=None, examples=[[{"1": 10}]])

    # Aliased variables
    penalty_range: Optional[int] = Field(
        default=-1,
        validation_alias=AliasChoices(
            "penalty_range",
            "repetition_range",
            "repetition_penalty_range",
        ),
    )

    def to_gen_params(self):
        """Converts to internal generation parameters."""
        # Convert stop to an array of strings
        if isinstance(self.stop, str):
            self.stop = [self.stop]

        return {
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "add_bos_token": self.add_bos_token,
            "ban_eos_token": self.ban_eos_token,
            "token_healing": self.token_healing,
            "logit_bias": self.logit_bias,
            "temperature": self.temperature,
            "temperature_last": self.temperature_last,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "top_a": self.top_a,
            "typical": self.typical,
            "min_p": self.min_p,
            "tfs": self.tfs,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "penalty_range": self.penalty_range,
            "repetition_decay": self.repetition_decay,
            "mirostat": self.mirostat_mode == 2,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta,
        }
