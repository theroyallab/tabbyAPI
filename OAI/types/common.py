from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union

class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[float] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Dict[str, float]] = Field(default_factory=list)

class UsageStats(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class CommonCompletionRequest(BaseModel):
    # Model information
    # This parameter is not used, the loaded model is used instead
    model: Optional[str] = None

    # Extra OAI request stuff
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    n: Optional[int] = 1
    suffix: Optional[str] = None
    user: Optional[str] = None

    # Generation info
    seed: Optional[int] = -1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = []

    # Default to 150 as 16 makes no sense as a default
    max_tokens: Optional[int] = 150

    # Aliased to repetition_penalty
    frequency_penalty: Optional[float] = 0.0

    # Sampling params
    token_healing: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 0
    top_p: Optional[float] = 1.0
    typical: Optional[float] = 0.0
    min_p: Optional[float] = 0.0
    tfs: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 1.0
    repetition_penalty_range: Optional[int] = 0
    repetition_decay: Optional[int] = 0
    mirostat_mode: Optional[int] = 0
    mirostat_tau: Optional[float] = 1.5
    mirostat_eta: Optional[float] = 0.1
    add_bos_token: Optional[bool] = True
    ban_eos_token: Optional[bool] = False

    # Converts to internal generation parameters
    def to_gen_params(self):
        # Convert stop to an array of strings
        if isinstance(self.stop, str):
            self.stop = [self.stop]
    
        # Set repetition_penalty to frequency_penalty if repetition_penalty isn't already defined
        if (self.repetition_penalty is None or self.repetition_penalty == 1.0) and self.frequency_penalty:
            self.repetition_penalty = self.frequency_penalty
    
        return {
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "add_bos_token": self.add_bos_token,
            "ban_eos_token": self.ban_eos_token,
            "token_healing": self.token_healing,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "typical": self.typical,
            "min_p": self.min_p,
            "tfs": self.tfs,
            "repetition_penalty": self.repetition_penalty,
            "repetition_penalty_range": self.repetition_penalty_range,
            "repetition_decay": self.repetition_decay,
            "mirostat": self.mirostat_mode == 2,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta
        }
