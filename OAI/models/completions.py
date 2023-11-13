from uuid import uuid4
from time import time
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from OAI.models.common import LogProbs, UsageStats

class CompletionRespChoice(BaseModel):
    finish_reason: str
    index: int
    logprobs: Optional[LogProbs] = None
    text: str

class CompletionRequest(BaseModel):
    # Model information
    model: str

    # Prompt can also contain token ids, but that's out of scope for this project.
    prompt: Union[str, List[str]]

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
    stop: Optional[Union[str, List[str]]] = None

    # Default to 150 as 16 makes no sense as a default
    max_tokens: Optional[int] = 150

    # Not supported sampling params
    presence_penalty: Optional[int] = 0

    # Aliased to repetition_penalty
    frequency_penalty: int = 0

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

    # Converts to internal generation parameters
    def to_gen_params(self):
        # Convert prompt to a string
        if isinstance(self.prompt, list):
            self.prompt = "\n".join(self.prompt)

        # Convert stop to an array of strings
        if isinstance(self.stop, str):
            self.stop = [self.stop]

        # Set repetition_penalty to frequency_penalty if repetition_penalty isn't already defined
        if (self.repetition_penalty is None or self.repetition_penalty == 1.0) and self.frequency_penalty:
            self.repetition_penalty = self.frequency_penalty

        return {
            "prompt": self.prompt,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
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
            "mirostat": True if self.mirostat_mode == 2 else False,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta
        }

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid4().hex}")
    choices: List[CompletionRespChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: str = "text-completion"

    # TODO: Add usage stats
    usage: Optional[UsageStats] = None
