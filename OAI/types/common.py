from pydantic import BaseModel, Field
from typing import List, Dict

class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[float] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Dict[str, float]] = Field(default_factory=list)

class UsageStats(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
