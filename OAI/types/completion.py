from uuid import uuid4
from time import time
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from OAI.types.common import LogProbs, UsageStats, CommonCompletionRequest

class CompletionRespChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: str
    logprobs: Optional[LogProbs] = None
    text: str

# Inherited from common request
class CompletionRequest(CommonCompletionRequest):
    # Prompt can also contain token ids, but that's out of scope for this project.
    prompt: Union[str, List[str]]

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid4().hex}")
    choices: List[CompletionRespChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: str = "text_completion"

    # TODO: Add usage stats
    usage: Optional[UsageStats] = None
