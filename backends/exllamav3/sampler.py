from dataclasses import dataclass, field
from typing import List
import torch
from exllamav3.generator.sampler import (
    CustomSampler,
    SS_Temperature,
    SS_RepP,
    SS_PresFreqP,
    SS_Argmax,
    SS_MinP,
    SS_TopK,
    SS_TopP,
    SS_Sample,
    SS_Base,
    SS_AdaptiveP,
)
from exllamav3.generator.sampler.custom import SS


class SS_BanTokens(SS_Base):
    """Sampling step that masks the given token IDs to negative infinity."""

    def __init__(self, token_ids):
        self.token_ids = list(token_ids)

    def run(self, state):
        match state.state:
            case SS.INIT:
                state.logits = state.in_logits.to(torch.float, copy=True)
                state.logits[:, self.token_ids] = float("-inf")
            case SS.LOGITS:
                state.logits[:, self.token_ids] = float("-inf")
            case _:
                raise ValueError("Sampling logic error")
        state.state = SS.LOGITS


@dataclass
class ExllamaV3SamplerBuilder:
    """
    Custom sampler chain/stack for TabbyAPI
    """

    stack: List[SS_Base] = field(default_factory=list)
    banned_tokens: List[int] = field(default_factory=list)

    def penalties(self, rep_p, freq_p, pres_p, penalty_range, rep_decay):
        self.stack += [
            SS_RepP(rep_p, penalty_range, rep_decay),
            SS_PresFreqP(pres_p, freq_p, penalty_range, rep_decay),
        ]

    def temperature(self, temp):
        self.stack.append(SS_Temperature(temp))

    def top_k(self, top_k):
        self.stack.append(SS_TopK(top_k))

    def top_p(self, top_p):
        self.stack.append(SS_TopP(top_p))

    def min_p(self, min_p):
        self.stack.append(SS_MinP(min_p))

    def greedy(self):
        self.stack.append(SS_Argmax())

    def adaptive_p(self, adaptive_target, adaptive_decay):
        self.stack.append(SS_AdaptiveP(adaptive_target, adaptive_decay))

    def ban_tokens(self, token_ids):
        self.banned_tokens = list(token_ids)

    def build(self, greedy):
        """Builds the final sampler from stack."""

        # A ban step runs first so masked tokens stay out of every later step,
        # including the greedy path that otherwise discards the stack.
        prefix = [SS_BanTokens(self.banned_tokens)] if self.banned_tokens else []

        # Adaptive-P does categorical sampling already
        if len(self.stack) and isinstance(self.stack[-1], SS_AdaptiveP):
            return CustomSampler(prefix + self.stack)

        # Use greedy if temp is 0
        if greedy:
            return CustomSampler(prefix + [SS_Argmax()])
        else:
            self.stack.append(SS_Sample())
            return CustomSampler(prefix + self.stack)
