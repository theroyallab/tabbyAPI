from dataclasses import dataclass, field
from typing import List
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
    SS_BanTokens,
    SS_XTC,
    SS_LogitBias,
)

# Logits-space steps that remain meaningful under greedy decoding: they can
# change which token has the highest logit, unlike the probability-shaping
# steps (temperature, top-k/p, min-p, XTC), which never alter the argmax
_GREEDY_KEPT_STEPS = tuple(
    step for step in (SS_LogitBias, SS_RepP, SS_PresFreqP, SS_BanTokens) if step is not None
)


@dataclass
class ExllamaV3SamplerBuilder:
    """
    Custom sampler chain/stack for TabbyAPI
    """

    stack: List[SS_Base] = field(default_factory=list)

    def logit_bias(self, logit_bias) -> bool:
        """Returns False when the installed exllamav3 lacks SS_LogitBias."""

        if SS_LogitBias is None:
            return False

        # Must run before the logits are transformed, so prepend it to the stack
        self.stack.insert(0, SS_LogitBias(logit_bias))
        return True

    def penalties(self, rep_p, freq_p, pres_p, penalty_range, rep_decay):
        self.stack += [
            SS_RepP(rep_p, penalty_range, rep_decay),
            SS_PresFreqP(pres_p, freq_p, penalty_range, rep_decay),
        ]

    def ban_tokens(self, banned_tokens):
        self.stack.append(SS_BanTokens(banned_tokens))

    def temperature(self, temp):
        self.stack.append(SS_Temperature(temp))

    def top_k(self, top_k):
        self.stack.append(SS_TopK(top_k))

    def top_p(self, top_p):
        self.stack.append(SS_TopP(top_p))

    def min_p(self, min_p):
        self.stack.append(SS_MinP(min_p))

    def xtc(self, xtc_probability, xtc_threshold, tokenizer):
        # The tokenizer supplies the default set of protected tokens
        # (newline pieces and special tokens)
        self.stack.append(SS_XTC(xtc_probability, xtc_threshold, tokenizer=tokenizer))

    def greedy(self):
        self.stack.append(SS_Argmax())

    def adaptive_p(self, adaptive_target, adaptive_decay):
        self.stack.append(SS_AdaptiveP(adaptive_target, adaptive_decay))

    def build(self, greedy):
        """Builds the final sampler from stack."""

        # Adaptive-P does categorical sampling already
        if len(self.stack) and isinstance(self.stack[-1], SS_AdaptiveP):
            return CustomSampler(self.stack)

        # Use greedy if temp is 0. Probability-shaping steps are dropped, but
        # logit biases, penalties and token bans still affect the argmax
        if greedy:
            kept = [s for s in self.stack if isinstance(s, _GREEDY_KEPT_STEPS)]
            return CustomSampler(kept + [SS_Argmax()])
        else:
            self.stack.append(SS_Sample())
            return CustomSampler(self.stack)
