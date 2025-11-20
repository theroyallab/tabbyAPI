from dataclasses import dataclass, field
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
)


@dataclass
class ExllamaV3SamplerBuilder:
    """
    Custom sampler chain/stack for TabbyAPI
    """

    stack: list[SS_Base] = field(default_factory=list)

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

    def build(self, greedy):
        """Builds the final sampler from stack."""

        # Use greedy if temp is 0
        if greedy:
            return CustomSampler([SS_Argmax()])
        else:
            self.stack.append(SS_Sample())
            return CustomSampler(self.stack)
