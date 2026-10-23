from dataclasses import dataclass, field
from typing import List, Tuple
from common.sampling import BaseSamplerRequest
from common.utils import coalesce, unwrap
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
    SS_DRY,
    dry_sequence_breaker_tokens,
)

# Logits-space steps that remain meaningful under greedy decoding: they can
# change which token has the highest logit, unlike the probability-shaping
# steps (temperature, top-k/p, min-p, XTC), which never alter the argmax
_GREEDY_KEPT_STEPS = tuple(
    step for step in (SS_LogitBias, SS_RepP, SS_PresFreqP, SS_DRY, SS_BanTokens) if step is not None
)


@dataclass
class ExllamaV3SamplerBuilder:
    """
    Custom sampler chain/stack for TabbyAPI
    """

    stack: List[SS_Base] = field(default_factory=list)

    # (request field, display value) for every setting that shaped the stack.
    # Neutral values are left out, so this lists what the sampler actually does.
    settings: List[Tuple[str, object]] = field(default_factory=list)

    @classmethod
    def from_params(cls, params: BaseSamplerRequest, tokenizer, max_seq_len: int):
        """Build the sampler stack for a request and record the settings in effect."""

        builder = cls()
        settings = builder.settings
        greedy = params.temperature == 0

        # Logit bias first so it lands ahead of the other steps
        if params.logit_bias and builder.logit_bias(params.logit_bias):
            settings.append(("logit_bias", f"{len(params.logit_bias)} tokens"))

        # Penalties. Range -1 means the whole context (exl3 takes a large number)
        penalty_range = unwrap(params.penalty_range, max_seq_len)
        if penalty_range < 0:
            penalty_range = int(10e7)

        fallback_decay = 0 if unwrap(params.penalty_range, -1) < 0 else params.penalty_range
        repetition_decay = coalesce(params.repetition_decay, fallback_decay, 0)

        builder.penalties(
            params.repetition_penalty,
            params.frequency_penalty,
            params.presence_penalty,
            penalty_range,
            repetition_decay,
        )

        penalties_active = False
        if params.repetition_penalty != 1.0:
            settings.append(("repetition_penalty", params.repetition_penalty))
            penalties_active = True
        if params.frequency_penalty:
            settings.append(("frequency_penalty", params.frequency_penalty))
            penalties_active = True
        if params.presence_penalty:
            settings.append(("presence_penalty", params.presence_penalty))
            penalties_active = True
        if penalties_active:
            if unwrap(params.penalty_range, -1) >= 0:
                settings.append(("penalty_range", params.penalty_range))
            if params.repetition_decay:
                settings.append(("repetition_decay", params.repetition_decay))

        # DRY works on the raw logits like the other penalties, so it goes before them
        # are transformed. An empty breaker list means the backend's default set
        if params.dry_multiplier > 0 and params.dry_base >= 1.0:
            breakers = None
            if params.dry_sequence_breakers:
                breakers = dry_sequence_breaker_tokens(
                    tokenizer, tuple(params.dry_sequence_breakers)
                )
            builder.dry(
                params.dry_multiplier,
                params.dry_base,
                params.dry_allowed_length,
                params.dry_range,
                breakers,
            )
            settings.append(("dry_multiplier", params.dry_multiplier))
            settings.append(("dry_base", params.dry_base))
            settings.append(("dry_allowed_length", params.dry_allowed_length))
            if params.dry_range:
                settings.append(("dry_range", params.dry_range))
            if params.dry_sequence_breakers:
                settings.append(
                    ("dry_sequence_breakers", f"{len(params.dry_sequence_breakers)} strings")
                )

        if params.banned_tokens:
            builder.ban_tokens(params.banned_tokens)
            settings.append(("banned_tokens", f"{len(params.banned_tokens)} tokens"))

        # Probability-shaping steps. Under greedy decoding these never change the
        # argmax and build() drops them, so they are not reported either
        shaping = []

        if not params.temperature_last:
            builder.temperature(params.temperature)

        # min-P, top-K, top-P is the order the backend's fused sampling kernel
        # accepts (and the order its own ComboSampler uses). Any other order with
        # two or more of them active falls back to the step-by-step path, which
        # sorts and renormalizes between steps and costs several times as much
        # per token
        builder.min_p(params.min_p)
        builder.top_k(params.top_k)
        builder.top_p(params.top_p)

        if params.temperature_last:
            builder.temperature(params.temperature)

        if params.temperature != 1.0:
            shaping.append(("temperature", params.temperature))
        if params.min_p > 0:
            shaping.append(("min_p", params.min_p))
        if params.top_k > 0:
            shaping.append(("top_k", params.top_k))
        if params.top_p < 1.0:
            shaping.append(("top_p", params.top_p))
        if params.temperature_last and params.temperature != 1.0:
            shaping.append(("temperature_last", True))

        if params.xtc_probability > 0.0:
            builder.xtc(params.xtc_probability, params.xtc_threshold, tokenizer)
            shaping.append(("xtc_probability", params.xtc_probability))
            shaping.append(("xtc_threshold", params.xtc_threshold))

        # Adaptive-P samples from the distribution, so temperature 0 wins over it
        # (as in the backend's ComboSampler): a temperature-0 step ahead of it
        # would divide the logits by zero
        if params.adaptive_target < 1.0 and not greedy:
            builder.adaptive_p(params.adaptive_target, params.adaptive_decay)
            shaping.append(("adaptive_target", params.adaptive_target))
            shaping.append(("adaptive_decay", params.adaptive_decay))

        if greedy:
            settings.append(("temperature", "0, greedy"))
        else:
            settings.extend(shaping)

        return builder

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

    def dry(self, multiplier, base, allowed_length, dry_range, sequence_breakers):
        self.stack.append(
            SS_DRY(
                dry_multiplier=multiplier,
                dry_base=base,
                dry_allowed_length=allowed_length,
                dry_range=dry_range,
                dry_sequence_breakers=sequence_breakers,
            )
        )

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

        # Use greedy if temp is 0. Probability-shaping steps (Adaptive-P included)
        # are dropped, but logit biases, penalties and token bans still affect
        # the argmax
        if greedy:
            kept = [s for s in self.stack if isinstance(s, _GREEDY_KEPT_STEPS)]
            return CustomSampler(kept + [SS_Argmax()])

        # Adaptive-P does categorical sampling already
        if len(self.stack) and isinstance(self.stack[-1], SS_AdaptiveP):
            return CustomSampler(self.stack)

        self.stack.append(SS_Sample())
        return CustomSampler(self.stack)
