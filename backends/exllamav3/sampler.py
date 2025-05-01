from exllamav3.generator.sampler.custom import (
    CustomSampler,
    SS_Argmax,
    SS_MinP,
    SS_PresFreqP,
    SS_RepP,
    SS_Sample,
    SS_Temperature,
    SS_TopK,
    SS_TopP,
)

from common.utils import coalesce, unwrap


class ExllamaV3SamplerBuilder(CustomSampler):

    def __init__(self, params, max_seq_len):
        """
        Initialize the ExllamaV3SamplerBuilder with all relevant sampling parameters.
        """

        stack = []

        # handle greedy if temperature is 0
        if params.temperature == 0.0:
            stack = [
                SS_Argmax()
            ]

        else:
            # Set penalty range

            penalty_range = unwrap(params.penalty_range, max_seq_len)
            # Exl3's version of including the entire context
            if penalty_range < 0:
                penalty_range = 10e7

            # Apply penalties
            # Always make sure the fallback is 0 if range < 0
            # It's technically fine to use -1, but this just validates the passed
            # fallback
            # Always default to 0 if something goes wrong
            if params.penalty_range < 0:
                fallback_decay = 0
            else:
                fallback_decay = params.penalty_range
            repetition_decay = coalesce(params.repetition_decay, fallback_decay, 0)

            if params.repetition_penalty != 1.0:
                stack.append(SS_RepP(
                    rep_p=params.repetition_penalty,
                    sustain_range=penalty_range,
                    decay_range=repetition_decay,
                ))

            if params.presence_penalty != 0 or params.frequency_penalty != 0:
                stack.append(
                    SS_PresFreqP(params.presence_penalty, params.frequency_penalty)
                )

            # Apply temperature
            if not params.temperature_last and params.temperature > 0:
                stack.append(SS_Temperature(params.temperature))

            # Apply alphabet samplers
            if params.top_k > 0:
                stack.append(SS_TopK(params.top_k))

            if params.top_p < 1.0:
                stack.append(SS_TopP(params.top_p))

            if params.min_p > 0:
                stack.append(SS_MinP(params.min_p))

            # Apply temperature again if needed
            if params.temperature_last and params.temperature > 0:
                stack.append(SS_Temperature(params.temperature))

            # Final sampling step
            stack.append(SS_Sample())

            # Initialize parent class with the constructed sampler list
        super().__init__(stack)
