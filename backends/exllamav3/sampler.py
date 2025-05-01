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


class ExllamaV3SamplerBuilder(CustomSampler):

    def __init__(self, params):
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
            # Apply penalties
            if params.repetition_penalty != 1.0:
                stack.append(SS_RepP(params.repetition_penalty))

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
