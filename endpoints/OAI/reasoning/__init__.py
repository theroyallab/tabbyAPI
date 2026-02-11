from endpoints.OAI.reasoning.abs_reasoning_parsers import (

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
    DeltaMessage,
    ReasoningParser,
    ReasoningParserManager,
)
from endpoints.OAI.reasoning.basic_parsers import BaseThinkingReasoningParser
from endpoints.OAI.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from endpoints.OAI.reasoning.deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser
from endpoints.OAI.reasoning.ernie45_reasoning_parser import Ernie45ReasoningParser
from endpoints.OAI.reasoning.exaone4_reasoning_parser import Exaone4ReasoningParser
from endpoints.OAI.reasoning.glm4_moe_reasoning_parser import Glm4MoeModelReasoningParser
from endpoints.OAI.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
from endpoints.OAI.reasoning.granite_reasoning_parser import GraniteReasoningParser
from endpoints.OAI.reasoning.holo2_reasoning_parser import Holo2ReasoningParser
from endpoints.OAI.reasoning.hunyuan_a13b_reasoning_parser import HunyuanA13BReasoningParser
from endpoints.OAI.reasoning.identity_reasoning_parser import IdentityReasoningParser
from endpoints.OAI.reasoning.kimi_k2_reasoning_parser import KimiK2ReasoningParser
from endpoints.OAI.reasoning.minimax_m2_reasoning_parser import (
    MiniMaxM2AppendThinkReasoningParser,
    MiniMaxM2ReasoningParser,
)
from endpoints.OAI.reasoning.mistral_reasoning_parser import MistralReasoningParser
from endpoints.OAI.reasoning.olmo3_reasoning_parser import Olmo3ReasoningParser
from endpoints.OAI.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from endpoints.OAI.reasoning.seedoss_reasoning_parser import SeedOSSReasoningParser
from endpoints.OAI.reasoning.step3_reasoning_parser import Step3ReasoningParser
from endpoints.OAI.reasoning.step3p5_reasoning_parser import Step3p5ReasoningParser


@ReasoningParserManager.register_module("identity")
class _IdentityParser(IdentityReasoningParser):
    pass


@ReasoningParserManager.register_module("basic")
class _BasicParser(DeepSeekR1ReasoningParser):
    pass


ReasoningParserManager.reasoning_parsers.update(
    {
        "deepseek_r1": DeepSeekR1ReasoningParser,
        "deepseek_v3": DeepSeekV3ReasoningParser,
        "ernie45": Ernie45ReasoningParser,
        "exaone4": Exaone4ReasoningParser,
        "glm45": Glm4MoeModelReasoningParser,
        "openai_gptoss": GptOssReasoningParser,
        "granite": GraniteReasoningParser,
        "holo2": Holo2ReasoningParser,
        "hunyuan_a13b": HunyuanA13BReasoningParser,
        "kimi_k2": KimiK2ReasoningParser,
        "minimax_m2": MiniMaxM2ReasoningParser,
        "minimax_m2_append_think": MiniMaxM2AppendThinkReasoningParser,
        "mistral": MistralReasoningParser,
        "olmo3": Olmo3ReasoningParser,
        "qwen3": Qwen3ReasoningParser,
        "seed_oss": SeedOSSReasoningParser,
        "step3": Step3ReasoningParser,
        "step3p5": Step3p5ReasoningParser,
    }
)


__all__ = [
    "BaseThinkingReasoningParser",
    "DeltaMessage",
    "DeepSeekR1ReasoningParser",
    "DeepSeekV3ReasoningParser",
    "Ernie45ReasoningParser",
    "Exaone4ReasoningParser",
    "Glm4MoeModelReasoningParser",
    "GptOssReasoningParser",
    "GraniteReasoningParser",
    "Holo2ReasoningParser",
    "HunyuanA13BReasoningParser",
    "IdentityReasoningParser",
    "KimiK2ReasoningParser",
    "MiniMaxM2AppendThinkReasoningParser",
    "MiniMaxM2ReasoningParser",
    "MistralReasoningParser",
    "Olmo3ReasoningParser",
    "Qwen3ReasoningParser",
    "ReasoningParser",
    "ReasoningParserManager",
    "SeedOSSReasoningParser",
    "Step3ReasoningParser",
    "Step3p5ReasoningParser",
]
