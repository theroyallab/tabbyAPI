from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from endpoints.OAI.reasoning.abs_reasoning_parsers import DeltaMessage, ReasoningParser
from endpoints.OAI.reasoning.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)
from endpoints.OAI.reasoning.identity_reasoning_parser import IdentityReasoningParser


class DeepSeekV3ReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = bool(chat_kwargs.get("thinking", False))
        enable_thinking = bool(chat_kwargs.get("enable_thinking", False))
        thinking = thinking or enable_thinking

        if thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self._parser.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: Any,
    ) -> tuple[str | None, str | None]:
        return self._parser.extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int],
        current_token_ids: list[int],
        delta_token_ids: list[int],
    ) -> DeltaMessage | None:
        return self._parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
