from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from endpoints.OAI.reasoning.abs_reasoning_parsers import (
    DeltaMessage,
    ReasoningParserManager,
)
from endpoints.OAI.reasoning.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)
from endpoints.OAI.reasoning.deepseek_v3_reasoning_parser import (
    DeepSeekV3ReasoningParser,
)
from endpoints.OAI.reasoning.identity_reasoning_parser import IdentityReasoningParser


@ReasoningParserManager.register_module("exaone4")
class Exaone4ReasoningParser(DeepSeekV3ReasoningParser):
    """
    EXAONE-specific parser.

    Important model behavior:
    - Uses only `enable_thinking` to toggle reasoning mode.
    - Chat template may prefill `<think>` so model output often starts directly
      with reasoning body and closes with `</think>`.
    """

    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        enable_thinking = bool(chat_kwargs.get("enable_thinking", False))
        self._reasoning_ended = False

        if enable_thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    def _thinking_enabled(self) -> bool:
        return isinstance(self._parser, DeepSeekR1ReasoningParser)

    def _strip_stray_end_token(self, text: str) -> str:
        if not text or not self._thinking_enabled():
            return text

        end_token = self._parser.end_token
        start_token = self._parser.start_token
        if start_token not in text and end_token in text:
            return text.replace(end_token, "")
        return text

    def extract_reasoning(
        self,
        model_output: str,
        request: Any,
    ) -> tuple[str | None, str | None]:
        if not self._thinking_enabled():
            return None, model_output

        start_token = self._parser.start_token
        end_token = self._parser.end_token

        if start_token in model_output:
            _, _, model_output = model_output.partition(start_token)

        if end_token in model_output:
            reasoning, _, content = model_output.partition(end_token)
            return reasoning or None, content or None

        return model_output or None, None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int],
        current_token_ids: list[int],
        delta_token_ids: list[int],
    ) -> DeltaMessage | None:
        if not self._thinking_enabled():
            return self._parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        if self._reasoning_ended:
            content = self._strip_stray_end_token(delta_text)
            return DeltaMessage(content=content) if content else None

        start_token = self._parser.start_token
        end_token = self._parser.end_token

        if end_token in delta_text:
            reasoning_part, _, content_part = delta_text.partition(end_token)
            if start_token in reasoning_part:
                _, _, reasoning_part = reasoning_part.partition(start_token)

            self._reasoning_ended = True
            return DeltaMessage(
                reasoning=reasoning_part or None,
                content=content_part or None,
            )

        reasoning_part = delta_text
        if start_token in reasoning_part:
            _, _, reasoning_part = reasoning_part.partition(start_token)
        return DeltaMessage(reasoning=reasoning_part or None) if reasoning_part else None
