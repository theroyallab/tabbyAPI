from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

from endpoints.OAI.reasoning.abs_reasoning_parsers import (
    DeltaMessage,
    ReasoningParser,
    ReasoningParserManager,
)


@ReasoningParserManager.register_module("exaone4")
class Exaone4ReasoningParser(ReasoningParser):
    """
    Reasoning parser for EXAONE 4.x models.

    Behavior notes:
    - EXAONE uses `enable_thinking` (not `thinking`) to control reasoning mode.
    - Templates may prefill `<think>`, so streamed/output text can start directly
      with reasoning text and close at `</think>`.
    """

    start_token = "<think>"
    end_token = "</think>"

    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = bool(chat_kwargs.get("enable_thinking", False))
        self.start_token_id = self.vocab.get(self.start_token)
        self.end_token_id = self.vocab.get(self.end_token)

    def _contains_token(
        self, token_id: int | None, token_text: str, token_ids: Sequence[int], text: str
    ) -> bool:
        if token_id is not None and token_id in token_ids:
            return True
        return token_text in text if text else False

    def _strip_reasoning_tokens(self, text: str) -> str:
        if not text:
            return ""
        return text.replace(self.start_token, "").replace(self.end_token, "")

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if not self.thinking_enabled:
            return True
        if self.end_token_id is None:
            return False
        return any(token_id == self.end_token_id for token_id in reversed(input_ids))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking_enabled:
            return input_ids
        if self.end_token_id is None or self.end_token_id not in input_ids[:-1]:
            return []
        return input_ids[input_ids.index(self.end_token_id) + 1 :]

    def extract_reasoning(
        self,
        model_output: str,
        request: Any,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            content = self._strip_reasoning_tokens(model_output)
            return None, content or None

        if self.start_token in model_output:
            _, _, model_output = model_output.partition(self.start_token)

        if self.end_token in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            content = self._strip_reasoning_tokens(content)
            return reasoning or None, content or None

        reasoning = model_output.replace(self.start_token, "")
        return reasoning or None, None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int],
        current_token_ids: list[int],
        delta_token_ids: list[int],
    ) -> DeltaMessage | None:
        if not delta_text and not delta_token_ids:
            return None

        if not self.thinking_enabled:
            content = self._strip_reasoning_tokens(delta_text)
            return DeltaMessage(content=content or None) if content else None

        end_in_prev = self._contains_token(
            self.end_token_id, self.end_token, previous_token_ids, previous_text
        )
        end_in_delta = self._contains_token(
            self.end_token_id, self.end_token, delta_token_ids, delta_text
        )

        if len(delta_token_ids) == 1 and (
            (self.start_token_id is not None and delta_token_ids[0] == self.start_token_id)
            or (self.end_token_id is not None and delta_token_ids[0] == self.end_token_id)
        ):
            return None

        if end_in_prev:
            content = self._strip_reasoning_tokens(delta_text)
            return DeltaMessage(content=content or None) if content else None

        if end_in_delta:
            reasoning_part, _, content_part = delta_text.partition(self.end_token)
            if self.start_token in reasoning_part:
                _, _, reasoning_part = reasoning_part.partition(self.start_token)

            reasoning = reasoning_part or None
            content = self._strip_reasoning_tokens(content_part) or None
            if reasoning is None and content is None:
                return None
            return DeltaMessage(reasoning=reasoning, content=content)

        # EXAONE can omit start token in stream when template prefills <think>.
        # While thinking mode is enabled, treat pre-end chunks as reasoning.
        reasoning_part = delta_text.replace(self.start_token, "")

        reasoning = reasoning_part or None
        return DeltaMessage(reasoning=reasoning) if reasoning else None
