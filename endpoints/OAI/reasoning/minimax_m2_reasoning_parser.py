from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

from endpoints.OAI.reasoning.abs_reasoning_parsers import DeltaMessage, ReasoningParser
from endpoints.OAI.reasoning.basic_parsers import BaseThinkingReasoningParser


class MiniMaxM2ReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.end_token_id:
            return None

        if self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)

        if self.end_token_id in delta_token_ids:
            end_index = delta_text.find(self.end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self.end_token) :]
            return DeltaMessage(reasoning=reasoning or None, content=content or None)

        return DeltaMessage(reasoning=delta_text)


class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.end_token_id = self.vocab.get("</think>")

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return any(input_id == self.end_token_id for input_id in reversed(input_ids))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if len(previous_token_ids) == 0:
            delta_text = "<think>" + delta_text
        return DeltaMessage(content=delta_text)

    def extract_reasoning(
        self, model_output: str, request: Any
    ) -> tuple[str | None, str | None]:
        return None, "<think>" + model_output
