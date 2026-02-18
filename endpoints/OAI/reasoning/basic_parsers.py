from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from typing import Any

from endpoints.OAI.reasoning.abs_reasoning_parsers import DeltaMessage, ReasoningParser


class BaseThinkingReasoningParser(ReasoningParser):
    @property
    @abstractmethod
    def start_token(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def end_token(self) -> str:
        raise NotImplementedError

    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.start_token_id = self.vocab.get(self.start_token)
        self.end_token_id = self.vocab.get(self.end_token)
        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError(
                f"{self.__class__.__name__} could not locate think tokens in tokenizer"
            )

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        for token_id in reversed(input_ids):
            if token_id == self.start_token_id:
                return False
            if token_id == self.end_token_id:
                return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.end_token_id not in input_ids[:-1]:
            return []
        return input_ids[input_ids.index(self.end_token_id) + 1 :]

    def extract_reasoning(
        self,
        model_output: str,
        request: Any,
    ) -> tuple[str | None, str | None]:
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.end_token not in model_output:
            return model_output or None, None

        reasoning, _, content = model_output.partition(self.end_token)
        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int],
        current_token_ids: list[int],
        delta_token_ids: list[int],
    ) -> DeltaMessage | None:
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0] in [self.start_token_id, self.end_token_id]
        ):
            return None

        if self.start_token_id in previous_token_ids:
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index] or None
                content = delta_text[end_index + len(self.end_token) :] or None
                return DeltaMessage(reasoning=reasoning, content=content)
            if self.end_token_id in previous_token_ids:
                return DeltaMessage(content=delta_text or None)
            return DeltaMessage(reasoning=delta_text or None)

        if self.start_token_id in delta_token_ids:
            if self.end_token_id in delta_token_ids:
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[start_index + len(self.start_token) : end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(reasoning=reasoning or None, content=content or None)
            return DeltaMessage(reasoning=delta_text or None)

        return DeltaMessage(content=delta_text or None)
