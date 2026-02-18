from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

from endpoints.OAI.reasoning.abs_reasoning_parsers import DeltaMessage, ReasoningParser


class Step3ReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.think_end_token = "</think>"
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        if self.think_end_token_id is None:
            raise RuntimeError(
                "Step3 reasoning parser could not locate think end token in tokenizer"
            )

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.think_end_token_id:
            return None

        if self.think_end_token_id in delta_token_ids:
            end_index = delta_text.find(self.think_end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self.think_end_token) :]
            return DeltaMessage(reasoning=reasoning, content=content or None)

        if self.think_end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)

        return DeltaMessage(reasoning=delta_text)

    def extract_reasoning(
        self, model_output: str, request: Any
    ) -> tuple[str | None, str | None]:
        if self.think_end_token not in model_output:
            return model_output or None, None

        end_index = model_output.find(self.think_end_token)
        reasoning = model_output[:end_index]
        content = model_output[end_index + len(self.think_end_token) :]
        return reasoning or None, content or None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self.think_end_token_id in input_ids

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        return self.think_end_token_id in delta_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.think_end_token_id not in input_ids[:-1]:
            return []
        return input_ids[input_ids.index(self.think_end_token_id) + 1 :]
