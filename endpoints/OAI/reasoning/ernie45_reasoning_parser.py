from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

from endpoints.OAI.reasoning.abs_reasoning_parsers import DeltaMessage
from endpoints.OAI.reasoning.basic_parsers import BaseThinkingReasoningParser


class Ernie45ReasoningParser(BaseThinkingReasoningParser):
    response_start_token: str = "<response>"
    response_end_token: str = "</response>"
    newline_token: str = "<0x0A>"

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.response_start_token_id = self.vocab.get(self.response_start_token)
        self.response_end_token_id = self.vocab.get(self.response_end_token)
        self.newline_token_id = self.vocab.get(self.newline_token)
        self.parser_token_ids = [self.end_token_id, self.response_end_token_id]

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0]
            in [
                self.start_token_id,
                self.end_token_id,
                self.response_start_token_id,
                self.response_end_token_id,
            ]
        ):
            return None

        if self.end_token_id in delta_token_ids:
            think_end_index = delta_text.find(self.end_token)
            reasoning = delta_text[:think_end_index]
            content = delta_text[think_end_index + len(self.end_token) :].lstrip("\n")
            response_start_idx = content.find(self.response_start_token)
            response_end_idx = content.rfind(self.response_end_token)
            if response_start_idx != -1:
                content = content[response_start_idx + len(self.response_start_token) :]
            if response_end_idx != -1:
                content = content[:response_end_idx]
            return DeltaMessage(reasoning=reasoning, content=content or None)

        if self.end_token_id in previous_token_ids:
            content = delta_text
            if self.response_start_token_id in delta_token_ids:
                content = content.lstrip("\n")
                response_start_idx = content.find(self.response_start_token)
                content = content[response_start_idx + len(self.response_start_token) :]
                response_end_idx = content.rfind(self.response_end_token)
                if response_end_idx != -1:
                    content = content[:response_end_idx]
            elif self.response_end_token_id in delta_token_ids:
                response_end_idx = content.rfind(self.response_end_token)
                content = content[:response_end_idx]

            if previous_token_ids and previous_token_ids[-1] in self.parser_token_ids:
                if delta_token_ids and delta_token_ids[0] == self.newline_token_id:
                    content = content.lstrip("\n")
            if len(previous_token_ids) > 1 and previous_token_ids[-2] == self.end_token_id:
                if delta_token_ids and delta_token_ids[0] == self.newline_token_id:
                    content = content.lstrip("\n")

            return DeltaMessage(content=content or None)

        return DeltaMessage(reasoning=delta_text)

    def extract_reasoning(
        self, model_output: str, request: Any
    ) -> tuple[str | None, str | None]:
        reasoning, content = super().extract_reasoning(model_output, request)
        if content:
            start_idx = content.find(self.response_start_token)
            end_idx = content.rfind(self.response_end_token)
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                content = content[start_idx + len(self.response_start_token) : end_idx]
        return reasoning, content or None
