from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

from endpoints.OAI.reasoning.abs_reasoning_parsers import DeltaMessage
from endpoints.OAI.reasoning.basic_parsers import BaseThinkingReasoningParser


class Step3p5ReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._pending_reasoning_newline = False
        self.end_offset = 1

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self.end_token_id in input_ids and self.end_offset > 0:
            self.end_offset -= 1
            return False
        return self.end_offset < 1

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        if self.end_token_id in input_ids and self.end_offset > 0:
            self.end_offset -= 1
            return False
        return self.end_offset < 1

    def extract_reasoning(
        self, model_output: str, request: Any
    ) -> tuple[str | None, str | None]:
        reasoning, content = super().extract_reasoning(model_output, request)
        if reasoning is not None:
            reasoning = reasoning.removesuffix("\n")
        if content is not None:
            content = content.removeprefix("\n")
        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if previous_text.endswith(self.end_token) and delta_text:
            if delta_text == "\n":
                return None
            if delta_text.startswith("\n"):
                remaining = delta_text.removeprefix("\n")
                return DeltaMessage(content=remaining) if remaining else None

        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if ret is None:
            return None

        if (
            self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                ret = DeltaMessage(reasoning=reasoning, content=content or None)
            elif self.end_token_id in previous_token_ids:
                ret = DeltaMessage(content=delta_text)
            else:
                ret = DeltaMessage(reasoning=delta_text)

        reasoning_to_output = ret.reasoning
        content_to_output = ret.content

        if reasoning_to_output is not None:
            if self._pending_reasoning_newline:
                reasoning_to_output = "\n" + reasoning_to_output
                self._pending_reasoning_newline = False

            if reasoning_to_output.endswith("\n"):
                reasoning_to_output = reasoning_to_output.removesuffix("\n")
                if self.end_token in delta_text:
                    self._pending_reasoning_newline = False
                else:
                    self._pending_reasoning_newline = True

        if content_to_output is not None:
            self.end_offset -= 1
            self._pending_reasoning_newline = False
            if self.end_token in delta_text and content_to_output.startswith("\n"):
                content_to_output = content_to_output.removeprefix("\n")

        reasoning_to_output = reasoning_to_output or None
        content_to_output = content_to_output or None
        if reasoning_to_output is None and content_to_output is None:
            return None

        return DeltaMessage(reasoning=reasoning_to_output, content=content_to_output)
