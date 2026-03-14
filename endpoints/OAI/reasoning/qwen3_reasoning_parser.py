from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from endpoints.OAI.reasoning.basic_parsers import BaseThinkingReasoningParser


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning(
        self, model_output: str, request: Any
    ) -> tuple[str | None, str | None]:
        if self.start_token not in model_output or self.end_token not in model_output:
            return None, model_output

        _, _, tail = model_output.partition(self.start_token)
        reasoning, _, content = tail.partition(self.end_token)
        return reasoning or None, content or None
