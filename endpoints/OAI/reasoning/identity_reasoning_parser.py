from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from endpoints.OAI.reasoning.abs_reasoning_parsers import DeltaMessage, ReasoningParser


class IdentityReasoningParser(ReasoningParser):
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int],
        current_token_ids: list[int],
        delta_token_ids: list[int],
    ) -> DeltaMessage | None:
        if not delta_text:
            return None
        return DeltaMessage(content=delta_text)

    def extract_reasoning(
        self,
        model_output: str,
        request: Any,
    ) -> tuple[str | None, str | None]:
        return None, model_output
