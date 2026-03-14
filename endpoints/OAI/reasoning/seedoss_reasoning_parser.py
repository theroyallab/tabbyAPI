from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from endpoints.OAI.reasoning.basic_parsers import BaseThinkingReasoningParser


class SeedOSSReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<seed:think>"

    @property
    def end_token(self) -> str:
        return "</seed:think>"
