from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

from endpoints.OAI.reasoning.basic_parsers import BaseThinkingReasoningParser


class MistralReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "[THINK]"

    @property
    def end_token(self) -> str:
        return "[/THINK]"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        has_eot = False
        for token_id in reversed(input_ids):
            if token_id == self.start_token_id:
                return has_eot
            if token_id == self.end_token_id:
                has_eot = True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        has_bot = False
        has_eot = False
        bot_idx = -1
        eot_idx = -1
        for i, token_id in enumerate(input_ids):
            if token_id == self.start_token_id and not has_bot:
                has_bot = True
                bot_idx = i
            elif token_id == self.end_token_id:
                has_eot = True
                eot_idx = i
                break

        if has_bot and not has_eot:
            return input_ids[:bot_idx]
        if not has_bot and not has_eot:
            return input_ids
        if has_bot and has_eot:
            return input_ids[:bot_idx] + input_ids[eot_idx + 1 :]
        return input_ids[:eot_idx] + input_ids[eot_idx + 1 :]

    def extract_reasoning(
        self, model_output: str, request: Any
    ) -> tuple[str | None, str | None]:
        if not model_output:
            return None, ""

        prefix, bot, post_bot = model_output.partition(self.start_token)
        has_bot = bool(bot)
        has_valid_eot = has_bot and self.end_token in post_bot

        if has_bot and has_valid_eot:
            reasoning, _, post_eot = post_bot.partition(self.end_token)
            content = prefix + post_eot
            return reasoning or None, content or None
        if has_bot:
            return post_bot or None, prefix or None

        if self.end_token in prefix:
            pre_eot, _, post_eot = prefix.partition(self.end_token)
            return None, (pre_eot + post_eot) or None

        return None, prefix
