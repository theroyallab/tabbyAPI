from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any

from endpoints.OAI.reasoning.abs_reasoning_parsers import DeltaMessage, ReasoningParser


NO_FUNC_REASONING_TAG = {
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "tags": [
            {
                "begin": "<|channel|>analysis<|message|>",
                "content": {"type": "any_text"},
                "end": "<|end|>",
            }
        ],
        "triggers": ["<|channel|>analysis"],
        "stop_after_first": False,
    },
}


class GptOssReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

    def _split_harmony(self, text: str) -> tuple[str | None, str | None]:
        # Minimal harmony-compatible splitter without vLLM parser dependency.
        analysis_tag = "<|channel|>analysis<|message|>"
        final_tag = "<|channel|>final<|message|>"
        end_tag = "<|end|>"

        a_idx = text.find(analysis_tag)
        f_idx = text.find(final_tag)
        if a_idx == -1 and f_idx == -1:
            return None, text or None

        reasoning = None
        content = None

        if a_idx != -1:
            a_start = a_idx + len(analysis_tag)
            a_end = text.find(end_tag, a_start)
            if a_end == -1:
                a_end = f_idx if f_idx != -1 else len(text)
            reasoning = text[a_start:a_end] or None

        if f_idx != -1:
            f_start = f_idx + len(final_tag)
            f_end = text.find(end_tag, f_start)
            if f_end == -1:
                f_end = len(text)
            content = text[f_start:f_end] or None

        return reasoning, content

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        text = self.model_tokenizer.decode(input_ids)
        return "<|channel|>final<|message|>" in text

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        _, content = self._split_harmony(self.model_tokenizer.decode(input_ids))
        if content is None:
            return []
        return self.model_tokenizer.encode(content)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        prev_reasoning, prev_content = self._split_harmony(previous_text)
        cur_reasoning, cur_content = self._split_harmony(current_text)

        reasoning_delta = None
        content_delta = None
        if cur_reasoning is not None:
            prev_r = prev_reasoning or ""
            reasoning_delta = (
                cur_reasoning[len(prev_r) :]
                if cur_reasoning.startswith(prev_r)
                else cur_reasoning
            ) or None
        if cur_content is not None:
            prev_c = prev_content or ""
            content_delta = (
                cur_content[len(prev_c) :]
                if cur_content.startswith(prev_c)
                else cur_content
            ) or None

        if reasoning_delta is None and content_delta is None:
            return None
        return DeltaMessage(reasoning=reasoning_delta, content=content_delta)

    def extract_reasoning(
        self,
        model_output: str,
        request: Any,
    ) -> tuple[str | None, str | None]:
        return self._split_harmony(model_output)

    def prepare_structured_tag(
        self, original_tag: str | None, tool_server: Any | None
    ) -> str | None:
        if original_tag is not None:
            return original_tag
        return json.dumps(NO_FUNC_REASONING_TAG)
