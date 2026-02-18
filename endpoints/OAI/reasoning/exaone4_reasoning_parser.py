from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

from endpoints.OAI.reasoning.abs_reasoning_parsers import (
    DeltaMessage,
    ReasoningParser,
    ReasoningParserManager,
)


@ReasoningParserManager.register_module("exaone4")
class Exaone4ReasoningParser(ReasoningParser):
    """
    Reasoning parser for EXAONE 4.x models.

    Behavior notes:
    - EXAONE uses `enable_thinking` (not `thinking`) to control reasoning mode.
    - Templates may prefill `<think>`, so streamed/output text can start directly
      with reasoning text and close at `</think>`.
    """

    start_token = "<think>"
    end_token = "</think>"
    # Tool-call starts supported by ToolCallProcessor parser families.
    # We use these as fallback reasoning boundaries when a model emits
    # tool syntax without closing </think>.
    tool_start_markers = (
        "<tool_call>",
        "<function=",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>",
        "<｜DSML｜function_calls>",
        "<｜DSML｜invoke",
        "<|python_tag|>",
    )

    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = bool(chat_kwargs.get("enable_thinking", False))
        self.start_token_id = self.vocab.get(self.start_token)
        self.end_token_id = self.vocab.get(self.end_token)

    def _strip_reasoning_tokens(self, text: str) -> str:
        if not text:
            return ""
        return text.replace(self.start_token, "").replace(self.end_token, "")

    def _trailing_overlap_len(self, text: str, token: str) -> int:
        """Longest suffix overlap of text with token prefix."""
        max_len = min(len(text), len(token) - 1)
        for size in range(max_len, 0, -1):
            if text.endswith(token[:size]):
                return size
        return 0

    def _find_first_marker(self, text: str, markers: Sequence[str]) -> tuple[int, str] | None:
        first_idx = -1
        first_marker = ""
        for marker in markers:
            idx = text.find(marker)
            if idx == -1:
                continue
            if first_idx == -1 or idx < first_idx:
                first_idx = idx
                first_marker = marker
        if first_idx == -1:
            return None
        return first_idx, first_marker

    def _max_trailing_overlap_len(self, text: str, markers: Sequence[str]) -> int:
        overlap = 0
        for marker in markers:
            overlap = max(overlap, self._trailing_overlap_len(text, marker))
        return overlap

    def _split_reasoning_content_streaming(
        self, text: str
    ) -> tuple[str | None, str | None]:
        """Split text into reasoning/content for streaming-safe diffing.

        Important: when end token is not yet complete, withhold a trailing
        overlap with `</think>` or tool-call prefixes to avoid leaking
        partial control-tag bytes into reasoning output. This prevents
        boundary-split regressions such as `</thi` + `nk>answer` and
        `<tool_c` + `all>{...}`.
        """
        if not self.thinking_enabled:
            content = self._strip_reasoning_tokens(text)
            return None, content or None

        body = text
        if self.start_token in body:
            _, _, body = body.partition(self.start_token)

        if self.end_token in body:
            reasoning, _, content = body.partition(self.end_token)
            return reasoning or None, self._strip_reasoning_tokens(content) or None

        marker_match = self._find_first_marker(body, self.tool_start_markers)
        if marker_match is not None:
            marker_index, _ = marker_match
            reasoning = body[:marker_index]
            content = body[marker_index:]
            return reasoning or None, self._strip_reasoning_tokens(content) or None

        reasoning = body.replace(self.start_token, "")
        overlap = max(
            self._trailing_overlap_len(reasoning, self.end_token),
            self._max_trailing_overlap_len(reasoning, self.tool_start_markers),
        )
        if overlap:
            reasoning = reasoning[:-overlap]
        return reasoning or None, None

    def _delta_from_previous(self, previous: str | None, current: str | None) -> str | None:
        if current is None:
            return None
        previous_text = previous or ""
        if current.startswith(previous_text):
            delta = current[len(previous_text) :]
        else:
            # Fallback for recovery paths where prefix alignment breaks.
            delta = current
        return delta or None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if not self.thinking_enabled:
            return True
        if self.end_token_id is None:
            return False
        return any(token_id == self.end_token_id for token_id in reversed(input_ids))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking_enabled:
            return input_ids
        if self.end_token_id is None or self.end_token_id not in input_ids[:-1]:
            return []
        return input_ids[input_ids.index(self.end_token_id) + 1 :]

    def extract_reasoning(
        self,
        model_output: str,
        request: Any,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            content = self._strip_reasoning_tokens(model_output)
            return None, content or None

        if self.start_token in model_output:
            _, _, model_output = model_output.partition(self.start_token)

        if self.end_token in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            content = self._strip_reasoning_tokens(content)
            return reasoning or None, content or None

        reasoning = model_output.replace(self.start_token, "")
        return reasoning or None, None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int],
        current_token_ids: list[int],
        delta_token_ids: list[int],
    ) -> DeltaMessage | None:
        if not delta_text and not delta_token_ids:
            return None

        if not self.thinking_enabled:
            prev_reasoning, prev_content = self._split_reasoning_content_streaming(
                previous_text
            )
            cur_reasoning, cur_content = self._split_reasoning_content_streaming(
                current_text
            )
            content_delta = self._delta_from_previous(prev_content, cur_content)
            if content_delta is None:
                return None
            return DeltaMessage(content=content_delta)

        if len(delta_token_ids) == 1 and (
            (self.start_token_id is not None and delta_token_ids[0] == self.start_token_id)
            or (self.end_token_id is not None and delta_token_ids[0] == self.end_token_id)
        ):
            return None

        prev_reasoning, prev_content = self._split_reasoning_content_streaming(previous_text)
        cur_reasoning, cur_content = self._split_reasoning_content_streaming(current_text)

        reasoning_delta = self._delta_from_previous(prev_reasoning, cur_reasoning)
        content_delta = self._delta_from_previous(prev_content, cur_content)

        if reasoning_delta is None and content_delta is None:
            return None
        return DeltaMessage(reasoning=reasoning_delta, content=content_delta)
