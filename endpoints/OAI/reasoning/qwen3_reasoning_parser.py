from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

from endpoints.OAI.reasoning.abs_reasoning_parsers import DeltaMessage
from endpoints.OAI.reasoning.basic_parsers import BaseThinkingReasoningParser


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3/Qwen3.5 model family.

    Qwen3.5 chat templates prefill `<think>` in the prompt, so streaming output
    usually begins with reasoning text and only later emits `</think>`.
    This parser mirrors vLLM behavior for that path while also honoring
    `enable_thinking=False` by routing all deltas as normal content.
    """

    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        enable_thinking = chat_kwargs.get("enable_thinking")
        if enable_thinking is None:
            enable_thinking = chat_kwargs.get("thinking")

        # Only force "prefilled <think>" behavior when the template explicitly
        # exposes a thinking switch. Templates like Qwen3-Next's tokenizer
        # config do not, and should fall back to normal tag-based parsing.
        self.thinking_enabled = (
            None if enable_thinking is None else bool(enable_thinking)
        )

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def _strip_reasoning_tags(self, text: str) -> str:
        if not text:
            return ""
        return text.replace(self.start_token, "").replace(self.end_token, "")

    def extract_reasoning(
        self, model_output: str, request: Any
    ) -> tuple[str | None, str | None]:
        if self.thinking_enabled is None:
            if self.start_token not in model_output or self.end_token not in model_output:
                return None, model_output or None

            _, _, tail = model_output.partition(self.start_token)
            reasoning, _, content = tail.partition(self.end_token)
            return reasoning or None, content or None

        if not self.thinking_enabled:
            content = self._strip_reasoning_tags(model_output)
            return None, content or None

        # Qwen3.5 templates prefill <think> in the prompt.
        # If <think> appears in output (legacy templates), strip it.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.end_token not in model_output:
            return None, model_output

        reasoning, _, content = model_output.partition(self.end_token)
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
        if self.thinking_enabled is None:
            return super().extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        if not self.thinking_enabled:
            cleaned = self._strip_reasoning_tags(delta_text)
            if not cleaned:
                return None
            return DeltaMessage(content=cleaned)

        # Handle old templates where model may generate <think>.
        if self.start_token_id in delta_token_ids:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token) :]

        if self.end_token_id in delta_token_ids:
            end_index = delta_text.find(self.end_token)
            if end_index >= 0:
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                if not reasoning and not content:
                    return None
                return DeltaMessage(
                    reasoning=reasoning if reasoning else None,
                    content=content if content else None,
                )
            # End token id is present but text was already stripped by backend.
            return None

        if not delta_text:
            return None
        if self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)
        return DeltaMessage(reasoning=delta_text)
