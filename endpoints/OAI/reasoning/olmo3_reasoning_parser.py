from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses as dt
import enum
from collections.abc import Sequence
from typing import Any

try:
    import regex as re
except ImportError:  # pragma: no cover
    import re

from endpoints.OAI.reasoning.abs_reasoning_parsers import DeltaMessage, ReasoningParser


class Olmo3ReasoningState(enum.Enum):
    REASONING = 1
    CONTENT = 2


@dt.dataclass(frozen=True)
class Indices:
    start: int
    end: int

    def __len__(self):
        return self.end - self.start


def string_overlap(a: str, b: str) -> tuple[Indices | None, Indices | None]:
    a, b, swap = (a, b, False) if len(a) < len(b) else (b, a, True)

    if a in b:
        ind_a = Indices(0, len(a))
        ind_b = Indices(b.index(a), b.index(a) + len(a))
        return (ind_b, ind_a) if swap else (ind_a, ind_b)

    for i in range(len(a) - 1, 0, -1):
        if a[-i:] == b[:i]:
            ind_a = Indices(len(a) - i, len(a))
            ind_b = Indices(0, i)
            return (ind_b, ind_a) if swap else (ind_a, ind_b)

    for i in range(len(a) - 1, 0, -1):
        if b[-i:] == a[:i]:
            ind_a = Indices(0, i)
            ind_b = Indices(len(b) - i, len(b))
            return (ind_b, ind_a) if swap else (ind_a, ind_b)

    return None, None


@dt.dataclass
class Olmo3ReasoningBuffer:
    think_start: str = "<think>"
    think_end: str = "</think>"
    buffer: str = ""
    state: Olmo3ReasoningState = Olmo3ReasoningState.REASONING

    def process_buffer(self) -> DeltaMessage | None:
        start_think_idx = self.buffer.find(self.think_start)
        if start_think_idx >= 0:
            self.state = Olmo3ReasoningState.REASONING
            pretext, self.buffer = (
                self.buffer[:start_think_idx],
                self.buffer[start_think_idx + len(self.think_start) :],
            )
            if start_think_idx > 0:
                return DeltaMessage(content=pretext)

        end_think_idx = self.buffer.rfind(self.think_end)
        if end_think_idx >= 0:
            self.state = Olmo3ReasoningState.CONTENT
            pretext, self.buffer = (
                self.buffer[:end_think_idx],
                self.buffer[end_think_idx + len(self.think_end) :],
            )
            if end_think_idx > 0:
                return DeltaMessage(reasoning=pretext)

        if self.state == Olmo3ReasoningState.REASONING:
            text_buffer, self.buffer = self.buffer, ""
            return DeltaMessage(reasoning=text_buffer)

        if self.state == Olmo3ReasoningState.CONTENT:
            text_buffer, self.buffer = self.buffer, ""
            return DeltaMessage(content=text_buffer)

        return None

    def add_text(self, delta_text: str) -> DeltaMessage | None:
        self.buffer += delta_text
        delta_message: DeltaMessage | None = None

        _, overlap_think_start = string_overlap(delta_text, self.think_start)
        _, overlap_think_end = string_overlap(delta_text, self.think_end)

        partial_overlap_start = overlap_think_start is not None and len(
            overlap_think_start
        ) < len(self.think_start)
        partial_overlap_end = overlap_think_end is not None and len(overlap_think_end) < len(
            self.think_end
        )

        if partial_overlap_start and self.think_start in self.buffer and not partial_overlap_end:
            delta_message = self.process_buffer()
        elif partial_overlap_end and self.think_end in self.buffer:
            delta_message = self.process_buffer()
        elif partial_overlap_start or partial_overlap_end:
            return None
        else:
            delta_message = self.process_buffer()

        return delta_message


class Olmo3ReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: Any, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        self.think_start = r"<think>"
        self.think_end = r"</think>"

        reasoning_expr = (
            rf"^(?:{self.think_start})?(?P<reasoning>.*?)" + rf"{self.think_end}(?P<content>.*)$"
        )
        self.reasoning_regex = re.compile(reasoning_expr, re.DOTALL)
        self.buffer = Olmo3ReasoningBuffer(
            think_start=self.think_start, think_end=self.think_end
        )

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        text = self.model_tokenizer.decode(input_ids)
        return self.think_end in text

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return []

    def extract_reasoning(
        self,
        model_output: str,
        request: Any,
    ) -> tuple[str | None, str | None]:
        re_match = self.reasoning_regex.match(model_output)
        if re_match:
            reasoning = re_match.group("reasoning") or None
            content = re_match.group("content") or None
            return reasoning, content
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        delta_message = self.buffer.add_text(delta_text)
        if delta_message is None and self.buffer.think_end in self.buffer.buffer:
            delta_message = self.buffer.process_buffer()
        return delta_message
