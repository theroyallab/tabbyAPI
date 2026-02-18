from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class DeltaMessage:
    content: str | None = None
    reasoning: str | None = None


class ReasoningParser(ABC):
    def __init__(self, tokenizer: Any, *args, **kwargs):
        self.model_tokenizer = tokenizer

    @property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()

    @abstractmethod
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        pass

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        return self.is_reasoning_end(input_ids)

    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        pass

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
        request: Any,
    ) -> tuple[str | None, str | None]:
        pass

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        pass

    def prepare_structured_tag(self, original_tag: str | None, tool_server: Any | None):
        return original_tag


class ReasoningParserManager:
    reasoning_parsers: dict[str, type[ReasoningParser]] = {}

    @classmethod
    def list_registered(cls) -> list[str]:
        return sorted(cls.reasoning_parsers.keys())

    @classmethod
    def get_reasoning_parser(cls, name: str) -> type[ReasoningParser]:
        parser = cls.reasoning_parsers.get(name)
        if parser is None:
            registered = ", ".join(cls.list_registered())
            raise KeyError(
                f"Reasoning parser '{name}' not found. Available parsers: {registered}"
            )
        return parser

    @classmethod
    def register_module(
        cls,
        module_name: str,
    ) -> Callable[[type[ReasoningParser]], type[ReasoningParser]]:
        def _decorator(module: type[ReasoningParser]) -> type[ReasoningParser]:
            cls.reasoning_parsers[module_name] = module
            return module

        return _decorator
