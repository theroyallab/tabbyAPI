import typing
from typing import List, Optional
import traceback

from exllamav3 import (
    Tokenizer,
    Filter,
    FormatronFilter,
)
from formatron.extractor import NonterminalExtractor
from formatron.formatter import FormatterBuilder
from formatron.schemas import json_schema
from common.logger import xlogger


class CFGExtractor(NonterminalExtractor):
    """Extractor class for KBNF context-free grammar"""

    def __init__(self, nonterminal: str, kbnf_string: str):
        super().__init__(nonterminal)
        self.kbnf_string = kbnf_string

    # Return the entire input string as the extracted string
    def extract(self, input_str: str) -> typing.Optional[tuple[str, typing.Any]]:
        return "", input_str

    @property
    def kbnf_definition(self) -> str:
        return self.kbnf_string.replace("start", self.nonterminal)


class ExLlamaV3Grammar:
    """ExLlamaV3 class for various grammar filters/parsers."""

    filters: List[Filter]

    def __init__(self):
        self.filters = []

    def add_json_schema_filter(
        self,
        schema: dict,
        tokenizer: Tokenizer,
        trigger_token_id: int = None,
    ):
        """Adds an ExllamaV3 filter based on a JSON schema."""

        leading_character = "[" if schema.get("type") == "array" else "{"

        try:
            # Get named schema nested in from OAI response format config
            if "schema" in schema and "name" in schema:
                schema = schema["schema"]

            # Add fields required by formatron if not present
            if "$id" not in schema:
                schema["$id"] = "https://example.com/example.json"
            if "$schema" not in schema:
                schema["$schema"] = "http://json-schema.org/draft-07/schema#"

            # Validate schema and create formatter
            schema = json_schema.create_schema(schema)

        except Exception:
            traceback.print_exc()
            xlogger.error(
                "Skipping because the JSON schema couldn't be parsed. "
                "Please read the above error for more information.",
                {"schema": schema, "exception": traceback.format_exc()},
            )
            return

        f = FormatterBuilder()
        f.append_line(f"{f.json(schema)}")
        self.filters.append(
            FormatronFilter(
                tokenizer,
                eos_after_completed=True,
                formatter_builder=f,
                trigger_token=trigger_token_id,
            )
        )

        # Additional constraint to force leading character
        f = FormatterBuilder()
        f.append_line(leading_character)
        self.filters.append(
            FormatronFilter(
                tokenizer,
                formatter_builder=f,
                trigger_token=trigger_token_id,
            )
        )

    def add_regex_filter(
        self,
        pattern: str,
        tokenizer: Tokenizer,
        trigger_token_id: Optional[int] = None,
    ):
        """Adds an ExllamaV3 filter based on a regular expression."""

        try:
            # Validate regex, create formatter and build the filter
            f = FormatterBuilder()
            f.append_line(f"{f.regex(pattern)}")

            lmfilter = FormatronFilter(
                tokenizer,
                eos_after_completed=True,
                formatter_builder=f,
                trigger_token=trigger_token_id,
            )
        except Exception:
            traceback.print_exc()
            xlogger.error(
                "Skipping because the regex pattern couldn't be parsed. "
                "Please read the above error for more information.",
                {"pattern": pattern, "exception": traceback.format_exc()},
            )
            return

        self.filters.append(lmfilter)

    def add_kbnf_filter(
        self,
        kbnf_string: str,
        tokenizer: Tokenizer,
        trigger_token_id: Optional[int] = None,
    ):
        """Adds an ExllamaV3 filter based on KBNF grammar."""

        try:
            # Validate KBNF, create formatter and build the filter
            f = FormatterBuilder()
            f.append_line(
                f"""{f.extractor(lambda nonterminal: CFGExtractor(nonterminal, kbnf_string))}"""
            )

            lmfilter = FormatronFilter(
                tokenizer,
                eos_after_completed=True,
                formatter_builder=f,
                trigger_token=trigger_token_id,
            )
        except Exception:
            traceback.print_exc()
            xlogger.error(
                "Skipping because the KBNF string couldn't be parsed. "
                "Please read the above error for more information.",
                {"kbnf_string": kbnf_string, "exception": traceback.format_exc()},
            )
            return

        self.filters.append(lmfilter)
