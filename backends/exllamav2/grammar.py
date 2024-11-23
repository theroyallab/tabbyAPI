import traceback
import typing
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.generator.filters import ExLlamaV2Filter
from loguru import logger
from typing import List

from formatron.formatter import FormatterBuilder
from formatron.schemas import json_schema
from formatron.integrations.exllamav2 import create_formatter_filter
from formatron.extractor import NonterminalExtractor


def clear_grammar_func_cache():
    """Flush tokenizer_data cache to avoid holding references to
    tokenizers after unloading a model"""

    # TODO: Unsure if this is needed with formatron
    pass


class ExLlamaV2Grammar:
    """ExLlamaV2 class for various grammar filters/parsers."""

    filters: List[ExLlamaV2Filter]

    def __init__(self):
        self.filters = []

    def add_json_schema_filter(
        self,
        schema: dict,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        """Adds an ExllamaV2 filter based on a JSON schema."""

        # Create the parser
        try:
            # Add fields required by formatron if not present
            if "$id" not in schema:
                schema["$id"] = "https://example.com/example.json"
            if "$schema" not in schema:
                schema["$schema"] = "http://json-schema.org/draft-07/schema#"

            # Validate schema and create formatter
            schema = json_schema.create_schema(schema)
            f = FormatterBuilder()
            f.append_line(f"{f.json(schema)}")
        except Exception:
            traceback.print_exc()
            logger.error(
                "Skipping because the JSON schema couldn't be parsed. "
                "Please read the above error for more information."
            )

            return

        lmfilter = create_formatter_filter(model, tokenizer, f)

        # Append the filters
        self.filters.append(lmfilter)

    def add_regex_filter(
        self,
        pattern: str,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        """Adds an ExllamaV2 filter based on regular expressions."""

        # Create the parser
        try:
            # Validate regex and create formatter
            f = FormatterBuilder()
            f.append_line(f"{f.regex(pattern)}")
        except Exception:
            traceback.print_exc()
            logger.error(
                "Skipping because the regex pattern couldn't be parsed. "
                "Please read the above error for more information."
            )

            return

        lmfilter = create_formatter_filter(model, tokenizer, f)

        # Append the filters
        self.filters.append(lmfilter)

    def add_kbnf_filter(
        self,
        kbnf_string: str,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        """Adds an ExllamaV2 filter based on KBNF grammar."""

        # Create the parser
        try:
            # Validate KBNF and create formatter
            f = FormatterBuilder()
            f.append_line(
                f"{f.extractor(
                    lambda nonterminal: CustomExtractor(nonterminal, kbnf_string)
                    )}"
            )
        except Exception:
            logger.error(
                "Skipping because the KBNF string couldn't be parsed. "
                "Please read the above error for more information."
            )

            return

        lmfilter = create_formatter_filter(model, tokenizer, f)

        # Append the filters
        self.filters.append(lmfilter)


class CustomExtractor(NonterminalExtractor):
    def __init__(self, nonterminal: str, kbnf_string: str):
        super().__init__(nonterminal)
        self.kbnf_string = kbnf_string

    # Fails without an extract function defined
    # No idea what it does or why it's needed, but this seems to work
    # TODO: Figure out how to do this properly
    def extract(self, input_str: str) -> typing.Optional[tuple[str, typing.Any]]:
        return input_str[len(input_str) :], input_str[: len(input_str)]

    @property
    def kbnf_definition(self) -> str:
        return self.kbnf_string.replace("start", self.nonterminal)
