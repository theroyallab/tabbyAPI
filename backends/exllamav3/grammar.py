from typing import List, Optional
import traceback

from exllamav3 import (
    Tokenizer,
    Filter,
    LLGuidanceFilter,
)
from common.logger import xlogger


class ExLlamaV3Grammar:
    """ExLlamaV3 class for various grammar filters/parsers."""

    filters: List[Filter]

    def __init__(self):
        self.filters = []

    def add_json_schema_filter(
        self,
        schema: dict,
        tokenizer: Tokenizer,
        trigger_token_id: Optional[int] = None,
    ):
        """Adds an ExllamaV3 filter based on a JSON schema."""

        # Unwrap a named schema nested in an OAI response format config
        if "schema" in schema and "name" in schema:
            schema = schema["schema"]

        try:
            lmfilter = LLGuidanceFilter(
                tokenizer,
                eos_after_completed=True,
                json_schema=schema,
                trigger_token=trigger_token_id,
            )
        except Exception:
            traceback.print_exc()
            xlogger.error(
                "Skipping because the JSON schema couldn't be parsed. "
                "Please read the above error for more information.",
                {"schema": schema, "exception": traceback.format_exc()},
            )
            return

        self.filters.append(lmfilter)

    def add_regex_filter(
        self,
        pattern: str,
        tokenizer: Tokenizer,
        trigger_token_id: Optional[int] = None,
    ):
        """Adds an ExllamaV3 filter based on a regular expression."""

        try:
            lmfilter = LLGuidanceFilter(
                tokenizer,
                eos_after_completed=True,
                regex=pattern,
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

    def add_grammar_filter(
        self,
        grammar_string: str,
        tokenizer: Tokenizer,
        trigger_token_id: Optional[int] = None,
    ):
        """Adds an ExllamaV3 filter based on a context-free grammar.

        Accepts Lark syntax or llama.cpp GBNF, distinguished by the rule
        definition operator (GBNF uses `::=`).
        """

        grammar_kind = "gbnf_grammar" if "::=" in grammar_string else "lark_grammar"

        try:
            lmfilter = LLGuidanceFilter(
                tokenizer,
                eos_after_completed=True,
                trigger_token=trigger_token_id,
                **{grammar_kind: grammar_string},
            )
        except Exception:
            traceback.print_exc()
            xlogger.error(
                "Skipping because the grammar couldn't be parsed. "
                "Please read the above error for more information.",
                {"grammar_string": grammar_string, "exception": traceback.format_exc()},
            )
            return

        self.filters.append(lmfilter)
