from typing import List, Optional
import traceback

from exllamav3 import (
    Tokenizer,
    Filter,
    LLGuidanceFilter,
)
from common.errors import GrammarParseError
from common.logger import xlogger


def _llguidance_ready() -> bool:
    """Whether the llguidance backend itself is importable and present.

    A missing backend is a server environment problem and keeps its own
    exception identity (server error), while a schema/regex/grammar the
    backend rejects is a client error.
    """

    try:
        from exllamav3.generator.filter.llguidance import llguidance_available
    except ImportError:
        # Flag moved in a future exllamav3; let construction errors decide.
        return True
    return llguidance_available


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
        except Exception as exc:
            xlogger.error(
                "JSON schema could not be compiled; rejecting the request instead "
                "of generating without the constraint.",
                {"schema": schema, "exception": traceback.format_exc()},
            )
            if not _llguidance_ready():
                raise
            raise GrammarParseError(f"The JSON schema could not be compiled: {exc}") from exc

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
        except Exception as exc:
            xlogger.error(
                "Regex pattern could not be compiled; rejecting the request instead "
                "of generating without the constraint.",
                {"pattern": pattern, "exception": traceback.format_exc()},
            )
            if not _llguidance_ready():
                raise
            raise GrammarParseError(f"The regex pattern could not be compiled: {exc}") from exc

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
        except Exception as exc:
            xlogger.error(
                f"{grammar_kind} could not be compiled; rejecting the request instead "
                "of generating without the constraint.",
                {"grammar_string": grammar_string, "exception": traceback.format_exc()},
            )
            if not _llguidance_ready():
                raise
            raise GrammarParseError(
                f"The grammar ({grammar_kind}) could not be compiled: {exc}"
            ) from exc

        self.filters.append(lmfilter)
