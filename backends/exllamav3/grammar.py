from typing import List, Optional
import traceback

from exllamav3 import (
    Tokenizer,
    Filter,
    LLGuidanceFilter,
)
from backends.exllamav3.whitespace_guard import (
    WhitespaceStallGuardFilter,
    build_whitespace_bitmask,
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

    def __init__(self, stall_tokens: Optional[int] = 8):
        self.filters = []
        self.stall_tokens = stall_tokens
        self._whitespace_bitmask = None

    def _wrap_with_stall_guard(self, lmfilter: Filter, tokenizer: Tokenizer) -> Filter:
        """Wrap the filter with the whitespace stall guard, building the
        vocab-wide whitespace bitmask once per handler."""

        if not self.stall_tokens or self.stall_tokens < 1:
            return lmfilter
        if self._whitespace_bitmask is None:
            try:
                self._whitespace_bitmask = build_whitespace_bitmask(tokenizer)
            except Exception as exc:
                xlogger.warning(
                    "Whitespace stall guard unavailable; constrained generation "
                    "may stall in legal whitespace runs.",
                    {"exception": str(exc)},
                )
                self._whitespace_bitmask = False
        if self._whitespace_bitmask is False:
            return lmfilter
        return WhitespaceStallGuardFilter(
            lmfilter, self._whitespace_bitmask, stall_tokens=self.stall_tokens
        )

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

        self.filters.append(self._wrap_with_stall_guard(lmfilter, tokenizer))

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

        self.filters.append(self._wrap_with_stall_guard(lmfilter, tokenizer))

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

        self.filters.append(self._wrap_with_stall_guard(lmfilter, tokenizer))
