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


# llguidance compile options applied to every JSON schema unless the schema
# brings its own "x-guidance" block. Optional whitespace is disabled: a grammar
# can forbid tokens but not compel progress, and with unlimited whitespace legal
# between JSON tokens a model that wants to stop can idle on whitespace until
# max_tokens without ever satisfying the schema. Without it the only legal
# continuations are the next JSON token or the literal ", " / ": " separators,
# so the output is compact and a stall is impossible. Clients that need
# pretty-printed output can send {"x-guidance": {"whitespace_flexible": true}}.
JSON_SCHEMA_GUIDANCE_DEFAULTS = {"whitespace_flexible": False}


def prepare_json_schema(schema):
    """Unwrap an OAI named schema and apply the default llguidance options."""

    # Unwrap a named schema nested in an OAI response format config
    if isinstance(schema, dict) and "schema" in schema and "name" in schema:
        schema = schema["schema"]

    if isinstance(schema, dict) and "x-guidance" not in schema:
        schema = {**schema, "x-guidance": dict(JSON_SCHEMA_GUIDANCE_DEFAULTS)}

    return schema


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

        schema = prepare_json_schema(schema)

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
