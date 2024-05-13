import traceback
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2Sampler
from exllamav2.generator.filters import ExLlamaV2Filter, ExLlamaV2PrefixFilter
from loguru import logger


class OutlinesTokenizerWrapper:
    """Wrapper for Outlines tokenizer"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        self.vocabulary = {piece: idx for idx, piece in enumerate(id_to_piece)}
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = id_to_piece[self.tokenizer.eos_token_id]
        self.special_tokens = list(self.tokenizer.extended_id_to_piece.keys())

    def convert_token_to_string(self, token):
        return token

    def decode(self, tokens):
        s = ""
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        for t in tokens:
            s += id_to_piece[t]
        return s


class ExLlamaV2EbnfFilter(ExLlamaV2Filter):
    """Filter class for context-free grammar via outlines"""

    def __init__(self, model, tokenizer, grammar):
        from outlines.fsm.fsm import CFGFSM

        super().__init__(model, tokenizer)

        self.wrapped_tokenizer = OutlinesTokenizerWrapper(tokenizer)
        self.fsm = CFGFSM(grammar, self.wrapped_tokenizer)
        self.state = self.fsm.first_state

    def begin(self, prefix_str=""):
        self.state = self.fsm.first_state

    def feed(self, token):
        self.state = self.fsm.next_state(self.state, token.item())

    def next(self):
        return self.fsm.allowed_token_ids(self.state), set()


class ExLlamaV2Grammar:
    """ExLlamaV2 class for various grammar filters/parsers."""

    def add_json_schema_filter(
        self,
        json_schema: dict,
        gen_settings: ExLlamaV2Sampler.Settings,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        """Adds an ExllamaV2 filter based on a JSON schema."""

        # Import optional dependencies
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.exllamav2 import (
                ExLlamaV2TokenEnforcerFilter,
            )
        except ImportError:
            logger.error(
                "Skipping JSON schema parsing because "
                "lm-format-enforcer is not installed.\n"
                "Please run the following command in your environment "
                "to reinstall dependencies:\n"
                "pip install -U ."
            )

            return

        # Create the parser
        try:
            schema_parser = JsonSchemaParser(json_schema)
        except Exception:
            traceback.print_exc()
            logger.error(
                "Skipping because the JSON schema couldn't be parsed. "
                "Please read the above error for more information."
            )

            return

        lmfilter = ExLlamaV2TokenEnforcerFilter(schema_parser, tokenizer)
        prefix_filter = ExLlamaV2PrefixFilter(model, tokenizer, "{")

        # Append the filters
        gen_settings.filters.extend([lmfilter, prefix_filter])
        gen_settings.filter_prefer_eos = True

    def add_regex_filter(
        self,
        pattern: str,
        gen_settings: ExLlamaV2Sampler.Settings,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        """Adds an ExllamaV2 filter based on regular expressions."""

        # Import optional dependencies
        try:
            from lmformatenforcer import RegexParser
            from lmformatenforcer.integrations.exllamav2 import (
                ExLlamaV2TokenEnforcerFilter,
            )
        except ImportError:
            logger.error(
                "Skipping regex parsing because "
                "lm-format-enforcer is not installed.\n"
                "Please run the following command in your environment "
                "to reinstall dependencies:\n"
                "pip install -U ."
            )

            return

        # Create the parser
        try:
            pattern_parser = RegexParser(pattern)
        except Exception:
            traceback.print_exc()
            logger.error(
                "Skipping because the regex pattern couldn't be parsed. "
                "Please read the above error for more information."
            )

            return

        lmfilter = ExLlamaV2TokenEnforcerFilter(pattern_parser, tokenizer)

        # Append the filters
        gen_settings.filters.extend([lmfilter])
        gen_settings.filter_prefer_eos = True

    def add_ebnf_filter(
        self,
        ebnf_string: str,
        gen_settings: ExLlamaV2Sampler.Settings,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        """
        Add an EBNF grammar filter.
        Possibly replace outlines with an in-house solution in the future.
        """

        try:
            ebnf_filter = ExLlamaV2EbnfFilter(model, tokenizer, ebnf_string)
        except ImportError:
            logger.error(
                "Skipping EBNF parsing because Outlines is not installed.\n"
                "Please run the following command in your environment "
                "to install extra packages:\n"
                "pip install -U .[extras]"
            )

            return

        gen_settings.filters.append(ebnf_filter)
        gen_settings.filter_prefer_eos = True
