from common.logger import init_logger
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2Sampler

# Temporary, remove once the exllama version is bumped
try:
    from exllamav2.generator.filters import ExLlamaV2PrefixFilter

    _exllama_filter_available = True
except ImportError:
    _exllama_filter_available = False

try:
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter

    _lmformatenforcer_available = True
except ImportError:
    _lmformatenforcer_available = False


logger = init_logger(__name__)


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

        # Check if the required dependencies can be imported
        if not _exllama_filter_available:
            logger.warning(
                "ExllamaV2PrefixFilter is not available "
                "in the currently installed ExllamaV2 version."
            )

            return

        if not _lmformatenforcer_available:
            logger.error(
                "lmformatenforcer must be installed to parse a json schema.\n"
                "Please run the following command: pip install lm-format-enforcer"
            )

            return

        # Create the parser
        schema_parser = JsonSchemaParser(json_schema)
        lmfilter = ExLlamaV2TokenEnforcerFilter(schema_parser, tokenizer)
        prefix_filter = ExLlamaV2PrefixFilter(model, tokenizer, "{")

        # Append the filters
        gen_settings.filters += [lmfilter, prefix_filter]
        gen_settings.filter_prefer_eos = True
