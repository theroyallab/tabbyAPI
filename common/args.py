"""Argparser for overriding config values"""

import argparse


def str_to_bool(value):
    """Converts a string into a boolean value"""

    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def init_argparser():
    """Creates an argument parser that any function can use"""

    parser = argparse.ArgumentParser(
        epilog="NOTE: These args serve to override parts of the config. "
        + "It's highly recommended to edit config.yml for all options and "
        + "better descriptions!"
    )
    add_network_args(parser)
    add_model_args(parser)
    add_logging_args(parser)
    add_developer_args(parser)
    add_sampling_args(parser)
    add_config_args(parser)

    return parser


def convert_args_to_dict(args: argparse.Namespace, parser: argparse.ArgumentParser):
    """Broad conversion of surface level arg groups to dictionaries"""

    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {}
        for arg in group._group_actions:
            value = getattr(args, arg.dest, None)
            if value is not None:
                group_dict[arg.dest] = value

            arg_groups[group.title] = group_dict

    return arg_groups


def add_config_args(parser: argparse.ArgumentParser):
    """Adds config arguments"""

    parser.add_argument(
        "--config", type=str, help="Path to an overriding config.yml file"
    )


def add_network_args(parser: argparse.ArgumentParser):
    """Adds networking arguments"""

    network_group = parser.add_argument_group("network")
    network_group.add_argument("--host", type=str, help="The IP to host on")
    network_group.add_argument("--port", type=int, help="The port to host on")
    network_group.add_argument(
        "--disable-auth",
        type=str_to_bool,
        help="Disable HTTP token authenticaion with requests",
    )
    network_group.add_argument(
        "--send-tracebacks",
        type=str_to_bool,
        help="Decide whether to send error tracebacks over the API",
    )


def add_model_args(parser: argparse.ArgumentParser):
    """Adds model arguments"""

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--model-dir", type=str, help="Overrides the directory to look for models"
    )
    model_group.add_argument("--model-name", type=str, help="An initial model to load")
    model_group.add_argument(
        "--use-dummy-models",
        type=str_to_bool,
        help="Add dummy OAI model names for API queries",
    )
    model_group.add_argument(
        "--use-as-default",
        type=str,
        nargs="+",
        help="Names of args to use as a default fallback for API load requests ",
    )
    model_group.add_argument(
        "--max-seq-len", type=int, help="Override the maximum model sequence length"
    )
    model_group.add_argument(
        "--override-base-seq-len",
        type=str_to_bool,
        help="Overrides base model context length",
    )
    model_group.add_argument(
        "--gpu-split-auto",
        type=str_to_bool,
        help="Automatically allocate resources to GPUs",
    )
    model_group.add_argument(
        "--autosplit-reserve",
        type=int,
        nargs="+",
        help="Reserve VRAM used for autosplit loading (in MBs) ",
    )
    model_group.add_argument(
        "--gpu-split",
        type=float,
        nargs="+",
        help="An integer array of GBs of vram to split between GPUs. "
        + "Ignored if gpu_split_auto is true",
    )
    model_group.add_argument(
        "--rope-scale", type=float, help="Sets rope_scale or compress_pos_emb"
    )
    model_group.add_argument("--rope-alpha", type=float, help="Sets rope_alpha for NTK")
    model_group.add_argument(
        "--cache-mode",
        type=str,
        help="Set the quantization level of the K/V cache. Options: (FP16, Q8, Q6, Q4)",
    )
    model_group.add_argument(
        "--cache-size",
        type=int,
        help="The size of the prompt cache (in number of tokens) to allocate",
    )
    model_group.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk size for prompt ingestion",
    )
    model_group.add_argument(
        "--max-batch-size",
        type=int,
        help="Maximum amount of prompts to process at one time",
    )
    model_group.add_argument(
        "--prompt-template",
        type=str,
        help="Set the jinja2 prompt template for chat completions",
    )
    model_group.add_argument(
        "--num-experts-per-token",
        type=int,
        help="Number of experts to use per token in MoE models",
    )
    model_group.add_argument(
        "--fasttensors",
        type=str_to_bool,
        help="Possibly increases model loading speeds",
    )


def add_logging_args(parser: argparse.ArgumentParser):
    """Adds logging arguments"""

    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument(
        "--log-prompt", type=str_to_bool, help="Enable prompt logging"
    )
    logging_group.add_argument(
        "--log-generation-params",
        type=str_to_bool,
        help="Enable generation parameter logging",
    )
    logging_group.add_argument(
        "--log-requests",
        type=str_to_bool,
        help="Enable request logging",
    )


def add_developer_args(parser: argparse.ArgumentParser):
    """Adds developer-specific arguments"""

    developer_group = parser.add_argument_group("developer")
    developer_group.add_argument(
        "--unsafe-launch", type=str_to_bool, help="Skip Exllamav2 version check"
    )
    developer_group.add_argument(
        "--disable-request-streaming",
        type=str_to_bool,
        help="Disables API request streaming",
    )
    developer_group.add_argument(
        "--cuda-malloc-backend",
        type=str_to_bool,
        help="Disables API request streaming",
    )


def add_sampling_args(parser: argparse.ArgumentParser):
    """Adds sampling-specific arguments"""

    sampling_group = parser.add_argument_group("sampling")
    sampling_group.add_argument(
        "--override-preset", type=str, help="Select a sampler override preset"
    )
