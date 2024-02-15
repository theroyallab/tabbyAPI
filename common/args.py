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
        epilog="These args are only for a subset of the config. "
        + "Please edit config.yml for all options!"
    )
    add_network_args(parser)
    add_model_args(parser)
    add_logging_args(parser)
    add_developer_args(parser)
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


def add_model_args(parser: argparse.ArgumentParser):
    """Adds model arguments"""

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--model-dir", type=str, help="Overrides the directory to look for models"
    )
    model_group.add_argument("--model-name", type=str, help="An initial model to load")
    model_group.add_argument(
        "--max-seq-len", type=int, help="Override the maximum model sequence length"
    )
    model_group.add_argument(
        "--override-base-seq-len",
        type=str_to_bool,
        help="Overrides base model context length",
    )
    model_group.add_argument(
        "--rope-scale", type=float, help="Sets rope_scale or compress_pos_emb"
    )
    model_group.add_argument("--rope-alpha", type=float, help="Sets rope_alpha for NTK")
    model_group.add_argument(
        "--prompt-template",
        type=str,
        help="Set the prompt template for chat completions",
    )
    model_group.add_argument(
        "--gpu-split-auto",
        type=str_to_bool,
        help="Automatically allocate resources to GPUs",
    )
    model_group.add_argument(
        "--gpu-split",
        type=float,
        nargs="+",
        help="An integer array of GBs of vram to split between GPUs. "
        + "Ignored if gpu_split_auto is true",
    )
    model_group.add_argument(
        "--num-experts-per-token",
        type=int,
        help="Number of experts to use per token in MoE models",
    )
    model_group.add_argument(
        "--use-cfg",
        type=str_to_bool,
        help="Enables CFG support",
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
