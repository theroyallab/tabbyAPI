"""Argparser for overriding config values"""

import argparse
from pydantic import BaseModel

from common.config_models import TabbyConfigModel
from common.utils import is_list_type, unwrap, unwrap_optional_type


def add_field_to_group(group, field_name, field_type, field) -> None:
    """
    Adds a Pydantic field to an argparse argument group.
    """

    kwargs = {
        "help": field.description if field.description else "No description available",
    }

    # If the inner type contains a list, specify argparse as such
    if is_list_type(field_type):
        kwargs["nargs"] = "+"

    group.add_argument(f"--{field_name}", **kwargs)


def init_argparser(
    existing_parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """
    Initializes an argparse parser based on a Pydantic config schema.

    If an existing provider is given, use that.
    """

    parser = unwrap(
        existing_parser, argparse.ArgumentParser(description="TabbyAPI server")
    )

    add_subcommands(parser)

    # Loop through each top-level field in the config
    for field_name, field_info in TabbyConfigModel.model_fields.items():
        field_type = unwrap_optional_type(field_info.annotation)
        group = parser.add_argument_group(
            field_name, description=f"Arguments for {field_name}"
        )

        # Check if the field_type is a Pydantic model
        if issubclass(field_type, BaseModel):
            for sub_field_name, sub_field_info in field_type.model_fields.items():
                sub_field_name = sub_field_name.replace("_", "-")
                sub_field_type = sub_field_info.annotation
                add_field_to_group(
                    group, sub_field_name, sub_field_type, sub_field_info
                )
        else:
            field_name = field_name.replace("_", "-")
            group.add_argument(f"--{field_name}", help=f"Argument for {field_name}")

    return parser


def add_subcommands(parser: argparse.ArgumentParser):
    """Adds subcommands to an existing argparser"""

    actions_subparsers = parser.add_subparsers(
        dest="actions", help="Extra actions that can be run instead of the main server."
    )

    # Calls download action
    download_parser = actions_subparsers.add_parser(
        "download", help="Calls the model downloader"
    )
    download_parser.add_argument("repo_id", type=str, help="HuggingFace repo ID")
    download_parser.add_argument(
        "--folder-name",
        type=str,
        help="Folder name where the model should be downloaded",
    )
    download_parser.add_argument(
        "--revision",
        type=str,
        help="Branch name in HuggingFace repo",
    )
    download_parser.add_argument(
        "--token", type=str, help="HuggingFace access token for private repos"
    )
    download_parser.add_argument(
        "--include", type=str, help="Glob pattern of files to include"
    )
    download_parser.add_argument(
        "--exclude", type=str, help="Glob pattern of files to exclude"
    )

    # Calls openapi action
    openapi_export_parser = actions_subparsers.add_parser(
        "export-openapi", help="Exports an OpenAPI compliant JSON schema"
    )
    openapi_export_parser.add_argument(
        "--export-path",
        help="Path to export the generated OpenAPI JSON (default: openapi.json)",
    )
    openapi_export_parser.add_argument(
        "--api-servers", nargs="+", help="Sets API servers to run when exporting"
    )

    # Calls config export action
    config_export_parser = actions_subparsers.add_parser(
        "export-config", help="Generates and exports a sample config YAML file"
    )
    config_export_parser.add_argument(
        "--export-path",
        help="Path to export the generated sample config (default: config_sample.yml)",
    )


def convert_args_to_dict(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict:
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
