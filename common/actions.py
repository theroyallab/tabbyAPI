import argparse
import asyncio
import json
import traceback

from common.logger import xlogger
from common.tabby_config import generate_config_file
from common.utils import unwrap


def download_action(args: argparse.Namespace):
    from common.downloader import hf_repo_download

    try:
        asyncio.run(
            hf_repo_download(
                repo_id=args.repo_id,
                folder_name=args.folder_name,
                revision=args.revision,
                token=args.token,
                include=args.include,
                exclude=args.exclude,
            )
        )
    except Exception:
        exception = traceback.format_exc()
        xlogger.error(exception)


def config_export_action(args: argparse.Namespace):
    export_path = unwrap(args.export_path, "config_sample.yml")
    generate_config_file(filename=export_path)


def openapi_export_action(args: argparse.Namespace):
    from endpoints.server import export_openapi

    export_path = unwrap(args.export_path, "openapi.json")
    openapi_json = export_openapi()

    with open(export_path, "w") as f:
        f.write(json.dumps(openapi_json))
        xlogger.info("Successfully wrote OpenAPI spec to " + f"{export_path}")


def run_subcommand(args: argparse.Namespace) -> bool:
    match args.actions:
        case "download":
            download_action(args)
            return True
        case "export-config":
            config_export_action(args)
            return True
        case "export-openapi":
            openapi_export_action(args)
            return True
        case _:
            return False
