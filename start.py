"""Utility to automatically upgrade and start the API"""

import asyncio
import argparse
import os
import pathlib
import subprocess
from common.args import convert_args_to_dict, init_argparser


def get_install_features():
    """Fetches the appropriate requirements file depending on the GPU"""
    install_features = None
    ROCM_PATH = os.environ.get("ROCM_PATH")
    CUDA_PATH = os.environ.get("CUDA_PATH")

    # TODO: Check if the user has an AMD gpu on windows
    if ROCM_PATH:
        install_features = "amd"

        # Also override env vars for ROCm support on non-supported GPUs
        os.environ["ROCM_PATH"] = "/opt/rocm"
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        os.environ["HCC_AMDGPU_TARGET"] = "gfx1030"
    elif CUDA_PATH:
        cuda_version = pathlib.Path(CUDA_PATH).name
        if "12" in cuda_version:
            install_features = "cu121"
        elif "11" in cuda_version:
            install_features = "cu118"

    return install_features


def add_start_args(parser: argparse.ArgumentParser):
    """Add start script args to the provided parser"""
    start_group = parser.add_argument_group("start")
    start_group.add_argument(
        "-iu",
        "--ignore-upgrade",
        action="store_true",
        help="Ignore requirements upgrade",
    )
    start_group.add_argument(
        "-nw",
        "--nowheel",
        action="store_true",
        help="Don't upgrade wheel dependencies (exllamav2, torch)",
    )


if __name__ == "__main__":
    subprocess.run(["pip", "-V"])

    # Create an argparser and add extra startup script args
    parser = init_argparser()
    add_start_args(parser)
    args = parser.parse_args()

    if args.ignore_upgrade:
        print("Ignoring pip dependency upgrade due to user request.")
    else:
        install_features = None if args.nowheel else get_install_features()
        features = f"[{install_features}]" if install_features else ""

        # pip install .[features]
        subprocess.run(["pip", "install", "-U", f".{features}"])

    # Import entrypoint after installing all requirements
    from main import entrypoint

    asyncio.run(entrypoint(convert_args_to_dict(args, parser)))
