"""Utility to automatically upgrade and start the API"""
import argparse
import os
import pathlib
import subprocess
from main import entrypoint


def get_requirements_file():
    """Fetches the appropriate requirements file depending on the GPU"""
    requirements_name = "requirements-nowheel"
    ROCM_PATH = os.environ.get("ROCM_PATH")
    CUDA_PATH = os.environ.get("CUDA_PATH")

    # TODO: Check if the user has an AMD gpu on windows
    if ROCM_PATH:
        requirements_name = "requirements-amd"
    elif CUDA_PATH:
        cuda_version = pathlib.Path(CUDA_PATH).name
        if "12" in cuda_version:
            requirements_name = "requirements"
        elif "11" in cuda_version:
            requirements_name = "requirements-cu118"

    return requirements_name


def get_argparser():
    """Fetches the argparser for this script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-iu",
        "--ignore-upgrade",
        action="store_true",
        help="Ignore requirements upgrade",
    )
    parser.add_argument(
        "-nw",
        "--nowheel",
        action="store_true",
        help="Don't upgrade wheel dependencies (exllamav2, torch)",
    )
    return parser


if __name__ == "__main__":
    subprocess.run("pip -V")

    parser = get_argparser()
    args = parser.parse_args()

    if args.ignore_upgrade:
        print("Ignoring pip dependency upgrade due to user request.")
    else:
        requirements_file = (
            "requirements-nowheel" if args.nowheel else get_requirements_file()
        )
        subprocess.run(f"pip install -U -r {requirements_file}.txt")

    entrypoint()
