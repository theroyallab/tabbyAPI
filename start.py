"""Utility to automatically upgrade and start the API"""

import asyncio
import argparse
import os
import pathlib
import platform
import subprocess
import sys
from common.args import convert_args_to_dict, init_argparser


def get_user_choice(question, options_dict):
    """
    Gets user input in a commandline script.

    Originally from: https://github.com/oobabooga/text-generation-webui/blob/main/one_click.py#L213
    """

    print()
    print(question)
    print()

    for key, value in options_dict.items():
        print(f"{key}) {value.get('pretty')}")

    print()

    choice = input("Input> ").upper()
    while choice not in options_dict.keys():
        print("Invalid choice. Please try again.")
        choice = input("Input> ").upper()

    return choice


def get_install_features():
    """Fetches the appropriate requirements file depending on the GPU"""
    install_features = None
    possible_features = ["cu121", "cu118", "amd"]

    # Try getting the GPU lib from a file
    saved_lib_path = pathlib.Path("gpu_lib.txt")
    if saved_lib_path.exists():
        with open(saved_lib_path.resolve(), "r") as f:
            lib = f.readline()

            # Assume default if the file is invalid
            if lib not in possible_features:
                print(
                    f"WARN: GPU library {lib} not found. "
                    "Skipping GPU-specific dependencies.\n"
                    "WARN: Please delete gpu_lib.txt and restart "
                    "if you want to change your selection."
                )
                return

            print(f"Using {lib} dependencies from your preferences.")
            install_features = lib
    else:
        # Ask the user for the GPU lib
        gpu_lib_choices = {
            "A": {"pretty": "NVIDIA Cuda 12.x", "internal": "cu121"},
            "B": {"pretty": "NVIDIA Cuda 11.8", "internal": "cu118"},
            "C": {"pretty": "AMD", "internal": "amd"},
        }
        user_input = get_user_choice(
            "Select your GPU. If you don't know, select Cuda 12.x (A)",
            gpu_lib_choices,
        )

        install_features = gpu_lib_choices.get(user_input, {}).get("internal")

        # Write to a file for subsequent runs
        with open(saved_lib_path.resolve(), "w") as f:
            f.write(install_features)
            print(
                "Saving your choice to gpu_lib.txt. "
                "Delete this file and restart if you want to change your selection."
            )

    if install_features == "amd":
        # Exit if using AMD and Windows
        if platform.system() == "Windows":
            print(
                "ERROR: TabbyAPI does not support AMD and Windows. "
                "Please use Linux and ROCm 5.6. Exiting."
            )
            sys.exit(0)

        # Override env vars for ROCm support on non-supported GPUs
        os.environ["ROCM_PATH"] = "/opt/rocm"
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        os.environ["HCC_AMDGPU_TARGET"] = "gfx1030"

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
        install_command = f"pip install -U .{features}"
        print(f"Running install command: {install_command}")
        subprocess.run(install_command.split(" "))

    # Import entrypoint after installing all requirements
    from main import entrypoint

    asyncio.run(entrypoint(convert_args_to_dict(args, parser)))
