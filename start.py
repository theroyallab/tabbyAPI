"""Utility to automatically upgrade and start the API"""

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
from shutil import copyfile

from common.args import convert_args_to_dict, init_argparser


start_options = {}


def get_user_choice(question: str, options_dict: dict):
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


def get_install_features(lib_name: str = None):
    """Fetches the appropriate requirements file depending on the GPU"""
    install_features = None
    possible_features = ["cu121", "cu118", "amd"]

    if not lib_name:
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

        lib_name = gpu_lib_choices.get(user_input, {}).get("internal")

        # Write to start options
        start_options["gpu_lib"] = lib_name
        print("Saving your choice to start options.")

    # Assume default if the file is invalid
    if lib_name and lib_name in possible_features:
        print(f"Using {lib_name} dependencies from your preferences.")
        install_features = lib_name
    else:
        print(
            f"WARN: GPU library {lib_name} not found. "
            "Skipping GPU-specific dependencies.\n"
            "WARN: Please delete gpu_lib.txt and restart "
            "if you want to change your selection."
        )
        return

    if install_features == "amd":
        # Exit if using AMD and Windows
        if platform.system() == "Windows":
            print(
                "ERROR: TabbyAPI does not support AMD and Windows. "
                "Please use Linux and ROCm 6.0. Exiting."
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
        "-ur",
        "--update-repository",
        action="store_true",
        help="Update local git repository to latest",
    )
    start_group.add_argument(
        "-ud",
        "--update-deps",
        action="store_true",
        help="Update all pip dependencies",
    )
    start_group.add_argument(
        "-nw",
        "--nowheel",
        action="store_true",
        help="Don't upgrade wheel dependencies (exllamav2, torch)",
    )
    start_group.add_argument(
        "--gpu-lib",
        type=str,
        help="Select GPU library. Options: cu121, cu118, amd",
    )


def migrate_gpu_lib():
    gpu_lib_path = pathlib.Path("gpu_lib.txt")

    if not gpu_lib_path.exists():
        return

    print("Migrating gpu_lib.txt to the new start_options.json")
    with open("gpu_lib.txt", "r") as gpu_lib_file:
        start_options["gpu_lib"] = gpu_lib_file.readline().strip()
        start_options["first_run_done"] = True

    # Remove the old file
    gpu_lib_path.unlink()

    print(
        "Successfully migrated gpu lib options to start_options. "
        "The old file has been deleted."
    )


if __name__ == "__main__":
    subprocess.run(["pip", "-V"])

    # Create an argparser and add extra startup script args
    parser = init_argparser()
    add_start_args(parser)
    args = parser.parse_args()
    script_ext = "bat" if platform.system() == "Windows" else "sh"

    start_options_path = pathlib.Path("start_options.json")
    if start_options_path.exists():
        with open(start_options_path) as start_options_file:
            start_options = json.load(start_options_file)

        if start_options.get("first_run_done"):
            first_run = False
    else:
        print(
            "It looks like you're running TabbyAPI for the first time. "
            "Getting things ready..."
        )

    # Migrate from old setting storage
    migrate_gpu_lib()

    # Set variables that rely on start options
    first_run = not start_options.get("first_run_done")

    if args.gpu_lib:
        print("Overriding GPU lib name from args.")
        gpu_lib = args.gpu_lib
    elif "gpu_lib" in start_options:
        gpu_lib = start_options.get("gpu_lib")
    else:
        gpu_lib = None

    # Create a config if it doesn't exist
    # This is not necessary to run TabbyAPI, but is new user proof
    config_path = (
        pathlib.Path(args.config) if args.config else pathlib.Path("config.yml")
    )
    if not config_path.exists():
        sample_config_path = pathlib.Path("config_sample.yml")
        copyfile(sample_config_path, config_path)

        print(
            "A config.yml wasn't found.\n"
            f"Created one at {str(config_path.resolve())}"
        )

    if args.update_repository:
        print("Pulling latest changes from Github.")
        pull_command = "git pull"
        subprocess.run(pull_command.split(" "))

    if first_run or args.update_deps:
        install_features = None if args.nowheel else get_install_features(gpu_lib)
        features = f"[{install_features}]" if install_features else ""

        # pip install .[features]
        install_command = f"pip install -U .{features}"
        print(f"Running install command: {install_command}")
        subprocess.run(install_command.split(" "))
        print("\n")

        if args.update_deps:
            print(
                f"Dependencies updated. Please run TabbyAPI with `start.{script_ext}`. "
                "Exiting."
            )
            sys.exit(0)
        else:
            print(
                f"Dependencies installed. Update them with `update_deps.{script_ext}` "
                "inside the `update_scripts` folder."
            )

    if first_run:
        start_options["first_run_done"] = True

    # Save start options
    with open("start_options.json", "w") as start_file:
        start_file.write(json.dumps(start_options))

        print(
            "Successfully wrote your start script options to `start_options.json`. \n"
            "If something goes wrong, editing or deleting the file "
            "will reinstall TabbyAPI as a first-time user."
        )

    # Import entrypoint after installing all requirements
    from main import entrypoint

    converted_args = convert_args_to_dict(args, parser)

    print("Starting TabbyAPI...")
    entrypoint(converted_args)
