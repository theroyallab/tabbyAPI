"""Utility to automatically upgrade and start the API"""

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import traceback
from shutil import copyfile, which
from typing import List

# Checks for uv installation
has_uv = which("uv") is not None

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
    possible_features = ["cu12", "amd"]

    if not lib_name:
        has_nvidia = which("nvidia-smi") is not None
        has_rocm = which("rocm-smi") is not None
        has_amd = which("amd-smi") is not None
        has_amd_gpu = has_rocm or has_amd

        if has_nvidia and not has_amd_gpu:
            lib_name = "cu12"
            print("Auto-detected NVIDIA GPU. Using CUDA 12.x backend.")
        elif has_amd_gpu and not has_nvidia:
            lib_name = "amd"
            print("Auto-detected AMD GPU. Using AMD backend.")
        else:
            gpu_lib_choices = {
                "A": {"pretty": "NVIDIA Cuda 12.x", "internal": "cu12"},
                "B": {"pretty": "AMD", "internal": "amd"},
            }
            print(
                "WARNING: Auto-detection failed. "
                "Please ensure you have either an NVIDIA GPU (with nvidia-smi) "
                "or an AMD GPU (with rocm-smi or amd-smi) installed."
            )
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
            "WARN: Please remove the `gpu_lib` key from start_options.json and restart "
            "if you want to change your selection."
        )
        return

    if install_features == "amd":
        # Exit if using AMD and Windows
        if platform.system() == "Windows":
            print(
                "ERROR: TabbyAPI does not support AMD and Windows. "
                "Please use Linux and ROCm 6.4. Exiting."
            )
            sys.exit(0)

        # Override env vars for ROCm support on non-supported GPUs
        os.environ["ROCM_PATH"] = "/opt/rocm"
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        os.environ["HCC_AMDGPU_TARGET"] = "gfx1030"

    return install_features


def create_argparser():
    try:
        from common.args import init_argparser

        return init_argparser()
    except ModuleNotFoundError:
        print(
            "Pydantic not found. Showing an abridged help menu.\n"
            "Run this script once to install dependencies.\n"
        )

        return argparse.ArgumentParser()


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
        "-fr",
        "--force-reinstall",
        action="store_true",
        help="Forces a reinstall of dependencies. Only works with --update-deps",
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


def migrate_start_options(start_options: dict):
    migrated = False

    # Migrate gpu_lib key
    gpu_lib = start_options.get("gpu_lib")
    if gpu_lib == "cu121" or gpu_lib == "cu118":
        print("GPU lib key is legacy, migrating to cu12")
        start_options["gpu_lib"] = "cu12"
        migrated = True

    return migrated


def run_pip(command: List[str]):
    if has_uv:
        command.insert(0, "uv")

    subprocess.run(command)


if __name__ == "__main__":
    # Create an argparser and add extra startup script args
    # Try creating a full argparser if pydantic is installed
    # Otherwise, create an abridged one solely for startup
    try:
        from common.args import init_argparser

        parser = init_argparser()
        has_full_parser = True
    except ModuleNotFoundError:
        parser = argparse.ArgumentParser(
            description="Abridged TabbyAPI start script parser.",
            epilog=(
                "Some dependencies were not found to display the full argparser. "
                "Run the script once to install/update them."
            ),
        )
        has_full_parser = False

    add_start_args(parser)
    args, _ = parser.parse_known_args()

    # Log pip version
    run_pip(["pip", "-V"])

    script_ext = "bat" if platform.system() == "Windows" else "sh"
    do_start_options_write = False

    start_options_path = pathlib.Path("start_options.json")
    if start_options_path.exists():
        with open(start_options_path) as start_options_file:
            start_options = json.load(start_options_file)
            print("Loaded your saved preferences from `start_options.json`")

            do_start_options_write = migrate_start_options(start_options)
        if start_options.get("first_run_done"):
            first_run = False
    else:
        print(
            "It looks like you're running TabbyAPI for the first time. "
            "Getting things ready..."
        )

    # Set variables that rely on start options
    first_run = not start_options.get("first_run_done")

    # Set gpu_lib for dependency install
    if args.gpu_lib:
        print("Overriding GPU lib name from args.")
        gpu_lib = args.gpu_lib
    elif "gpu_lib" in start_options:
        gpu_lib = start_options.get("gpu_lib")
    else:
        gpu_lib = None

    # Pull from GitHub
    if args.update_repository:
        print("Pulling latest changes from Github.")
        pull_command = "git pull"
        subprocess.run(pull_command.split(" "))

    # Install/update dependencies
    if first_run or args.update_deps:
        install_command = ["pip", "install", "-U"]

        # Force a reinstall of the updated dependency if needed
        if args.force_reinstall:
            install_command.append("--force-reinstall")

        install_features = None if args.nowheel else get_install_features(gpu_lib)
        features = f".[{install_features}]" if install_features else "."
        install_command.append(features)

        # pip install .[features]
        print(f"Running install command: {' '.join(install_command)}")
        run_pip(install_command)
        print()

        if first_run:
            start_options["first_run_done"] = True

            # Save start options on first run
            do_start_options_write = True

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

    if do_start_options_write:
        with open("start_options.json", "w") as start_file:
            start_file.write(json.dumps(start_options))

            print(
                "Successfully wrote your start script options to "
                "`start_options.json`. \n"
                "If something goes wrong, editing or deleting the file "
                "will reinstall TabbyAPI as a first-time user."
            )

    # Expand the parser if it's not fully created
    if not has_full_parser:
        from common.args import init_argparser

        parser = init_argparser(parser)
        args = parser.parse_args()

    # Assume all dependencies are installed from here
    try:
        from main import entrypoint

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

        print("Starting TabbyAPI...")
        entrypoint(args, parser)
    except (ModuleNotFoundError, ImportError):
        traceback.print_exc()
        print(
            "\n"
            "This error was raised because a package was not found.\n"
            "Update your dependencies by running update_scripts/"
            f"update_deps.{'bat' if platform.system() == 'Windows' else 'sh'}\n\n"
        )
