"""Utility to automatically upgrade and start the API"""

import argparse
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import traceback
from shutil import copyfile, which
from typing import List

# Checks for uv installation
has_uv = which("uv") is not None

start_options = {}

# AMD's pip-native ROCm distribution: torch and its per-GPU device packages on one
# index, the ROCm runtime on another, both flat (pin by version, as the rocm extra
# does). TABBY_ROCM_INDEX_URLS (comma-separated) overrides them for a mirror or an
# offline copy.
ROCM_INDEX_URLS = [
    url.strip()
    for url in os.environ.get(
        "TABBY_ROCM_INDEX_URLS",
        "https://stable.repo.amd.com/rocm/pytorch/whl-next/,"
        "https://stable.repo.amd.com/rocm/core/whl-next/",
    ).split(",")
    if url.strip()
]


def rocm_device_package(target: str) -> str:
    """
    The per-GPU torch device package, e.g. gfx1100 -> amd-torch-device-gfx1100. It
    holds torch's device code for that chip and pulls in the ROCm libraries' code
    objects (rocm-sdk-device-*), which exllamav3's hipBLAS path needs. One per chip
    on a box with several different AMD GPUs.
    """

    return f"amd-torch-device-{target.lower()}"


def detect_amd_gpu_targets() -> List[str]:
    """
    The gfx targets of the AMD GPUs in this machine, read from the kernel's KFD
    topology (no ROCm needed): gfx_target_version 110000 is gfx1100, 110501 is
    gfx1151, 120001 is gfx1201, 100300 is gfx1030. CPU nodes report 0.
    """

    targets = []
    for node in sorted(pathlib.Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties")):
        try:
            text = node.read_text()
        except OSError:
            continue
        match = re.search(r"^gfx_target_version\s+(\d+)$", text, re.MULTILINE)
        if not match or int(match.group(1)) == 0:
            continue
        version = int(match.group(1))
        major, minor, step = version // 10000, (version // 100) % 100, version % 100
        target = f"gfx{major}{minor:x}{step:x}"
        if target not in targets:
            targets.append(target)
    return targets


def print_commit_hash():
    """Prints the commit hash of the current branch or a
    placeholder if git is not available (probably windows users)"""
    try:
        commit_hash = subprocess.check_output(["git",
                                               "rev-parse",
                                               "--short",
                                               "HEAD"]).decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit_hash = "placeholder"
    
    with open("endpoints/OAI/_commit.py", "w") as commit_file:
        contents = f"""commit_hash = "{commit_hash}" """
        commit_file.write(contents)

print_commit_hash()


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
    possible_features = ["cu12", "cu13", "rocm"]

    if not lib_name:
        has_nvidia = which("nvidia-smi") is not None
        has_amd = platform.system() == "Linux" and pathlib.Path("/dev/kfd").exists()

        if has_nvidia:
            lib_name = "cu12"
            print("Auto-detected NVIDIA GPU. Using CUDA 12.x backend.")
        elif has_amd:
            lib_name = "rocm"
            print("Auto-detected AMD GPU. Using the ROCm backend.")
        else:
            gpu_lib_choices = {
                "A": {"pretty": "NVIDIA Cuda 12.x", "internal": "cu12"},
                "B": {"pretty": "NVIDIA Cuda 13.x", "internal": "cu13"},
                "C": {"pretty": "AMD ROCm (RDNA2 to RDNA4, Linux)", "internal": "rocm"},
            }
            print(
                "WARNING: Auto-detection failed. "
                "Please ensure you have an NVIDIA GPU (with nvidia-smi) "
                "or an AMD GPU (with /dev/kfd) installed."
            )
            user_input = get_user_choice(
                "Select your GPU. If you don't know, select Cuda 12.x (A)",
                gpu_lib_choices,
            )
            lib_name = gpu_lib_choices.get(user_input, {}).get("internal")

        # Write to start options
        start_options["gpu_lib"] = lib_name
        print("Saving your choice to start options.")

    if lib_name == "rocm" and not start_options.get("rocm_targets"):
        targets = detect_amd_gpu_targets()
        if targets:
            print(f"Auto-detected AMD GPU target(s): {', '.join(targets)}")
        else:
            print(
                "Could not read the AMD GPU targets from /sys/class/kfd. Enter the gfx "
                "target(s) of your GPU(s), comma-separated, e.g. gfx1100 for a Radeon RX "
                "7900 XTX, gfx1201 for an RX 9070 XT, gfx1151 for Strix Halo."
            )
            targets = [t.strip() for t in input("Input> ").split(",") if t.strip()]
            while not all(re.fullmatch(r"gfx[0-9a-f]{3,4}", t) for t in targets) or not targets:
                print("Invalid target. Please try again (e.g. gfx1100).")
                targets = [t.strip() for t in input("Input> ").split(",") if t.strip()]
        start_options["rocm_targets"] = targets
        print("Saving your GPU target(s) to start options.")

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
        help="Don't upgrade wheel dependencies (exllamav3, torch)",
    )
    start_group.add_argument(
        "--gpu-lib",
        type=str,
        help="Select GPU library. Options: cu12, cu13, rocm",
    )
    start_group.add_argument(
        "--rocm-targets",
        type=str,
        help=(
            "AMD GPU target(s) for the ROCm device packages, comma-separated (e.g. gfx1100, "
            "gfx1201, gfx1151). Detected from /sys/class/kfd when not given. Only with "
            "--gpu-lib rocm"
        ),
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

    subprocess.run(command, check=True)


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

    # Log pip/uv version
    if has_uv:
        subprocess.run(["uv", "-V"])
    else:
        subprocess.run(["pip", "-V"])

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
        print("It looks like you're running TabbyAPI for the first time. Getting things ready...")

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

    if args.rocm_targets:
        start_options["rocm_targets"] = [
            t.strip() for t in args.rocm_targets.split(",") if t.strip()
        ]
        do_start_options_write = True

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

        if install_features == "rocm":
            # torch and the ROCm runtime only exist on AMD's indexes, and the per-GPU
            # device packages have to be asked for by name
            for url in ROCM_INDEX_URLS:
                install_command += ["--extra-index-url", url]
            install_command += [
                rocm_device_package(target) for target in start_options["rocm_targets"]
            ]

        # pip install .[features]
        print(f"Running install command: {' '.join(install_command)}")

        try:
            run_pip(install_command)
            print()
        except subprocess.CalledProcessError:
            print("\nDependency installation failed. Please check the logs and run again.\n")
            sys.exit(1)

        if first_run:
            start_options["first_run_done"] = True

            # Save start options on first run
            do_start_options_write = True

        if args.update_deps:
            print(f"Dependencies updated. Please run TabbyAPI with `start.{script_ext}`. Exiting.")
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
        config_path = pathlib.Path(args.config) if args.config else pathlib.Path("config.yml")
        if not config_path.exists():
            sample_config_path = pathlib.Path("config_sample.yml")
            copyfile(sample_config_path, config_path)

            print(f"A config.yml wasn't found.\nCreated one at {str(config_path.resolve())}")

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
