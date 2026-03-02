"""Test if the wheels are installed correctly."""

from importlib.metadata import version
from importlib.util import find_spec
from packaging import version as package_version

successful_packages = []
errored_packages = []

if find_spec("flash_attn") is not None:
    torch_version = version("torch").split("+")[0] if find_spec("torch") else "0"
    if package_version.parse(torch_version) >= package_version.parse("2.10.0"):
        print(
            f"Flash attention 2 detected with torch {torch_version}. "
            "This combination is unsupported for the flashinfer migration."
        )
        errored_packages.append("flash_attn")
    else:
        print(
            "Flash attention 2 is installed. "
            "ExLlamaV3 now uses flashinfer."
        )

if find_spec("flashinfer") is not None:
    print(
        "FlashInfer on version "
        f"{version('flashinfer-python')} successfully imported"
    )
    successful_packages.append("flashinfer")
else:
    print("FlashInfer is not found in your environment.")
    errored_packages.append("flashinfer")

if find_spec("exllamav2") is not None:
    print(f"Exllamav2 on version {version('exllamav2')} successfully imported")
    successful_packages.append("exllamav2")
else:
    print("Exllamav2 is not found in your environment (optional).")

if find_spec("torch") is not None:
    print(f"Torch on version {version('torch')} successfully imported")
    successful_packages.append("torch")
else:
    print("Torch is not found in your environment.")
    errored_packages.append("torch")

if find_spec("jinja2") is not None:
    print(f"Jinja2 on version {version('jinja2')} successfully imported")
    successful_packages.append("jinja2")
else:
    print("Jinja2 is not found in your environment.")
    errored_packages.append("jinja2")

print(f"\nSuccessful imports: {', '.join(successful_packages)}")
print(f"Errored imports: {''.join(errored_packages)}")

if len(errored_packages) > 0:
    print(
        "If all packages are installed, but not found "
        "on this test, please check the wheel versions for the "
        "correct python version and CUDA version (if "
        "applicable)."
    )
else:
    print("All wheels are installed correctly.")
