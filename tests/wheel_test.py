""" Test if the wheels are installed correctly. """
from importlib.metadata import version
from importlib.util import find_spec

successful_packages = []
errored_packages = []

if find_spec("flash_attn") is not None:
    print(f"Flash attention on version {version('flash_attn')} successfully imported")
    successful_packages.append("flash_attn")
else:
    print("Flash attention 2 is not found in your environment.")
    errored_packages.append("flash_attn")

if find_spec("exllamav2") is not None:
    print(f"Exllamav2 on version {version('exllamav2')} successfully imported")
    successful_packages.append("exllamav2")
else:
    print("Exllamav2 is not found in your environment.")
    errored_packages.append("exllamav2")

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

print(
    f"\nSuccessful imports: {', '.join(successful_packages)}",
    f"\nErrored imports: {''.join(errored_packages)}",
)

if len(errored_packages) > 0:
    print(
        "\nIf packages are installed, but not found on this test, please "
        "check the wheel versions for the correct python version and CUDA "
        "version (if applicable)."
    )
