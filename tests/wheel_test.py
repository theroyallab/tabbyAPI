import traceback
from importlib.metadata import version

successful_packages = []
errored_packages = []

try:
    import flash_attn
    print(f"Flash attention on version {version('flash_attn')} successfully imported")
    successful_packages.append("flash_attn")
except:
    print("Flash attention could not be loaded because:")
    print(traceback.format_exc())
    errored_packages.append("flash_attn")

try:
    import exllamav2
    print(f"Exllamav2 on version {version('exllamav2')} successfully imported")
    successful_packages.append("exllamav2")
except:
    print("Exllamav2 could not be loaded because:")
    print(traceback.format_exc())
    errored_packages.append("exllamav2")

try:
    import torch
    print(f"Torch on version {version('torch')} successfully imported")
    successful_packages.append("torch")
except:
    print("Torch could not be loaded because:")
    print(traceback.format_exc())
    errored_packages.append("torch")

try:
    import fastchat
    print(f"Fastchat on version {version('fschat')} successfully imported")
    successful_packages.append("fastchat")
except:
    print("Fastchat is only needed for chat completions with message arrays. Ignore this error if this isn't your usecase.")
    print("Fastchat could not be loaded because:")
    print(traceback.format_exc())
    errored_packages.append("fastchat")

print(
    f"\nSuccessful imports: {', '.join(successful_packages)}",
    f"\nErrored imports: {''.join(errored_packages)}"
)
