from typing import Optional

import torch

# Oldest NVIDIA compute capability ExLlamaV3's kernels run on: Turing (20 series).
# Volta is 7.0 and lacks instructions they rely on
MIN_CUDA_COMPUTE_CAPABILITY = (7, 5)


def torch_backend() -> Optional[str]:
    """The GPU backend the installed PyTorch was built for: "cuda", "rocm" or None."""

    if torch.version.hip:
        return "rocm"
    if torch.version.cuda:
        return "cuda"
    return None


def torch_gpu_problem() -> Optional[str]:
    """
    A message explaining why the installed PyTorch can't run models on a GPU, or
    None if it can. Catches the two setups that otherwise fail in confusing
    ways: a CPU-only PyTorch build (typically from a plain `pip install torch`
    replacing the CUDA or ROCm one), and a GPU build that finds no device.
    """

    backend = torch_backend()
    if backend is None:
        return (
            f"The installed PyTorch ({torch.__version__}) is a CPU-only build, with neither "
            "CUDA nor ROCm support, so ExLlamaV3 cannot use a GPU. This usually happens when "
            "torch was installed or upgraded with a plain `pip install torch`, replacing the "
            "GPU build. Reinstall the dependencies with the update script in update_scripts/ "
            "or `pip install --upgrade .[cu12]` (or `.[cu13]`), or install a ROCm build of "
            "PyTorch for AMD GPUs."
        )

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        if backend == "rocm":
            built_for = f"ROCm (HIP {torch.version.hip})"
            hint = "an AMD GPU with a working ROCm installation"
        else:
            built_for = f"CUDA {torch.version.cuda}"
            hint = "an NVIDIA GPU with a current driver"
        return (
            f"The installed PyTorch ({torch.__version__}) is built for {built_for} but found "
            f"no usable GPU. Check that {hint} is present, and that CUDA_VISIBLE_DEVICES or "
            "HIP_VISIBLE_DEVICES, if set, don't hide it."
        )

    return None


def hardware_supports_exllamav3(gpu_device_list: list[int]):
    """
    Check whether all GPUs in the list can run ExLlamaV3.

    On CUDA builds every device needs compute capability 7.5 (Turing) or higher.
    ROCm builds skip the check: the capability number is NVIDIA's and means
    nothing for AMD devices, and ExLlamaV3 does its own validation there.
    """

    if torch_backend() == "rocm":
        return True

    if not gpu_device_list:
        return False

    min_compute_capability = min(
        torch.cuda.get_device_capability(device=device_idx) for device_idx in gpu_device_list
    )

    return min_compute_capability >= MIN_CUDA_COMPUTE_CAPABILITY
