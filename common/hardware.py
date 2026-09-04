import torch


def hardware_supports_exllamav3(gpu_device_list: list[int]):
    """
    Check whether all GPUs in the list can run ExLlamaV3.

    ExLlamaV3 requires compute capability 7.5 (Turing) or higher
    and doesn't support ROCm. Volta is 7.0 and lacks instructions the
    kernels rely on, so it stays rejected.
    """

    min_compute_capability = min(
        torch.cuda.get_device_capability(device=device_idx) for device_idx in gpu_device_list
    )

    if torch.version.hip or min_compute_capability < (7, 5):
        return False
    else:
        return True
