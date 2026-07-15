import torch


def hardware_supports_exllamav3(gpu_device_list: list[int]):
    """
    Check whether all GPUs in the list can run ExLlamaV3.

    ExLlamaV3 requires compute capability 8.0 (Ampere) or higher
    and doesn't support ROCm.
    """

    min_compute_capability = min(
        torch.cuda.get_device_capability(device=device_idx)[0] for device_idx in gpu_device_list
    )

    if torch.version.hip or min_compute_capability < 8:
        return False
    else:
        return True
