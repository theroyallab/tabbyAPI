import torch


def hardware_supports_flash_attn(gpu_device_list: list[int]):
    """
    Check whether all GPUs in list support FA2

    Compute capability < 8 is not supported by FA2
    AMD is also unsupported until ROCm updates its FA2 fork
    """

    min_compute_capability = min(
        torch.cuda.get_device_capability(device=device_idx)[0]
        for device_idx in gpu_device_list
    )

    if torch.version.hip or min_compute_capability < 8:
        return False
    else:
        return True
