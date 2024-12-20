from typing import Any

import psutil
import torch
from other_libs.log import log_error

from .base import MemoryInfo

if torch.cuda.is_available():
    from .cuda import get_cuda_memory_info
else:
    def get_cuda_memory_info(**kwargs) -> dict:
        raise NotImplementedError()


def get_device_memory_info(
    device: torch.device | None = None, consider_cache: bool = True
) -> dict[torch.device, MemoryInfo]:
    """
        Retrieve memory info for the specified device
    """
    device_type: str = ""
    device_idx: int | None = None
    if device is not None:
        device_type = device.type.lower()
        device_idx = device.index
    else:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    match device_type:
        case "cuda":
            return get_cuda_memory_info(
                device_idx=device_idx, consider_cache=consider_cache
            )
        case "cpu" | "mps" | "vulkan" | "xpu":
            device = torch.device(type=device_type, index=0)
            vm = psutil.virtual_memory()
            return {
                device: MemoryInfo(
                    free=vm.available,
                    total=vm.total,
                    used=vm.used,
                )
            }
    raise NotImplementedError(device_type)


def get_cpu_device() -> torch.device:
    return torch.device("cpu")


class DeviceGreedyAllocator:
    @classmethod
    def get_devices(cls, max_needed_bytes: int | None = None) -> list[torch.device]:
        """
            Return a list of devices sorted by available memory
            Optionally filtering based on the required memory
        """
        memory_info = get_device_memory_info()
        memory_to_device: dict = {}
        for device, info in memory_info.items():
            if max_needed_bytes is not None and info.free < max_needed_bytes:
                continue
            if info.used / info.total > 0.95:
                continue
            if info.free not in memory_to_device:
                memory_to_device[info.free] = []
            memory_to_device[info.free].append(device)
        devices = []
        for k in sorted(memory_to_device.keys(), reverse=True):
            devices += memory_to_device[k]
        return devices

    @classmethod
    def get_device(cls, **kwargs: Any) -> torch.device:
        """
            return the single best devices based on available memory
        """
        return cls.get_devices(**kwargs)[0]


def get_devices(max_needed_bytes: None | int = None) -> list[torch.device]:
    """
        Wrapped function: call the 'get_devices' method of DeviceGreedyAllocator class
    """
    devices = DeviceGreedyAllocator.get_devices(max_needed_bytes=max_needed_bytes)
    if "cpu" not in devices[0].type.lower():
        return devices
    log_error(
        "max_needed_bytes is %s, only CPU device is available, which we don't want",
        max_needed_bytes,
    )
    return devices


def get_device(**kwargs: Any) -> torch.device:
    """
        Wrapped function: call the 'get_device' method of DeviceGreedyAllocator class
    """
    return get_devices(**kwargs)[0]
