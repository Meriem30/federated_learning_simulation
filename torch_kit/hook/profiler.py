import functools
import torch

from . import Hook
from other_libs.log import log_info


class CUDAMemoryProfiler(Hook):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__hooks: list = []
        self.__used_memory: dict = {}
        self.__last_used_memory = 0

    def _before_execute(self, **kwargs) -> None:
        self.__hooks = []
        self.__used_memory = {}
        self.__last_used_memory = 0

    def _before_batch(self, executor, batch_index, **kwargs) -> None:
        assert not self.__hooks
        if batch_index != 2:
            return
        for module_name, module in executor.model.named_modules():
            if not module_name:
                continue
            if not any(True for _ in module.parameters()):
                continue
            self.__hooks.append(
                module.register_forward_hook(
                    functools.partial(
                        self.__compute_gpu_memory_assumption,
                        module_name,
                        len(self.__hooks),
                    )
                )
            )

    def __compute_gpu_memory_assumption(
        self, module_name, hook_idx, module, _, __
    ) -> None:
        cur_used_memory = torch.cuda.memory_allocated()
        if not self.__used_memory:
            self.__used_memory[module_name] = float(cur_used_memory) / 1024 / 1024
            log_info(
                "%.1f MB CUDA memory is used for first module %s",
                self.__used_memory[module_name],
                module_name,
            )
        else:
            self.__used_memory[module_name] = (
                float(cur_used_memory - self.__last_used_memory) / 1024 / 1024
            )
            log_info(
                "%.1f MB CUDA memory is used for module %s",
                self.__used_memory[module_name],
                module_name,
            )
        self.__hooks[hook_idx].remove()
        self.__last_used_memory = cur_used_memory


class CUDAStreamProfiler(Hook):
    def _before_execute(self, executor, **kwargs) -> None:
        if executor.device.type.lower() == "cuda":
            torch.cuda.set_sync_debug_mode("warn")

    def _after_execute(self, **kwargs) -> None:
        torch.cuda.set_sync_debug_mode("default")


class Profiler(Hook):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if torch.cuda.is_available():
            self.cuda_stream_profiler = CUDAStreamProfiler()
            # self.__cuda_memory_profiler = CUDAMemoryProfiler()