from typing import Any

from other_libs.concurrency import ThreadContext

from .torch_task_queue import TorchTaskQueue


class TorchThreadTaskQueue(TorchTaskQueue):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(mp_ctx=ThreadContext(), **kwargs)