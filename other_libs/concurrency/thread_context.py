import queue
import threading
from typing import Any

from .context import ConcurrencyContext


class ThreadContext(ConcurrencyContext):
    def create_queue(self) -> queue.Queue:
        return queue.Queue()

    def create_pipe(self) -> Any:
        raise NotImplementedError()

    def create_event(self) -> threading.Event:
        return threading.Event()

    def create_worker(self, name, target, args, kwargs) -> threading.Thread:
        return self.create_thread(name=name, target=target, args=args, kwargs=kwargs)
