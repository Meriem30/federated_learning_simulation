import threading
from typing import Any


class ConcurrencyContext:
    def create_queue(self) -> Any:
        raise NotImplementedError()

    def support_pipe(self) -> bool:
        return False

    def create_pipe(self) -> Any:
        raise NotImplementedError()

    def create_event(self) -> Any:
        raise NotImplementedError()

    def create_thread(self, name: str, target, args, kwargs) -> threading.Thread:
        return threading.Thread(name=name, target=target, args=args, kwargs=kwargs)

    def create_worker(self, name, target, args, kwargs):
        raise NotImplementedError()
