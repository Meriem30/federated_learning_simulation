from typing import Any

from ..concurrency import ProcessContext
from .topology import Topology


class CentralTopology(Topology):
    def __init__(self, worker_num: int) -> None:
        self.worker_num = worker_num

    def get_from_server(self, worker_id: int) -> Any:
        raise NotImplementedError()

    def get_from_worker(self, worker_id: int) -> Any:
        raise NotImplementedError()

    def has_data_from_server(self, worker_id: int) -> bool:
        raise NotImplementedError()

    def has_data_from_worker(self, worker_id: int) -> bool:
        raise NotImplementedError()

    def send_to_server(self, worker_id: int, data: Any) -> None:
        raise NotImplementedError()

    def send_to_worker(self, worker_id: int, data: Any) -> None:
        raise NotImplementedError()

    def close_server_channel(self) -> None:
        raise NotImplementedError()

    def close_worker_channel(self, worker_id: int) -> None:
        raise NotImplementedError()


class ProcessPipeCentralTopology(CentralTopology):
    def __init__(
        self, *args: Any, mp_context: ProcessContext | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.__pipes: dict = {}
        if mp_context is None:
            mp_context = ProcessContext()
        self.__context = mp_context
        for worker_id in range(self.worker_num):
            # 0 => server 1=> worker
            self.__pipes[worker_id] = self.__context.create_pipe()

    def get_from_worker(self, worker_id: int) -> Any:
        assert 0 <= worker_id < self.worker_num
        return self.__pipes[worker_id][0].recv()

    def send_to_worker(self, worker_id: int, data: Any) -> None:
        self.__pipes[worker_id][0].send(data)

    def has_data_from_worker(self, worker_id: int) -> bool:
        assert 0 <= worker_id < self.worker_num
        return self.__pipes[worker_id][0].poll()

    def get_from_server(self, worker_id: int) -> Any:
        assert 0 <= worker_id < self.worker_num
        return self.__pipes[worker_id][1].recv()

    def has_data_from_server(self, worker_id: int) -> bool:
        assert 0 <= worker_id < self.worker_num
        return self.__pipes[worker_id][1].poll()

    def send_to_server(self, worker_id: int, data: Any) -> None:
        assert 0 <= worker_id < self.worker_num
        return self.__pipes[worker_id][1].send(data)

    def close_server_channel(self) -> None:
        for p in self.__pipes.values():
            p[1].close()

    def close_worker_channel(self, worker_id: int) -> None:
        self.__pipes[worker_id][0].close()


class ProcessQueueCentralTopology(CentralTopology):
    def __init__(
        self, *args: Any, mp_context: ProcessContext | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.__queues: dict = {}
        if mp_context is None:
            mp_context = ProcessContext()
        self.__context = mp_context
        for worker_id in range(self.worker_num):
            self.__queues[worker_id] = [
                self.__context.create_queue(),
                self.__context.create_queue(),
            ]

    def get_from_worker(self, worker_id: int) -> Any:
        assert 0 <= worker_id < self.worker_num
        return self.__queues[worker_id][1].get()

    def has_data_from_worker(self, worker_id: int) -> bool:
        assert 0 <= worker_id < self.worker_num
        return not self.__queues[worker_id][1].empty()

    def send_to_worker(self, worker_id: int, data: Any) -> None:
        self.__queues[worker_id][0].put(data)

    def get_from_server(self, worker_id: int) -> Any:
        assert 0 <= worker_id < self.worker_num
        return self.__queues[worker_id][0].get()

    def has_data_from_server(self, worker_id: int) -> bool:
        assert 0 <= worker_id < self.worker_num
        return not self.__queues[worker_id][0].empty()

    def send_to_server(self, worker_id: int, data: Any) -> None:
        assert 0 <= worker_id < self.worker_num
        self.__queues[worker_id][1].put(data)

    def close_server_channel(self) -> None:
        pass

    def close_worker_channel(self, worker_id: int) -> None:
        self.__queues.pop(worker_id)
