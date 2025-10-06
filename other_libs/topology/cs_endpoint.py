from typing import Any, Iterable

from .central_topology import CentralTopology
from .endpoint import Endpoint
from federated_learning_simulation_lib.message import ParameterMessageBase, ParameterMessage, DeltaParameterMessage
from torch_kit import ModelParameter
import copy


class ServerEndpoint(Endpoint):
    """
        used by the server to handle communication with clients
    """
    @property
    def topology(self) -> CentralTopology:
        assert isinstance(self._topology, CentralTopology)
        return self._topology

    @property
    def worker_num(self) -> int:
        """
            return the number of workers
        """
        return self.topology.worker_num

    def has_data(self, worker_id: int) -> bool:
        """
            check if data available from a specific worker
        """
        return self.topology.has_data_from_worker(worker_id=worker_id)

    def get(self, worker_id: int) -> Any:
        """
            retrieve from a specific worker
        """
        return self.topology.get_from_worker(worker_id=worker_id)

    def send(self, worker_id: int, data: Any) -> None:
        """
            send data to a specific worker
        """
        self.topology.send_to_worker(worker_id=worker_id, data=data)

    def broadcast(self, data: Any, worker_ids: None | Iterable = None) -> None:
        """
            send data to all or a subset of workers
        """
        all_worker_ids = set(range(self.worker_num))
        if worker_ids is None:
            worker_ids = all_worker_ids
        else:
            worker_ids = set(worker_ids).intersection(all_worker_ids)
        for worker_id in worker_ids:
            self.send(worker_id=worker_id, data=data)

    def close(self) -> None:
        """
            clode the server communication channel
        """
        assert isinstance(self._topology, CentralTopology)
        self._topology.close_server_channel()


class ClientEndpoint(Endpoint):
    """
        used by the client to handle communication with the server
    """
    def __init__(self, topology: CentralTopology, worker_id: int) -> None:
        """
            initialise the worker endpoint with a central endpoint and a worker ID
        """
        super().__init__(topology=topology)
        assert isinstance(self._topology, CentralTopology)
        self.__worker_id: int = worker_id

    @property
    def topology(self) -> CentralTopology:
        """
            return the instance of CentralTopology (if correct type)
        """
        assert isinstance(self._topology, CentralTopology)
        return self._topology

    def get(self) -> Any:
        """
            retrieve data from the server for this client
        """
        return self.topology.get_from_server(self.__worker_id)

    def has_data(self) -> bool:
        """
            check if the data available from the server for this client
        """
        return self.topology.has_data_from_server(self.__worker_id)

    def send(self, data: Any) -> None:
        """
            send data to the server
        """
        self.topology.send_to_server(worker_id=self.__worker_id, data=data)

    def close(self) -> None:
        """
            close the client communication channel
        """
        self.topology.close_worker_channel(worker_id=self.__worker_id)
