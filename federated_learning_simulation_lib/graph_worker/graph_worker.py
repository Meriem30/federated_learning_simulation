import networkx as nx
from federated_learning_simulation_lib.worker.worker import Worker
from .client_state import ClientState
from typing import List
from other_libs.log import log_info
from typing import Any, Dict


class GraphWorker(Worker):
    def __init__(self, **kwargs):
        Worker.__init__(self, **kwargs)
        self._state = ClientState(self.worker_id)

    @property
    def state(self):
        return self._state

    def _initialize_worker_state(self) -> None:
        families = self.config.family_number
        self.state.initialize_state(families=families)

    def _update_worker_state(self, current_state: ClientState) -> None:
        pass

    def _compute_worker_state(self) -> ClientState:
        pass

    def _send_worker_state_to_server(self):
        pass

    def _get_client_state(self, worker_id: int) -> ClientState:
        return self._state

    def _set_client_state(self, worker_id: int, state: ClientState) -> None:
        pass

    def _set_worker_family(self, worker_id: int, family: int) -> None:
        pass

    def _get_worker_family(self, worker_id: int) -> int:
        pass

