from functools import cached_property

from other_libs.topology import Endpoint
from torch_kit import Trainer

from ..protocol import ExecutorProtocol
from ..util import ModelCache


class WorkerProtocol(ExecutorProtocol):
    """
        define an interface for a worker in distributed learning
        must have two properties (endpoint and trainer) and define a pause method
    """
    @property
    def endpoint(self) -> Endpoint: ...

    @property
    def trainer(self) -> Trainer: ...

    def pause(self) -> None: ...


class AggregationWorkerProtocol(WorkerProtocol):
    """
        extend WorkerProtocol to define additional properties for the server
        must have the round idx for the current round
    """

    @property
    def round_index(self) -> int: ...

    @property
    def model_cache(self) -> ModelCache: ...

    @property
    def trainer(self) -> Trainer: ...


class GraphWorkerProtocol(WorkerProtocol):
    @property
    def worker_number(self) -> int:
        ...



    @cached_property
    def training_node_indices(self) -> set: ...


class GraphAggregationWorkerProtocol(AggregationWorkerProtocol):
    @cached_property
    def training_node_indices(self) -> set: ...
