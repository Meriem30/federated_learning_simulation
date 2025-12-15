from federated_learning_simulation_lib import (CentralizedAlgorithmFactory,
                                                AggregationWorker)
import logging
from federated_learning_simulation_lib.server import ClusteredSamplingServerAlgo1
from federated_learning_simulation_lib.algorithm import ClusteredSamplingAlgo1

logger = logging.getLogger(__name__)

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="clustered_sampling_algo1",
    client_cls=AggregationWorker, # Standard worker
    server_cls=ClusteredSamplingServerAlgo1, # Custom server
    algorithm_cls=ClusteredSamplingAlgo1,
)

logger.debug("Registered clustered_sampling_algo1 algorithm")