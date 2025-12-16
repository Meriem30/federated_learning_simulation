from federated_learning_simulation_lib import (CentralizedAlgorithmFactory,
                                                AggregationWorker)
import logging
from federated_learning_simulation_lib.server.clustered_sampling_server_algo2 import ClusteredSamplingServerAlgo2
from federated_learning_simulation_lib.algorithm.clustered_sampling_algo2 import ClusteredSamplingAlgo2

logger = logging.getLogger(__name__)

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="clustered_sampling_algo2",
    client_cls=AggregationWorker, # Standard worker
    server_cls=ClusteredSamplingServerAlgo2, # Custom server with gradient tracking
    algorithm_cls=ClusteredSamplingAlgo2,
)

logger.debug("Registered clustered_sampling_algo2 algorithm")