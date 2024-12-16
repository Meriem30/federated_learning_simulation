from federated_learning_simulation_lib import (CentralizedAlgorithmFactory,  # noqa: F401
                                               AggregationServer,  # noqa: F401
                                                AggregationWorker,  # noqa: F401
                                                FedAVGAlgorithm)

import logging

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_avg",
    client_cls=AggregationWorker,
    server_cls=AggregationServer,
    algorithm_cls=FedAVGAlgorithm,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.debug("Registered fed_avg algorithm")
