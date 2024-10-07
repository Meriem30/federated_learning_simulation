from federated_learning_simulation_lib import (CentralizedAlgorithmFactory,  # noqa: F401
                                               GraphAggregationWorker,  # noqa: F401
                                               GraphAggregationServer,
                                               GraphFedAVGAlgorithm)

import logging

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="graph_fed_avg",
    client_cls=GraphAggregationWorker,
    server_cls=GraphAggregationServer,
    algorithm_cls=GraphFedAVGAlgorithm,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Registered graph_fed_avg algorithm")
