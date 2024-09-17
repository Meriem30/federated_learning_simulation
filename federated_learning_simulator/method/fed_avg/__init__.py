from federated_learning_simulation_lib import (AggregationServer,  # noqa: F401
                                             AggregationWorker,  # noqa: F401
                                             CentralizedAlgorithmFactory,  # noqa: F401
                                             FedAVGAlgorithm)

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_avg",
    client_cls=AggregationWorker,
    server_cls=AggregationServer,
    algorithm_cls=FedAVGAlgorithm,
)
