from federated_learning_simulation_lib import FedAVGAlgorithm

class ClusteredSamplingAlgo2(FedAVGAlgorithm):
    """
    Clustered Sampling Algorithm 2.
    Inherits from FedAVGAlgorithm.
    The main logic changes are in the Server selection phase
    (gradient tracking and clustering).
    """
    def __init__(self) -> None:
        super().__init__()
