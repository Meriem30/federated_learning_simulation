from federated_learning_simulation_lib import FedAVGAlgorithm

class ClusteredSamplingAlgo1(FedAVGAlgorithm):
    """
    Clustered Sampling Algorithm 1.
    Inherits from FedAVGAlgorithm.
    The main logic changes are in the Server selection phase.
    """
    def __init__(self) -> None:
        super().__init__()