from typing import Any

from torch_kit import ModelGradient

from federated_learning_simulation_lib.graph_worker.graph_worker import GraphWorker
from federated_learning_simulation_lib.worker.gradient_worker import GradientWorker
from federated_learning_simulation_lib.worker.client import ClientMixin
# add NodeSelectionMixin


class GraphGradientWorker(GraphWorker, GradientWorker, ClientMixin):
    def __init__(self, **kwargs):
        GraphWorker.__init__(self, **kwargs)
        GradientWorker.__init__(self, **kwargs)
        ClientMixin.__init__(self)
        
    def _process_gradient(self, gradient_dict: ModelGradient) -> ModelGradient:
        # send data to the server in a ParameterMessage
        result = super()._process_gradient(gradient_dict)
        # add logic
        return result

