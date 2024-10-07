from federated_learning_simulation_lib.worker.gradient_worker import GradientWorker
from federated_learning_simulation_lib.graph_worker.graph_worker import GraphWorker
# add NodeSelectionMixin


class GraphGradientWorker(GraphWorker, GradientWorker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process_gradient(self, gradient_dict):
        result = super()._process_gradient(gradient_dict)
        # add logic
        return result
