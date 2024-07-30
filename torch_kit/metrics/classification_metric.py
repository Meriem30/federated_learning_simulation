import torch

from ..ml_type import ModelType
from .metric import Metric


class ClassificationMetric(Metric):
    """
        Handle metrics for classification tasks
    """
    def _before_execution(self, **kwargs) -> None:
        executor = kwargs["executor"]
        # Disable the metric if the model is not for classification task
        if executor.running_model_evaluator.model_type != ModelType.Classification:
            self.disable()

    def _get_output(self, result: dict) -> torch.Tensor:
        # Return output in suitable format for further metric calculation
        output = result.get("logits", None)
        if output is None:
            output = result["original_output"]
        assert isinstance(output, torch.Tensor)
        output = output.detach()
        output = torch.where(torch.any(output < 0), output.sigmoid(), output)
        if len(output.shape) == 2 and output.shape[1] == 1:
            output = torch.stack((1 - output, output), dim=2).squeeze()
        return output
