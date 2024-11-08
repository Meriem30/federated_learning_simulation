from typing import Any

import torch

from .metric import Metric


class AccuracyMetric(Metric):
    # Keep track of the number of correct predictions
    __correct_count: int | torch.Tensor | None = None
    # keep track of the total number of samples processed
    __dataset_size: int | torch.Tensor = 0

    def _before_epoch(self, **kwargs: Any) -> None:
        self.__dataset_size = 0
        self.__correct_count = None

    @torch.no_grad()
    def _after_batch(self, result: dict, **kwargs: Any) -> None:
        """
            Calculate the number of correct predictions for each batch
            Update the dataset size
        """
        output = result["model_output"]
        logits = result.get("logits", None)
        targets = result["targets"]
        if logits is not None:
            output = logits
        correct_count: int | torch.Tensor = 0
        if output.shape == targets.shape:
            if len(targets.shape) == 2:
                for idx, maxidx in enumerate(torch.argmax(output, dim=1)):
                    if targets[idx][maxidx] == 1:
                        correct_count += 1
            elif len(targets.shape) <= 1:
                correct_count = (
                    torch.eq(torch.round(output.sigmoid()), targets).view(-1).sum()
                )
            else:
                raise NotImplementedError()
        else:
            assert len(output.shape) == 2
            correct_count = (
                torch.eq(torch.max(output, dim=1)[1], targets.view(-1)).view(-1).sum()
            )
        if self.__correct_count is None:
            self.__correct_count = correct_count
        else:
            self.__correct_count += correct_count
        self.__dataset_size += targets.shape[0]

    def _after_epoch(self, **kwargs) -> None:
        """
            Calculate and record accuracy after each epoch
        """
        epoch = kwargs["epoch"]
        assert self.__correct_count is not None
        accuracy = self.__correct_count / self.__dataset_size
        self._set_epoch_metric(epoch, "accuracy", accuracy)
