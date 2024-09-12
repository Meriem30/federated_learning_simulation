from typing import Any

import torch
from torch_kit import TensorDict

from ..message import ParameterMessageBase
from .aggregation_worker import AggregationWorker


class ErrorFeedbackWorker(AggregationWorker):
    """
        extend the functionalities of AggregationWorker
        include error feedback mechanisms for more sophisticated params updates
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert self._send_parameter_diff
        # this dict will store the error tensors for each parameter
        self.__error: TensorDict = {}

    def _get_sent_data(self) -> ParameterMessageBase:
        """
            pass the data from the super method to sparsify method to apply error feedback before returning it
        """
        return self._sparsify(super()._get_sent_data())

    def _sparsify(self, sent_data: ParameterMessageBase) -> ParameterMessageBase:
        raise NotImplementedError()

    def _get_error(self, name: str, param: torch.Tensor) -> torch.Tensor:
        """
            retrieve the error tensor for a given params
        """
        if name not in self.__error:
            # initialize it with zero tensor with the same shape as the param tensor
            self.__error[name] = torch.zeros_like(param)
        return self.__error[name]

    def _set_error(self, name: str, error: torch.Tensor) -> None:
        """
            set the error tensor for a given param name
        """
        self.__error[name] = error
