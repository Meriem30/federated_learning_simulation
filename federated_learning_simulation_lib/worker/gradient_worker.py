from typing import Any, Callable

import torch
from torch_kit import ExecutorHookPoint, ModelEvaluator, ModelGradient
from torch_kit.tensor import tensor_to

from ..message import Message, ParameterMessage
from .client import ClientMixin
from .worker import Worker


class GradientModelEvaluator:
    def __init__(
        self,
        evaluator: ModelEvaluator,
        gradient_fun: Callable,
        aggregation_indicator_fun: Callable,
    ) -> None:
        assert torch.cuda.is_available()
        # initialize the model evaluator
        self.evaluator: ModelEvaluator = evaluator
        # initialize the functions
        self.__gradient_fun: Callable = gradient_fun
        self.__aggregation_indicator_fun: Callable = aggregation_indicator_fun

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.evaluator.__call__(*args, **kwargs)

    def __getattr__(self, name):
        if name in ("evaluator", "gradient_fun"):
            raise AttributeError()
        return getattr(self.evaluator, name)

    def backward_and_step(
        self,
        loss,
        optimizer: torch.optim.Optimizer,
        **backward_kwargs,
    ) -> None:
        # combine backward pass and optimizer step
        self.backward(loss=loss, optimizer=optimizer, **backward_kwargs)
        optimizer.step()

    def backward(
        self,
        loss,
        optimizer: torch.optim.Optimizer,
        **backward_kwargs,
    ) -> Any:
        # perform the backward pass using the evaluator
        self.evaluator.backward(loss=loss, optimizer=optimizer, **backward_kwargs)
        if not self.__aggregation_indicator_fun():
            return
        # process the model gradients of the evaluator to be assigned to gradient_dict
        gradient_dict: ModelGradient = self.__gradient_fun(
            self.evaluator.model_util.get_gradient_dict()
        )
        # load the processed model gradients back into the model evaluator
        # convert the tensor to the specified device
        self.evaluator.model_util.load_gradient_dict(
            tensor_to(gradient_dict, device=loss.device)
        )


class GradientWorker(Worker, ClientMixin):
    """
        integrate gradient processing and aggregation functionalities
    """
    def __init__(self, **kwargs) -> None:
        # call the constructor of both parent classes
        Worker.__init__(self, **kwargs)
        ClientMixin.__init__(self, **kwargs)
        # keep track of the iterations
        self.__cnt = 0
        # how frequent the aggregation should occur
        self.__aggregation_interval = self.config.algorithm_kwargs.get("interval", 1)
        # replace the current model evaluator with a new one after processing it
        self.trainer.replace_model_evaluator(
            lambda evaluator: GradientModelEvaluator(
                evaluator=evaluator,
                gradient_fun=self._process_gradient,
                aggregation_indicator_fun=self._should_aggregate,
            )
        )
        # add a hook to the trainer calling the __report_end method
        self.trainer.append_named_hook(
            ExecutorHookPoint.AFTER_EXECUTE, "end_training", self.__report_end
        )

    def __report_end(self, **kwargs: Any) -> None:
        self._send_data_to_server(Message(end_training=True))

    def _should_aggregate(self) -> bool:
        # if the counter is divided by the interval (==0)
        res = self.__cnt % self.__aggregation_interval
        self.__cnt += 1
        return res

    def _process_gradient(self, gradient_dict: ModelGradient) -> ModelGradient:
        # send data to the server in a ParameterMessage
        self._send_data_to_server(
            ParameterMessage(
                parameter=gradient_dict,
                in_round=True,
                aggregation_weight=self.trainer.dataset_size,
            )
        )
        # receive the processed gradient
        result = self._get_data_from_server()
        # type verification
        assert isinstance(result, ParameterMessage)
        # return the result params
        return result.parameter
