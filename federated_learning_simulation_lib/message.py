import copy
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

import torch
from torch_kit import ModelParameter
from torch_kit.tensor import recursive_tensor_op

"""
    define different types of messages to be used in training process
"""


@dataclass(kw_only=True)
class Message:
    """
        the base class for messages
    """
    other_data: dict = field(default_factory=lambda: {})
    # indicate which round the mssg is part of
    in_round: bool = False
    end_training: bool = False
    # optional float for aggregation
    aggregation_weight: float | None = None


@dataclass(kw_only=True)
class ParameterMessageBase(Message):
    """
        extend Message
    """
    # if this is the initial parameter message
    is_initial: bool = False


@dataclass(kw_only=True)
class ParameterMessage(ParameterMessageBase):
    """
        extend ParameterMessageBase
    """
    # add a ModelParameter object
    parameter: ModelParameter

    def complete(self, other_parameter: ModelParameter) -> None:
        """
            complete the parameter dict with another parameter dict
        """
        for k, v in other_parameter.items():
            if k not in self.parameter:
                self.parameter[k] = v


@dataclass(kw_only=True)
class DeltaParameterMessage(ParameterMessageBase):
    """
        extend ParameterMessage
    """
    # a ModelParameter obj to store the change in parameters
    delta_parameter: ModelParameter
    # optional ModelParameter objs for old and new parameter
    old_parameter: ModelParameter | None = None
    new_parameter: ModelParameter | None = None

    def restore(self, parameter: ModelParameter) -> ParameterMessage:
        """
            apply the delta to a given parameter
            verify correctness (the new calculated params match the expected ones)
            return: ParameterMessage obj
        """
        # deep copy of the passed parameters
        restored_parameter = copy.deepcopy(parameter)
        if self.old_parameter is not None:
            # if old_parameter is provided, ensure it matches the restored params
            assert len(self.old_parameter) == len(restored_parameter)
            # verify that also all the values match in CPU memery
            for k, v in self.old_parameter.items():
                assert (v.cpu() == restored_parameter[k]).all().item()
        # length check
        assert len(self.delta_parameter) == len(parameter)

        # for each pair in delta_params
        for k, v in self.delta_parameter.items():
            # add the value to the copied params after converting
            restored_parameter[k] = restored_parameter[k].to(dtype=torch.float64) + v
            if self.new_parameter is not None:
                v2 = self.new_parameter[k].to(dtype=torch.float64, device="cpu")
                # element-wise comparison for the values
                # (if the elements are close to each other; within a relative/absolute tolerance)
                if not torch.allclose(v2, restored_parameter[k]):
                    # if not equal, debug info
                    print("key is", k)
                    print("delta is", v)
                    print("result", restored_parameter[k])
                    # the expected value
                    print("gt", v2)
                assert torch.allclose(v2, restored_parameter[k])
        # create the ParameterMessage obj and initialize it  with the restored_params
        msg = ParameterMessage(parameter=restored_parameter)
        # loops over the fields of the dataclass self
        for f in fields(self):
            # set the value of the field name in the new msg
            # to the value retrieved from the current instance
            setattr(msg, f.name, getattr(self, f.name))
        # returned the new created msg MessageParameter
        msg.parameter = restored_parameter
        return msg


@dataclass(kw_only=True)
class FeatureMessage(Message):
    """
        extend Message
    """
    # add an optional tensor for feature
    feature: torch.Tensor | None


@dataclass(kw_only=True)
class MultipleWorkerMessage(Message):
    """
        extend Message
    """
    # add a mapping from worker ID to messages
    worker_data: Mapping[int, Message]


def get_message_size(msg: Message) -> int:
    cnt: int = 0

    def count(data: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        nonlocal cnt
        cnt += data.element_size() * data.numel()
        return data

    recursive_tensor_op(msg, fun=count)
    assert cnt > 0
    return cnt
