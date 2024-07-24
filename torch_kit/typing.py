from typing import Iterable, TypeAlias

import torch

"""
    Create shorthand for complex used type definitions
"""

OptionalTensor: TypeAlias = torch.Tensor | None  # torch.tensor or None
TensorDict: TypeAlias = dict[str, torch.Tensor]  # dict: key (str) & value (torch.tensor)
OptionalTensorDict: TypeAlias = TensorDict | None  # TensorDict or None
# list of tuples, each tuple contains a str and a torch.nn.Module
BlockType: TypeAlias = list[tuple[str, torch.nn.Module]]
IndicesType: TypeAlias = Iterable[int]  # Iterable of integers
OptionalIndicesType: TypeAlias = IndicesType | None  # Iterable of ints or None
ModelGradient: TypeAlias = TensorDict  # Represent a TensorDict
SampleTensors: TypeAlias = dict[int, torch.Tensor]  # dict: key (int) & value (torch.Tensor)
# dict: [key (int) & value (dict:{key (str) & value (torch.Tensor)})]
SampleGradients: TypeAlias = dict[int, ModelGradient]
ModelParameter: TypeAlias = TensorDict  # Represent a TensorDict
