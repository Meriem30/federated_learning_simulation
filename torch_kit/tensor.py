import dataclasses
import functools
import pickle
from collections.abc import Iterable
from typing import Any, Callable

import torch
from other_libs.algorithm.mapping_op import (
    get_mapping_items_by_key_order, get_mapping_values_by_key_order)

from .typing import TensorDict


def cat_tensors_to_vector(tensors: Iterable) -> torch.Tensor:
    """
        Concatenates a list of tensors into a single vector
    """
    return torch.cat([t.view(-1) for t in tensors])


def cat_tensor_dict(tensor_dict: dict) -> torch.Tensor:
    """
        Concatenates the values of a dictionary of tensors into a single vector
    """
    return cat_tensors_to_vector(get_mapping_values_by_key_order(tensor_dict))


def decompose_like_tensor_dict(tensor_dict: dict, tensor: torch.Tensor) -> dict:
    """
        Decompose a single tensor (tensor) back into a dict with the same shapes (keys) as the original dict (tensor_dict)
    """
    result = {}
    bias = 0
    for key, component in get_mapping_items_by_key_order(tensor_dict):
        param_element_num = torch.prod(component.shape).item()
        result[key] = tensor[bias: bias + param_element_num].view(*component.shape)
        bias += param_element_num
    assert bias == tensor.shape[0]
    return result


def decompose_tensor_to_list(shapes: list, tensor: torch.Tensor) -> list:
    """
        Decompose a single tensor back into a list of tensors with specified shape
    """
    result = []
    bias = 0
    for shape in shapes:
        param_element_num = torch.prod(torch.tensor(shape, dtype=torch.long)).item()
        result.append(tensor[bias: bias + param_element_num].view(*shape))
        bias += param_element_num
    assert bias == tensor.shape[0]
    return result


def get_tensor_serialization_size(data):
    """
        Get the size of the serialized tensor object
    """
    return len(pickle.dumps(data))


class __RecursiveCheckPoint:
    """
        A helper class used for recursive operations on tensors
    """
    def __init__(self, data: Any) -> None:
        self.data: Any = data


def recursive_tensor_op(data: Any, fun: Callable, **kwargs: Any) -> Any:
    """
        Recursively apply a function to a nested structure containing tensors
    """
    match data:
        case __RecursiveCheckPoint():
            if kwargs.pop("__check_recursive_point", False):
                return fun(data.data, **kwargs)
            return data
        case torch.Tensor():
            if kwargs.get("__check_recursive_point", False):
                return data
            return fun(data, **kwargs)
        case list():
            return [recursive_tensor_op(element, fun, **kwargs) for element in data]
        case tuple():
            return tuple(
                recursive_tensor_op(element, fun, **kwargs) for element in data
            )
        case dict():
            return {k: recursive_tensor_op(v, fun, **kwargs) for k, v in data.items()}
        case functools.partial():
            return functools.partial(
                data.func,
                *recursive_tensor_op(data.args, fun, **kwargs),
                **recursive_tensor_op(data.keywords, fun, **kwargs),
            )
    try:
        for field in dataclasses.fields(data):
            setattr(
                data,
                field.name,
                recursive_tensor_op(getattr(data, field.name), fun, **kwargs),
            )
        return data
    # pylint: disable=broad-exception-caught
    except BaseException:
        pass
    if hasattr(data, "data"):
        data.data = recursive_tensor_op(data.data, fun, **kwargs)
    return data


def tensor_to(
    data: Any, non_blocking: bool = True, check_slowdown: bool = False, **kwargs: Any
) -> Any:
    """
        Move tensors to a specified device with options for non-blocking transfers and slowdown checks
    """
    def fun(data, check_slowdown, **kwargs):
        if check_slowdown:
            device = kwargs.get("device", None)
            non_blocking = kwargs.get("non_blocking", True)
            if (
                str(data.device) == "cpu"
                and device is not None
                and str(device) != str(data.device)
            ):
                # if not data.is_pinned():
                #     raise RuntimeError("tensor is not pinned")
                if not non_blocking:
                    raise RuntimeError(
                        "copy is blocking",
                    )
            else:
                if device is not None and not kwargs.get("non_blocking", True):
                    raise RuntimeError(
                        "device to device copy is blocking",
                    )
            assert str(device) != str(data.device)
        return data.to(**kwargs)

    return recursive_tensor_op(
        data, fun, non_blocking=non_blocking, check_slowdown=check_slowdown, **kwargs
    )


def tensor_clone(data: Any, detach: bool = True) -> Any:
    """
        Clone tensors within a nested structure and optionally detache them
    """
    def fun(data, detach):
        new_data = data.clone()
        if detach:
            new_data = new_data.detach()
        return new_data

    return recursive_tensor_op(data, fun, detach=detach)


def assemble_tensors(data: Any) -> tuple[torch.Tensor | None, Any]:
    """
        Assemble all tensors within a nested structure into a single concatenated tensor
    """
    tensor_list = []
    offset = 0

    def fun(data: torch.Tensor) -> __RecursiveCheckPoint:
        nonlocal offset
        if data.numel() == 0:
            return __RecursiveCheckPoint(data=(data,))
        shape = list(data.shape)
        if not shape:
            return __RecursiveCheckPoint(data=(data.item(),))
        if data.dtype != torch.float32:
            return __RecursiveCheckPoint(data=(data,))
        old_offset = offset
        tensor_list.append(data.view(-1))
        offset += data.numel()
        return __RecursiveCheckPoint(data=(shape, old_offset))

    res = recursive_tensor_op(data, fun)
    if offset == 0:
        assert not tensor_list
        return None, res
    assert tensor_list
    return cat_tensors_to_vector(tensor_list), res


def disassemble_tensor(
    concatenated_tensor: torch.Tensor, data: Any, clone: bool = True
) -> Any:
    """
        Disassemble a concatenated tensor back into the original nested structure
    """
    def fun(data: torch.Tensor) -> Any:
        if len(data) == 1:
            return data[0]
        shape, offset = data
        tensor = concatenated_tensor[
            offset: offset + torch.prod(torch.tensor(shape, dtype=torch.long)).item()
        ].view(*shape)
        if clone:
            tensor = tensor.clone()
        return tensor

    if concatenated_tensor is None:
        return data

    return recursive_tensor_op(data, fun, __check_recursive_point=True)


def dot_product(a: TensorDict | torch.Tensor, b: TensorDict | torch.Tensor) -> float:
    """
        Compute the dot product between two tensors or dictionaries of tensors
    """
    match b:
        case dict():
            assert isinstance(a, dict)
            product = 0
            for k, v in b.items():
                if v.device == a[k].device:
                    product += v.view(-1).dot(a[k].view(-1)).item()
                else:
                    product += v.cpu().view(-1).dot(a[k].cpu().view(-1)).item()
            return product
        case _:
            assert isinstance(a, torch.Tensor)
            a = a.view(-1)
            b = b.view(-1)
            if a.device == b.device:
                return a.dot(b).item()
            return a.cpu().dot(b.cpu()).item()
        