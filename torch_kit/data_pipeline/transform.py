import copy
from collections.abc import Iterable
from typing import Any, Callable, Self


import torch.utils.data
from other_libs.log import log_debug
from torch.utils.data import default_collate

from ..ml_type import TransformType
from ..tensor import tensor_to
from .common import default_data_extraction
from .dataset import select_item


class Transforms:
    def __init__(self) -> None:
        self.__transforms: dict = {}
        # Create an empty dictionary to store different types of transformations
        # and append a default data extraction transformation
        self.append(key=TransformType.ExtractData, transform=default_data_extraction)

    def has_transform(self) -> bool:
        # Check if any transformations have been added
        return any(self.__transforms.values())

    def clear(self, key: TransformType) -> None:
        # Clear all transformers of a specified type
        self.__transforms.pop(key, None)

    def append(self, key: TransformType, transform: Callable) -> None:
        # Add a new transformation to the list of transformations for a specified type
        if key not in self.__transforms:
            self.__transforms[key] = []
        self.__transforms[key].append(transform)

    def get(self, key: TransformType) -> list:
        # Retrieve the list of transformations of a specific type
        return self.__transforms.get(key, [])

    def get_input_transforms_in_order(self, include_random: bool = True) -> list:
        # Combine text, input, and optionally random input transformations into
        # a single list to be returned
        res = self.get(TransformType.InputText) + self.get(TransformType.Input)
        if include_random:
            res += self.get(TransformType.RandomInput)
        return res

    def get_target_transforms_in_order(self) -> list:
        # Return the list of target transformations
        return self.get(TransformType.Target)

    def transform_text(self, text):
        # Iterate and apply the list of text transformations sequentially to the input
        # Return the new text
        for f in self.get(TransformType.InputText):
            text = f(text)
        return text

    def extract_data(self, data):
        # Iterate and apply the list of data extraction transformations sequentially
        # Return the new data
        for f in self.get(TransformType.ExtractData):
            data = f(data)
        return data

    def transform_input(self, sample_input: Any, apply_random: bool = True) -> Any:
        # Iterate and apply input transformations in order, individually to a sample input
        # Return the new input
        for f in self.get_input_transforms_in_order(include_random=apply_random):
            sample_input = f(sample_input)
        return sample_input

    def transform_inputs(self, inputs: Iterable) -> Any:
        # Iterate and apply the list of batch transformations to a batch of input (an iterable)
        # Return the new input batch
        batch_transforms = self.get(TransformType.InputBatch)
        if not batch_transforms:
            batch_transforms.append(default_collate)
        for f in batch_transforms:
            inputs = f(inputs)
        return inputs

    def transform_target(self, target: Any, index: int | None = None) -> Any:
        # Iterate and apply target transformations, individually to a target
        # Return the new target
        for f in self.get(TransformType.Target):
            target = f(target, index)
        return target

    def transform_targets(self, targets: Iterable) -> Any:
        # Iterate and apply batch transformations to a batch of targets
        # Return the new batch of targets
        batch_transforms = self.get(TransformType.TargetBatch)
        if not batch_transforms:
            batch_transforms.append(default_collate)
        for f in batch_transforms:
            targets = f(targets)
        return targets

    def collate_batch(self, batch: Iterable) -> dict:
        # Collate a batch of data by applying input and target transformations
        inputs = []
        targets = []
        other_info: list = []
        # Iterate and apply the input and target transformations to each sample input data
        for data in batch:
            # Create a shallow copy of the extracted data with shared references of nested objs
            # Duplicate the original object but do not create copies for nested objects
            data = copy.copy(self.extract_data(data))
            sample_input = self.transform_input(data.pop("input"))
            inputs.append(sample_input)
            targets.append(
                self.transform_target(
                    target=data.pop("target"), index=data.get("index", None)
                )
            )
            # Append additional information if exists
            other_info.append(data)
        batch_size = len(inputs)
        inputs = self.transform_inputs(inputs)
        targets = self.transform_targets(targets)
        # Combines the new input, target, and batch size in a res dict
        res = {
            "batch_size": batch_size,
            "inputs": inputs,
            "targets": targets,
        }
        if other_info:
            # default_collate takes a list of dicts and merges them into a single batch
            tmp: dict = default_collate(other_info)
            assert isinstance(tmp, dict)
            if "index" in tmp:
                # Rename 'index' field to 'sample_indices'
                tmp["sample_indices"] = tmp.pop("index")
            # Use union operator to merge 'tmp' dict to 'res'
            res |= tmp
        return res

    def cache_transforms(
        self, dataset: torch.utils.data.Dataset, device: torch.device
    ) -> tuple[dict, Self]:
        """
            Caches transformations and applies them to the data,
            Optionally, transfers data to the specific device
            :return: the transformed dataset and the modified 'new_transforms'
        """
        log_debug("cache dataset to device: %s", device)
        # Initialize a dict for the transformed data
        transformed_dataset: dict = {}
        # Iterate abd apply input and target data
        for k, item in select_item(dataset):
            item = self.extract_data(item)
            item["input"] = self.transform_input(item["input"], apply_random=False)
            item["target"] = self.transform_target(item["target"], index=k)
            # Transfer the transformed data
            if device is not None:
                item["input"] = tensor_to(
                    item["input"], device=device, non_blocking=True
                )
                item["target"] = tensor_to(
                    item["target"], device=device, non_blocking=True
                )
            # Add indices to each item
            transformed_dataset[k] = item
        # Create a deep copy for the current 'Transforms' object
        # Create a new objects and recursively copy all nested objects
        new_transforms = copy.deepcopy(self)
        # Remove existing Extract Data transformations
        new_transforms.clear(TransformType.ExtractData)
        # Reset the Extract Data transformations to be reused (clean slate)
        new_transforms.append(
            key=TransformType.ExtractData, transform=default_data_extraction
        )
        # Remove other transformations
        new_transforms.clear(TransformType.InputText)
        new_transforms.clear(TransformType.Input)
        new_transforms.clear(TransformType.Target)
        return transformed_dataset, new_transforms

    def __str__(self) -> str:
        desc = []
        # Iterate over a tuple containing enum members of TransformType
        for k in (
            TransformType.ExtractData,
            TransformType.InputText,
            TransformType.Input,
            TransformType.RandomInput,
            TransformType.InputBatch,
            TransformType.Target,
        ):
            # retrieve the list of transformations of each transformType
            transforms = self.__transforms.get(k, [])
            if transforms:
                # If not empty, append the str representation of each transformer to the desc
                desc.append(str(k) + "=>")
                # Iterate over the extracted list of transformations of a specific type
                # Convert it to a str and append it to the desc
                for t in transforms:
                    desc.append(str(t))
        # Join all the strings in the desc list with newline character and return it
        return "\n".join(desc)

