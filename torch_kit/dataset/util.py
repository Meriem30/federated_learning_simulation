import copy
import functools
from collections.abc import Iterable
from typing import Any, Generator, Type

import torch
import torch.nn.functional
import torch.utils.data

from ..data_pipeline.dataset import get_dataset_size, select_item, subset_dp
from ..data_pipeline.transform import Transforms
from ..factory import Factory
from ..typing import IndicesType, OptionalIndicesType


class DatasetUtil:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        name: None | str = None,
        transforms: Transforms | None = None,
        cache_dir: None | str = None,
    ) -> None:
        self.dataset: torch.utils.data.Dataset = dataset
        self.__len: None | int = None
        self._name: str = name if name else ""
        self._transforms: Transforms | None = transforms
        self._cache_dir = cache_dir

    def __len__(self) -> int:
        if self.__len is None:
            self.__len = get_dataset_size(self.dataset)
        return self.__len

    def decompose(self) -> None | dict:
        return None

    def get_subset(self, indices: IndicesType) -> torch.utils.data.MapDataPipe:
        # Return a DataPipe object containing a subset of dataset based on provided indices
        return subset_dp(self.dataset, indices)

    def get_raw_samples(self, indices: OptionalIndicesType = None) -> Generator:
        """
            Lazily fetch samples from dataset
            If indices is None, extract all raw samples
            The generator yields the retrieved samples and their indices
        """
        return select_item(dataset=self.dataset, indices=indices)

    def get_samples(self, indices: OptionalIndicesType = None) -> Generator:

        """
            Process fetched samples from the dateset
            Apply Extract Data transformations
            The generator yields the processed samples and their indices
        """
        raw_samples = self.get_raw_samples(indices=indices)
        for idx, sample in raw_samples:
            if self._transforms is not None:
                sample = self._transforms.extract_data(sample)
            yield idx, sample

    def get_sample(self, index: int) -> Any:
        for _, sample in self.get_samples(indices=[index]):
            return sample
        return None

    @classmethod
    def __decode_target(cls: Type, target: Any) -> set:
        """
        Handle various type of target data
        :return: a set of decoded target labels
        """
        match target:
            case int() | str():
                # If int or str, convert it to a set
                return {target}
            case torch.Tensor():
                # If a Tensor with only one element
                if target.numel() == 1:
                    # Convert it to a python scalar
                    return {target.item()}
                # one hot vector (values={0,1})
                # find all the indices where the tensor has a value of 1
                if (target <= 1).all().item() and (target >= 0).all().item():
                    # Store these indices in a set
                    return set(target.nonzero().view(-1).tolist())
                raise NotImplementedError(f"Unsupported target {target}")
            case dict():
                if "labels" in target:
                    return cls.__decode_target(target["labels"].tolist())
                # If all dict values are numeric string
                if all(isinstance(s, str) and s.isnumeric() for s in target):
                    return cls.__decode_target({int(s) for s in target})
            case Iterable():
                # If an iterable, convert it to a set
                return set(target)
        raise RuntimeError("can't extract labels from target: " + str(target))

    @classmethod
    def replace_target(cls, old_target: Any, new_target: dict) -> Any:
        """
            Replace the value in an existing target with new values of target (new_target)
        """
        match old_target:
            case int() | str():
                # Convert the target to a set containing the target value
                old_target_value = list(cls.__decode_target(old_target))[0]
                # Find a replacement
                new_target_value = new_target.get(old_target_value, None)
                if new_target_value is None:
                    return old_target
                assert len(new_target_value) == 1
                # If it is found, return the new value cast to the original type
                return type(old_target)(list(new_target_value)[0])
            case torch.Tensor():
                old_target_value = cls.__decode_target(old_target)
                # If the tensor has a single element,
                if old_target.numel() == 1:
                    old_target_value = list(old_target_value)[0]
                    if old_target_value not in new_target:
                        return old_target
                    new_target_tensor = old_target.clone()
                    # Replace it if a corresponding new value is found
                    new_target_tensor = new_target[old_target_value]
                    return new_target_tensor
                # one hot vector
                if (0 <= old_target <= 1).all().item():
                    old_target_value = {
                        new_target.get(old_t, old_t) for old_t in old_target_value
                    }
                    # Replace the indices where the tensor is 1
                    return torch.nn.functional.one_hot(
                        torch.tensor(list(old_target_value)),
                        num_classes=old_target.shape[-1],
                    )
                raise NotImplementedError(f"Unsupported target {old_target}")
            case dict():
                if "labels" in old_target:
                    new_target_dict = copy.deepcopy(old_target)
                    new_target_dict["labels"] = cls.replace_target(
                        new_target_dict["labels"], new_target
                    )
                    return new_target_dict
            case list() | tuple():
                old_target_value = cls.__decode_target(old_target)
                return type(old_target)(
                    new_target.get(old_t, old_t) for old_t in old_target_value
                )

        raise RuntimeError(f"can't convert labels {new_target} for target {old_target}")

    def _get_sample_input(self, index: int, apply_transform: bool = True) -> Any:
        """
        Extract and transform (optionally) the input data of a specific sample
        :return: the extracted [transformed] input data of a sample
        """
        sample = self.get_sample(index)
        sample_input = sample["input"]
        if apply_transform:
            assert self._transforms is not None
            sample_input = self._transforms.transform_input(
                sample_input, apply_random=False
            )
        return sample_input

    def get_batch_labels(
        self, indices: OptionalIndicesType = None
    ) -> Generator[tuple[int, set], None, None]:
        """
        Extract and transform the labels from a batch of samples
        :return: a generator that yields tuples (sample_idx, decoded (set) target labels)
        """
        for idx, sample in self.get_samples(indices):
            target = sample["target"]
            if self._transforms is not None:
                target = self._transforms.transform_target(target)
            yield idx, DatasetUtil.__decode_target(target)

    def get_sample_label(self, index: int) -> set:
        """
        Retrieve the label of a single sample (the value without the idx)
        :return: a set of label(s) of a specific sample
        """
        return list(self.get_batch_labels(indices=[index]))[0][1]

    def get_labels(self) -> set:
        """
        Retrieve labels for all samples, then extract only the unique label values
        :return: a set of unique labels
        """
        return set().union(*tuple(set(labels) for _, labels in self.get_batch_labels()))

    def get_original_dataset(self) -> torch.utils.data.Dataset:
        """
        Retrieve the original dataset from the current dataset obj
        Access and get the value of "original_dataset" key
        """
        return self.dataset[0].get("original_dataset", self.dataset)

    def get_label_names(self) -> dict:
        """
        Extract and return all label names of a dataset indexed
        :return: a dict: key (int) & value (label name: str)
        """
        original_dataset = self.get_original_dataset()
        if (
            hasattr(original_dataset, "classes")
            and original_dataset.classes
            and isinstance(original_dataset.classes[0], str)
        ):
            return dict(enumerate(original_dataset.classes))

        # If the dataset does not have the attribute 'classes'
        # The next aggregate the label names from the dataset samples
        def get_label_name(container: set, idx_and_labels: tuple[int, set]) -> set:
            container.update(idx_and_labels[1])
            return container
        # Accumulate all unique label names
        label_names: set = functools.reduce(
            get_label_name, self.get_batch_labels(), set()
        )
        if label_names:
            return dict(enumerate(sorted(label_names)))
        raise RuntimeError("no label names detected")


global_dataset_util_factor = Factory()