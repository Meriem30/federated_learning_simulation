import copy
import os
import threading
from typing import Any, Callable, Generator, Iterable

import torch
import torch.utils.data
from other_libs.fs.ssd import is_ssd
from other_libs.log import log_debug, log_warning
from other_libs.storage import get_cached_data
from other_libs.system_info import OSType, get_operating_system_type

from ..data_pipeline import (Transforms, append_transforms_to_dc,
                             dataset_with_indices)
from ..ml_type import DatasetType, MachineLearningPhase, TransformType
from .sampler import DatasetSampler
from .util import DatasetUtil, global_dataset_util_factor

"""
    Encapsulate various operation related to dataset used in various machine learning phases
"""


class DatasetCollection:
    def __init__(
        self,
        datasets: dict[MachineLearningPhase, Any], # MODIFIED to handle pneumonia # torch.utils.data.Dataset
        dataset_type: DatasetType | None = None,
        name: str | None = None,
        dataset_kwargs: dict | None = None,
        add_index: bool = True,
    ) -> None:
        self.__name: str = "" if name is None else name
        # Initialize a dict mapping a 'MachineLearningPhase' to 'torch.utils.data.Dataset' obj
        self.__datasets: dict[MachineLearningPhase, Any] = datasets # torch.utils.data.Dataset
        # Add indices to dataset if indicated
        if add_index:
            for k, v in self.__datasets.items():
                self.__datasets[k] = dataset_with_indices(v)
        self.__dataset_type: DatasetType | None = dataset_type
        # Initialize a dict mapping 'MachineLearningPhase' to a list of corresponding transforms
        self.__transforms: dict[MachineLearningPhase, Transforms] = {}
        for phase in MachineLearningPhase:
            self.__transforms[phase] = Transforms()
        self.__dataset_kwargs: dict = (
            copy.deepcopy(dataset_kwargs) if dataset_kwargs else {}
        )
        # Attach the appropriate data transformations to each dataset in DatasetCollection instance
        append_transforms_to_dc(self)

    def __copy__(self):
        # Create a deep copy of the entire DatasetCollection instance
        new_obj = copy.deepcopy(self)
        # Replace the deeply copied '__datasets' attribute with a shallow copy
        # (avoid duplicate datasets)
        new_obj.__datasets = self.__datasets.copy()
        return new_obj

    @property
    def name(self) -> str:
        return self.__name

    @property
    def dataset_kwargs(self) -> dict:
        return self.__dataset_kwargs

    @property
    def dataset_type(self) -> DatasetType:
        assert self.__dataset_type is not None
        return self.__dataset_type

    def foreach_dataset(self) -> Generator:
        yield from self.__datasets.values()

    def has_dataset(self, phase: MachineLearningPhase) -> bool:
        return phase in self.__datasets

    def transform_dataset(
        self, phase: MachineLearningPhase, transformer: Callable
    ) -> None:
        """
            Apply a transformation to a dataset for a specific phase
        """
        dataset = self.get_dataset(phase)
        dataset_util = self.get_dataset_util(phase)
        self.__datasets[phase] = transformer(dataset, dataset_util, phase)

    def transform_all_datasets(self, transformer: Callable) -> None:
        """
            Apply transformations to all datasets of all phases
        """
        for phase in self.__datasets:
            self.transform_dataset(phase, transformer)

    def set_subset(self, phase: MachineLearningPhase, indices: set) -> None:
        """
            Set a subset of a dataset for a specific phase based on indices
        """
        self.transform_dataset(
            phase=phase,
            transformer=lambda _, dataset_util, *__: dataset_util.get_subset(indices),
        )

    def remove_dataset(self, phase: MachineLearningPhase) -> None:
        """
            Remove a dataset for a specific phase
        """
        log_debug("remove dataset %s", phase)
        self.__datasets.pop(phase, None)

    def get_dataset(self, phase: MachineLearningPhase) -> torch.utils.data.Dataset:
        """
            Retrieve the dataset for a specific phase
        """
        return self.__datasets[phase]

    def get_transforms(self, phase: MachineLearningPhase) -> Transforms:
        """
            Retrieve the transforms for a specific phase
        """
        return self.__transforms[phase]

    def get_dataset_util(
        self, phase: MachineLearningPhase = MachineLearningPhase.Test
    ) -> DatasetUtil:
        """
            Retrieve a DatasetUtil instance for a specific phase
        """
        return global_dataset_util_factor.get(self.dataset_type)(
            dataset=self.get_dataset(phase),
            transforms=self.__transforms[phase],
            name=self.name,
            cache_dir=self._get_dataset_cache_dir(),
        )

    def get_original_dataset(
        self, phase: MachineLearningPhase
    ) -> torch.utils.data.Dataset:
        """
            Get the original dataset for a specific phase
        """
        return self.get_dataset_util(phase=phase).get_original_dataset()

    def foreach_transform(self) -> Generator:
        """
            Yield each transform in the collection instance
        """
        yield from self.__transforms.items()

    def append_transform(
        self, transform: Callable, key: TransformType, phases: None | Iterable = None
    ) -> None:
        """
            Append a transform to the collection of a specific ML phase
        """
        for phase in MachineLearningPhase:
            if phases is not None and phase not in phases:
                continue
            self.__transforms[phase].append(key, transform)

    # Define a class-level variable to create a path for storing dataset
    # (expanding the user's home directory)
    _dataset_root_dir: str = os.path.join(os.path.expanduser("~"), "pytorch_dataset")
    # Define a class-level reentrant lock (acquire the lock multiple time)
    lock = threading.RLock()

    @classmethod
    def get_dataset_root_dir(cls) -> str:
        with cls.lock:
            return os.getenv("pytorch_dataset_root_dir", cls._dataset_root_dir)

    @classmethod
    def set_dataset_root_dir(cls, root_dir: str) -> None:
        with cls.lock:
            cls._dataset_root_dir = root_dir

    @classmethod
    def get_dataset_dir(cls, name: str) -> str:
        dataset_dir = os.path.join(cls.get_dataset_root_dir(), name)
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        if get_operating_system_type() != OSType.Windows and not is_ssd(dataset_dir):
            log_warning("dataset %s is not on a SSD disk: %s", name, dataset_dir)
        return dataset_dir

    def _get_dataset_cache_dir(self) -> str:
        cache_dir = os.path.join(self.get_dataset_dir(self.name), "cache")
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def is_classification_dataset(self) -> bool:
        return True
        # if self.dataset_type == DatasetType.Graph:
        #     return True
        # labels = list(
        #     self.get_dataset_util(phase=MachineLearningPhase.Training).get_batch_labels(
        #         indices=[0]
        #     )
        # )[1]
        # if len(labels) != 1:
        #     return False
        # match next(iter(labels)):
        #     case int():
        #         return True
        # return False

    def iid_split(
        self,
        from_phase: MachineLearningPhase,
        parts: dict[MachineLearningPhase, float],
    ) -> None:
        """
            Split the dataset into IID split
        """
        assert self.has_dataset(phase=from_phase)
        assert parts
        log_debug("split %s dataset for %s", from_phase, self.name)
        part_list = list(parts.items())
        #print("part_list", part_list)

        sampler = DatasetSampler(dataset_util=self.get_dataset_util(phase=from_phase))
        datasets = sampler.iid_split([part for (_, part) in part_list])
        for idx, (phase, _) in enumerate(part_list):
            self.__datasets[phase] = datasets[idx]

    def add_transforms(self, model_evaluator: Any) -> None:
        """
            Add transforms to the DatasetCollection instance
        """
        append_transforms_to_dc(dc=self, model_evaluator=model_evaluator)

    def get_cached_data(self, file: str, computation_fun: Callable) -> Any:
        """
            Retrieve data from cache or compute it is not already cached
        """
        with DatasetCollection.lock:
            assert self.name is not None
            cache_dir = self._get_dataset_cache_dir()
            return get_cached_data(os.path.join(cache_dir, file), computation_fun)

