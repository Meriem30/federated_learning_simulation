import copy
import functools
from typing import Self

from ..ml_type import MachineLearningPhase
from .collection import DatasetCollection


class ClassificationDatasetCollection:
    def __init__(self, dc: DatasetCollection) -> None:
        self.__dc = dc

    @property
    def dc(self) -> DatasetCollection:
        # Access the dataset instance dc of DatasetCollection type
        return self.__dc

    def __copy__(self) -> Self:
        # Create a shallow copy
        # of ClassificationDatasetCollection class instance (using the dc copy)
        return type(self)(dc=copy.copy(self.dc))

    def __getattr__(self, name):
        if name == "dc":
            raise AttributeError()
        return getattr(self.dc, name)

    @functools.cached_property
    def label_number(self) -> int:
        # Calculate the number of unique labels
        return len(self.get_labels())

    def get_labels(self, use_cache: bool = True) -> set:
        """
            Extract (or fetch from cache) the set of all unique labels in dataset collection
            Aggregate all labels from all ML phase (Training,Validation,Test)
        """
        def computation_fun() -> set:
            if self.name.lower() == "imagenet":
                # For imagenet, return a set of predefined a set of labels (0-999)
                return set(range(1000))
            labels = set()
            for phase in (
                MachineLearningPhase.Training,
                MachineLearningPhase.Validation,
                MachineLearningPhase.Test,
            ):
                # If dataset exists for the current phase, retrieve the existing labels
                # & add them to 'labels' set (unique ones)
                if self.dc.has_dataset(phase):
                    labels |= self.dc.get_dataset_util(phase).get_labels()
            return labels

        if not use_cache:
            return computation_fun()

        return self.get_cached_data("labels.pk", computation_fun)

    def is_mutilabel(self) -> bool:
        # Check if dataset samples are multi-label
        def computation_fun() -> bool:
            if self.name.lower() == "imagenet":
                return False
            for _, labels in self.__get_first_dataset_util().get_batch_labels():
                if len(labels) > 1:
                    return True
            return False

        if not self.has_dataset(MachineLearningPhase.Training):
            return computation_fun()

        return self.get_cached_data("is_mutilabel.pk", computation_fun)

    def get_label_names(self) -> dict:
        """
            Extract (or fetch from cache) a dict of dataset label names: key (label indices) & value (label names)
            Only those used in during the training phase
        """
        def computation_fun():
            label_names = self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_label_names()
            if not label_names:
                raise NotImplementedError(f"failed to get label names for {self.name}")
            return label_names

        return self.get_cached_data("label_names.pk", computation_fun)

    def __get_first_dataset_util(self):
        for phase in (
            MachineLearningPhase.Training,
            MachineLearningPhase.Validation,
            MachineLearningPhase.Test,
        ):
            if self.dc.has_dataset(phase):
                return self.dc.get_dataset_util(phase)
        raise RuntimeError("no dataset")
