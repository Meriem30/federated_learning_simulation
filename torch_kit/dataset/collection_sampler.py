import copy
import functools
from typing import Any

from ..ml_type import Factory, MachineLearningPhase, TransformType
from .classification_collection import ClassificationDatasetCollection
from .collection import DatasetCollection
from .sampler import DatasetSampler
from .util import DatasetUtil


class Base:
    def __init__(
        self, dataset_collection: DatasetCollection | ClassificationDatasetCollection
    ) -> None:
        self._dc = dataset_collection
        self._samplers: dict[MachineLearningPhase, DatasetSampler] = {}
        self.set_dataset_collection(dataset_collection)

    @property
    def dataset_collection(self) -> DatasetCollection | ClassificationDatasetCollection:
        return self._dc

    def set_dataset_collection(
        self, dataset_collection: DatasetCollection | ClassificationDatasetCollection
    ) -> None:
        """
            Update the dataset collection _dc and
            & Initialize samplers for each phase
        """
        self._dc = dataset_collection
        self._samplers = {
            phase: DatasetSampler(dataset_collection.get_dataset_util(phase))
            for phase in MachineLearningPhase
        }

    def __getstate__(self) -> dict:
        # capture what is normally pickled
        # Prepare the object for pickling
        # by coping its dictionary and setting _sampler to None
        state = self.__dict__.copy()
        state["_samplers"] = None
        # Return the object's state for pickling
        return state


class SamplerBase(Base):
    """
        Maintain indices for the copied dataset, across different ML phases
    """
    def __init__(
        self,
        dataset_collection: DatasetCollection | ClassificationDatasetCollection,
    ) -> None:
        super().__init__(dataset_collection=dataset_collection)
        # Initialize a dict to store dataset indices for all phases
        self._dataset_indices: dict[MachineLearningPhase, set] = {}

    def sample(self) -> DatasetCollection | ClassificationDatasetCollection:
        dc = copy.copy(self._dc)
        assert id(dc.get_dataset(phase=MachineLearningPhase.Training)) == id(
            self._dc.get_dataset(phase=MachineLearningPhase.Training)
        )
        for phase in MachineLearningPhase:
            # Extract the data indices of each phase
            indices = self._dataset_indices[phase]
            # Set a subset for the corresponding phase
            assert indices
            dc.set_subset(phase=phase, indices=indices)
        return dc


class SplitBase(Base):
    """
        Maintain indices for only one part ID of the copied dataset, across different ML phases
    """
    def __init__(
        self,
        dataset_collection: DatasetCollection | ClassificationDatasetCollection,
        part_number: int,
    ) -> None:
        super().__init__(dataset_collection=dataset_collection)
        # The number of parts to split the dataset into
        self._part_number = part_number
        # The dataset indices for all phases
        self._dataset_indices: dict[MachineLearningPhase, dict] = {}

    def sample(
        self, part_id: int
    ) -> DatasetCollection | ClassificationDatasetCollection:
        dc = copy.copy(self._dc)
        assert id(dc.get_dataset(phase=MachineLearningPhase.Training)) == id(
            self._dc.get_dataset(phase=MachineLearningPhase.Training)
        )
        for phase in MachineLearningPhase:
            # Take only one part ID of the dataset indices for each phase
            indices = self._dataset_indices[phase][part_id]
            assert indices
            dc.set_subset(phase=phase, indices=indices)
        return dc


class DatasetCollectionSplit(SplitBase):
    """
        Proportionally split the indices of dataset, across different phases
    """
    def __init__(
        self,
        dataset_collection: DatasetCollection | ClassificationDatasetCollection,
        # Initialize a list of dict, each dict contains the proportions of splitting the dataset
        part_proportions: list[dict[Any, float]],
    ) -> None:
        super().__init__(
            dataset_collection=dataset_collection, part_number=len(part_proportions)
        )
        for phase in MachineLearningPhase:
            # Calculate & store the indices for different PARTS based the provided proportions
            self._dataset_indices[phase] = dict(
                enumerate(
                    self._samplers[phase].split_indices(
                        part_proportions=part_proportions
                    )
                )
            )


class IIDSplit(SplitBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Create a list of equal parts based on the _part_number
        parts: list[float] = [1] * self._part_number
        for phase in MachineLearningPhase:
            # Calculate and store indices by IID splitting the dataset of each phase (equal parts)
            self._dataset_indices[phase] = dict(
                enumerate(self._samplers[phase].iid_split_indices(parts))
            )


class IIDSplitWithFlip(IIDSplit):
    """
        Simulating label noise
        Randomly changes a percentage of labels in the dataset for specific parts
    """
    def __init__(
        self,
        dataset_collection: ClassificationDatasetCollection,
        part_number: int,
        flip_percent: float | list[dict[Any, float]],
    ) -> None:
        super().__init__(dataset_collection=dataset_collection, part_number=part_number)
        self._flipped_indices: dict = {}
        for phase, part_indices in self._dataset_indices.items():
            if phase != MachineLearningPhase.Training:
                continue
            if isinstance(flip_percent, (dict, list)):
                assert len(flip_percent) == part_number
                for part_index, indices in part_indices.items():
                    self._samplers[phase].checked_indices = indices
                    self._flipped_indices |= self._samplers[
                        phase
                    ].randomize_label_by_class(
                        percent=flip_percent[part_index],
                        all_labels=dataset_collection.get_labels(),
                    )
            else:
                assert isinstance(flip_percent, float)
                for indices in part_indices.values():
                    self._flipped_indices |= self._samplers[phase].randomize_label(
                        indices=indices,
                        percent=flip_percent,
                        all_labels=dataset_collection.get_labels(),
                    )
        assert self._flipped_indices

    @classmethod
    def __transform_target(cls, flipped_indices, target, index):
        if index in flipped_indices:
            return DatasetUtil.replace_target(target, {target: flipped_indices[index]})
        return target

    def sample(
        self, part_id: int
    ) -> DatasetCollection | ClassificationDatasetCollection:
        dc = super().sample(part_id=part_id)
        for phase in MachineLearningPhase:
            if phase != MachineLearningPhase.Training:
                continue
            indices = self._dataset_indices[phase][part_id]
            assert indices
            index_list = sorted(indices)
            new_flipped_dict = {}
            for new_idx, idx in enumerate(sorted(index_list)):
                if idx in self._flipped_indices:
                    new_flipped_dict[new_idx] = self._flipped_indices[idx]
            assert new_flipped_dict
            dc.append_transform(
                transform=functools.partial(self.__transform_target, new_flipped_dict),
                key=TransformType.Target,
                phases=[phase],
            )
        return dc


class IIDSplitWithSample(IIDSplit):
    """
        Updates the indices, for each phase and part, by applying the specified sampling probabilities
    """
    def __init__(
        self,
        dataset_collection: ClassificationDatasetCollection,
        part_number: int,
        # Initialize a list of dict representing the sampling probabilities for each part
        sample_probs: list[dict[Any, float]],
    ) -> None:
        assert len(sample_probs) == part_number
        super().__init__(dataset_collection=dataset_collection, part_number=part_number)
        for phase, part_indices in self._dataset_indices.items():
            # Adjust the indices for each part of each phase
            # based on the provided sampling probabilities
            for part_index, indices in part_indices.items():
                self._samplers[phase].checked_indices = indices
                part_indices[part_index] = self._samplers[phase].sample_indices(
                    parts=[sample_probs[part_index]],
                )[0]


class RandomSplit(SplitBase):
    """
        Random equal splitting of the dataset, across different phases
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Initialize a list of equal parts
        parts: list[float] = [1] * self._part_number
        for phase in MachineLearningPhase:
            # Calculate and store indices of random splitting the dataset, for each phase
            self._dataset_indices[phase] = dict(
                enumerate(self._samplers[phase].random_split_indices(parts))
            )


class ProbabilitySampler(SamplerBase):
    """
        Update the dataset indices, for each phase, based on the provided probabilities
    """
    def __init__(
        self,
        dataset_collection: DatasetCollection,
        sample_prob: dict[Any, float],
    ) -> None:
        super().__init__(dataset_collection=dataset_collection)
        for phase in MachineLearningPhase:
            self._dataset_indices[phase] = self._samplers[phase].sample_indices(
                parts=[sample_prob]
            )[0]


global_sampler_factory = Factory()
global_sampler_factory.register("iid", IIDSplit)
global_sampler_factory.register("iid_split_and_flip", IIDSplitWithFlip)
global_sampler_factory.register("iid_split_and_sample", IIDSplitWithSample)
global_sampler_factory.register("random", RandomSplit)
global_sampler_factory.register("prob_sampler", ProbabilitySampler)


def get_dataset_collection_split(
    name: str, dataset_collection: DatasetCollection | ClassificationDatasetCollection, **kwargs
) -> SplitBase:
    """
        Create an instance of the dataset splitting class based on the provided name (strategy)
        :return: a SplitBase subclass
    """
    constructor = global_sampler_factory.get(name.lower())
    if constructor is None:
        raise NotImplementedError(name)
    return constructor(dataset_collection=dataset_collection, **kwargs)


def get_dataset_collection_sampler(
    name: str, dataset_collection: DatasetCollection, **kwargs
) -> SamplerBase:
    """
        Create an instance of the dataset sampling class based on the provided name (operation)
        :return: a SamplerBase subclass
    """
    constructor = global_sampler_factory.get(name.lower())
    if constructor is None:
        raise NotImplementedError(name)
    return constructor(dataset_collection=dataset_collection, **kwargs)