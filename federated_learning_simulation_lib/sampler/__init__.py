import random
from typing import Any

import torch
from other_libs.algorithm.mapping_op import (
    get_mapping_items_by_key_order, get_mapping_values_by_key_order)
from other_libs.log import log_info
from torch_kit import (ClassificationDatasetCollection,
                               DatasetCollection, DatasetCollectionSplit,
                               MachineLearningPhase, SplitBase)
from torch_kit.dataset import (  # noqa
    get_dataset_collection_sampler, get_dataset_collection_split,
    global_sampler_factory)


class RandomLabelIIDSplit(SplitBase):
    """
        Each worker gets a random subset of labels
        the distribution is IID
    """
    def __init__(
        self,
        dataset_collection: DatasetCollection,
        part_number: int,  # nbr of workers
        sampled_class_number: int,  # nbr of labels each worker will be assigned
    ) -> None:
        super().__init__(dataset_collection=dataset_collection, part_number=part_number)
        assert isinstance(dataset_collection, ClassificationDatasetCollection)
        assert not dataset_collection.is_mutilabel()
        # all unique labels in the dataset
        labels = dataset_collection.get_labels()
        assert sampled_class_number < len(labels)
        # randomly assign a specific nbr of labels to each worker
        assigned_labels = [
            random.sample(list(labels), sampled_class_number)
            for _ in range(part_number)
        ]

        # assure that every label on the dataset is assigned to at least one worker
        assert len(labels) == len(set(sum(assigned_labels, start=[])))

        # phase-wise dataset splitting
        for phase in MachineLearningPhase:
            # Iterate over diff phases: _dataset_indices {('phase', dict())}
            self._dataset_indices[phase] = dict(
                # generate key-value pairs: ('worker_ID', List(dataset indices) )
                enumerate(
                    # self._samplers is a dict: key: 'phase', value: 'sampler_obj' (for actual dataset split)
                    # call split_indices() method of sampler_obj:
                    self._samplers[phase].split_indices(
                        part_proportions=[
                            {label: 1 for label in labels} for labels in assigned_labels
                        ]
                    )
                )
            )

        for worker_id, labels in enumerate(assigned_labels):
            log_info("worker %s has assigned labels %s", worker_id, labels)
            training_set_size = len(
                self._dataset_indices[MachineLearningPhase.Training][worker_id]
            )
            log_info("worker %s has training set size %s", worker_id, training_set_size)


class DirichletSplit(DatasetCollectionSplit):
    def __init__(
        self,
        dataset_collection: DatasetCollection,
        concentration: float | list[dict[Any, float]],
        part_number: int,
    ) -> None:
        if not isinstance(concentration, list):
            assert isinstance(dataset_collection, ClassificationDatasetCollection)
            all_labels = dataset_collection.get_labels()
            concentration = [
                {label: float(concentration) for label in all_labels}
            ] * part_number
        assert isinstance(concentration, list)
        assert len(concentration) == part_number
        part_proportions: list[dict] = []
        for worker_concentration in concentration:
            concentration_tensor = torch.tensor(
                list(get_mapping_values_by_key_order(worker_concentration))
            )
            prob = torch.distributions.dirichlet.Dirichlet(
                concentration_tensor
            ).sample()
            part_proportions.append({})
            for (k, _), label_prob in zip(
                get_mapping_items_by_key_order(worker_concentration), prob
            ):
                part_proportions[-1][k] = label_prob

        super().__init__(
            dataset_collection=dataset_collection, part_proportions=part_proportions
        )


global_sampler_factory.register("random_label_iid", RandomLabelIIDSplit)
global_sampler_factory.register("dirichlet_split", DirichletSplit)