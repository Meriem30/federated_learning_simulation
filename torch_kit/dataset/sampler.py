import copy
import functools
import random
from typing import Any, Callable

from other_libs.log import log_warning, log_debug

from .util import DatasetUtil
"""
    Define various methods for sampling and splitting indices of a dataset 
"""


class DatasetSampler:
    def __init__(self, dataset_util: DatasetUtil) -> None:
        self.__dataset_util: DatasetUtil = dataset_util
        self._excluded_indices: set = set()
        self.checked_indices: set | None = None

    def set_excluded_indices(self, excluded_indices):
        self._excluded_indices = excluded_indices

    # cached properties to avoid recalculation
    @functools.cached_property
    def sample_label_dict(self) -> dict[int, set]:
        """
            Compute and return a dict: key (idx) & value ( set of sample labels)
        """
        return dict(self.__dataset_util.get_batch_labels())

    @functools.cached_property
    def label_sample_dict(self) -> dict[Any, set]:
        """
            Compute and return a dict: key (label) & value (set of indices where this label exists)
        """
        label_sample_dict: dict = {}
        for index, labels in self.sample_label_dict.items():
            for label in labels:
                if label not in label_sample_dict:
                    label_sample_dict[label] = {index}
                else:
                    label_sample_dict[label].add(index)
        return label_sample_dict

    def get_subsets(self, index_list: list) -> list:
        """
            Retrieve a list of dataset subsets, based on the input list
            :input: a list, each element is a list of indices
            :return: a list, each element is torch.utils.data.MapDataPipe object
        """
        log_debug("constructing subsets ..")
        return [
            self.__dataset_util.get_subset(indices=indices) for indices in index_list
        ]

    def split_indices(
        self,
        part_proportions: list[dict[Any, float]],
        labels: list | None = None,
        is_iid: bool = False,
    ) -> list[set]:
        """
            Partition the indices of a dataset into subsets
            based on the proportions provided for each label
            or split it IID
            :return: sub_index_list:a list of set,
                    each contains indices corresponding to a particular proportion
        """
        assert part_proportions
        # Create one set for each partition item
        sub_index_list: list[set] = [set()] * len(part_proportions)

        def __split_per_label(label, indices):
            """
                Handle splitting of indices for each label
                :input: the current label being processed
                        the set of indices of dataset samples associated with this label
                :return: the indices of the current label
            """
            nonlocal part_proportions
            # Extract all partitions (floats) of a specific label in a list
            label_part = [
                part_proportion.get(label, 0) for part_proportion in part_proportions
            ]
            # If the list of partition is empty, return an empty set
            if sum(label_part) == 0:
                return set()
            # If not, return a list of lists, each inner list indicates:
            # the indices of one subset: the data sample holding a specific label for one client
            part_index_lists = self.__split_index_list(
                label_part, list(indices), is_iid=is_iid
            )
            # Update sub_index_list to contain the indices for the current label
            for i, part_index_list in enumerate(part_index_lists):
                sub_index_list[i] = sub_index_list[i] | set(part_index_list)
            return indices
        # Invoke split_per_label
        self.__check_sample_by_label(
            callback=__split_per_label,
            labels=labels,
        )
        return sub_index_list

    def sample_indices(
        self,
        parts: list[dict[Any, float]],
        labels: list | None = None,
    ) -> list[set]:
        assert parts

        sub_index_list: list[set] = [set()] * len(parts)

        def __sample_per_label(label: Any, indices: set):
            nonlocal parts
            label_part = [part.get(label, 0) for part in parts]
            if sum(label_part) == 0:
                return set()
            part_index_lists = self.__split_index_list(
                label_part, list(indices), is_iid=False
            )
            sampled_indices = set()
            for i, part_index_list in enumerate(part_index_lists):
                part_index_set = set(
                    part_index_list[: int(label_part[i] * len(indices))]
                )
                sampled_indices.update(part_index_set)
                sub_index_list[i] = sub_index_list[i] | part_index_set
            return sampled_indices

        self.__check_sample_by_label(
            callback=__sample_per_label,
            labels=labels,
        )
        return sub_index_list

    def iid_split_indices(
        self,
        parts: list[float],
        labels: list | None = None,
    ) -> list[set]:
        assert parts

        if labels is None:
            labels = list(self.label_sample_dict.keys())
        return self.split_indices(
            part_proportions=[{label: part for label in labels} for part in parts],
            labels=labels,
        )

    def random_split_indices(
        self,
        parts: list[float],
        labels: list | None = None,
    ) -> list[list]:
        collected_indices = set()

        def __collect(label: Any, indices: set):
            collected_indices.update(indices)
            return indices

        self.__check_sample_by_label(callback=__collect, labels=labels)
        return self.__split_index_list(parts, list(collected_indices), is_iid=False)

    def iid_split(self, parts: list[float], labels: list | None = None) -> list:
        return self.get_subsets(self.iid_split_indices(parts, labels=labels))

    def iid_sample_indices(self, percent: float) -> set:
        labels = list(self.label_sample_dict.keys())
        return self.sample_indices(
            [{label: percent for label in labels}], labels=labels
        )[0]

    def randomize_label(
        self, indices: list, percent: float, all_labels: set | None = None
    ) -> dict[int, set]:
        randomized_label_map: dict[int, set] = {}
        if all_labels is None:
            all_labels = set(self.label_sample_dict.keys())

        flipped_indices = random.sample(list(indices), k=int(len(indices) * percent))
        for index in flipped_indices:
            other_labels = list(all_labels - self.sample_label_dict[index])
            randomized_label_map[index] = set(
                random.sample(
                    other_labels,
                    min(len(other_labels), len(self.sample_label_dict[index])),
                )
            )

        return randomized_label_map

    def randomize_label_by_class(
        self, percent: float | dict[Any, float], all_labels: set | None = None, **kwargs
    ) -> dict[int, set]:
        randomized_label_map: dict[int, set] = {}

        def __randomize(label: set, indices):
            nonlocal randomized_label_map
            nonlocal percent
            if not indices:
                return indices

            if isinstance(percent, dict):
                new_percent = percent[label]
            else:
                new_percent = percent
            assert isinstance(new_percent, float | int)

            randomized_label_map |= self.randomize_label(
                indices=indices,
                percent=new_percent,
                all_labels=all_labels,
            )

            return indices

        self.__check_sample_by_label(callback=__randomize, **kwargs)

        return randomized_label_map

    @classmethod
    def __split_index_list(
        cls, parts: list[float], index_list: list, is_iid: bool
    ) -> list[list]:
        """
            Split the indices of a specific label into subsets for different clients
            based on the proportion list (each item is the prop of that label for one client)
        """
        assert index_list
        if len(parts) == 1:
            assert parts[0] != 0
            return [index_list]
        random.shuffle(index_list)
        part_lens: list[int] = []
        first_assert = True
        index_num = len(index_list)

        for part in parts:
            assert part > 0
            part_len = int(index_num * part / sum(parts))
            if part_len == 0 and is_iid:
                if sum(part_lens, start=0) < index_num:
                    part_len = 1
                elif first_assert:
                    first_assert = False
                    log_warning(
                        "has zero part when splitting list, %s %s",
                        index_num,
                        parts,
                    )
            part_lens.append(part_len)
        for _ in range(index_num - sum(part_lens)):
            idx = random.choice(range(len(part_lens)))
            part_lens[idx] = part_lens[idx] + 1
        assert sum(part_lens) == index_num
        part_indices = []
        for part_len in part_lens:
            if part_len != 0:
                part_indices.append(index_list[0:part_len])
                index_list = index_list[part_len:]
            else:
                part_indices.append([])
        return part_indices

    def __check_sample_by_label(
        self,
        callback: Callable,  # a function to be specified in run time to process indices
        labels: list | None = None,
    ) -> None:
        """
            Calculate the set of indices to be processed, for each label from labels list
            The callback function is applied to the returned indices
        """
        excluded_indices = copy.copy(self._excluded_indices)

        if labels is None:
            labels = list(self.label_sample_dict.keys())
        for label in labels:
            indices = self.label_sample_dict[label] - excluded_indices
            if self.checked_indices is not None:
                indices = indices.intersection(self.checked_indices)
            if indices:
                resulting_indices = callback(label=label, indices=indices)
                excluded_indices.update(resulting_indices)
