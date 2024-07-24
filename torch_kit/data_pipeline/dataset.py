from typing import Any, Generator

import torch
import torch.utils.data
import torch.utils.data.datapipes
import torch.utils.data.dataset

from ..typing import OptionalIndicesType


def get_dataset_size(dataset: Any) -> int:
    """
    Determine the dataset size, using structural pattern matching (Python 3.10)
    to handle different dataset structures
    :param dataset: any
    :return: int : dataset size
    """
    match dataset:
        # if dataset is a dictionary with a key 0
        # that maps to another dictionary containing a key "mask"
        case {0: {"mask": mask}}:
            # return the sum of mask ({0: {"mask": torch.tensor([1, 0, 1])}})
            return mask.sum()
        # if dataset is a list containing a single dictionary with a key "mask"
        case [{"mask": mask}]:
            # return the number of elements in mask (the sum)
            return mask.sum()
    # if none of the above, but dataset has len attribute, return it
    if hasattr(dataset, "__len__"):
        return len(dataset)
    match dataset:
        # if data is IterableDataset() to be loaded on the fly, rather than fully loaded in memory
        case torch.utils.data.IterableDataset():
            # count the number of its elements
            return sum(1 for _ in dataset)
    raise NotImplementedError(dataset)


class KeyPipe(torch.utils.data.MapDataPipe):
    """
    MapDataPipe is map-style dataset (Pytorch class for processing dataset)
    when iterated over, returns tuples where the first element is the index of the item
    and the second element is the item itself.
    """
    def __init__(self, dp: Any) -> None:
        super().__init__()
        self.__dp = dp

    def __getitem__(self, index) -> tuple:
        item = self.__dp[index]
        # this transforms each item in the dataset into a tuple
        return (index, item)

    def __len__(self) -> int:
        return len(self.__dp)


def __add_index_to_map_item(item) -> dict:
    """
    :param item: takes one item (tuple/list), unpacks it into two variables,
    :return: a dictionary with keys "index" and "data"
    """
    key, value = item[0], item[1]
    return {"index": key, "data": value}


def dataset_with_indices(
    dataset: torch.utils.data.Dataset,
) -> torch.utils.data.Dataset:
    """
     Converts a dataset to include indices for each item.
    :param dataset: a dataset of type torch.utils.data.Dataset
    :return: a modified dataset of the same type
    """
    old_dataset = dataset
    # Initialize a pattern matching to handle different types of datasets
    match dataset:
        case list():
            return dataset
        # Convert the iterable dataset into a data pipe
        case torch.utils.data.IterableDataset():
            dataset = torch.utils.data.datapipes.iter.IterableWrapper(dataset)
    # Initialize another block of pattern matching to further process the new dataset
    match dataset:
        # If it is an instance of IterDataPipe call enumerate to add indices to each item
        case torch.utils.data.IterDataPipe():
            dataset = dataset.enumerate()
        # For all other types, it applies a Mapper
        case _:
            # KeyPipe(dataset): wraps the dataset with KeyPipe, paring each item with each index
            # Mapper (,): convert each (index, item) pair into a dictionary with keys "index" and "data"
            dataset = torch.utils.data.datapipes.map.Mapper(
                KeyPipe(dataset), __add_index_to_map_item
            )
    # Add "original_dataset" attribute to store the old version of dataset
    assert not hasattr(dataset, "original_dataset")
    setattr(dataset, "original_dataset", old_dataset)
    # Return the new dataset with the indices added
    return dataset


def select_item(dataset: Any, indices: OptionalIndicesType = None) -> Generator:
    """
        This generator selects items form dataset as provided in indices
        :param dataset: Any dataset type
        :param indices: A list or set of indices to select from the dataset, if None: all item are selected
        :return:
    """
    if indices is not None:
        indices = set(indices)  # for faster membership tracking
    match dataset:
        case torch.utils.data.IterableDataset():
            # Convert the dataset to an iterator to sequentially access its element
            iterator = iter(dataset)
            for idx, item in enumerate(iterator):
                # Loop over the iterator and yield the pair (idx, item) if the idx is in indices
                if indices is None or idx in indices:
                    # Instead of returning a single value and terminating,
                    # a generator can yield multiple values, one at a time (resuming where it stopped).
                    yield idx, item
                    # Remove each idx once it's yielded
                    if indices is not None:
                        indices.remove(idx)
        # Default case: return all the dataset element
        case _:
            if indices is None:
                indices = list(range(get_dataset_size(dataset)))
            # Iterate over indices and yield (idx, data) for each index in the set
            for idx in indices:
                yield idx, dataset[idx]


def subset_dp(
    dataset: torch.utils.data.Dataset, indices: OptionalIndicesType = None
) -> torch.utils.data.MapDataPipe:
    # original_dataset = getattr(dataset, "dataset", None)
    # if has_hugging_face:
    #     match original_dataset:
    #         case hugging_face_datasets.arrow_dataset.Dataset():
    #             pass

    # select_item(dataset, indices): returns a generator that yields (index, item) pairs from the dataset
    # dict(select_item(dataset, indices)): Converts the generator into a dictionary
    # list(dict(select_item(dataset, indices)).values()): Extracts the values from the dict
    # and converts them into a list
    # torch.utils.data.datapipes.map.SequenceWrapper: Wraps this list into a SequenceWrapper (a type of MapDataPipe)
    return torch.utils.data.datapipes.map.SequenceWrapper(
        list(dict(select_item(dataset, indices)).values()), deepcopy=False
    )
