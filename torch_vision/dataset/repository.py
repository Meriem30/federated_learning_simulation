import torch
import torch.utils.data
import torchvision.datasets
from other_libs.reflection import get_class_attrs
from torch_kit import DatasetType
from torch_kit.dataset.repository import register_dataset_constructors


def register_constructors() -> None:
    repositories = [
        torchvision.datasets,
    ]
    dataset_constructors: dict = {}
    for repository in repositories:
        dataset_constructors |= get_class_attrs(
            repository,
            filter_fun=lambda _, v: issubclass(v, torch.utils.data.Dataset),
        )

    for name, constructor in dataset_constructors.items():
        register_dataset_constructors(DatasetType.Vision, name, constructor)
