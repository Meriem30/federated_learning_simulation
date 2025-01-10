import torch
import torch.utils.data
import torchvision.datasets
from other_libs.reflection import get_class_attrs
from torch_kit import DatasetType
from torch_kit.dataset.repository import register_dataset_constructors
from .pneumonia_data import PneumoniaDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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
    register_dataset_constructors(DatasetType.Vision,"PNEUMONIA",
                                  lambda: PneumoniaDataset(
                                      csv_file=f"C:/Users/Home/pytorch_dataset/PNEUMONIA/img__data.csv",
                                      image_dir="C:/Users/Home/pytorch_dataset/PNEUMONIA/stage_2_train_images/",
                                      transform=transform,
                                  ))

