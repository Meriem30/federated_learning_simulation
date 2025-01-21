from .pneumonia import PneumoniaDataset
from .util import *
from torch_kit.ml_type import DatasetType
from torch_kit.dataset.repository import register_dataset_constructors
from torchvision import transforms

transform = transforms.ToTensor()

def register_medical_constructors() -> None:
    #repositories = [
    #    "PNEUMONIA",
    #]
    #dataset_constructors: dict = {}
    #for repository in repositories:
    #     dataset_constructors.update(
    #        repository,
    #        filter_fun=lambda: PneumoniaDataset,
    #    )
    csv_path = os.path.join(os.path.expanduser("~"), "nvx_data_test.csv"),
    image_dir = os.path.join(os.path.expanduser("~"), "rsna-pneumonia-detection-challenge/stage_2_train_images/")
    #for name, constructor in dataset_constructors.items():
    register_dataset_constructors(DatasetType.Medical,"PNEUMONIA", PneumoniaDataset)
    # Define the constructor_kwargs in the appropriate place

    #lambda: PneumoniaDataset(
    #    csv_path=os.path.join(os.path.expanduser("~"), "nvx_data_test.csv"),
    #    root=os.path.join(os.path.expanduser("~"), "rsna-pneumonia-detection-challenge/stage_2_train_images/")
    #),