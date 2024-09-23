from torch_kit import DatasetType
from torch_kit.dataset.util import global_dataset_util_factor

from .repository import register_constructors
from .util import VisionDatasetUtil

global_dataset_util_factor.register(DatasetType.Vision, VisionDatasetUtil)
register_constructors()
