import functools
from torch_kit import DatasetType, Factory
from torch_kit.data_pipeline import global_data_transform_factory
from torch_kit.dataset.util import global_dataset_util_factor
from torch_kit.model import global_model_evaluator_factory, global_model_factory


from torch_vision.data_pipeline import append_transforms_to_dc
from torch_vision.dataset import register_constructors, VisionDatasetUtil
from torch_vision.model import get_model_evaluator, get_model_constructors, get_model, VisionModelEvaluator


def register_vision_transforms():
    global_data_transform_factory.register(DatasetType.Vision, append_transforms_to_dc)


def register_vision_datasets():
    global_dataset_util_factor.register(DatasetType.Vision, VisionDatasetUtil)
    register_constructors()


def register_vision_models():
    global_model_evaluator_factory.register(DatasetType.Vision, VisionModelEvaluator)
    if DatasetType.Vision not in global_model_factory:
        global_model_factory[DatasetType.Vision] = Factory()
    for name, constructor_info in get_model_constructors().items():
        global_model_factory[DatasetType.Vision].register(
            name, functools.partial(get_model, constructor_info)
        )
    global_model_evaluator_factory.register(DatasetType.Vision, get_model_evaluator)

