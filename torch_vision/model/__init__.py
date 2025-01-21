import functools

from torch_kit import DatasetCollection, DatasetType, Factory
from torch_kit.model import create_model
from torch_kit.model.repositary import (get_model_info,
                                        get_torch_hub_model_info)

from ..dataset.util import VisionDatasetUtil
from .evaluator import VisionModelEvaluator


def get_model_evaluator(model, **kwargs) -> VisionModelEvaluator:
    return VisionModelEvaluator(model=model, **kwargs)


#global_model_evaluator_factory.register(DatasetType.Vision, VisionModelEvaluator)


def get_model(
    model_constructor_info: dict, dataset_collection: DatasetCollection, **kwargs
) -> dict:
    final_model_kwargs: dict = kwargs
    dataset_util = dataset_collection.get_dataset_util()
    assert isinstance(dataset_util, VisionDatasetUtil)
    for k in ("input_channels", "channels"):
        if k not in kwargs:
            final_model_kwargs |= {
                k: dataset_util.channel,
            }
    model = create_model(model_constructor_info["constructor"], **final_model_kwargs)
    return {"model": model, "repo": model_constructor_info.get("repo", None)}


def get_model_constructors() -> dict:
    model_info: dict = {}
    github_repos: list = [
        "huggingface/pytorch-image-models:main",
        "pytorch/vision:main",
    ]

    for repo in github_repos:
        model_info |= get_torch_hub_model_info(repo)
    model_info |= get_model_info()[DatasetType.Vision]
    return model_info


#if DatasetType.Vision not in global_model_factory:
#    global_model_factory[DatasetType.Vision] = Factory()
#for name, constructor_info in get_model_constructors().items():
#    global_model_factory[DatasetType.Vision].register(
#        name, functools.partial(get_model, constructor_info)
#    )

#global_model_evaluator_factory.register(DatasetType.Vision, get_model_evaluator)
