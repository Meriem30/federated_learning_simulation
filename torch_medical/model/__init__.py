import functools

from torch_kit import DatasetCollection, DatasetType, Factory
from torch_medical.model.evaluator import MedicalModelEvaluator

from ..dataset.util import MedicalDatasetUtil

from torch_kit.model import create_model
from torch_kit.model.repositary import (get_model_info,
                                        get_torch_hub_model_info)

def get_medical_model_evaluator(model, **kwargs) -> MedicalModelEvaluator:
    return MedicalModelEvaluator(model=model, **kwargs)

def get_medical_model(
    model_constructor_info: dict, dataset_collection: DatasetCollection, **kwargs
) -> dict:
    final_model_kwargs: dict = kwargs
    dataset_util = dataset_collection.get_dataset_util()
    assert isinstance(dataset_util, MedicalDatasetUtil)
    for k in ("input_channels", "channels"):
        if k not in kwargs:
            final_model_kwargs |= {
                k: dataset_util.channel,
            }
    model = create_model(model_constructor_info["constructor"], **final_model_kwargs)
    return {"model": model, "repo": model_constructor_info.get("repo", None)}



def get_medical_model_constructors() -> dict:
    model_info: dict = {}
    github_repos: list = [ # reconsider
        "huggingface/pytorch-image-models:main",
        "pytorch/vision:main",
    ]

    for repo in github_repos:
        model_info |= get_torch_hub_model_info(repo)
    model_info |= get_model_info()[DatasetType.Medical]
    return model_info

