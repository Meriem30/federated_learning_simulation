from ..factory import Factory
from .transform import *
from .dataset import *

global_data_transform_factory = Factory()


def append_transforms_to_dc(dc, model_evaluator=None) -> None:
    # Get the transformation constructor based the dataset type of the 'DatasetCollection' instance
    constructor = global_data_transform_factory.get(dc.dataset_type)
    assert constructor is not None
    # Call the constructor passing the dc instance
    return constructor(dc=dc, model_evaluator=model_evaluator)