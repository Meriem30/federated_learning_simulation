#from torch_kit import DatasetType
#from torch_kit.data_pipeline import global_data_transform_factory

from .transform import add_vision_extraction, add_vision_transforms


def append_transforms_to_dc(dc, model_evaluator=None) -> None:
    if model_evaluator is None:
        add_vision_extraction(dc=dc)
    else:
        add_vision_transforms(dc=dc, model_evaluator=model_evaluator)
    return


#global_data_transform_factory.register(DatasetType.Vision, append_transforms_to_dc)
