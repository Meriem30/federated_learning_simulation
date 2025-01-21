from .transform import *

def append_medical_transforms_to_dc(dc, model_evaluator=None) -> None:
    if model_evaluator is None:
        add_medical_vision_extraction(dc=dc)
    else:
        add_medical_vision_transforms(dc=dc, model_evaluator=model_evaluator)

    return

