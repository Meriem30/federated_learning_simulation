import os

import torch
import torch.utils.data
from other_libs.log import log_debug

from ..concurrency import TorchProcessContext
from ..dataset.collection import DatasetCollection
from ..factory import Factory
from ..hyper_parameter import HyperParameter
from ..ml_type import DatasetType, MachineLearningPhase
from ..model import ModelEvaluator

# Instance of Factory class, used to create dataloader objects based on the dataset type
global_dataloader_factory = Factory()


def __prepare_dataloader_kwargs(
    dc: DatasetCollection,
    phase: MachineLearningPhase,
    hyper_parameter: HyperParameter,
    device: torch.device,
    cache_transforms: str | None = None, # Specifies whether to cache transforms and where (CPU or device)
    **kwargs,
) -> dict:
    """
        Prepares the keyword arguments for creating a PyTorch DataLoader
        based on the dataset, phase, and various other parameters
        :return: kwargs for data loading
    """
    dataset = dc.get_dataset(phase=phase)
    transforms = dc.get_transforms(phase=phase)
    data_in_cpu: bool = True
    if dc.dataset_type == DatasetType.Graph:
        cache_transforms = None
    transformed_dataset: dict | torch.utils.data.Dataset = dataset
    match cache_transforms:
        case "cpu":
            transformed_dataset, transforms = transforms.cache_transforms(
                dataset=dataset, device=torch.device("cpu")
            )
        case "device":
            data_in_cpu = False
            assert device is not None
            transformed_dataset, transforms = transforms.cache_transforms(
                dataset=dataset, device=device
            )
        case None:
            pass
        case _:
            raise RuntimeError(cache_transforms)
    use_process: bool = "USE_THREAD_DATALOADER" not in os.environ
    if dc.dataset_type == DatasetType.Graph:
        # don't pass large graphs around processes
        use_process = False
    if cache_transforms is not None:
        use_process = False
    use_process = False
    if use_process:
        kwargs["prefetch_factor"] = 2
        kwargs["num_workers"] = 1
        if not data_in_cpu:
            kwargs["multiprocessing_context"] = TorchProcessContext().get_ctx()
        kwargs["persistent_workers"] = True
    else:
        log_debug("use threads")
        kwargs["num_workers"] = 0
        kwargs["prefetch_factor"] = None
        kwargs["persistent_workers"] = False
    kwargs["batch_size"] = hyper_parameter.batch_size
    kwargs["shuffle"] = phase == MachineLearningPhase.Training
    kwargs["pin_memory"] = False
    kwargs["collate_fn"] = transforms.collate_batch
    kwargs["dataset"] = transformed_dataset
    return kwargs


def get_dataloader(
    dc: DatasetCollection,
    phase: MachineLearningPhase,
    model_evaluator: ModelEvaluator,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Creates and returns a Pytorch DataLoader for the specified dataset and phase
    """
    # calls the function that prepares the keyword arguments for the dataloader
    dataloader_kwargs = __prepare_dataloader_kwargs(
        dc=dc,
        phase=phase,
        **kwargs,
    )
    # Uses 'global_dataloader_factory' to get a constructor for the dataset type
    constructor = global_dataloader_factory.get(dc.dataset_type)
    # If the constructor is found, it uses it to create a DataLoader, then return it
    if constructor is not None:
        return constructor(
            dc=dc, model_evaluator=model_evaluator, phase=phase, **dataloader_kwargs
        )
    # If no constructor is found, it falls back to creating a default Pytorch DataLoader
    return torch.utils.data.DataLoader(**dataloader_kwargs)
