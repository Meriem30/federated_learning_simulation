import abc
import contextlib
import copy
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

import torch
import torch.cuda
import torch.utils.data
from other_libs.log import log_debug, log_warning
from torch import Stream

from .data_pipeline.loader import get_dataloader
from .dataset import DatasetCollection, DatasetUtil
from .device import get_device
from .hook import HookCollection
from .hook.config import HookConfig
from .hyper_parameter import HyperParameter, lr_scheduler_step_after_batch
from .metric_visualizers import MetricVisualizer
from .metrics import PerformanceMetric
from .ml_type import EvaluationMode, ExecutorHookPoint, MachineLearningPhase
from .model import ModelEvaluator, ModelUtil


class Executor(HookCollection, abc.ABC):
    def __init__(
        self,
        model_evaluator: ModelEvaluator,
        dataset_collection: DatasetCollection,
        phase: MachineLearningPhase,
        hyper_parameter: HyperParameter,
        hook_config: HookConfig | None = None,
        dataloader_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self._data: dict = {}
        self.__model_evaluator: ModelEvaluator = model_evaluator
        self.__dataset_collection: DatasetCollection = dataset_collection
        self.__phase: MachineLearningPhase = phase
        self.__hyper_parameters: dict = {phase: copy.deepcopy(hyper_parameter)}
        if not hook_config:
            hook_config = HookConfig()
        self.hook_config: HookConfig = copy.deepcopy(hook_config)
        self.__device: None | torch.device = None
        self.__device_fun: Callable = get_device
        self.__dataloader: None | torch.utils.data.DataLoader = None
        self.__dataloader_kwargs: dict = (
            copy.deepcopy(dataloader_kwargs) if dataloader_kwargs is not None else {}
        )
        self.__stream: None | Stream | torch.cpu.Stream = None
        self.__save_dir: None | str = None
        self.__visualizer_prefix: str = ""

    @property
    def dataset_collection(self) -> DatasetCollection:
        """
            Return the dataset collection
        """
        return self.__dataset_collection

    @property
    def mutable_dataset_collection(self) -> DatasetCollection:
        """
            Return the dataset collection &
            Reset the dataloader to None
        """
        self.__dataloader = None
        return self.__dataset_collection

    @property
    def device(self) -> torch.device:
        """
            Return the available device (CPU,CUDA)
            Initialize it if not already
        """
        if self.__device is None:
            self.set_device(self.__device_fun())
        assert self.__device is not None
        return self.__device

    @property
    def device_context(self) -> AbstractContextManager:
        """
            Return the appropriate device manager for  cuda / xpu
            Return null for other devices
        """
        match self.device.type.lower():
            case "cuda":
                return torch.cuda.device(device=self.device)
            case "xpu":
                return torch.xpu.device(device=self.device)
        return contextlib.nullcontext()

    @property
    def stream(self) -> torch.cpu.Stream | Stream:
        """
            Return the stream property for the current device
            Create it if necessary
        """
        if self.__stream is None:
            match self.device.type.lower():
                case "cuda":
                    self.__stream = torch.cuda.Stream(device=self.device)
                case "cpu":
                    self.__stream = torch.cpu.Stream()
                case _:
                    raise RuntimeError(self.device)
        assert self.__stream is not None
        return self.__stream

    @property
    def stream_context(
        self,
    ) -> torch.cuda.StreamContext | torch.cpu.StreamContext:
        """
            Return the appropriate stream context for the current device
        """
        match self.device.type.lower():
            case "cuda":
                return torch.cuda.stream(self.stream)
            case "cpu":
                return torch.cpu.stream(self.stream)
        raise RuntimeError(self.device)

    @property
    def dataloader_kwargs(self) -> dict:
        """
            Return the keyword args for the data loader
        """
        return self.__dataloader_kwargs

    @property
    def hyper_parameter(self) -> HyperParameter:
        """
            Return the hyper params for the current phase
        """
        return self.__hyper_parameters[self.phase]

    def set_phase(self, phase: MachineLearningPhase) -> None:
        """
            Set the current phase
        """
        self.__phase = phase

    def set_hyper_parameter(
        self, hyper_parameter: HyperParameter, phase: MachineLearningPhase | None = None
    ) -> None:
        """
            Set the hyper_params for the current phase
        """
        if phase is None:
            phase = self.phase
        self.__hyper_parameters[phase] = hyper_parameter

    @property
    def performance_metric(self) -> PerformanceMetric:
        """
            Return the performance metric hook
        """
        hook = self.get_hook("performance_metric")
        assert isinstance(hook, PerformanceMetric)
        return hook

    @property
    def phase(self) -> MachineLearningPhase:
        """
            Return the current phase
        """
        return self.__phase

    def exec_hooks(self, hook_point: ExecutorHookPoint, **kwargs: Any) -> None:
        """
            Execute hooks at a given hook point
            passing the executor itself as a keyword arg
        """
        kwargs["executor"] = self
        super().exec_hooks(hook_point=hook_point, **kwargs)

    def set_save_dir(self, save_dir: str) -> None:
        """
            Set the save directory for the executor and any sub_executors
            Update the directory for metric visualizer hook
        """
        self.__save_dir = save_dir
        data_dir = os.path.join(save_dir, "visualizer")
        for hook in self._hook_objs.values():
            if isinstance(hook, MetricVisualizer):
                hook.set_data_dir(data_dir)
        for executor in self._foreach_sub_executor():
            executor.set_save_dir(save_dir)

    def set_visualizer_prefix(self, prefix: str) -> None:
        """
            Set the visualizer prefix for the executor and metric visualizer hook
        """
        self.__visualizer_prefix = prefix
        for hook in self._hook_objs.values():
            if isinstance(hook, MetricVisualizer):
                hook.set_prefix(prefix)

    @property
    def visualizer_prefix(self) -> str:
        return self.__visualizer_prefix

    @property
    def save_dir(self) -> None | str:
        return self.__save_dir

    @property
    def dataset(self):
        """
            Return the dataset of the current phase
        """
        return self.dataset_collection.get_dataset(phase=self.__phase)

    @property
    def dataset_util(self) -> DatasetUtil:
        """
            Return dataset collection (phase) utilities
        """
        return self.dataset_collection.get_dataset_util(phase=self.__phase)

    @property
    def dataset_size(self) -> int:
        """
            Return the dataset size of a given phase
        """
        if "dataset_size" not in self._data:
            self.__refresh_dataset_size()
        return self._data["dataset_size"]

    def __refresh_dataset_size(self) -> None:
        """
            Update the sorted dataset size
        """
        self._data["dataset_size"] = len(self.dataset_util)

    def remove_dataloader_kwargs(self, key: str) -> None:
        """
            Remove a keyword arg from the data loader configuration
            Reset the data loader
        """
        self.__dataloader_kwargs.pop(key, None)
        self.__dataloader = None

    def update_dataloader_kwargs(self, **kwargs: Any) -> None:
        """
            Update the DataLoader args with new keyword
            Reset the DataLoader
        """
        self.__dataloader_kwargs.update(kwargs)
        self.__dataloader = None

    @property
    def dataloader(self) -> torch.utils.data.DataLoader:
        """
            Initialize lazily and return the DataLoader for the current phase
        """
        if self.__dataloader is None:
            self.__dataloader = get_dataloader(
                dc=self.dataset_collection,
                phase=self.__phase,
                hyper_parameter=self.hyper_parameter,
                device=self.device,
                model_evaluator=self.running_model_evaluator,
                **self.__dataloader_kwargs,
            )
        return self.__dataloader

    @property
    def running_model_evaluator(self) -> ModelEvaluator:
        return self.__model_evaluator

    @property
    def model_evaluator(self) -> ModelEvaluator:
        """
            Ensure the current stream is synchronized (all GPUs ops are completed)
            before returning the model evaluator
        """
        self.wait_stream()
        return self.__model_evaluator

    @property
    def model_util(self) -> ModelUtil:
        """
            Provide access to model utilities from the running model evaluator
        """
        return self.running_model_evaluator.model_util

    @property
    def loss_fun(self) -> Callable:
        """
            Return the loss function used by the model evaluator
        """
        return self.running_model_evaluator.loss_fun

    @property
    def model(self) -> torch.nn.Module:
        """
            Return  the PyTorch model being evaluated
        """
        return self.running_model_evaluator.model

    def replace_model(self, fun: Callable) -> None:
        """
        Replace the current model with a new one
        generated by the provided function
        """
        self.running_model_evaluator.set_model(fun(self.model))

    def replace_model_evaluator(self, fun: Callable) -> None:
        """
            Replace the current model evaluator with a new one
            generated by the provided function
        """
        self.wait_stream()
        self.__model_evaluator = fun(self.model_evaluator)

    def _prepare_execution(self) -> None:
        """
            Prepare the executor for execution
            by setting up hooks, save directory, and visualizer prefix
        """
        self.hook_config.set_hooks(self)
        if self.save_dir:
            self.set_save_dir(self.save_dir)
        if self.__visualizer_prefix:
            self.set_visualizer_prefix(self.__visualizer_prefix)
        self.exec_hooks(hook_point=ExecutorHookPoint.BEFORE_EXECUTE)

    def set_device_fun(self, device_fun: Callable) -> None:
        """
            Set the function that determine the device to be used
        """
        self.__device_fun = device_fun

    def set_device(self, device: torch.device) -> None:
        """
            Set the device for the execution and propagate the same to sub-executors
        """
        if self.__device != device:
            # Ensure all operations in CUDA stream are completed before switching device
            self.wait_stream()
            self.__device = device
            log_warning("%s use device %s", str(self.__phase), self.__device)
            self.__stream = None
            self.__dataloader = None

        for executor in self._foreach_sub_executor():
            executor.set_device(device)

    def __getstate__(self) -> dict[str, Any]:
        # prepare the state for pickling (serialize and deserialize) => all the object can be pickled safely
        state = self.__dict__.copy()
        state["_Executor__device"] = None
        state["_Executor__stream"] = None
        state["_Executor__dataloader"] = None
        return state

    def wait_stream(self) -> None:
        # Synchronize the current stream to ensure all operations are completed
        if isinstance(self.__stream, Stream):
            self.__stream.synchronize()
            assert self.__stream.query()

    def set_dataset_collection(self, dc: DatasetCollection) -> None:
        """
            Set the dataset collection, allow updating dynamically for different phases
        """
        self.wait_stream()
        self.__dataset_collection = dc

    def set_model_evaluator(self, model_evaluator: ModelEvaluator) -> None:
        self.wait_stream()
        self.__model_evaluator = model_evaluator

    def load_model(self, model_path: str) -> None:
        """
            Load a model state from the specified path
        """
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )

    def _foreach_sub_executor(self) -> Generator:
        yield from []

    def save_model(self, model_path: str) -> None:
        """
            Save the current model on the specified path
        """
        torch.save(self.model.state_dict(), model_path)

    def offload_from_device(self) -> None:
        """
            Offload the model from the device (ex: GPU to CPU) to free up GPU memory
        """
        self.model_evaluator.offload_from_device()
        torch.cuda.empty_cache()
        for executor in self._foreach_sub_executor():
            executor.offload_from_device()

    def has_optimizer(self) -> bool:
        # Check if an optimizer is present in the data
        return "optimizer" in self._data

    def has_lr_scheduler(self) -> bool:
        # Check if a learning rate scheduler is present in the data
        return "lr_scheduler" in self._data

    def get_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError()

    def get_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """
            Ensure that the learning rate scheduler is properly initialized and available
        """
        if "lr_scheduler" not in self._data:
            self._data["lr_scheduler"] = self.hyper_parameter.get_lr_scheduler(self)
        return self._data["lr_scheduler"]

    def __execute_batch(
        self,
        batch_index: int,
        batch: dict,
        epoch: int,
        evaluation_mode: EvaluationMode,
    ) -> None:
        """
            Execute a single batch of data, including forward pass, loss calculation, and backpropagation
            during any ML phase
        """
        # Skip the batch if the batch size is 1 and Batch Normalization is used
        # because if performs behaves with very small batch sizes
        if (
            evaluation_mode == EvaluationMode.Training
            and self.hyper_parameter.batch_size != 1
            and batch.get("batch_size", None) == 1
            and self.running_model_evaluator.model_util.have_module(
                module_type=torch.nn.BatchNorm2d
            )
        ):
            log_debug("drop last one-sized batch for batch norm")
            return
        # Add additional information to the batch (the data of this batch)
        batch |= {
            "batch_index": batch_index,
            "phase": self.phase,
            "device": self.device,
            "evaluation_mode": evaluation_mode,
            "non_blocking": True,
        }
        # Execute any hooks that should run before the batch is processed
        self.exec_hooks(
            hook_point=ExecutorHookPoint.BEFORE_BATCH,
            epoch=epoch,
            **batch,
        )
        # Forward pass
        evaluation_kwargs = batch
        forward_result: dict = {}
        # Execute a MODEL_FORWARD hook if any => set forward_result in self._data
        if self.has_hook(ExecutorHookPoint.MODEL_FORWARD):
            self.exec_hooks(
                hook_point=ExecutorHookPoint.MODEL_FORWARD,
                evaluation_kwargs=evaluation_kwargs,
            )
            forward_result = self._data.pop("forward_result")
        # If not, directly call running_model_evaluator with the batch data to get forward_result
        else:
            forward_result = self.running_model_evaluator(**evaluation_kwargs)
        # Compute the normalized batch loss, considering the total dataset size
        forward_result["normalized_batch_loss"] = (
            self.running_model_evaluator.get_normalized_batch_loss(
                dataset_size=self.dataset_size, forward_result=forward_result
            )
        )
        # Update the batch dict with the new calculated forward_result
        batch |= forward_result
        # Backward Pass and optimization, if not in test mode
        if evaluation_mode != EvaluationMode.Test:
            if evaluation_mode == EvaluationMode.Training:
                # Retrieve the optimizer
                optimizer = self.get_optimizer()
                # Perform a backward pass + an optimization step
                self.running_model_evaluator.backward_and_step(
                    loss=forward_result["loss"], optimizer=optimizer
                )
            # If validation mode
            else:
                # Perform only a backward pass to compute gradients
                self.running_model_evaluator.backward(
                    loss=forward_result["normalized_batch_loss"]
                )
            # If training mode & step the lr scheduler if it needs to be after each batch
            if evaluation_mode == EvaluationMode.Training:
                lr_scheduler = self.get_lr_scheduler()
                if lr_scheduler_step_after_batch(lr_scheduler):
                    log_debug("adjust lr after batch")
                    lr_scheduler.step()
        # Execute any AFTER_BATCH hooks, passing the necessary data
        self.exec_hooks(
            hook_point=ExecutorHookPoint.AFTER_BATCH,
            epoch=epoch,
            result=forward_result,
            **batch,
        )

    def _execute_epoch(
        self,
        epoch: int,
        evaluation_mode: EvaluationMode,
    ) -> None:
        """
            Manage the training, validation, and test over an entire epoch
        """
        # Execute any hooks that should be run before the epoch starts, passing the current epoch
        self.exec_hooks(
            hook_point=ExecutorHookPoint.BEFORE_EPOCH,
            epoch=epoch,
        )
        # Update the internal representation of the dataset
        self.__refresh_dataset_size()
        # Execute any hooks that should run before fetching the first batch, passing the initial batch index
        self.exec_hooks(hook_point=ExecutorHookPoint.BEFORE_FETCH_BATCH, batch_index=0)
        # Iterate over each batch in the dataloader
        for batch_index, batch in enumerate(self.dataloader):
            # Execute any hooks that should run after fetching the first batch, passing the batch index
            self.exec_hooks(
                hook_point=ExecutorHookPoint.AFTER_FETCH_BATCH,
                batch_index=batch_index,
            )
            # Process the current batch by calling the function
            self.__execute_batch(
                batch_index=batch_index,
                batch=batch,
                epoch=epoch,
                evaluation_mode=evaluation_mode,
            )
            # Execute any hooks that should run before fetching the next batch, passing the next batch index
            self.exec_hooks(
                hook_point=ExecutorHookPoint.BEFORE_FETCH_BATCH,
                batch_index=batch_index + 1,
            )
        # If in training phase
        if evaluation_mode == EvaluationMode.Training:
            # Adjust the lr scheduler if necessary
            lr_scheduler = self.get_lr_scheduler()
            if not lr_scheduler_step_after_batch(lr_scheduler):
                match lr_scheduler:
                    # If the lr scheduler is ReduceLROnPlateau, it steps based on the training loss
                    case torch.optim.lr_scheduler.ReduceLROnPlateau():
                        training_loss = self.performance_metric.get_loss(
                            epoch, to_item=False
                        )
                        log_debug(
                            "call ReduceLROnPlateau for training loss %s",
                            training_loss,
                        )
                        lr_scheduler.step(training_loss)
                    # Otherwise, it steps normally
                    case _:
                        lr_scheduler.step()
        # Execute any hooks that should run after the epoch ends
        self.exec_hooks(
            hook_point=ExecutorHookPoint.AFTER_EPOCH,
            epoch=epoch,
        )


@dataclass(kw_only=True)
class ExecutorConfig:
    """
        This is a configuration class for creating an executor object
    """
    # For storing the configuration for hooks
    hook_config: HookConfig = HookConfig()
    # Additional dict for keyword arguments to be passed to the dataloader
    dataloader_kwargs: dict = field(default_factory=lambda: {})
    # Specifying the type of cache transforms
    cache_transforms: None | str = None

    def create_executor(
        self,
        # Callable => the class of the executor
        cls: Callable,
        # Instance of dataCollection => containing the dataset
        dataset_collection: DatasetCollection,
        # Instance of the model evaluator => in which phase the model is evaluated
        model_evaluator: ModelEvaluator,
        # Instance of Hyperparameter => storing hyperparameter for training
        hyper_parameter: HyperParameter,
    ) -> Any:
        """
            Create and configurate an executor object
        """
        # Add transforms to dataset collection
        dataset_collection.add_transforms(
            model_evaluator=model_evaluator,
        )
        # Cache transforms configuration
        if (
            self.cache_transforms is not None
            and "cache_transforms" not in self.dataloader_kwargs
        ):
            self.dataloader_kwargs["cache_transforms"] = self.cache_transforms
        # Instantiate the executor using the provided cls, with the given configuration
        executor = cls(
            model_evaluator=model_evaluator,
            dataset_collection=dataset_collection,
            hyper_parameter=hyper_parameter,
            hook_config=self.hook_config,
            dataloader_kwargs=self.dataloader_kwargs,
        )
        # Return the newly created and configured executor instance
        return executor

