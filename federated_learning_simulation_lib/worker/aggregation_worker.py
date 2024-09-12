import os
from typing import Any

import torch
from other_libs.log import log_debug, log_info
from torch_kit import (ExecutorHookPoint, MachineLearningPhase,  # noqa
                               ModelParameter, StopExecutingException, # noqa
                               tensor_to)
from torch_kit.hook.keep_model import KeepModelHook

from ..message import (DeltaParameterMessage, Message, ParameterMessage,
                       ParameterMessageBase)
from ..util import ModelCache, load_parameters
from .client import ClientMixin
from .worker import Worker


class AggregationWorker(Worker, ClientMixin):
    """
        inherit from both classes
    """
    def __init__(self, **kwargs: Any) -> None:
        Worker.__init__(self, **kwargs)
        ClientMixin.__init__(self)
        # when to perform aggregation
        self._aggregation_time: ExecutorHookPoint = ExecutorHookPoint.AFTER_EXECUTE
        self._reuse_learning_rate: bool = False
        self.__choose_model_by_validation: bool | None = None
        # whether to send parameter differences instead of full parameters
        self._send_parameter_diff: bool = False
        # whether to keep a cache for model params
        self._keep_model_cache: bool = False
        # whether to send loss information
        self._send_loss: bool = False
        self._model_cache: ModelCache = ModelCache()
        # optional custom function for loading model
        self._model_loading_fun = None

    @property
    def model_cache(self) -> ModelCache:
        """
            return the model cache instance
        """
        return self._model_cache

    @property
    def distribute_init_parameters(self) -> bool:
        """
            determine if initial parameters should be distributed based on the configuration
        """
        return self.config.algorithm_kwargs.get("distribute_init_parameters", True)

    def _before_training(self) -> None:
        # invokes the parent method
        super()._before_training()
        # remove test data set from the trainer
        self.trainer.dataset_collection.remove_dataset(phase=MachineLearningPhase.Test)
        # determine if the model should be chosen by validation
        choose_model_by_validation = self.__choose_model_by_validation
        # default it to True in epoch > 1
        if choose_model_by_validation is None:
            choose_model_by_validation = self.config.hyper_parameter_config.epoch > 1
        # enable or disable it
        if choose_model_by_validation:
            self.enable_choosing_model_by_validation()
        else:
            self.disable_choosing_model_by_validation()
        if not self.__choose_model_by_validation and not self.config.use_validation:
            # if validation is not used, remove the validation dataset from the trainer
            self.trainer.dataset_collection.remove_dataset(
                phase=MachineLearningPhase.Validation
            )
        # free up resources
        self.trainer.offload_from_device()
        # load initial parameters from the server
        if self.distribute_init_parameters:
            self.__get_result_from_server()
            if self._stopped():
                return
        # register the aggregation hook to be executed
        self._register_aggregation()

    def _register_aggregation(self) -> None:
        log_debug("use aggregation_time %s", self._aggregation_time)
        # remove any existing aggregation hook from the trainer
        self.trainer.remove_named_hook(name="aggregation")

        def __aggregation_impl(**kwargs) -> None:
            # if not stopped, perform the aggregation with the sent data
            if not self._stopped():
                self._aggregation(sent_data=self._get_sent_data(), **kwargs)
        # append the aggregation implementation as a hook to be executed
        self.trainer.append_named_hook(
            self._aggregation_time,
            "aggregation",
            __aggregation_impl,
        )

    def _aggregation(self, sent_data: Message, **kwargs: Any) -> None:
        # send data
        self._send_data_to_server(sent_data)
        self._offload_from_device()
        # retrieve the result
        self.__get_result_from_server()

    def enable_choosing_model_by_validation(self) -> None:
        """
            enable the selection of the best model by validation
        """
        self.__choose_model_by_validation = True
        hook = KeepModelHook()
        hook.keep_best_model = True
        assert self.trainer.dataset_collection.has_dataset(
            phase=MachineLearningPhase.Validation
        )
        self.trainer.remove_hook("keep_model_hook")
        self.trainer.append_hook(hook, "keep_model_hook")

    def disable_choosing_model_by_validation(self) -> None:
        """
            disable the selection of the best model based on validation
        """
        self.__choose_model_by_validation = False
        self.trainer.remove_hook("keep_model_hook")

    @property
    def best_model_hook(self) -> KeepModelHook | None:
        """
            provide access to the KeepModlHook if exists
        """
        if not self.trainer.has_hook_obj("keep_model_hook"):
            return None
        hook = self.trainer.get_hook("keep_model_hook")
        assert isinstance(hook, KeepModelHook)
        return hook

    def _get_sent_data(self) -> ParameterMessageBase:
        """
            prepare the data to be sent to the server
        """
        # select the best model (& epoch) on validation if enabled
        if self.__choose_model_by_validation:
            assert self.best_model_hook is not None
            parameter = self.best_model_hook.best_model["parameter"]
            best_epoch = self.best_model_hook.best_model["epoch"]
            log_debug("use best model best_epoch %s", best_epoch)
        # otherwise use the current model
        else:
            parameter = self.trainer.model_util.get_parameters()
            best_epoch = self.trainer.hyper_parameter.epoch
            log_debug(
                "use best model best_epoch %s acc %s parameter size %s",
                best_epoch,
                self.trainer.performance_metric.get_epoch_metric(
                    best_epoch, "accuracy"
                ),
                len(parameter),
            )
        # convert the model params to the CPU
        parameter = tensor_to(parameter, device="cpu", dtype=torch.float64)
        # prepare other data
        other_data = {}
        # add training loss to other_data id necessary
        if self._send_loss:
            other_data["training_loss"] = (
                self.trainer.performance_metric.get_epoch_metric(best_epoch, "loss")
            )
            assert other_data["training_loss"] is not None
        # create ParameterMessage or DeltaParameterMessage
        # based on the _send_parameter_diff
        message: ParameterMessageBase = ParameterMessage(
            aggregation_weight=self.trainer.dataset_size,
            parameter=parameter,
            other_data=other_data,
        )
        if self._send_parameter_diff:
            assert self._model_cache.has_data
            message = DeltaParameterMessage(
                aggregation_weight=self.trainer.dataset_size,
                other_data=other_data,
                # old_parameter=self._model_cache.parameter,
                # new_parameter=parameter,
                delta_parameter=self._model_cache.get_parameter_diff(parameter),
            )
        # discard the model cache if necessary
        if not self._keep_model_cache:
            self._model_cache.discard()
        # returned the prepared message
        return message

    def _load_result_from_server(self, result: Message) -> None:
        """
            load the result received from the server and apply it to the model
        """
        # define the path to save the model
        model_path = os.path.join(
            self.save_dir, "aggregated_model", f"round_{self.round_index}.pk"
        )
        # initialize the parameter
        parameter: ModelParameter = {}
        # check the result message type
        match result:
            case ParameterMessage():
                parameter = result.parameter
                # cache the model
                if self._keep_model_cache or self._send_parameter_diff:
                    self._model_cache.cache_parameter(result.parameter, path=model_path)
            case DeltaParameterMessage():
                assert self._model_cache.has_data
                self._model_cache.add_parameter_diff(
                    result.delta_parameter, path=model_path
                )
                parameter = self._model_cache.parameter
            case _:
                raise NotImplementedError()
        # load params into the trainer
        load_parameters(
            trainer=self.trainer,
            parameter=parameter,
            reuse_learning_rate=self._reuse_learning_rate,
            loading_fun=self._model_loading_fun,
        )
        # stop execution if end_training is set in the result
        if result.end_training:
            self._force_stop = True
            raise StopExecutingException()

    def _offload_from_device(self, in_round: bool = False) -> None:
        # save or discard the model cache
        if self._model_cache.has_data:
            if self._keep_model_cache:
                self._model_cache.save()
            else:
                self._model_cache.discard()
        # clear the best_model_hook (not in the middle of a round)
        if self.best_model_hook is not None:
            assert not in_round
            self.best_model_hook.clear()
        # call the super class
        super()._offload_from_device()

    def __get_result_from_server(self) -> None:
        """
            retrieve and process the result data from the server
        """
        # continue to request data from the server until valid data is received
        while True:
            result = super()._get_data_from_server()
            log_debug("get result from server %s", type(result))
            if result is None:
                log_info("skip round %s", self.round_index)
                self._send_data_to_server(None)
                self._round_index += 1
                if self._stopped():
                    return
                continue
            # load the result from the server if not none
            self._load_result_from_server(result=result)
            # exit
            break
        return
