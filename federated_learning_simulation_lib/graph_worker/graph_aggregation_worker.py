import os
import torch
from federated_learning_simulation_lib.worker.aggregation_worker import AggregationWorker, Worker
from federated_learning_simulation_lib.graph_worker import GraphWorker, ClientState
from other_libs.log import log_debug, log_info
from torch_kit import (ExecutorHookPoint, MachineLearningPhase,  # noqa
                               ModelParameter, StopExecutingException, # noqa
                               tensor_to)
from ..message import (DeltaParameterMessage, Message, ParameterMessage,
                       ParameterMessageBase)
from ..util import ModelCache, load_parameters
# add nodeSelectionMixin to be inherited here


class GraphAggregationWorker(GraphWorker, AggregationWorker):
    def __init__(self, **kwargs):
        # explicitly cal parent __init__ func
        GraphWorker.__init__(self, **kwargs)
        AggregationWorker.__init__(self, **kwargs)
        self._communicate_node_state: bool = True
        self.__choose_model_by_validation: bool | None = None
        self.__model_cache: ModelCache = ModelCache()

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
        # ADDED to handle Graphs
        log_debug("communicate node state to server with sent data: ", self._communicate_node_state)
        if self._communicate_node_state:
            other_data["node_state"] = (
                self._get_client_state(self.worker_id)
            )
            log_info("worker %s node_state added to other data", self.worker_id)
            assert other_data["node_state"] is not None
        # create ParameterMessage or DeltaParameterMessage
        # based on the _send_parameter_diff
        message: ParameterMessageBase = ParameterMessage(
            aggregation_weight=self.trainer.dataset_size,
            parameter=parameter,
            other_data=other_data,
        )
        if self._send_parameter_diff:
            assert self.__model_cache.has_data
            message = DeltaParameterMessage(
                aggregation_weight=self.trainer.dataset_size,
                other_data=other_data,
                # old_parameter=self.__model_cache.parameter,
                # new_parameter=parameter,
                delta_parameter=self.__model_cache.get_parameter_diff(parameter),
            )
        # discard the model cache if necessary
        if not self._keep_model_cache:
            self.__model_cache.discard()
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
                    self.__model_cache.cache_parameter(result.parameter, path=model_path)
            case DeltaParameterMessage():
                assert self.__model_cache.has_data
                self.__model_cache.add_parameter_diff(
                    result.delta_parameter, path=model_path
                )
                parameter = self.__model_cache.parameter
            case _:
                raise NotImplementedError()
        # ADDED to handle graphs
        # check if family assignment has changed
        other_data = result.other_data
        log_debug("load family assignments data from server to check changes")
        new_family = self._load_family_assignment_from_server(other_data)
        # change it in the client state
        if new_family != 0 & new_family != self.state.family:
            log_info("change family assignment for worker %s. old:  %s => new: %s",
                     self.worker_id,
                     self.state.family,
                     new_family)
            self.state.set_family(new_family)
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

    def _load_family_assignment_from_server(self, data: dict) -> int:
        assert data is not None
        if "family_assignment" in data.keys():
            family = data["family_assignment"][self.worker_id]
            return family
        else:
            return 0
