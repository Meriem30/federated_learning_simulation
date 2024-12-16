import os
import pickle
from typing import Any

from other_libs.log import log_debug, log_info
from torch_kit import Inferencer, ModelParameter
from torch_kit.tensor import tensor_to

from ..algorithm.aggregation_algorithm import AggregationAlgorithm
from ..message import (DeltaParameterMessage, Message, MultipleWorkerMessage,
                       ParameterMessage, ParameterMessageBase)
from ..util.model_cache import ModelCache
from .performance_mixin import PerformanceMixin
from .round_selection_mixin import RoundSelectionMixin
from .server import Server


class AggregationServer(Server, PerformanceMixin, RoundSelectionMixin):
    """
        extend multiple classes
        manage the aggregation of model updates
    """
    def __init__(self, algorithm: AggregationAlgorithm, **kwargs: Any) -> None:
        # initialize parent classes
        Server.__init__(self, **kwargs)
        PerformanceMixin.__init__(self)
        RoundSelectionMixin.__init__(self)
        self._round_index: int = 1
        self._compute_stat: bool = True
        # whether to stop straining
        self._stop = False
        self.__model_cache: ModelCache = ModelCache()
        # track which worker have submitted data
        self.__worker_flag: set = set()
        algorithm.set_config(self.config)
        # instance of the aggregation algorithm
        self.__algorithm: AggregationAlgorithm = algorithm
        self._need_init_performance = False

    @property
    def early_stop(self) -> bool:
        return self.config.algorithm_kwargs.get("early_stop", False)

    @property
    def algorithm(self):
        return self.__algorithm

    @property
    def round_index(self) -> int:
        return self._round_index

    def get_tester(self, copy_tester: bool = False) -> Inferencer:
        """
            retrieve the inferencer
            set a visualiser prefix with round nbr
        """
        tester = super().get_tester(copy_tester=copy_tester)
        tester.set_visualizer_prefix(f"round: {self.round_index},")
        return tester

    def __get_init_model(self) -> ModelParameter:
        """
            load initial model parameter,
            from a specified path or by obtaining parameter from the tester
        """
        parameter: ModelParameter = {}
        init_global_model_path = self.config.algorithm_kwargs.get(
            "global_model_path", None
        )
        if init_global_model_path is not None:
            with open(os.path.join(init_global_model_path), "rb") as f:
                parameter = pickle.load(f)
        else:
            parameter = self.get_tester().model_util.get_parameters()
        return parameter

    @property
    def distribute_init_parameters(self) -> bool:
        return self.config.algorithm_kwargs.get("distribute_init_parameters", True)

    def _before_start(self) -> None:
        """
            send initial model params to workers
        """
        if self.distribute_init_parameters:
            self._send_result(
                ParameterMessage(
                    in_round=True, parameter=self.__get_init_model(), is_initial=True
                )
            )

    def _send_result(self, result: Message) -> None:
        """
            send results (of aggregation models) to workers
            depending on the type of mssg
        """
        # pre-send hook
        self._before_send_result(result=result)
        # match to determine how to send
        match result:
            case MultipleWorkerMessage():
                # if contain specific data to each worker
                for worker_id, data in result.worker_data.items():
                    # iterate over and send
                    self._endpoint.send(worker_id=worker_id, data=data)
            case ParameterMessageBase():
                # if ParameterMessage or any subclass, perform worker selection
                selected_workers = self._select_cluster_workers(list(range(self.worker_number)))
                if len(selected_workers) < self.config.worker_number:
                    # increment the worker_round if selected < total
                    worker_round = self.round_index + 1
                    # unless it’s the initial round
                    if result.is_initial:
                        assert self.round_index == 1
                        worker_round = 1
                    log_info(
                        "chosen round %s workers %s", worker_round, selected_workers
                    )
                # if there was a selection, broadcast the mssg
                if selected_workers:
                    self._endpoint.broadcast(data=result, worker_ids=selected_workers)
                # for unselected workers, broadcast None
                unselected_workers = set(range(self.worker_number)) - selected_workers
                if unselected_workers:
                    self._endpoint.broadcast(data=None, worker_ids=unselected_workers)
            case _:
                # if other result type, broadcast to all workers
                self._endpoint.broadcast(data=result)
        # post-send hook
        self._after_send_result(result=result)

    def _server_exit(self) -> None:
        # clean up when the server exits
        self.__algorithm.exit()

    def _process_worker_data(self, worker_id: int, data: Message | None) -> None:
        """
            process the data received from a worker
            update the aggregation
            rend the result back to worker
        """
        log_debug("before processing worker data, ensure that the server set the algo correctly", self.__algorithm)
        assert 0 <= worker_id < self.worker_number
        log_debug("getting data from worker %s", worker_id)
        if data is not None:
            if data.end_training:
                # set the stop flag if the training ended
                self._stop = True
                if not isinstance(data, ParameterMessageBase):
                    return
            # retrieve the current cached server model
            old_parameter = self.__model_cache.parameter
            match data:
                case DeltaParameterMessage():
                    assert old_parameter is not None
                    # add the worker delta to the server old_parameter model
                    # resulting in the new worker model
                    data = data.restore(old_parameter)
                case ParameterMessage():
                    if old_parameter is not None:
                        # complete the parameter using the old ones
                        data.complete(old_parameter)
                    data.parameter = tensor_to(data.parameter, device="cpu")
        # process the worker data using the aggregation algorithm
        self.__algorithm.process_worker_data(worker_id=worker_id, worker_data=data)
        # add the ID of this worker to the set of processed worker
        self.__worker_flag.add(worker_id)
        if len(self.__worker_flag) == self.worker_number:
            # aggregate the data from all worker once the data from them is processed
            result = self._aggregate_worker_data()
            # send the aggregated model to server
            self._send_result(result)
            # clear the set of worker flag
            self.__worker_flag.clear()
        else:
            # log the status
            log_debug(
                "we have %s committed, and we need %s workers,skip",
                len(self.__worker_flag),
                self.worker_number,
            )

    def _aggregate_worker_data(self) -> Any:
        """
            set the old model params for the aggregation algorithm (the current one)
            call the aggregation algorithm
        """
        self.__algorithm.set_old_parameter(self.__model_cache.parameter)
        return self.__algorithm.aggregate_worker_data()

    def _before_send_result(self, result: Message) -> None:
        """
            prior to sending the resulted model to worker
            perform necessary checks
        """
        if not isinstance(result, ParameterMessageBase):
            # only process ParameterMessageBase messages
            return
        assert isinstance(result, ParameterMessage)
        if self._need_init_performance:
            # if initial  performance need to be recorded
            assert self.distribute_init_parameters
        if self._need_init_performance and result.is_initial:
            # record performance of the initial round
            self.record_performance_statistics(result)
        elif self._compute_stat and not result.is_initial and not result.in_round:
            # record performance for non-initial non-round (subround) resulted models
            self.record_performance_statistics(result)
            if not result.end_training and self.early_stop and self.convergent():
                # checks for convergence and potentially stop the training early
                log_info("stop early")
                self._stop = True
                result.end_training = True
        elif result.end_training:
            # record performance for the final model
            self.record_performance_statistics(result)
        # construct the file path to save the model
        model_path = os.path.join(
            self.config.save_dir,
            "aggregated_model",
            f"round_{self.round_index}.pk",
        )
        # cache the model params
        self.__model_cache.cache_parameter(result.parameter, model_path)

    def _after_send_result(self, result: Any) -> None:
        """
            operations after sending the aggregated model to the workers
            increment the round index (if not intermediate round)
            clear data from the algorithm
            if end training: save the cached model
        """
        if not result.in_round:
            self._round_index += 1
        self.__algorithm.clear_worker_data()
        if result.end_training or self._stopped():
            assert self.__model_cache.has_data
            self.__model_cache.save()

    def _stopped(self) -> bool:
        """
            check the maximum round number
        """
        return self.round_index > self.config.round or self._stop




