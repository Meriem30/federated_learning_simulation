import copy
import os
import pickle
import time
from typing import Any

from other_libs.log import log_debug, log_info
from other_libs.topology import ServerEndpoint
from torch_kit import Inferencer, MachineLearningPhase, ModelParameter

from ..executor import Executor, ExecutorContext
from ..message import Message, ParameterMessage
from .round_selection_mixin import RoundSelectionMixin
from .node_selection_mixin import NodeSelectionMixin


class Server(Executor, RoundSelectionMixin, NodeSelectionMixin):
    """
        extend both Executor and RoundSelectionMixin classes
        handle communication, selection, evaluation, processing
    """

    def __init__(self, task_id: int, endpoint: ServerEndpoint, **kwargs: Any) -> None:
        # initialize the server with a given task_id
        name: str = "server"
        if task_id is not None:
            name = f"server of {task_id}"
        # call both parent constructors
        super().__init__(**kwargs, name=name)
        RoundSelectionMixin.__init__(self)
        # initialize the server with an endpoint (attribute)
        self._endpoint: ServerEndpoint = endpoint
        # for model evaluation
        self.__tester: Inferencer | None = None

    @property
    def worker_number(self) -> int:
        """
            return the number of workers configured with the task
        """
        return self.config.worker_number

    @property
    def selected_worker_number(self) -> int:
        if not self.config.algorithm_kwargs.get("node_sample_percent", 1.0):
            return int(round(self.config.algorithm_kwargs.get("node_sample_percent") * self.worker_number))
        elif not self.config.algorithm_kwargs.get("random_client_number", True):
            return self.config.algorithm_kwargs.get("random_client_number")
        return self.worker_number

    def get_tester(self, copy_tester: bool = False) -> Inferencer:
        """
            retrieve or create a tester (Inferencer)
        """
        if self.__tester is not None and not copy_tester:
            return self.__tester
        # if copy_tester, create a new one
        tester = self.config.create_inferencer(
            phase=MachineLearningPhase.Test, inherent_device=False
        )
        # remove the training and the validation dataset from the tester
        tester.dataset_collection.remove_dataset(phase=MachineLearningPhase.Training)
        tester.dataset_collection.remove_dataset(phase=MachineLearningPhase.Validation)
        tester.hook_config.summarize_executor = False
        # caches the created tester
        self.__tester = tester
        return tester

    def load_parameter(self, tester: Inferencer, parameter: ModelParameter) -> None:
        """
            load model params into the tester
        """
        tester.model_util.load_parameters(parameter)

    def get_metric(
        self,
        parameter: ModelParameter | ParameterMessage,
        log_performance_metric: bool = True,
        copy_tester: bool = False,
    ) -> dict:
        """
            evaluate the model (parameter) performance
        """
        if isinstance(parameter, ParameterMessage):
            parameter = parameter.parameter
        # configure the tester
        tester = self.get_tester(copy_tester=copy_tester)
        self.load_parameter(tester=tester, parameter=parameter)
        tester.model_util.disable_running_stats()
        tester.hook_config.use_performance_metric = True
        tester.hook_config.log_performance_metric = log_performance_metric
        tester.hook_config.save_performance_metric = True
        # adjust the batch size
        batch_size: int | None = None
        if "server_batch_size" in tester.dataloader_kwargs:
            batch_size = tester.dataloader_kwargs["server_batch_size"]
            tester.remove_dataloader_kwargs("server_batch_size")
        elif "batch_number" in tester.dataloader_kwargs:
            batch_size = min(
                int(tester.dataset_size / tester.dataloader_kwargs["batch_number"]),
                100,
            )
        if batch_size is not None:
            assert batch_size > 0
            log_info("server uses batch_size %s", batch_size)
            tester.remove_dataloader_kwargs("batch_number")
            tester.update_dataloader_kwargs(batch_size=batch_size)
        if tester.has_hook_obj("performance_metric"):
            tester.performance_metric.clear_metric()
            metric: dict = tester.performance_metric.get_epoch_metrics(1)
            assert not metric
        # run inference on the tester
        tester.inference()
        # retrieve performance metrics
        metric = tester.performance_metric.get_epoch_metrics(1)
        assert metric
        # free up the resources
        tester.offload_from_device()
        return metric

    def start(self) -> None:
        """
            the main loop for the server
            manage the communication with the workers
        """
        # set the server context
        ExecutorContext.set_name(self.name)
        # save the configuration to a file
        with open(os.path.join(self.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        self._before_start()
        # check continuously (until _stopped()) for data from worker & process it
        # remove the worker ID from the copied dataset after being processed
        worker_set: set = set()
        while not self._stopped():
            if not worker_set:
                worker_set = set(range(min(self._endpoint.worker_num, self.selected_worker_number())))
                #worker_set = set(range(self._endpoint.worker_num))
            assert self._endpoint.worker_num == self.config.worker_number
            for worker_id in copy.copy(worker_set):
                has_data: bool = self._endpoint.has_data(worker_id)
                if has_data:
                    log_info(
                        "get result from worker_id %s",
                        worker_id,)
                    self._process_worker_data(
                        worker_id, self._endpoint.get(worker_id=worker_id)
                    )
                    worker_set.remove(worker_id)
            if worker_set:
                log_info("wait for other %s workers ", len(worker_set))

            if worker_set and not self._stopped():
                time.sleep(5)
        # close the endpoint
        self._endpoint.close()
        # clean up
        self._server_exit()
        log_info("end server")

    def _before_start(self) -> None:
        pass

    def _server_exit(self) -> None:
        pass

    def _process_worker_data(self, worker_id: int, data: Message) -> None:
        raise NotImplementedError()

    def _stopped(self) -> bool:
        raise NotImplementedError()
