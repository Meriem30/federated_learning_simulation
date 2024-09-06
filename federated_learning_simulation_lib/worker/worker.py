import functools
import os
from functools import cached_property
from typing import Any

import dill
from other_libs.log import log_debug
from other_libs.topology.endpoint import Endpoint
from torch_kit import ExecutorHookPoint, Trainer

from ..executor import Executor, ExecutorContext
from ..practitioner import Practitioner


class Worker(Executor):
    """
        handle tasks related to model training (start, pause, stop ...)
        inherit from Executor class
    """
    def __init__(
        self,
        task_id: int | None,
        endpoint: Endpoint,
        practitioner: Practitioner,
        **kwargs: Any,
    ) -> None:
        worker_id = practitioner.worker_id
        name = f"worker {worker_id}"
        if task_id is not None:
            name = f"worker {worker_id} of {task_id}"
        super().__init__(name=name, **kwargs)
        self.__practitioner: Practitioner = practitioner
        # store the endpoint instance for communication
        self._endpoint = endpoint
        # initialize the training round communication to 0
        self._round_index = 0
        # flag: if the worker should stop
        self._force_stop = False

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def round_index(self):
        return self._round_index

    @property
    def worker_id(self):
        return self.__practitioner.worker_id

    @cached_property
    def trainer(self) -> Trainer:
        return self.__new_trainer()

    def __new_trainer(self) -> Trainer:
        """
            create a new trainer instance
            after removing server_batch_size from the configuration
        """
        if "server_batch_size" in self.config.trainer_config.dataloader_kwargs:
            self.config.trainer_config.dataloader_kwargs.pop("server_batch_size")
        return self.__practitioner.create_trainer(self.config)

    def _offload_from_device(self) -> None:
        self.trainer.offload_from_device()

    def _before_training(self) -> None:
        pass

    def _after_training(self) -> None:
        """
            save the trainer's hyperparams to a file after training
        """
        with open(os.path.join(self.save_dir, "hyper_parameter.pk"), "wb") as f:
            dill.dump(
                self.trainer.hyper_parameter,
                f,
            )

    def _stopped(self) -> bool:
        """
            condition control to stop (based on round index or force stop flag)
        """
        return self._round_index > self.config.round or self._force_stop

    def pause(self) -> None:
        """
            pause the training by waiting  the current stream
            and releasing the device lock
        """
        self.trainer.wait_stream()
        self._release_device_lock()

    def start(self, **kwargs: Any) -> None:
        """
            main training loop:
            start the training process
            execute training rounds
            until the stopping condition are met
        """
        first_training: bool = True
        self._round_index = 1
        self._force_stop = False
        while not self._stopped():
            # in case worker changes round number
            # ensure the worker has exclusive access during critical operations
            with ExecutorContext(self.name):
                if first_training:
                    # if first round, perform pre-training setup
                    self._before_training()
                    first_training = False
                    # in case worker changes round number
                    if self._stopped():
                        break
                    # set the trainer device and hooks
                    self.trainer.set_device_fun(
                        functools.partial(
                            self._get_device,
                            lock_callback=lambda: self.trainer.append_named_hook(
                                ExecutorHookPoint.AFTER_BATCH,
                                "_release_device_lock",
                                self._release_device_lock,
                            ),
                        )
                    )
                else:
                    self.trainer.hook_config.summarize_executor = False
                # if performance metric should be logged
                self.trainer.hook_config.log_performance_metric = (
                    self.config.enable_training_log
                )
                self.trainer.disable_hook("batch_loss_logger")
                self.trainer.set_visualizer_prefix(
                    prefix=f"round: {self._round_index},"
                )
                # execute training
                self.trainer.train(
                    **kwargs,
                )
                # increment the round index
                self._round_index += 1
        # clean up once the loop is completed
        with ExecutorContext(self.name):
            log_debug("finish worker")
            self.endpoint.close()
            log_debug("close endpoint")
            self._after_training()
            log_debug("end worker")
