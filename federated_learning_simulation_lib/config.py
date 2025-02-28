import datetime
import importlib
import os
import uuid
from typing import Any

import omegaconf
from other_libs.log import log_debug, log_warning
from torch_kit import ClassificationDatasetCollection, Config
from torch_kit.device import get_device_memory_info

from .practitioner import Practitioner
from .sampler import get_dataset_collection_split


class DistributedTrainingConfig(Config):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.distributed_algorithm: str = ""
        self.algorithm_kwargs: dict = {}
        self.worker_number: int = 0
        self.round: int = 0
        self.dataset_sampling: str = "iid"
        self.dataset_sampling_kwargs: dict[str, Any] = {}
        self.endpoint_kwargs: dict = {}
        self.exp_name: str = "selection_exps"
        self.log_file: str = ""
        self.enable_training_log: bool = False
        self.use_validation: bool = True
        self.worker_number_per_process: int = 0
        # added to consider graphs
        self.graph_worker: bool = False
        # self.enable_clustering: bool = True
        self.family_number: int = 0
        # ADDED to handle Spectral Clustering
        self.graph_type = None
        self.num_neighbor: int = 0
        self.threshold: float = 0.1
        self.similarity_function = None
        self.laplacian_type = None

    def load_config_and_process(self, conf: Any) -> None:
        # load the config
        self.load_config(conf)
        # reset the session
        self.reset_session()
        # import necessary dep
        import_dependencies(
            dataset_type=self.dc_config.dataset_kwargs.get("dataset_type", None)
        )

    def get_worker_number_per_process(self) -> int:
        """
            determine the number of workers per process
            depending on the available memery on devices
        """
        if self.worker_number_per_process != 0:
            return self.worker_number_per_process

        memory_info = get_device_memory_info()
        refined_memory_info: dict = {}
        MB = 1024 * 1024
        GB = MB * 1024
        for device, info in memory_info.items():
            if info.total / GB >= 20:
                if info.free / GB < 5:
                    continue
            if info.used / info.total > 0.9:
                continue
            free_GB = int(info.free / GB)
            if free_GB == 0:
                continue
            # the devices that pass all memory checks are stored in refined_memory_info
            refined_memory_info[device] = info.free
        #assert refined_memory_info
        log_warning("Use devices %s", list(refined_memory_info.keys()))
        # if true, each worker can run on a separate device process
        if self.worker_number <= len(refined_memory_info):
            return 1
        # small scale training
        if self.worker_number <= 50:
            # spread the worker evenly cross the valid devices
            return int(self.worker_number / len(refined_memory_info))
        # large scale training
        # sum the free memory of all worker
        total_bytes = sum(refined_memory_info.values())
        # calculates the memory allocated in MB per worker, should not exceed 10 GB
        MB_per_worker = min(total_bytes / MB / self.worker_number, 10 * GB)
        # log the memory per worker and the minimum free memory among the valid devices
        log_debug(
            "MB_per_worker %s other %s",
            MB_per_worker,
            min(refined_memory_info.values()) / MB,
        )
        # calculate the number of workers per process
        worker_number_per_process = int(
            min(refined_memory_info.values()) / MB / MB_per_worker
        )
        assert worker_number_per_process > 0
        return worker_number_per_process

    def reset_session(self) -> None:
        """
            reset the session by creating new directories for logs and session data
        """
        task_time = datetime.datetime.now()
        date_time = f"{task_time:%Y-%m-%d_%H_%M_%S}"
        # get the dataset from the kwargs
        # if it does not exist, defaults it to name
        dataset_name = self.dc_config.dataset_kwargs.get(
            "name", self.dc_config.dataset_name
        )
        # construct dir_suffix for paths directory
        dir_suffix = os.path.join(
            # add the distributed algo
            self.distributed_algorithm,
            (
                f"{dataset_name}_{self.dataset_sampling}"
                if isinstance(self.dataset_sampling, str)
                else f"{dataset_name}_{'_'.join(self.dataset_sampling)}"  # if it is a list join elements
            ),
            # add the model name
            self.model_config.model_name,
            date_time,
            # generate a random identifier
            str(uuid.uuid4().int + os.getpid()),
        )
        # if the experiment is set, add it
        if self.exp_name:
            dir_suffix = os.path.join(self.exp_name, dir_suffix)
        # create the full path and assign it to save_dir
        self.save_dir = os.path.join("session", dir_suffix)
        # create the full path for logs
        self.log_file = str(os.path.join("log", dir_suffix)) + ".log"

    def create_practitioners(self) -> set[Practitioner]:
        """
            create and initialize a practitioner instance
        """
        # initialize a set of empty practitioners
        practitioners: set[Practitioner] = set()
        # create a data set collection (ML phases: data, ..)
        dataset_collection = self.create_dataset_collection()
        assert isinstance(dataset_collection, ClassificationDatasetCollection)
        # initialize the sampler according to name:'dataset_sampling' and other params
        sampler = get_dataset_collection_split(
            name=self.dataset_sampling,
            dataset_collection=dataset_collection,
            part_number=self.worker_number,
            **self.dataset_sampling_kwargs,
        )
        # iterate over the number of worker
        for practitioner_id in range(self.worker_number):
            # create a practitioner for each worker
            practitioner = Practitioner(
                practitioner_id=practitioner_id,
            )
            # set the sampler for the current practitioner
            practitioner.set_sampler(sampler=sampler)
            # add the new practitioner to the set to simulate one worker
            practitioners.add(practitioner)
        assert practitioners
        return practitioners


def load_config(conf_obj: Any, global_conf_path: str) -> DistributedTrainingConfig:
    """
        merge configuration objects (global and conf_obj)
        load & process the configuration
    """
    config: DistributedTrainingConfig = DistributedTrainingConfig()
    while "dataset_name" not in conf_obj and len(conf_obj) == 1:
        conf_obj = next(iter(conf_obj.values()))
    result_conf = omegaconf.OmegaConf.load(global_conf_path)
    result_conf.merge_with(conf_obj)
    config.load_config_and_process(result_conf)
    return config


def load_config_from_file(
    config_file: str, global_conf_path: str
) -> DistributedTrainingConfig:
    """
        load a conf from a file
        merge & process the global conf
    """
    return load_config(omegaconf.OmegaConf.load(config_file), global_conf_path)


import_result: dict = {}


def import_dependencies(dataset_type: str | None = None) -> None:
    global import_result
    if import_result:
        return
    libs = ["torch_text", "torch_vision"]
    if dataset_type is not None:
        match dataset_type.lower():
            case "medical":
                libs = ["torch_medical"]
            case "vision":
                libs = ["torch_vision"]
            case "text":
                libs = ["torch_text"]
            case _:
                raise NotImplementedError(dataset_type)
    for dependency in libs:
        try:
            importlib.import_module(dependency)
            import_result[dependency] = True
        except ModuleNotFoundError:
            pass
