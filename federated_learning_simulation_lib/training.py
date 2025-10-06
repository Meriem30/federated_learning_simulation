import copy
import multiprocessing
import os
import time
# to save memory in large-scale training
import uuid

import gevent
from other_libs.concurrency.process_initialization import get_process_data
from other_libs.log import add_file_handler, log_debug, log_info
from other_libs.time_counter import TimeCounter
from torch_kit.concurrency import TorchProcessPool

from .algorithm_factory import get_worker_config
from .config import DistributedTrainingConfig
from .worker import Worker

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["USE_THREAD_DATALOADER"] = "1"


def start_server(task_id: int | None, server_config: dict) -> dict:
    """
        initialize and start a server process
    """
    # retrieve shared process data
    device_lock = get_process_data()["device_lock"]
    topology = get_process_data()["topology"]
    log_debug("task_id %s topology id %d", task_id, id(topology))

    # construct the server object
    server = server_config["constructor"](
        extra_kwargs={
            "task_id": task_id,
            "device_lock": device_lock,
        },
        extra_endpoint_kwargs={
            "topology": topology,
        },
    )
    # start the server
    server.start()
    log_info("stop server")

    res: dict = {}
    # no shapley value
    # if hasattr(server.algorithm, "shapley_values"):
    #    res["sv"] = server.algorithm.shapley_values
    # collect performance statistics
    res |= {"performance": server.performance_stat}
    return res


def start_workers(
    task_id: int | None,
    worker_configs: list[dict],
) -> None:
    """
        initialize and start worker processes
    """
    # retrieve shared process data
    device_lock = get_process_data()["device_lock"]
    log_lock = get_process_data()["log_lock"]
    topology = get_process_data()["topology"]
    workers: list[Worker] = []
    assert worker_configs

    for worker_config in worker_configs:
        # construct worker objects and append them in a list
        worker_instance = worker_config["constructor"](
                extra_kwargs={
                    "task_id": task_id,
                    "device_lock": device_lock,
                    "log_lock": log_lock,
                },
                extra_endpoint_kwargs={
                    "topology": topology,
                },
            )
        # Append the worker instance to the workers list
        workers.append(worker_instance)

    # log workers info
    log_info(
        "run workers %s in the same process for task %s",
        [worker.worker_id for worker in workers],
        task_id,
    )
    # run the start method for each worker concurrently
    # waiting for all greenlets to complete
    gevent.joinall([gevent.spawn(worker.start) for worker in workers], raise_error=True)

    log_debug("stop workers")


tasks: dict = {}
task_results: dict = {}


def train(
    config: DistributedTrainingConfig,
    practitioners: None | set = None,
) -> int | None:
    """
        the main function to start the training process
    """
    # deepcopy config & practitioner for concurrent training
    config = copy.deepcopy(config)
    practitioners = copy.deepcopy(practitioners)
    # reset and apply global config
    config.reset_session()
    config.apply_global_config()
    # to measure the training duration
    timer = TimeCounter()
    task_id = None
    if practitioners is None:
        # add a file handlr for logging
        add_file_handler(config.log_file)
        #print("add  file handler", handler.baseFilename)
    else:
        # otherwise, generate a unique task id
        task_id = uuid.uuid4().int + os.getpid()
    # get worker configuration & retrieve topology
    worker_config = get_worker_config(config, practitioners=practitioners)
    log_debug("here is all workers config: ", worker_config)
    topology = worker_config.pop("topology")
    # initialize multiprocessing Manager
    manager = multiprocessing.Manager()
    device_lock = manager.RLock()
    # control access to logging resources
    log_lock = manager.Semaphore()
    assert topology.worker_num == config.worker_number
    # create & initialize the process pool to manage the concurrent processes
    process_pool: TorchProcessPool = TorchProcessPool(
        initargs={
            "process_data": {
                "device_lock": device_lock,
                "log_lock": log_lock,
                "topology": topology,
            }
        }
    )
    process_pool.catch_exception()
    # iterate over the config of each group of worker that should run together
    for worker_configs in worker_config["worker"]:
        # submit a group task to the pool passing the start_workers function to be executed
        process_pool.submit(
            start_workers, task_id=task_id, worker_configs=worker_configs
        )
    # submit server task if applicable
    server_config = worker_config.get("server", None)
    if server_config is not None:
        process_pool.submit(
            start_server,
            task_id=task_id,
            server_config=server_config,
        )
    # if practitioners is provided, store the task info in the tasks dict
    if practitioners is not None:
        tasks[task_id] = {
            "process_pool": process_pool,
            "practitioner_ids": {practitioner.id for practitioner in practitioners},
            "config": config,
        }
        # initialize an empty result entry in task_results to be updated during the training
        task_results[task_id] = {}
        return task_id
    # wait for all tasks
    process_pool.wait_results(timeout=None)
    # shutdown the process pool
    process_pool.shutdown(wait=True)
    log_info("training took %s seconds", timer.elapsed_milliseconds() / 1000)
    #time.sleep(20)
    return None


def get_training_result(task_id: int, timeout: None | float = None) -> None | dict:
    """
        retrieve the result of a specific task
        wait for the task to complete; aggregate the results; shutdown the process pool of this task
    """
    task = tasks[task_id]
    process_pool = task["process_pool"]
    results, not_done = process_pool.wait_results(timeout=timeout)
    for result in results.values():
        if result is not None:
            task_results[task_id] |= result
    if not_done:
        # true indicate that the function couldn't collect all results within the timeout specified
        return None
    # terminate all worker processes of this pool
    process_pool.shutdown()
    # remove the task entry from the global tasks
    tasks.pop(task_id)
    log_info("finish task %s", task_id)
    # initialize a dict to hold the aggregate results
    stats: dict = {}
    practitioner_ids = task["practitioner_ids"]
    config = task["config"]
    assert practitioner_ids is not None
    # iterate through the items in  task_results for a given task id
    for k, v in task_results[task_id].items():
        # capy the value in the stats result dict
        if k != "sv":
            stats[k] = v
            continue
        sv_dict: dict = {}
        # no shapley value
        #for round_number, tmp_sv_dict in v.items():
        #    sv_dict[round_number] = {}
        #    for practitioner_id, worker_id in zip(
        #        sorted(practitioner_ids), range(config.worker_number)
        #    ):
        #        sv_dict[round_number][practitioner_id] = tmp_sv_dict[worker_id]
        #stats[k] = sv_dict
    # remove the result entry from the global task_results as it has been processed
    task_results.pop(task_id)
    return stats
