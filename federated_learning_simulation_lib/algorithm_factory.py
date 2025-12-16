import functools
import os
from typing import Callable

from other_libs.log import log_warning, log_debug
from other_libs.system_info import OSType, get_operating_system_type
from other_libs.topology.central_topology import (
    ProcessPipeCentralTopology, ProcessQueueCentralTopology)
from other_libs.topology.cs_endpoint import ClientEndpoint, ServerEndpoint
from torch_kit.concurrency import TorchProcessContext
from .graph_worker import GraphWorker, GraphAggregationWorker, GraphGradientWorker
from .config import DistributedTrainingConfig


class CentralizedAlgorithmFactory:
    """
        store the config for different algorithms
        (key: algo_name, value: config_dict)
    """
    config: dict[str, dict] = {}

    @classmethod
    def register_algorithm(
        cls,
        algorithm_name: str,
        client_cls: Callable,
        server_cls: Callable,
        client_endpoint_cls: None | Callable = None,
        server_endpoint_cls: None | Callable = None,
        algorithm_cls: None | Callable = None, # the core algorithm logic
    ) -> None:
        """
            register a new algorithm with the factory
            add information about how to create its components
        """
        # assert is not already registered
        assert algorithm_name not in cls.config
        # set the client and server endpoints if not provided
        if client_endpoint_cls is None:
            client_endpoint_cls = ClientEndpoint
        if server_endpoint_cls is None:
            server_endpoint_cls = ServerEndpoint
        # add the algo (name as the key) to the conf dict
        cls.config[algorithm_name] = {
            "client_cls": client_cls,
            "server_cls": server_cls,
            "client_endpoint_cls": client_endpoint_cls,
            "server_endpoint_cls": server_endpoint_cls,
        }
        if algorithm_cls is not None:
            cls.config[algorithm_name]["algorithm_cls"] = algorithm_cls

    @classmethod
    def has_algorithm(cls, algorithm_name: str) -> bool:
        """
            check if an algo is registered in the factory
        """
        return algorithm_name in cls.config

    @classmethod
    def create_client(
        cls,
        algorithm_name: str,
        kwargs: dict,  # the main args to create the client
        endpoint_kwargs: dict,
        extra_kwargs: dict | None = None,
        extra_endpoint_kwargs: dict | None = None,  # optional additional arguments
    ) -> None:
        """
            streamline the creation of the client instances
            for different algorithm
            abstract the creation of endpoint and client instances
        """
        # retrieve the configuration
        config = cls.config[algorithm_name]
        # if extra args are not provided, they are initialized as empty dict
        if extra_kwargs is None:
            extra_kwargs = {}
        if extra_endpoint_kwargs is None:
            extra_endpoint_kwargs = {}
        # create the client endpoint instance merging the two dicts
        endpoint = config["client_endpoint_cls"](
            **(endpoint_kwargs | extra_endpoint_kwargs)
        )
        # create and return the client instance using the created endpoint and other args
        client_instance = config["client_cls"](endpoint=endpoint, **(kwargs | extra_kwargs))
        # ADDED to handle graphs
        # Attach GraphWorker if required
        graph_worker_class = extra_kwargs.pop("graph_worker_class", None)
        if graph_worker_class:
            graph_worker = graph_worker_class(
                task_id=kwargs["task_id"],
                endpoint=endpoint,
                practitioner=kwargs["practitioner"],
                config=kwargs["config"],
            )
            client_instance.graph_worker = graph_worker

        return client_instance

    @classmethod
    def create_server(
        cls,
        algorithm_name: str,
        kwargs: dict,
        endpoint_kwargs: dict,
        extra_kwargs: dict | None = None,
        extra_endpoint_kwargs: dict | None = None,
    ) -> None:
        """
            create the server instance
            abstract the creation of the endpoint, the algo, and the server instances
        """
        config = cls.config[algorithm_name]
        if extra_kwargs is None:
            extra_kwargs = {}
        if extra_endpoint_kwargs is None:
            extra_endpoint_kwargs = {}
        # create an endpoint instance
        endpoint = config["server_endpoint_cls"](
            **(endpoint_kwargs | extra_endpoint_kwargs)
        )
        algorithm = None
        # create the algo if specified in the conf and add it to extra params
        if "algorithm_cls" in config:
            algorithm = config["algorithm_cls"]()
            assert "algorithm" not in extra_kwargs
            extra_kwargs["algorithm"] = algorithm
        # create and return the server instance using merging args
        server_instance = config["server_cls"](endpoint=endpoint, **(kwargs | extra_kwargs))

        # ADDED to handle graphs
        # Attach GraphAggregationWorker if required
        graph_server_class = extra_kwargs.pop("graph_server_class", None)
        if graph_server_class:
            graph_aggregation_worker = graph_server_class(
                task_id=kwargs["task_id"],
                endpoint=endpoint,
                config=kwargs["config"],
            )
            server_instance.graph_aggregation_worker = graph_aggregation_worker

        return server_instance


def get_worker_config(
    config: DistributedTrainingConfig, practitioners: None | set = None
) -> dict:
    """
        generate a configuration dict for the distributed learning,
        include: topology, server, worker, practitioner
    """
    # create and  initialize practitioner if not provided
    if practitioners is None:
        log_debug("creating practitioners ...")
        practitioners = config.create_practitioners()
    else:
        # set the worker number and sort practitioner by their IDs
        log_debug("we already have practitioners:", len(practitioners))
        config.worker_number = len(practitioners)
        for worker_id, practitioner in enumerate(
            sorted(practitioners, key=lambda p: p.id)
        ):
            # ensure each practitioner has the required dataset
            assert practitioner.has_dataset(config.dc_config.dataset_name)
            # assign IDs to practitioner
            practitioner.set_worker_id(worker_id)
    assert practitioners
    # check the algo specified in conf exists in the CentralizedAlgorithmFactory
    assert CentralizedAlgorithmFactory.has_algorithm(config.distributed_algorithm)
    # set the default topology class to Pipe
    #topology_class = ProcessPipeCentralTopology
    topology_class = ProcessQueueCentralTopology
    if get_operating_system_type() == OSType.Windows or "no_pipe" in os.environ:
        # if true, change it to Queue
        topology_class = ProcessQueueCentralTopology
        log_warning("use ProcessQueueCentralTopology")
    # create an instance of the topology class
    topology = topology_class(
        mp_context=TorchProcessContext(), worker_num=config.worker_number
    )
    # initialize a result dict with the topology
    result: dict = {"topology": topology}
    # server configuration
    # partially apply the create_server method
    result["server"] = {}
    result["server"]["constructor"] = functools.partial(
        CentralizedAlgorithmFactory.create_server,
        algorithm_name=config.distributed_algorithm,
        endpoint_kwargs=config.endpoint_kwargs.get("server", {}),
        kwargs={"config": config},
    )
    # ADDED to handle graphs
    #if config.graph_worker:
    #    result["server"]["constructor"].keywords["extra_kwargs"] = {
    #        "graph_server_class": GraphAggregationWorker}
    # client configuration
    worker_number_per_process = config.get_worker_number_per_process()
    log_warning(
        "There are %s workers in total, and %s workers form a group",
        len(practitioners),
        worker_number_per_process,
    )
    client_config: list[list[dict]] = []
    tmp = list(practitioners)
    # divide practitioners into batch based on the number of workers per group
    while tmp:
        batch = tmp[:worker_number_per_process]
        tmp = tmp[worker_number_per_process:]
        client_config.append(
            [
                {
                    "constructor": functools.partial(
                        CentralizedAlgorithmFactory.create_client,
                        algorithm_name=config.distributed_algorithm,
                        endpoint_kwargs=config.endpoint_kwargs.get("worker", {})
                        | {
                            "worker_id": practitioner.worker_id,
                        },
                        kwargs={
                            "config": config,
                            "practitioner": practitioner,
                        },
                    ),
                }
                for practitioner in batch
            ]
        )
        # ADDED to handle graphs
        #if config.graph_worker:
        #    for batch_config in client_config[-1]:
        #        batch_config["constructor"].keywords["extra_kwargs"] = {"graph_worker_class": GraphWorker}
    assert client_config
    result["worker"] = client_config
    return result
