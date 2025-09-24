import random
from typing import Any, MutableMapping, Tuple
import numpy as np
import torch
from other_libs.log import log_error, log_debug, log_info, log_warning
from torch_kit import ModelParameter

from ..message import Message, ParameterMessage
from .aggregation_algorithm import AggregationAlgorithm
from .fed_avg_algorithm import FedAVGAlgorithm
from .spectral_clustering import SpectralClustering, SimilarityType, GraphType, LaplacianType
#from .shapley_value import EnhancedGTGShapleyValueAlgorithm

class GraphFedAVGAlgorithm(AggregationAlgorithm):

    def __init__(self) -> None:
        super().__init__()
        # whether to accumulate the updates from the workers or compute the average directly
        self.accumulate: bool = True
        # whether to aggregate loss information
        self.aggregate_loss: bool = False
        # dict to store total weights for each model
        self.__total_weights: dict[str, float] = {}
        # store accumulated parameter  updates
        # track workers whose data was not received
        self.skipped_workers: set[int] = set()
        self.__parameter: ModelParameter = {}
        self.__client_states_array = None
        self._assign_family: bool = True
        self._enable_clustering: bool = True
        self._enum_converted: bool = False
        self._adjacency_matrix = None
        # ADDED to handle svs
        self._assign_egtg_sv: bool = True

    @property
    def adjacency_sc_matrix(self):
        return self._adjacency_matrix

    def process_worker_data(
            self,
            worker_id: int,
            worker_data: Message | None,
    ) -> bool:
        """
            add more functionalities to the parent method
        """
        # initialize the parent method for processing data
        res = super().process_worker_data(worker_id=worker_id, worker_data=worker_data)
        if not res:
            return False
        if not self.accumulate:
            # if accumulate is not required, the method returns early
            return True
        worker_data = self._all_worker_data.get(worker_id, None)
        if worker_data is None:
            return True
        if not isinstance(worker_data, ParameterMessage):
            return True
        # if the all worker data is not None
        for k, v in worker_data.parameter.items():
            # iterate over each parameter update from the worker and apply the weight
            assert not v.isnan().any().cpu()
            # retrieve the weight associated with the param v
            weight = self._get_weight(worker_data, name=k, parameter=v)
            # store the weighted parameter
            tmp = v.to(dtype=torch.float64) * weight
            if k not in self.__parameter:
                self.__parameter[k] = tmp
            else:
                # if the param key already exists, add the weighted update value
                self.__parameter[k] += tmp
            #
            if k not in self.__total_weights:
                # similarly, add or initialize the weight for the param key
                self.__total_weights[k] = weight
            else:
                self.__total_weights[k] += weight
        # release to reduce memory pressure
        worker_data.parameter = {}
        return True

    def _get_weight(
            self, worker_data: ParameterMessage, name: str, parameter: Any
    ) -> Any:
        return worker_data.aggregation_weight

    def _apply_total_weight(
            self, name: str, parameter: torch.Tensor, total_weight: Any
    ) -> torch.Tensor:
        """
            device the parameter tensor by the total weight to compute average
        """
        return parameter / total_weight

    def aggregate_worker_data(self) -> ParameterMessage:
        if not self.accumulate:
            # if not required, directly aggregate
            parameter = self.aggregate_parameter(self._all_worker_data)
        else:
            # if required, process accumulated parameter & normalize them using total weight
            log_info("this is graph fed_avg algorithm about to aggregate parameters")
            assert self.__parameter
            parameter = self.__parameter
            self.__parameter = {}
            for k, v in parameter.items():
                assert not v.isnan().any().cpu()
                parameter[k] = self._apply_total_weight(
                    name=k, parameter=v, total_weight=self.__total_weights[k]
                )
                assert not parameter[k].isnan().any().cpu()
            self.__total_weights = {}
        # ADDED to handle spectral_clustering
        clustering_results = None
        if self._enable_clustering and self.config.round > 1:
            self.__client_states_array = self._create_workers_matrix(self._all_worker_data)
            # call the appropriate class,function passing the matrix of data points (client_states)
            clustering_results, self._adjacency_matrix = self._perform_clustering(self.__client_states_array)
        other_data: dict[str, Any] = {}
        if self.aggregate_loss:
            # if true, compute aggregated loss values
            other_data |= self.__aggregate_loss(self._all_worker_data)
        # ADDED to handle graphs
        if self._assign_family and self.config.round != 1:
            log_info("update family is enabled for all workers")
            #log_info("this is other data before calling _update_family_assignment function %s", self._all_worker_data)
            other_data |= self.__update_family_assignment(self._all_worker_data, clustering_results)
            log_debug("family assignments updated by graph fed_avg algorithm")
        # ensure consistency
        other_data |= self.__check_and_reduce_other_data(self._all_worker_data)
        # return the aggregated params and other data
        return ParameterMessage(
            parameter=parameter,
            end_training=next(iter(self._all_worker_data.values())).end_training,
            in_round=next(iter(self._all_worker_data.values())).in_round,
            other_data=other_data,
        )

    def _create_workers_matrix(self, all_worker_data: MutableMapping[int, Message]) -> np.ndarray:
        """Extract client states and use them to initialize a numpy matrix """
        log_info("here is create worker matrix")
        num_clients = self.config.worker_number
        log_info("workers number: %s", num_clients)
        first_key = next(iter(all_worker_data))
        num_properties = all_worker_data[first_key].other_data["node_state"].get_number_of_properties()
        log_info("state worker properties number: %s", num_properties)

        # initialize a zero matrix of shape (num_clients, num_properties)
        client_states = np.zeros((num_clients, num_properties))
        #log_info("client_states before uploading client info: \n %s", client_states)

        # collect data from participating clients
        # extract the participating_clients_ids for that round
        participating_clients = np.array([worker_id for worker_id in all_worker_data.keys()
                                          if worker_id not in self.skipped_workers])
        if len(participating_clients) > 0:
            """states = np.array([
                [float(all_worker_data[worker_id].other_data["node_state"]._battery),
                 float(all_worker_data[worker_id].other_data["node_state"]._energy_consumption),
                 float(all_worker_data[worker_id].other_data["node_state"]._memory_occupation)]
                for worker_id in participating_clients
            ])"""
            for worker_id in range(num_clients):
                if worker_id in participating_clients:
                    feature = float(all_worker_data[worker_id].other_data["node_state"].mi)
                    states = np.array(feature)
                    client_states[worker_id, :] = states
                elif worker_id in list(range(num_clients)):
                    client_states[worker_id, :] = np.array(0.0)


        # update previous states in Graph_Aggregation_Server not here
        # for worker_id in participating_clients:
        #         self.graph_client_states[worker_id] = all_worker_data[worker_id].other_data["node_state"]

        # Retain previous states for clients not in the current round
        # missing_clients = np.setdiff1d(np.arange(num_clients), participating_clients)
        # for worker_id in missing_clients:
            # if worker_id in self.previous_states:
            #    prev_state = self.previous_states[worker_id]
            #    client_states[worker_id, 0] = float(prev_state._battery)
            #    client_states[worker_id, 1] = float(prev_state._energy_consumption)
            #    client_states[worker_id, 2] = float(prev_state._memory_occupation)

        #log_info("Here is client states matrix \n %s", client_states)
        return client_states

    @classmethod
    def __update_family_assignment(cls, all_worker_data: MutableMapping[int, Message], results: np.ndarray) -> dict[str, dict]:
        assert all_worker_data
        family_data = {}
        for worker_data in all_worker_data.values():
            if "node_state" in worker_data.other_data:
                family_data["family_assignment"] = cls._assign_new_families(all_worker_data, results)
            break
        log_info("new family assignments dict %s added by the graph algo", family_data)

        for worker_data in all_worker_data.values():
            if "node_state" in worker_data.other_data:
                # remove node_state data from other_data att of worker
                worker_data.other_data.pop("node_state", None)
        log_debug("released node state from other worker data")
        return family_data

    @classmethod
    def _assign_new_families(cls, all_worker_data: MutableMapping[int, Message], families: np.ndarray) -> dict[int, int]:

        family_dict = {}
        # log_debug("passed arg type for compute_new_families", type(all_worker_data)) -> dict
        for worker_id, worker_data in all_worker_data.items():
            family_dict[worker_id] = int(families[worker_id])
        log_info("this is the family dict after updating the families %s", family_dict)
        return family_dict

    @classmethod
    def _compute_new_family(cls, worker_data: Message) -> int:
        # logic for calculating new family
        state = worker_data.other_data["node_state"]
        current_family = state.family
        log_info("this is the worker family %s", current_family)
        new_family = random.choice(list(range(1, 4)))
        log_info("this is the new worker family %s", new_family)
        return new_family

    # ADDED to handle spectral clustering
    def _process_clustering_results(self, labels):
        # process the clustering labels
        # use `labels` to assign workers to different groups/families or whatever the final use is
        for worker_id, label in enumerate(labels):
            # Example: assign the client to a new family or group based on the clustering result
            self._graph_client_states[worker_id].set_family(label)
            log_info(f"Worker {worker_id} assigned to family {label}")

    # ADDED to handle spectral clustering
    def _convert_enum_properties(self) -> None:
        """
        Converts string values in the config dictionary to their corresponding enum members.
        This ensures type safety and prevents errors in downstream classes.
        """
        enum_mappings = {
            'similarity_function': SimilarityType,
            'graph_type': GraphType,
            'laplacian_type': LaplacianType,
        }
        for attr, enum_type in enum_mappings.items():
            if hasattr(self.config, attr):
                value = getattr(self.config, attr)
                try:
                    if isinstance(value, str):
                        # normalize case, try matching by value first
                        enum_value = None
                        try:
                            enum_value = enum_type(value.lower())  # match enum value (e.g. 'gaussian')
                        except ValueError:
                            try:
                                enum_value = enum_type[value.capitalize()]  # match enum name (e.g. 'Gaussian')
                            except KeyError:
                                pass
                        if enum_value is None:
                            raise ValueError(f"Unknown {attr}: {value}")
                    elif isinstance(value, enum_type):
                        enum_value = value  # already correct type
                    else:
                        raise ValueError(f"Invalid type for {attr}: {value}")

                    setattr(self.config, attr, enum_value)
                    self._enum_converted = True

                except Exception as e:
                    raise ValueError(f"Failed to convert {attr}={value} → {enum_type}: {e}")


    # ADDED to handle spectral clustering
    def _perform_clustering(self, client_states_array: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Performs spectral clustering on the client states array

        Args:
            client_states_array: The input data for clustering
        Returns:
            A tuple containing the cluster labels and the adjacency matrix
        """
        # Step 1: Convert string config values to enums
        self._convert_enum_properties()

        # Step 2: Initialize the spectral clustering class with the config dictionary
        # This single-argument initialization is clean and scalable.
        sc_config = {
            "graph_type" : self.config.graph_type,  # for graph construction
            "num_neighbors" : self.config.num_neighbor,  # Number of neighbors for KNN
            "threshold" : self.config.threshold,
            "laplacian_type" : self.config.laplacian_type,  # Type of Laplacian
            "similarity_function" : self.config.similarity_function,  # Similarity function
            "num_clusters" : self.config.family_number  # Number of clusters (K)
        }
        clustering = SpectralClustering(sc_config)

        # Step 3: Perform spectral clustering and get the labels.
        labels = clustering.fit(self.__client_states_array)
        self._adjacency_matrix = clustering.adjacency_matrix

        # Step 4: Log results efficiently and concisely.
        log_info("Labels returned from clustering: %s", labels)
        # The adjacency matrix can be very large. Log its shape and sparsity instead of the full matrix.
        if self._adjacency_matrix is not None:
            log_info(
                f"Adjacency matrix returned from clustering: Shape={self._adjacency_matrix.shape}") # NNZ={self._adjacency_matrix.nnz}

        #log_warning("this is the centroids returned from the clustering : %s", centroids)
        #log_info(" this is the adjacency matrix returned from the clustering :\n %s", self._adjacency_matrix)
        # Process the result
        #self._process_clustering_results(labels)
        return labels, self._adjacency_matrix

    @classmethod
    def aggregate_parameter(
            cls, all_worker_data: MutableMapping[int, Any]
    ) -> ModelParameter:
        """
            aggregate (compute the weighted average) parameters from all worker
            use weighted_avg parent method
        """
        assert all_worker_data
        assert all(
            isinstance(parameter, ParameterMessage)
            for parameter in all_worker_data.values()
        )
        parameter = AggregationAlgorithm.weighted_avg(
            all_worker_data,
            AggregationAlgorithm.get_ratios(all_worker_data),
        )
        assert parameter
        return parameter

    @classmethod
    def __aggregate_loss(cls, all_worker_data: MutableMapping[int, Message]) -> dict:
        """
            compute the weighted average for training and validation loss values
            from worker data
        """
        assert all_worker_data
        loss_dict = {}
        for worker_data in all_worker_data.values():
            for loss_type in ("training_loss", "validation_loss"):
                if loss_type in worker_data.other_data:
                    loss_dict[loss_type] = AggregationAlgorithm.weighted_avg_for_scalar(
                        all_worker_data,
                        AggregationAlgorithm.get_ratios(all_worker_data),
                        scalar_key=loss_type,
                    )
            break
        assert loss_dict
        # for worker_data in all_worker_data.values():
        #    for loss_type in ("training_loss", "validation_loss"):
        #        # remove loss data from other_data
        #        worker_data.other_data.pop(loss_type, None)
        return loss_dict

    @classmethod
    def __check_and_reduce_other_data(
            cls, all_worker_data: MutableMapping[int, Message]
    ) -> dict:
        """
            check other_data fields across all worker data are consistent (no discrepancies)
        """
        result: dict = {}
        for worker_data in all_worker_data.values():
            for k, v in worker_data.other_data.items():
                if k not in result:
                    result[k] = v
                    continue
                if v != result[k]:
                    log_error("different values on key %s", k)
                    raise RuntimeError(f"different values on key {k}")
        return result
