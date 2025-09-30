import os
import pickle
from typing import Any
from datetime import datetime

import networkx as nx
from other_libs.log import log_debug, log_info, log_warning, log_error
from torch_kit import Inferencer, ModelParameter
from torch_kit.tensor import tensor_to
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ..algorithm.aggregation_algorithm import AggregationAlgorithm
from ..message import (DeltaParameterMessage, Message, MultipleWorkerMessage,
                       ParameterMessage, ParameterMessageBase)
from ..util.model_cache import ModelCache
from .performance_mixin import PerformanceMixin
from .round_selection_mixin import RoundSelectionMixin
from .server import Server
from federated_learning_simulation_lib.graph_worker.client_state import ClientState


class GraphAggregationServer(Server, PerformanceMixin, RoundSelectionMixin):
    def __init__(self, algorithm: AggregationAlgorithm, **kwargs: Any) -> None:
        # initialize parent classes
        Server.__init__(self, **kwargs)
        PerformanceMixin.__init__(self)
        RoundSelectionMixin.__init__(self)
        self._round_index: int = 1
        self._compute_stat: bool = True
        # whether to stop straining
        self._stop = False
        self._model_cache: ModelCache = ModelCache()
        # track which worker have submitted data
        self.__worker_flag: set = set()
        algorithm.set_config(self.config)
        # instance of the aggregation algorithm
        self.__algorithm: AggregationAlgorithm | GraphFedAVGAlgorithm = algorithm
        self._need_init_performance = True
        self._network = nx.Graph()
        self._graph_client_states = {
            worker_id: ClientState(worker_id)
            for worker_id in range(self.config.worker_number)
        }
        self._families = {i: [] for i in range(self.config.family_number)} if self.config.family_number != 0 else {}
        self._initialize_network()
        self.__root_graph_folder = os.path.join("graph_spectral_clustering_images", datetime.now().strftime("%Y_%m_%d_%H-%M-%S"))

    @property
    def early_stop(self) -> bool:
        return self.config.algorithm_kwargs.get("early_stop", False)

    @property
    def graph_client_states(self):
        return self._graph_client_states

    @property
    def algorithm(self):
        return self.__algorithm

    @property
    def round_index(self) -> int:
        return self._round_index

    @property
    def selected_worker_number(self) -> int:
        return len(self.get_selected_workers())

    def get_selected_workers(self) -> set[int]:
        if self._families.values() is not None and self.round_index != 1:
            selected_workers  = self._select_workers_from_clusters(self._families)
            return selected_workers
        return range(self.worker_number)
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
       return self.config.algorithm_kwargs.get("distribute_init_parameters", True )

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
        log_warning("*********** distribute init params. Round %s send_results to workers", self.round_index)

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
                selected_workers = self.get_selected_workers()
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
                unselected_workers = set(range(self.worker_number)) - set(selected_workers)
                if unselected_workers:
                    self._endpoint.broadcast(data=None, worker_ids=unselected_workers)
            case _:
                # if other result type, broadcast to all workers
                self._endpoint.broadcast(data=result)
        # post-send hook
        log_warning("_sent_result from server ends !")
        self._after_send_result(result=result)

    def _server_exit(self) -> None:
        # clean up when the server exits
        self.__algorithm.exit()

    def _initialize_network(self):
        for worker_id in range(self.config.worker_number):
            self._network.add_node(worker_id, state=self._graph_client_states[worker_id])
        log_info("Network initialized with nodes for round: %s ", self.round_index)

    def _update_network(self):
        """
        Update the graph's nodes, edges, and clusters based on the latest client states and similarity matrix.
        """
        # Update client states
        for worker_id, client_state in self._graph_client_states.items():
            if self._network.has_node(worker_id):
                self._network.nodes[worker_id]['state'] = client_state
            else:
                self._network.add_node(worker_id, state=client_state)

        # Update edges based on the new similarity matrix
        adjacency_matrix = self.__algorithm.adjacency_sc_matrix
        if adjacency_matrix is not None:
            num_clients = self.worker_number
            self._network.clear_edges()  # Remove old edges
            for i in range(num_clients):
                for j in range(i + 1, num_clients):
                    weight = adjacency_matrix[i][j]
                    if weight > 0:
                        self._network.add_edge(i, j, weight=weight)
            #self._save_and_print_graph()
            #print("Graph updated with new nodes and edges.")
        else:
            log_warning("adjacency_matrix is %s ", adjacency_matrix)

    def _save_and_print_graph(self):
        """
        Save and print the current state of the graph, coloring nodes by cluster and displaying edge weights.
        """
        # Assign colors to nodes based on their family (cluster)
        families = set(client_state.family for _, client_state in self._graph_client_states.items())
        # Map family to an index
        family_to_color = {family: idx for idx, family in enumerate(families)}
        # Use a colormap with enough distinct colors
        cmap = plt.cm.Set1
        # Assign the same color to the clients members of the same family
        node_colors = [cmap(family_to_color[self._network.nodes[node]['state'].family]) for node in self._network.nodes]

        # Create a layout for the graph with more spacing
        pos = nx.spring_layout(self._network, k=0.8, seed=42)  # k controls spacing, seed ensures reproducibility

        # Draw the graph
        plt.figure(figsize=(16, 10))
        nx.draw(
            self._network,
            pos,
            with_labels=True,
            labels={node: f"{node}:(f{self._network.nodes[node]['state'].family})" for node in self._network.nodes},
            node_color=node_colors,
            node_size=950,
            font_size=12,
            font_color='black',
            font_weight='bold',
            edge_color='gray',
            width=1.0
        )

        # Draw the edges
        nx.draw_networkx_edges(self._network, pos, alpha=0.7)
        edge_labels = {
            (u, v): f"{d['weight']:.2f}" for u, v, d in self._network.edges(data=True)
        }


        # Draw the edges labels with weights formatted to 2 decimal place
        nx.draw_networkx_edge_labels(self._network, pos, edge_labels=edge_labels, font_size=10)


        # Save the graph image
        save_dir = self.__root_graph_folder
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"graph_iteration_{self.round_index}.png")
        # Save the figure without padding
        plt.savefig(filepath, bbox_inches='tight')
        plt.show()
        # Clear the plot
        #plt.clf()
        print(f"Graph saved as {filepath}")


    def _process_worker_data(self, worker_id: int, data: Message | None) -> None:
        """
            process the data received from a worker
            update the aggregation
            rend the result back to worker
        """
        log_warning("before processing worker data, ensure that the server set the algo %s correctly", self.__algorithm)
        assert 0 <= worker_id < self.worker_number
        log_debug("getting data from worker %s ..", worker_id)
        if data is not None:
            if data.end_training:
                # set the stop flag if the training ended
                self._stop = True
                if not isinstance(data, ParameterMessageBase):
                    return
            # retrieve the current cached server model
            old_parameter = self._model_cache.parameter
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
            assert data.other_data
            client_state = data.other_data["node_state"]
            log_warning("server has extracted the worker %s state", worker_id)
            log_warning(repr(client_state))
            self._graph_client_states[worker_id] = client_state
            self._network.nodes[worker_id]['state'] = client_state
            log_info("server graph network has been updated with the new worker %s state", worker_id)
        # process the worker data using the aggregation algorithm
        log_info("this is before calling process_worker_data of the algo")
        self.__algorithm.process_worker_data(worker_id=worker_id, worker_data=data)
        log_info("this is after calling process_worker_data of the algo")
        # add the ID of this worker to the set of processed worker
        self.__worker_flag.add(worker_id)
        if len(self.__worker_flag) == self.selected_worker_number:
            log_info("here is after processing all worker data: len worker_flag %s , len worker_number %s ",
                      len(self.__worker_flag),
                      self.worker_number)
            # aggregate the data from all worker once the data from them is processed
            result = self._aggregate_worker_data()
            # assert result have a "family_assignment" key in other_data
            log_info("this is the other data %s after aggregation and family processing", result.other_data)
            assert result.other_data is not None
            # update the stored graph data stored in the server (we may use _graph_client_states to assign families)
            self._update_graph_data(result)
            # send the aggregated model to server
            log_debug("here is before calling _send_result function")
            self._send_result(result)
            log_debug("here is after calling _send_result function")
            #log_info("************************************  adj matrix from server \n %s ", self.__algorithm.adjacency_sc_matrix)

            log_info("************************************  adj matrix shape & type %s \n %s ",
                     self.__algorithm.adjacency_sc_matrix.shape, type(self.__algorithm.adjacency_sc_matrix))
            # clear the set of worker flag
            self.__worker_flag.clear()
        else:
            # log the status
            log_debug(
                "we have %s committed, and we need %s workers,skip",
                len(self.__worker_flag),
                self.selected_worker_number,
            )

    def _update_graph_data(self, result) -> None:
        # logic ADDED
        self._update_families(result.other_data)
        for node_idx, node_state in self._graph_client_states.items():
            for family, list_nodes in self._families.items():
                if node_idx in list_nodes:
                    self._graph_client_states[node_idx].set_family(family)
        log_info("graph data (families) is updated")

    def _update_families(self, other_data: dict[str, Any]) -> None:
        """
        Update the families based on the 'family_assignment' dict in other_data
        """
        if "family_assignment" not in other_data:
            log_info("family_assignment is not set")
            return

        family_assignments = other_data["family_assignment"]

        for worker_id, new_family in family_assignments.items():
            # find current family of the worker
            current_family = None
            for family, members in self._families.items():
                if worker_id in members:
                    current_family = family
                    break

            # only update if the new family assignment is different
            if current_family != new_family:
                # remove worker_id from the current family if it exists
                if current_family is not None:
                    self._families[current_family].remove(worker_id)

                # assign worker_id to the new family
                self._families[new_family].append(worker_id)

    def _aggregate_worker_data(self) -> Any:
        """
            set the old model params for the aggregation algorithm (the current one)
            call the aggregation algorithm
        """
        self.__algorithm.set_old_parameter(self._model_cache.parameter)
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
        self._model_cache.cache_parameter(result.parameter, model_path)

    def _after_send_result(self, result: Any) -> None:
        """
            operations after sending the aggregated model to the workers
            increment the round index (if not intermediate round)
            clear data from the algorithm
            if end training: save the cached model
        """
        self._update_network()
        if not result.in_round:
            self._round_index += 1
        self.__algorithm.clear_worker_data()
        # delete the condition in order to get the model saved after each iteration
        #if result.end_training or self._stopped():
        assert self._model_cache.has_data
        self._model_cache.save()

    def _stopped(self) -> bool:
        """
            check the maximum round number
        """
        return self.round_index > self.config.round or self._stop
