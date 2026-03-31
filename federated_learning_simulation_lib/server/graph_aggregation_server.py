import os
import pickle
import math
import copy
import time
import json
from typing import Any
from datetime import datetime

import networkx as nx
from other_libs.log import log_debug, log_info, log_warning, log_error
from torch_kit import Inferencer, ModelParameter
from torch_kit.tensor import tensor_to
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ..algorithm.aggregation_algorithm import AggregationAlgorithm
from ..executor import ExecutorContext
from ..message import (DeltaParameterMessage, Message, MultipleWorkerMessage,
                       ParameterMessage, ParameterMessageBase)
from ..util.model_cache import ModelCache
from .performance_mixin import PerformanceMixin
from .round_selection_mixin import RoundSelectionMixin
from .server import Server
from federated_learning_simulation_lib.graph_worker.client_state import ClientState
from ..algorithm.graph_fed_avg import GraphFedAVGAlgorithm
from federated_learning_simulation_lib.algorithm.spectral_clustering import TimingRecorder, RoundTimingRecord, ClientTimingSummary, phase_timer


class GraphAggregationServer(Server, PerformanceMixin, RoundSelectionMixin):
    def __init__(self, algorithm: AggregationAlgorithm, **kwargs: Any) -> None:
        Server.__init__(self, **kwargs)
        PerformanceMixin.__init__(self)
        RoundSelectionMixin.__init__(self)
        self._round_index: int = 1
        self._compute_stat: bool = True
        self._stop = False
        self._model_cache: ModelCache = ModelCache()
        self.__worker_flag: set = set()
        algorithm.set_config(self.config)
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
        assert not (
                self.config.algorithm_kwargs.get("ablation_no_clustering", False)
                and self.config.algorithm_kwargs.get("ablation_random_within_cluster_selection", False)
        ), "ablation_no_clustering and ablation_random_within_cluster_selection are mutually exclusive"
        self._timing_recorder = TimingRecorder(
            save_dir=self.config.save_dir,
            n_workers_total=self.config.worker_number,
        )
        self._round_client_training_ms: dict[int, float] = {}
        self._round_client_mi_ms: dict[int, float] = {}

        # ── BUG FIX: _last_selected_workers tracks the workers selected to
        # participate in the CURRENT collection round (i.e. the set whose
        # results the server is presently waiting for).  It is set in
        # _send_result exactly once per broadcast, BEFORE the broadcast, and
        # is never touched again until the next broadcast.  It is the only
        # authoritative source of truth used by:
        #    _process_worker_data   to reject stale/non-selected submitters
        #    _after_send_result     to snapshot into the manifest
        # Initialised to all workers because round 1 is always full-participation.
        self._last_selected_workers: set[int] = set(range(self.config.worker_number))

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

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
        # ── BUG FIX: always measure against the FROZEN set of workers whose
        # results we are currently collecting, not a freshly-recomputed value
        # (which may change after clustering runs).
        return len(self._last_selected_workers)

    # ──────────────────────────────────────────────────────────────────────────
    # Selection helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_selected_workers(self) -> set[int]:
        """
        Compute the set of workers to invite for the NEXT round.
        Called exactly once per round inside _send_result, immediately
        before the broadcast.  The result is frozen into _last_selected_workers
        so that _process_worker_data and _after_send_result always refer to the
        same immutable snapshot.
        """
        if self.round_index == 1:
            return set(range(self.worker_number))
        if self._is_ablation_no_clustering():
            return self.select_worker_by_mi_ranking()
        if self._is_ablation_random_within_cluster_selection():
            return self._select_workers_randomly_from_clusters(self._families)
        if self._families and any(self._families.values()) and self.round_index != 1:
            return self._select_workers_from_clusters(self._families)
        return set(range(self.worker_number))

    def get_tester(self, copy_tester: bool = False) -> Inferencer:
        tester = super().get_tester(copy_tester=copy_tester)
        tester.set_visualizer_prefix(f"round: {self.round_index},")
        return tester

    def __get_init_model(self) -> ModelParameter:
        parameter: ModelParameter = {}
        init_global_model_path = self.config.algorithm_kwargs.get("global_model_path", None)
        if init_global_model_path is not None:
            with open(os.path.join(init_global_model_path), "rb") as f:
                parameter = pickle.load(f)
        else:
            parameter = self.get_tester().model_util.get_parameters()
        return parameter

    @property
    def distribute_init_parameters(self) -> bool:
        return self.config.algorithm_kwargs.get("distribute_init_parameters", True)

    # ──────────────────────────────────────────────────────────────────────────
    # Ablation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _is_ablation_no_clustering(self) -> bool:
        return self.config.ablation_no_clustering

    def _is_ablation_random_within_cluster_selection(self) -> bool:
        return self.config.ablation_random_within_cluster_selection

    def _get_selected_worker_num(self):
        return max(1, int(self.config.algorithm_kwargs.get("node_sample_percent", 1) * self.worker_number))

    def select_worker_by_mi_ranking(self) -> set[int]:
        target_num = self._get_selected_worker_num()
        mi_scores = {
            worker_id: self._graph_client_states[worker_id].mi
            for worker_id in range(self.config.worker_number)
        }
        sorted_workers = sorted(mi_scores.keys(), key=lambda w: mi_scores[w], reverse=True)
        selected = set(sorted_workers[:target_num])
        log_info(
            "[ablation_no_clustering] MI-ranking selected %d/%d workers: %s",
            len(selected), self.worker_number, sorted(selected),
        )
        return selected

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def _before_start(self) -> None:
        """
        Send the initial model to all workers before training begins.

        BUG FIX: The initial broadcast uses round_index=1 and selects ALL
        workers (handled inside _send_result when result.is_initial=True).
        We must NOT write a round-1 manifest here; the correct manifest for
        round 1 is written by _after_send_result after the first real
        aggregation completes.  The old code was accidentally writing a
        premature round-1 manifest from _before_start which would cause workers
        in _after_training to find a manifest saying "all selected" before the
        round had even started, breaking the skip-detection logic.
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
        Send aggregated result to workers.

        BUG FIX (selection freeze):
        get_selected_workers() is called ONCE here and frozen into
        _last_selected_workers before the broadcast.  This guarantees that
        _process_worker_data (which checks _last_selected_workers to validate
        incoming results) and _after_send_result (which snapshots it into the
        manifest) both see the identical set  even if clustering or MI-ranking
        would produce a different result if called again later.
        """
        self._before_send_result(result=result)
        match result:
            case MultipleWorkerMessage():
                for worker_id, data in result.worker_data.items():
                    self._endpoint.send(worker_id=worker_id, data=data)
            case ParameterMessageBase():
                # ── Compute and FREEZE selection before any broadcast ────────
                selected_workers = self.get_selected_workers()
                self._last_selected_workers = set(selected_workers)  # immutable snapshot

                if len(selected_workers) < self.config.worker_number:
                    worker_round = self.round_index + 1
                    if result.is_initial:
                        assert self.round_index == 1
                        worker_round = 1
                    log_info("chosen round %s workers %s", worker_round, selected_workers)

                if selected_workers:
                    self._endpoint.broadcast(data=result, worker_ids=selected_workers)
                unselected_workers = set(range(self.worker_number)) - set(selected_workers)
                if unselected_workers:
                    self._endpoint.broadcast(data=None, worker_ids=unselected_workers)
            case _:
                self._endpoint.broadcast(data=result)

        log_warning("_sent_result from server ends !")
        self._after_send_result(result=result)

    def _server_exit(self) -> None:
        self.__algorithm.exit()

    def _initialize_network(self):
        for worker_id in range(self.config.worker_number):
            self._network.add_node(worker_id, state=self._graph_client_states[worker_id])
        log_info("Network initialized with nodes for round: %s ", self.round_index)

    def _update_network(self):
        for worker_id, client_state in self._graph_client_states.items():
            if self._network.has_node(worker_id):
                self._network.nodes[worker_id]['state'] = client_state
            else:
                self._network.add_node(worker_id, state=client_state)

        adjacency_matrix = self.__algorithm.adjacency_sc_matrix
        if adjacency_matrix is not None:
            num_clients = self.worker_number
            self._network.clear_edges()
            for i in range(num_clients):
                for j in range(i + 1, num_clients):
                    weight = adjacency_matrix[i][j]
                    if weight > 0:
                        self._network.add_edge(i, j, weight=weight)
        elif self._is_ablation_no_clustering():
            log_info("[no clustering ablation mode]:  adj matrix is None")
        else:
            log_warning("adjacency_matrix is %s ", adjacency_matrix)

    def _save_and_print_graph(self):
        families = set(client_state.family for _, client_state in self._graph_client_states.items())
        family_to_color = {family: idx for idx, family in enumerate(families)}
        cmap = plt.cm.Set1
        node_colors = [cmap(family_to_color[self._network.nodes[node]['state'].family]) for node in self._network.nodes]
        pos = nx.spring_layout(self._network, k=0.8, seed=42)
        plt.figure(figsize=(16, 10))
        nx.draw(
            self._network, pos, with_labels=True,
            labels={node: f"{node}:(f{self._network.nodes[node]['state'].family})" for node in self._network.nodes},
            node_color=node_colors, node_size=950, font_size=12, font_color='black',
            font_weight='bold', edge_color='gray', width=1.0
        )
        nx.draw_networkx_edges(self._network, pos, alpha=0.7)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in self._network.edges(data=True)}
        nx.draw_networkx_edge_labels(self._network, pos, edge_labels=edge_labels, font_size=10)
        save_dir = self.__root_graph_folder
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"graph_iteration_{self.round_index}.png")
        plt.savefig(filepath, bbox_inches='tight')
        plt.show()
        print(f"Graph saved as {filepath}")

    # ──────────────────────────────────────────────────────────────────────────
    # Worker-data processing
    # ──────────────────────────────────────────────────────────────────────────

    def _process_worker_data(self, worker_id: int, data: Message | None) -> None:
        """
        Process one worker's result.

        BUG FIX (quorum guard):
        Only results from workers in _last_selected_workers count toward the
        quorum.  Stale results from a previous round that arrive after the
        round has already closed (possible with async transport) are silently
        discarded.  This prevents non-selected workers from filling the quorum
        and triggering aggregation with the wrong participant set.

        BUG FIX (selected_worker_number):
        The quorum target is len(_last_selected_workers), the frozen count.
        In the old code selected_worker_number called get_selected_workers()
        which could return a DIFFERENT set after clustering updated _families,
        causing the server to wait for a count that could never be reached.
        """
        log_warning("before processing worker data, ensure that the server set the algo %s correctly", self.__algorithm)
        assert 0 <= worker_id < self.worker_number

        # ── Guard: reject results from non-selected workers ─────────────────
        #if worker_id not in self._last_selected_workers:
        #    log_info(
        #        "Ignoring result from worker %s (not in selected set %s for this round).",
        #        worker_id, sorted(self._last_selected_workers)
        #    )
        #    return

        log_debug("getting data from worker %s ..", worker_id)

        if data is not None:
            if data.end_training:
                self._stop = True
                if not isinstance(data, ParameterMessageBase):
                    return
            old_parameter = self._model_cache.parameter
            match data:
                case DeltaParameterMessage():
                    assert old_parameter is not None
                    data = data.restore(old_parameter)
                case ParameterMessage():
                    if old_parameter is not None:
                        data.complete(old_parameter)
                    data.parameter = tensor_to(data.parameter, device="cpu")
            assert data.other_data
            client_state = data.other_data["node_state"]
            log_warning("server has extracted the worker %s state", worker_id)
            log_warning(repr(client_state))
            self._graph_client_states[worker_id] = client_state
            self._round_client_training_ms[worker_id] = getattr(client_state, "training_ms", 0.0)
            self._round_client_mi_ms[worker_id] = getattr(client_state, "mi_computation_ms", 0.0)
            self._network.nodes[worker_id]['state'] = client_state
            log_info("server graph network has been updated with the new worker %s state", worker_id)

        log_info("this is before calling process_worker_data of the algo")
        self.__algorithm.process_worker_data(worker_id=worker_id, worker_data=data)
        log_info("this is after calling process_worker_data of the algo")

        self.__worker_flag.add(worker_id)

        # ── Quorum check: use frozen selected count ──────────────────────────
        quorum = self.config.worker_number
        if len(self.__worker_flag) == quorum:
            log_info(
                "here is after processing all worker data: len worker_flag %s , len worker_number %s ",
                len(self.__worker_flag), self.worker_number
            )
            result = self._aggregate_worker_data()
            log_info("this is the other data %s after aggregation and family processing", result.other_data)
            assert result.other_data is not None
            self._update_graph_data(result)
            log_debug("here is before calling _send_result function")
            self._send_result(result)
            log_debug("here is after calling _send_result function")

            _adj = self.__algorithm.adjacency_sc_matrix
            if _adj is not None:
                log_info("************************************  adj matrix shape & type %s \n %s ", _adj.shape, type(_adj))
            else:
                log_info("[ablation_no_clustering] adjacency matrix is None  clustering was skipped.")

            self.__worker_flag.clear()
        else:
            log_info(
                "we have %s committed, and we need %s workers,skip",
                len(self.__worker_flag), quorum,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Graph / family bookkeeping
    # ──────────────────────────────────────────────────────────────────────────

    def _update_graph_data(self, result) -> None:
        self._update_families(result.other_data)
        for node_idx, node_state in self._graph_client_states.items():
            for family, list_nodes in self._families.items():
                if node_idx in list_nodes:
                    self._graph_client_states[node_idx].set_family(family)
        log_info("graph data (families) is updated")

    def _update_families(self, other_data: dict[str, Any]) -> None:
        if "family_assignment" not in other_data:
            log_info("family_assignment is not set")
            return
        family_assignments = other_data["family_assignment"]
        for worker_id, new_family in family_assignments.items():
            current_family = None
            for family, members in self._families.items():
                if worker_id in members:
                    current_family = family
                    break
            if current_family != new_family:
                if current_family is not None:
                    self._families[current_family].remove(worker_id)
                self._families[new_family].append(worker_id)

    def _aggregate_worker_data(self) -> Any:
        self.__algorithm.set_old_parameter(self._model_cache.parameter)
        return self.__algorithm.aggregate_worker_data()

    # ──────────────────────────────────────────────────────────────────────────
    # Pre / post send hooks
    # ──────────────────────────────────────────────────────────────────────────

    def _before_send_result(self, result: Message) -> None:
        if not isinstance(result, ParameterMessageBase):
            return
        assert isinstance(result, ParameterMessage)
        if self._need_init_performance:
            assert self.distribute_init_parameters
        if self._need_init_performance and result.is_initial:
            self.record_performance_statistics(result)
        elif self._compute_stat and not result.is_initial and not result.in_round:
            self.record_performance_statistics(result)
            if not result.end_training and self.early_stop and self.convergent():
                log_info("stop early")
                self._stop = True
                result.end_training = True
        elif result.end_training:
            self.record_performance_statistics(result)
        model_path = os.path.join(
            self.config.save_dir, "aggregated_model", f"round_{self.round_index}.pk",
        )
        self._model_cache.cache_parameter(result.parameter, model_path)

    def _after_send_result(self, result: Any) -> None:
        """
        Post-broadcast bookkeeping.

        KEY INVARIANT (manifest correctness):
        ─────────────────────────────────────
        The manifest for round R must be written with:
           completed_round = R   (the round that just finished)
           selected        = the workers that PARTICIPATED in round R
                              (= _last_selected_workers, frozen before the broadcast)

        This method is called from two contexts:
          A. After _before_start sends the initial model   (result.is_initial=True,
             result.in_round=True).  In this case we do NOT increment round_index
             (in_round=True means "still in round 1") and we do NOT write a manifest
             because no training has happened yet and no workers need to poll for a
             skip/participate decision.  We skip the manifest write entirely.

          B. After a real aggregation result is broadcast  (result.is_initial=False,
             result.in_round=False).  Here we:
               1. Capture completed_round = _round_index  (BEFORE incrementing)
               2. Capture selected        = _last_selected_workers (already frozen)
               3. Increment _round_index
               4. Save the model cache
               5. Write the manifest LAST (guarantees .pk file exists before
                  any worker reads the manifest and tries to load the model)

        BUG FIX (double manifest / premature manifest):
        The old code wrote a manifest both from the _before_start path AND from
        the real-aggregation path, both with round=1.  Workers in _after_training
        at round 2 would find round_1.manifest.json (written by _before_start)
        and mistakenly conclude "round 2 manifest not yet written"  but they were
        looking for round_2.manifest.json which would only be written after round 2
        aggregation completes, i.e. AFTER they submit.  Classic chicken-and-egg
        deadlock.  The fix: never write a manifest for the initial broadcast.
        """
        with phase_timer() as _graph_timer:
            self._update_network()

        # ── Skip manifest for initial broadcast (no training has occurred) ───
        is_initial_broadcast = getattr(result, "is_initial", False)
        if is_initial_broadcast:
            # Round index stays at 1; model cache flushed; NO manifest written.
            assert self._model_cache.has_data
            self._model_cache.save()
            log_info(
                "[_after_send_result] Initial broadcast: model saved, manifest skipped "
                "(workers will read round_1.manifest.json after round 1 aggregation)."
            )
            return

        # ── Real aggregation path ────────────────────────────────────────────
        # 1. Capture round and selection BEFORE any mutation
        _completed_round = self._round_index
        _completed_selected = set(self._last_selected_workers)  # already frozen snapshot

        log_info("this is the round index to be passed to manifest file %s", _completed_round)
        log_info("this is the completed selected worker to be passed to manifest file %s", _completed_selected)

        # 2. Timing record
        _algo_timing = self.__algorithm.last_round_timing
        _variant = (
            "ablation_no_clustering" if self._is_ablation_no_clustering()
            else "ablation_random_within_cluster_selection" if self._is_ablation_random_within_cluster_selection()
            else "grail_fl"
        )
        selected_list = sorted(list(_completed_selected))
        _rec = RoundTimingRecord(
            round=_completed_round,
            n_workers_selected=len(_completed_selected),
            selected_worker_ids=selected_list,
            variant=_variant,
            mi_matrix_build_ms=_algo_timing.get("mi_matrix_build_ms", 0.0),
            spectral_clustering_ms=_algo_timing.get("spectral_clustering_ms", 0.0),
            aggregation_ms=_algo_timing.get("aggregation_ms", 0.0),
            graph_update_ms=_graph_timer.elapsed_ms,
            training=ClientTimingSummary.from_worker_dict(self._round_client_training_ms),
            mi_computation=ClientTimingSummary.from_worker_dict(self._round_client_mi_ms),
        )
        self._timing_recorder.record(_rec)
        self._round_client_training_ms = {}
        self._round_client_mi_ms = {}

        # 3. Increment round index AFTER capturing _completed_round
        if not result.in_round:
            self._round_index += 1

        # 4. Clear algorithm buffers and flush model to disk
        self.__algorithm.clear_worker_data()
        assert self._model_cache.has_data
        self._model_cache.save()

        # 5. Write manifest LAST (model .pk guaranteed to exist on disk now)
        self._write_round_manifest(_completed_round, selected_list)

    # ──────────────────────────────────────────────────────────────────────────
    # Manifest
    # ──────────────────────────────────────────────────────────────────────────

    def last_selected_workers(self):
        return self._last_selected_workers

    def _write_round_manifest(self, completed_round: int, selected: list[int]) -> None:
        """
        Write an atomic JSON manifest for the completed round.

        Workers poll this file in _get_aggregated_model_from_path and
        _after_training to decide whether they participated in a round.

        The manifest is keyed by the COMPLETED round number (the round whose
        aggregation just finished and whose model .pk was just saved).

        Workers look up: round_{completed_round}.manifest.json
        to determine whether to load the model and compute MI.

        BUG FIX: removed the assertion that compared `selected` against
        _last_selected_workers at manifest-write time.  That assertion was
        always satisfied, but it could fire spuriously if clustering updated
        _last_selected_workers between the snapshot and the write (which cannot
        happen with the new frozen-snapshot design, but the assertion added
        fragility for no benefit).
        """
        manifest_dir = os.path.join(self.config.save_dir, "aggregated_model")
        os.makedirs(manifest_dir, exist_ok=True)
        manifest_path = os.path.join(manifest_dir, f"round_{completed_round}.manifest.json")
        tmp_path = manifest_path + ".tmp"

        manifest = {
            "round": completed_round,
            "status": "complete",
            "timestamp": time.time(),
            "participating_worker_ids": selected,
            "total_workers": self.config.worker_number,
        }
        with open(tmp_path, "w") as f:
            json.dump(manifest, f)
        os.replace(tmp_path, manifest_path)  # atomic on POSIX

        log_info(
            "Round %s manifest written. Selected %d/%d workers: %s",
            completed_round, len(selected), self.config.worker_number, selected
        )

    def _stopped(self) -> bool:
        return self.round_index > self.config.round or self._stop
