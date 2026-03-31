import os
import json
import torch
import dill
import pickle
import time
from federated_learning_simulation_lib.worker.aggregation_worker import AggregationWorker, Worker
from federated_learning_simulation_lib.graph_worker import GraphWorker
from other_libs.log import log_debug, log_info, log_warning
from torch_kit import (ExecutorHookPoint, MachineLearningPhase,  # noqa
                       ModelParameter, StopExecutingException,  # noqa
                       tensor_to, Inferencer, EvaluationMode, ModelEvaluator, DatasetCollection)
import torch
import numpy as np
import copy
from torch_medical import transform
from ..message import (DeltaParameterMessage, Message, ParameterMessage,
                       ParameterMessageBase)
from ..util import ModelCache, load_parameters
from ..worker.client import ClientMixin
from ..executor import ExecutorContext

class GraphAggregationWorker(GraphWorker, AggregationWorker, ClientMixin):  # AggregationWorker
    def __init__(self, **kwargs):
        GraphWorker.__init__(self, **kwargs)
        AggregationWorker.__init__(self, **kwargs)
        ClientMixin.__init__(self)
        self._communicate_node_state: bool = True
        self.__choose_model_by_validation: bool | None = None
        self.__model_cache: ModelCache = ModelCache()
        self.__worker_mi_evaluator: Inferencer
        self.__global_mi_evaluator: Inferencer
        self._was_selected_this_round: bool = True
        self.__mi: float = 0.0
        self._training_start_time: float = 0.0
        self._training_ms: float = 0.0
        self._mi_computation_ms: float = 0.0
        self._last_trained_round: int = 0

    def num_mi_trials(self) -> int:
        return self.config.num_mi_trials

    def mi_update_interval(self) -> int:
        return self.config.mi_update_interval

    def _before_training(self) -> None:
        """
        Hook called just before local SGD starts.
        Records wall-clock start time for training-duration measurement.
        """
        self._training_start_time = time.perf_counter()
        if hasattr(super(), "_before_training"):
            super()._before_training()

    def _after_training(self) -> None:
        """
        Hook called after local training completes.

        `self.round_index` is NOT reliable here: AggregationWorker.__get_result_from_server
        increments _round_index for every round the worker is skipped (not selected),
        and those skips happen inside __get_result_from_server which is called from
        _aggregation (AFTER_EXECUTE hook) BEFORE _after_training runs.  By the time
        _after_training is called, round_index may have advanced past the actual
        training round.

        Fix: _get_sent_data (called earlier in the same _aggregation hook, before
        __get_result_from_server) captures self.round_index into self._last_trained_round.
        That gives us the correct round without any manifest polling.

        If _last_trained_round == 0 the worker never actually trained (cleanup call
        at end of loop) — defer to the base class only.
        """
        current_round = self._last_trained_round

        if current_round == 0:
            # Cleanup / end-of-training call; nothing to do here.
            AggregationWorker._after_training(self)
            return

        # ── Training wall-clock ───────────────────────────────────────────────
        if self._training_start_time > 0.0:
            self._training_ms = (time.perf_counter() - self._training_start_time) * 1_000.0
            self._training_start_time = 0.0
        else:
            self._training_ms = 0.0

        # The worker reached _after_training because _get_sent_data was called,
        # which only happens when the AFTER_EXECUTE hook fires — i.e. the worker
        # actually trained this round and was selected.
        self._was_selected_this_round = True

        AggregationWorker._after_training(self)

        # ── MI computation ────────────────────────────────────────────────────
        mi_interval = self.mi_update_interval()
        _should_update_mi = (
            current_round % mi_interval == 0
            or current_round == 1
            or self._state.mi == 0.0
        )
        if self.config.round > current_round:
            if _should_update_mi:
                _mi_start = time.perf_counter()
                self._compute_worker_mi(for_round=current_round)
                self._mi_computation_ms = (time.perf_counter() - _mi_start) * 1_000.0
                log_info(
                    "Worker %s MI updated at round %s (interval=%s)",
                    self.worker_id, current_round, mi_interval
                )
            else:
                self.__mi = self._state.mi if self._state.mi is not None else 0.0
                self._mi_computation_ms = 0.0
                log_info(
                    "Worker %s MI reused (round %s, interval=%s, cached=%.4f)",
                    self.worker_id, current_round, mi_interval, self.__mi
                )
        else:
            self._mi_computation_ms = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Manifest helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _poll_for_manifest(self, manifest_path: str, timeout: float = 90.0) -> dict | None:
        """
        Block until the manifest file appears or timeout expires.
        Uses short exponential backoff to minimise filesystem pressure.
        Returns the parsed manifest dict, or None on timeout.
        """
        initial_wait = 2.0
        max_wait = 2.0
        backoff = 1.5
        start = time.time()
        wait = initial_wait
        attempt = 0

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                log_warning(
                    "Worker %s: manifest not found at %s after %.0fs.",
                    self.worker_id, manifest_path, elapsed
                )
                return None

            attempt += 1
            manifest = self._try_load_manifest(manifest_path)
            if manifest is not None:
                return manifest

            if attempt % 5 == 1:
                log_info(
                    "Worker %s: waiting for manifest %s (attempt %d, elapsed %.0fs).",
                    self.worker_id, manifest_path, attempt, elapsed
                )
            time.sleep(min(wait, max_wait))
            wait = min(wait * backoff, max_wait)

    def _try_load_manifest(self, manifest_path: str) -> dict | None:
        """
        Safely load a round manifest JSON.
        Returns None if not yet available or being written (mid-write race).
        Never raises.
        """
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Global model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _get_aggregated_model_from_path(self, round_idx: int) -> ModelParameter | None:
        """
        Load the aggregated global model for use in MI computation.

        BUG FIX (model index):
        MI compares the worker's LOCAL model (post-training) against the GLOBAL
        model that was broadcast at the START of this round.  That global model
        was produced by aggregation round (round_idx - 1) and saved as
        round_{round_idx-1}.pk.

        Old code had:
            round_idx = round_idx - 1 if round_idx != 1 else round_idx
        which is correct for R>1 but for R=1 it loaded round_1.pk which
        doesn't exist at the time _compute_worker_mi runs (round 1 aggregation
        hasn't happened yet).  For round 1, MI against the initial broadcast
        model is not meaningful anyway; the caller (_after_training) skips MI
        computation at round 1 entirely, but _compute_worker_mi itself also
        calls this function.  We guard that case explicitly.

        Protocol:
          Poll for round_{round_idx-1}.pk and load it.
          The caller already determined the worker was selected (it trained).
        """
        base_dir = os.path.join(self.config.save_dir, "aggregated_model")

        # The global model snapshot that was sent at the start of round round_idx
        # was produced by aggregating round (round_idx - 1).
        completed_round = round_idx - 1  # the aggregation whose result we want to load
        if completed_round < 1:
            # round_idx == 1: there is no previous aggregation; caller should not
            # reach here because _after_training skips MI for round 1.
            log_warning(
                "Worker %s: _get_aggregated_model_from_path called with round_idx=1. "
                "No previous model to load.  Returning None.",
                self.worker_id
            )
            return None

        model_path = os.path.join(base_dir, f"round_{completed_round}.pk")

        initial_wait: float = 0.5
        max_wait: float = 30.0
        backoff: float = 1.3
        hard_timeout: float = 900.0
        log_interval: int = 1

        start = time.time()
        wait = initial_wait
        attempt = 0
        previous_size = None

        log_info(
            "Round %s. Worker %s waiting for global model (round_%s.pk, timeout: %ds).",
            round_idx, self.worker_id, completed_round, hard_timeout
        )

        while True:
            elapsed = time.time() - start
            attempt += 1

            if elapsed > hard_timeout:
                raise TimeoutError(
                    f"Round {round_idx}. Worker {self.worker_id} gave up after "
                    f"{elapsed:.0f}s ({attempt} attempts). Model round_{completed_round}.pk "
                    f"not available. Check server logs."
                )

            # ── Load .pk file ────────────────────────────────────────────────
            # The caller already knows this worker was selected for round_idx
            # (it trained, therefore _get_sent_data was called, therefore
            # _last_trained_round was set).  We just need the model file.
            if os.path.exists(model_path):
                try:
                    current_size = os.path.getsize(model_path)
                except OSError as e:
                    log_warning("Round %s. stat() failed: %s. Retrying...", round_idx, e)
                    current_size = None

                if current_size and current_size > 0:
                    if current_size == previous_size:
                        try:
                            with open(model_path, "rb") as f:
                                model_data = pickle.load(f)
                            log_info(
                                "Round %s. Model loaded (attempt %d, %.1fs, %d bytes).",
                                round_idx, attempt, elapsed, current_size
                            )
                            return model_data
                        except (pickle.UnpicklingError, EOFError) as e:
                            log_warning(
                                "Round %s. Corrupt pickle (attempt %d): %s. Retrying...",
                                round_idx, attempt, e
                            )
                            previous_size = None
                        except Exception as e:
                            log_warning(
                                "Round %s. Unexpected read error (attempt %d): %s.",
                                round_idx, attempt, e
                            )
                            previous_size = None
                    else:
                        if attempt % log_interval == 1:
                            log_warning(
                                "Round %s. File size: %s → %d bytes. Still writing...",
                                round_idx, previous_size, current_size
                            )
                        previous_size = current_size
                else:
                    previous_size = current_size
            else:
                if attempt % log_interval == 1:
                    log_warning(
                        "Round %s. Worker %s: model round_%s.pk not found yet "
                        "(attempt %d, elapsed %.0fs).",
                        round_idx, self.worker_id, completed_round, attempt, elapsed
                    )
                previous_size = None

            time.sleep(min(wait, max_wait))
            wait = min(wait * backoff, max_wait)

    # ──────────────────────────────────────────────────────────────────────────
    # Data helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_cached_model(self) -> ModelParameter:
        parameter = self.trainer.model_util.model
        log_debug("************************* worker parameter", parameter)
        model = tensor_to(parameter, device="cpu", dtype=torch.float64)
        log_debug("************************* worker model", model)
        return model

    def _get_test_data_subset(self, ratio: float = 0.5):
        """
        Extract a subset of the validation dataset for MI estimation.
        """
        assert 0 < ratio <= 1, "Ratio must be between 0 and 1."

        if not self.trainer.dataset_collection.has_dataset(MachineLearningPhase.Validation):
            raise ValueError("Validation dataset is not available for this worker.")

        test_dataset = self.trainer.dataset_collection.get_dataset(MachineLearningPhase.Validation)

        if not hasattr(test_dataset, "__getitem__"):
            raise TypeError(f"Dataset {type(test_dataset)} does not support indexing.")

        original_indices = list(range(len(test_dataset)))
        total = len(original_indices)

        log_warning("Total available local validation samples: %d", total)

        if total == 0:
            raise RuntimeError(
                "Worker has 0 validation samples  cannot compute MI. "
                "Consider reducing the number of clients or increasing the dataset size."
            )

        batch_size = self.trainer.hyper_parameter.batch_size
        log_warning("Batch size from trainer: %d", batch_size)

        try:
            num_classes = len(self.trainer.dataset_util.get_label_names())
        except Exception:
            num_classes = 2
        min_reliable = max(batch_size, 5 * num_classes)

        desired = int(ratio * total)
        aligned = (desired // batch_size) * batch_size

        if aligned < batch_size:
            aligned = batch_size
            log_warning(
                "Desired subset (%d) is smaller than one batch (%d) after alignment. "
                "Falling back to one batch.",
                desired, batch_size
            )
        elif aligned != desired:
            log_warning(
                "Subset size adjusted from %d to %d to align with batch_size %d.",
                desired, aligned, batch_size
            )

        if aligned >= total:
            log_warning(
                "Requested subset size (%d) >= available samples (%d). "
                "Using all available samples without random sampling.",
                aligned, total
            )
            sampled_indices = np.array(original_indices)
        else:
            sampled_indices = np.random.choice(original_indices, size=aligned, replace=False)

        self._mi_estimate_reliable = len(sampled_indices) >= min_reliable
        if not self._mi_estimate_reliable:
            log_warning(
                "MI estimate reliability: LOW  only %d samples available, "
                "recommended minimum is %d (%d classes × 5, or 1 batch). "
                "MI value will be computed but should be treated with caution.",
                len(sampled_indices), min_reliable, num_classes
            )
        else:
            log_warning(
                "MI estimate reliability: OK  %d samples selected "
                "(minimum recommended: %d).",
                len(sampled_indices), min_reliable
            )

        log_warning("Final subset size for MI estimation: %d", len(sampled_indices))
        return original_indices, sampled_indices

    def _get_model_results(self, data: torch.utils.data.Dataset, model: ModelParameter) -> np.ndarray:
        inferencer = self.trainer.get_inferencer(
            phase=MachineLearningPhase.Test, deepcopy_model=False
        )
        results = []
        return results

    @staticmethod
    def compare_model_parameters(model1: dict, model2: dict, rtol: float = 1e-5, atol: float = 1e-7) -> bool:
        if model1.keys() != model2.keys():
            missing_in_2 = [k for k in model1.keys() if k not in model2]
            missing_in_1 = [k for k in model2.keys() if k not in model1]
            log_warning("Model parameter keys do not match. missing_in_2=%s missing_in_1=%s", missing_in_2, missing_in_1)
            return False

        all_exact_equal = True
        for key in model1:
            t1 = model1[key]
            t2 = model2[key]
            if t1.shape != t2.shape:
                log_warning("Shape mismatch at '%s': %s vs %s", key, t1.shape, t2.shape)
                return False
            if not torch.allclose(t1, t2, rtol=rtol, atol=atol):
                diff = (t1 - t2).detach()
                max_abs = diff.abs().max().item()
                mean_abs = diff.abs().mean().item()
                l2 = torch.linalg.norm(diff).item()
                log_warning("Value mismatch at '%s': max_abs=%g mean_abs=%g l2=%g rtol=%g atol=%g", key, max_abs, mean_abs, l2, rtol, atol)
                try:
                    numel = diff.numel()
                    if numel > 0:
                        flat = diff.view(-1)
                        topk = min(5, flat.numel())
                        vals, idx = torch.topk(flat.abs(), k=topk)
                        log_debug("Top-%d abs diffs for '%s': %s", topk, key, vals.tolist())
                except Exception:
                    pass
                return False
            if not torch.equal(t1, t2):
                all_exact_equal = False

        if all_exact_equal:
            log_info("All model parameters are bitwise-identical.")
        else:
            log_info("All model parameters are numerically equal within tolerances (not bitwise-identical).")
        return all_exact_equal

    # ──────────────────────────────────────────────────────────────────────────
    # MI computation
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_worker_mi(self, for_round: int | None = None) -> None:
        """
        Compute MI between the worker's model and the global model.

        `for_round` is the actual training round (captured in _last_trained_round).
        Falls back to self.round_index if not provided (legacy callers).
        _get_aggregated_model_from_path loads round_{for_round-1}.pk, which is the
        global model that was broadcast at the start of `for_round`.
        """
        round_idx = for_round if for_round is not None else self.round_index
        log_info("Computing mutual information (MI) for round %s ...", round_idx)

        num_trials = self.num_mi_trials()
        estimated_values = np.zeros(num_trials)
        trainer_inferencer = self.trainer.get_inferencer(
            phase=MachineLearningPhase.Validation,
            inherent_device=True,
            deepcopy_model=True,
        )

        for i in range(num_trials):
            # ── Step 1: worker inferencer ────────────────────────────────────
            self.__worker_mi_evaluator = copy.deepcopy(trainer_inferencer)
            current_local_params = trainer_inferencer.running_model_evaluator.model_util.get_parameters()
            self.__worker_mi_evaluator.running_model_evaluator.model_util.load_parameters(current_local_params)
            trainer_device = trainer_inferencer.device
            log_debug("Using trainer device for worker MI evaluator: %s", trainer_device)
            try:
                self.__worker_mi_evaluator.running_model_evaluator.model_util.to_device(trainer_device)
            except Exception as e:
                log_warning("Failed to move worker model to %s: %s, using CPU", trainer_device, e)
                self.__worker_mi_evaluator.running_model_evaluator.model_util.to_device("cpu")

            assert self.__worker_mi_evaluator.dataset_collection.has_dataset(MachineLearningPhase.Validation)
            model1_params = self.__worker_mi_evaluator.running_model_evaluator.model_util.get_parameters()

            # ── Step 2: global inferencer ────────────────────────────────────
            self.__global_mi_evaluator = copy.deepcopy(self.__worker_mi_evaluator)

            global_model_state_dict = self._get_aggregated_model_from_path(round_idx)
            if global_model_state_dict is None:
                log_warning(
                    "Round %s. Worker %s: global model unavailable, carrying forward MI: %s",
                    round_idx, self.worker_id, self._state.mi
                )
                self.__mi = self._state.mi if self._state.mi is not None else 0.0
                self._state.set_mi(self.__mi)
                return

            self.__global_mi_evaluator.running_model_evaluator.model_util.load_parameters(global_model_state_dict)
            log_warning("Using same device for global MI evaluator: %s", trainer_device)
            try:
                self.__global_mi_evaluator.running_model_evaluator.model_util.to_device(trainer_device)
            except Exception as e:
                log_warning("Failed to move global model to %s: %s, using CPU", trainer_device, e)
                self.__global_mi_evaluator.running_model_evaluator.model_util.to_device("cpu")

            assert self.__global_mi_evaluator.dataset_collection.has_dataset(MachineLearningPhase.Validation)

            # ── Step 3: data subset ──────────────────────────────────────────
            original_indices, sampled_indices = self._get_test_data_subset(0.3)
            self.__worker_mi_evaluator.dataset_collection.set_subset(MachineLearningPhase.Validation, sampled_indices)
            self.__global_mi_evaluator.dataset_collection.set_subset(MachineLearningPhase.Validation, sampled_indices)

            # ── Step 4: compare parameters ───────────────────────────────────
            model2_params = self.__global_mi_evaluator.running_model_evaluator.model_util.get_parameters()
            log_warning("Comparing worker/global inferencer params: %s",
                        self.compare_model_parameters(model1=model1_params, model2=model2_params))

            # ── Step 5: dataset sizes ────────────────────────────────────────
            worker_dataset = self.__worker_mi_evaluator.dataset_collection.get_dataset(MachineLearningPhase.Validation)
            global_dataset = self.__global_mi_evaluator.dataset_collection.get_dataset(MachineLearningPhase.Validation)
            log_warning("Worker dataset size: %d", len(worker_dataset))
            log_warning("Global dataset size: %d", len(global_dataset))

            # ── Step 6: inference ────────────────────────────────────────────
            worker_results = self.__worker_mi_evaluator.get_model_outputs_and_targets(EvaluationMode.Test)
            worker_outputs = worker_results["model_output"].clone().detach().cpu().numpy()
            worker_targets = worker_results["targets"].clone().detach().cpu().numpy()

            server_results = self.__global_mi_evaluator.get_model_outputs_and_targets(EvaluationMode.Test)
            server_outputs = server_results["model_output"].clone().detach().cpu().numpy()
            server_targets = server_results["targets"].clone().detach().cpu().numpy()

            assert server_targets.shape == worker_targets.shape
            assert server_outputs.shape == worker_outputs.shape

            # ── Step 7: MI ───────────────────────────────────────────────────
            labels = self.__worker_mi_evaluator.dataset_util.get_label_names()
            estimated_values[i] = self.estimate_mutual_information(
                X=worker_outputs, Y=server_outputs, possible_values=labels
            )
            log_info("This is resulted calculated MI %s for estimation round %s ", estimated_values[i], round_idx)

            # ── Step 8: reset dataset subset ─────────────────────────────────
            trainer_inferencer.dataset_collection.set_subset(MachineLearningPhase.Validation, original_indices)

        self.__mi = np.mean(estimated_values)
        self._state.set_mi(self.__mi)
        log_warning("this is worker mi %s", self._state.mi)

    # ──────────────────────────────────────────────────────────────────────────
    # MI estimation math
    # ──────────────────────────────────────────────────────────────────────────

    def estimate_mutual_information(self, X, Y, possible_values: list):
        def to_numpy(array):
            if isinstance(array, torch.Tensor):
                return array.cpu().numpy()
            elif isinstance(array, list):
                return np.array(array)
            elif isinstance(array, np.ndarray):
                return array
            elif isinstance(array, dict):
                return np.array(array.values())
            else:
                raise TypeError(f"Unsupported data type: {type(array)}")

        X = to_numpy(X)
        Y = to_numpy(Y)
        assert X.shape == Y.shape, "The shapes of X and Y must match."
        assert possible_values is not None
        possible_values = np.array(list(possible_values))

        if X.ndim > 1:
            X = X.flatten()
        if Y.ndim > 1:
            Y = Y.flatten()

        log_warning("X (worker) unique values: %s, shape: %s", np.unique(X), X.shape)
        log_warning("Y (global) unique values: %s, shape: %s", np.unique(Y), Y.shape)
        log_warning("Possible values from dataset: %s", possible_values)

        n = possible_values.shape[0]
        P = np.zeros((n, n))
        for i, x in enumerate(possible_values):
            for j, y in enumerate(possible_values):
                P[i, j] = np.sum((X == x) * (Y == y))
        P = P + 1e-10
        P = P / (len(X) + n * n * 1e-10)

        P_x = np.sum(P, axis=1)
        P_y = np.sum(P, axis=0)

        MI = 0
        for i, x in enumerate(possible_values):
            for j, y in enumerate(possible_values):
                if P[i, j] > 1e-10:
                    ratio = P[i, j] / (P_x[i] * P_y[j])
                    if ratio > 0:
                        MI += P[i, j] * np.log2(ratio)

        if MI < 0:
            log_warning("Warning: MI is negative (%f), this indicates an error in calculation", MI)
            log_warning("P matrix:\n%s", P)
            log_warning("P_x: %s", P_x)
            log_warning("P_y: %s", P_y)
            MI = max(0, MI)

        return MI

    def _calculate_mutual_information(self, X, Y, log_base: int = 2, unique_values=None) -> float:
        def to_numpy(array):
            if isinstance(array, torch.Tensor):
                return array.cpu().numpy()
            elif isinstance(array, list):
                return np.array(array)
            elif isinstance(array, np.ndarray):
                return array
            elif isinstance(array, dict):
                return np.array(array.values())
            else:
                raise TypeError(f"Unsupported data type: {type(array)}")

        X = to_numpy(X)
        Y = to_numpy(Y)
        assert X.shape == Y.shape, "The shapes of X and Y must match."
        assert unique_values is not None

        X_values = np.array(list(unique_values))
        X_counts = np.zeros(X_values.shape[0], dtype=int)
        for i in range(X.shape[0]):
            for j in range(X_values.shape[0]):
                if X[i] == X_values[j]:
                    X_counts[j] += 1
        log_warning("This is X_values %s, and X_counts %s", X_values, X_counts)
        P_X = X_counts / X_counts.sum()
        H_X = -np.sum(P_X * np.log(P_X + 1e-10) / np.log(log_base))

        unique_Y = np.unique(Y)
        H_X_given_Y = 0
        for y in unique_Y:
            X_given_Y = X[Y == y]
            X_values_given_Y, X_counts_given_Y = np.unique(X_given_Y, return_counts=True)
            P_X_given_Y = X_counts_given_Y / X_counts_given_Y.sum()
            H_X_given_Y += -np.sum(P_X_given_Y * np.log(P_X_given_Y + 1e-10) / np.log(log_base))

        MI = H_X - H_X_given_Y
        return MI

    # ──────────────────────────────────────────────────────────────────────────
    # Message preparation / loading
    # ──────────────────────────────────────────────────────────────────────────

    def _get_sent_data(self) -> ParameterMessageBase:
        # Capture the round this worker actually trained in BEFORE __get_result_from_server
        # can increment _round_index for skipped future rounds.
        self._last_trained_round = self.round_index
        if self.__choose_model_by_validation:
            assert self.best_model_hook is not None
            parameter = self.best_model_hook.best_model["parameter"]
            best_epoch = self.best_model_hook.best_model["epoch"]
            log_debug("use best model best_epoch %s", best_epoch)
        else:
            parameter = self.trainer.model_util.get_parameters()
            best_epoch = self.trainer.hyper_parameter.epoch
            log_debug(
                "use best model best_epoch %s acc %s parameter size %s",
                best_epoch,
                self.trainer.performance_metric.get_epoch_metric(best_epoch, "accuracy"),
                len(parameter),
            )
        parameter = tensor_to(parameter, device="cpu", dtype=torch.float64)
        other_data = {}
        if self._send_loss:
            other_data["training_loss"] = (
                self.trainer.performance_metric.get_epoch_metric(best_epoch, "loss")
            )
            assert other_data["training_loss"] is not None
        self.__model_cache.save()
        log_debug("enabled flag to communicate node_state to server along with sent data: %s", self._communicate_node_state)
        if self._communicate_node_state:
            other_data["node_state"] = self._get_client_state(self.worker_id)
            node_state = other_data["node_state"]
            if hasattr(node_state, "training_ms"):
                node_state.training_ms = self._training_ms
            if hasattr(node_state, "mi_computation_ms"):
                node_state.mi_computation_ms = self._mi_computation_ms
            log_debug(
                "worker %s timing: training=%.1f ms  MI=%.1f ms",
                self.worker_id, self._training_ms, self._mi_computation_ms,
            )
            log_debug("worker %s node_state added to other data: %s", self.worker_id, other_data)
            assert other_data["node_state"] is not None

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
                delta_parameter=self.__model_cache.get_parameter_diff(parameter),
            )
        if not self._keep_model_cache:
            self.__model_cache.discard()
        return message

    def _load_result_from_server(self, result: Message) -> None:
        model_path = os.path.join(
            self.save_dir, "aggregated_model", f"round_{self.round_index}.pk"
        )
        parameter: ModelParameter = {}
        match result:
            case ParameterMessage():
                parameter = result.parameter
                if self._keep_model_cache or self._send_parameter_diff:
                    self.__model_cache.cache_parameter(result.parameter, path=model_path)
            case DeltaParameterMessage():
                assert self.__model_cache.has_data
                self.__model_cache.add_parameter_diff(result.delta_parameter, path=model_path)
                parameter = self.__model_cache.parameter
            case _:
                raise NotImplementedError()

        other_data = result.other_data
        log_warning("load family assignments data from server to check changes !")
        log_info("current family: %s, for worker: %s", self.state.family, self.worker_id)
        new_family = self._load_family_assignment_from_server(other_data)
        log_info("attributed family: %s, for worker: %s", new_family, self.worker_id)
        if new_family != 0 and new_family != self._get_worker_family(self.worker_id):
            log_warning("change to be made for worker %s family", self.worker_id)
            self._set_worker_family(self.worker_id, new_family)
            log_warning("new set family for worker %s is: %s",
                        self.worker_id, self._get_worker_family(self.worker_id))
        load_parameters(
            trainer=self.trainer,
            parameter=parameter,
            reuse_learning_rate=self._reuse_learning_rate,
            loading_fun=self._model_loading_fun,
        )
        if result.end_training:
            self._force_stop = True
            raise StopExecutingException()

    def _load_family_assignment_from_server(self, data: dict) -> int:
        assert data is not None
        if "family_assignment" in data.keys():
            if self.worker_id in data["family_assignment"]:
                family = data["family_assignment"][self.worker_id]
                if family is not None:
                    return family
                else:
                    log_warning("family_assignment not found for worker %s", self.worker_id)
        return 0
