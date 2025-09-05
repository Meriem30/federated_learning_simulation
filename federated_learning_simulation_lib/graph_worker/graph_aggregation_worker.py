import os
import torch
import dill
import pickle
import time
from federated_learning_simulation_lib.worker.aggregation_worker import AggregationWorker, Worker
from federated_learning_simulation_lib.graph_worker import GraphWorker
from other_libs.log import log_debug, log_info, log_error, log_warning
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
#from federated_learning_simulation_lib.algorithm_factory import CentralizedAlgorithmFactory # circular import error
# add nodeSelectionMixin to be inherited here
from ..executor import ExecutorContext

class GraphAggregationWorker(GraphWorker, AggregationWorker, ClientMixin):  # AggregationWorker
    def __init__(self, **kwargs):
        # explicitly cal parent __init__ func
        GraphWorker.__init__(self, **kwargs)
        AggregationWorker.__init__(self, **kwargs)
        ClientMixin.__init__(self)
        self._communicate_node_state: bool = True
        self._num_mi_trials: int = self.config.num_mi_trials
        self.__choose_model_by_validation: bool | None = None
        self.__model_cache: ModelCache = ModelCache()
        self.__worker_mi_evaluator: Inferencer
        self.__global_mi_evaluator: Inferencer
        self.__mi: float = 0.0


    def _after_training(self) -> None:
        """
            super(): save the trainer's hyperparams to a file after training
            for graph_aggregation_worker
        """
        AggregationWorker._after_training(self)
        if self.config.round > self.round_index:
            self._compute_worker_mi()

    def _get_aggregated_model_from_path(self, round_idx: int) -> ModelParameter:
        """
        Load the aggregated model from a saved file, retrying until the file becomes available.
        """
        aggregated_model_path = os.path.join(
            self.config.save_dir,
            "aggregated_model",
            f"round_{round_idx}.pk",  # Adjust round index as needed
        )

        max_retries = 20  # Maximum number of retries
        wait_time = 5  # Wait time in seconds between retries
        previous_size = -1
        for attempt in range(max_retries):
            if os.path.exists(aggregated_model_path):
                #with ExecutorContext(f"File Access Round {round_idx}", is_file_lock=True):  # Lock file access
                    current_size = os.path.getsize(aggregated_model_path)

                    # Ensure the file is stable
                    if current_size == previous_size and current_size > 0:
                        try:
                            with open(aggregated_model_path, "rb") as f:
                                model_data = pickle.load(f)
                                log_info("Round %s. Successfully loaded model from %s",
                                             round_idx, aggregated_model_path)
                                return model_data
                        except (pickle.UnpicklingError, EOFError):
                            log_warning("Round %s. Corrupt file. Retrying (%d/%d)...",
                                            round_idx, attempt + 1, max_retries)
                    else:
                        log_warning("Round %s. File size is changing. Waiting...",
                                        round_idx)

                    previous_size = current_size
                    time.sleep(wait_time)
        # If the file is still not found after retries, raise an error
        raise FileNotFoundError(
            f"Round {self.round_index}. Aggregated model not found or corrupted at {aggregated_model_path} after {max_retries} retries."
        )

    def _get_cached_model(self) -> ModelParameter:
        """
        Retrieve the cached model.
        """
        parameter = self.trainer.model_util.model
        log_debug("************************* worker parameter", parameter)
        model = tensor_to(parameter, device="cpu", dtype=torch.float64)
        log_debug("************************* worker model", model)
        return model

    def _get_test_data_subset(self, ratio: float = 0.5):
        """
        Extract a subset of the test dataset.
        """
        assert 0 < ratio <= 1, "Ratio must be between 0 and 1."
        if not self.trainer.dataset_collection.has_dataset(MachineLearningPhase.Validation):
            raise ValueError("Training dataset is not available.")
        test_dataset = self.trainer.dataset_collection.get_dataset(MachineLearningPhase.Validation)
        # Ensure dataset supports indexing
        if not hasattr(test_dataset, "__getitem__"):
            raise TypeError(f"Dataset {type(test_dataset)} does not support indexing.")

        # Extract only the indices that exist within this worker's dataset
        original_indices = list(range(len(test_dataset)))  # Fix: Get local dataset indices

        log_warning("Total available local worker indices: %d", len(original_indices))

        # Randomly select a subset of indices from the local worker dataset
        subset_size = int(ratio * len(original_indices))
        sampled_indices = np.random.choice(original_indices, size=subset_size, replace=False)

        #log_warning("Selected worker dataset subset indices: %s", sampled_indices)

        # Create the subset using these indices
        #subset = subset_dp(test_dataset, sampled_indices)  # Ensure proper selection
        #subset = torch.utils.data.Subset(test_dataset, sampled_indices)

        log_warning("Extracted test data subset of size: %d", len(sampled_indices))
        return original_indices, sampled_indices

    def _get_model_results(self, data: torch.utils.data.Dataset, model: ModelParameter) -> np.ndarray:
        """
        Perform inference using the given model on the provided data.
        Can be done: calling the tester object, adding a simple forward pass in its logic
        """
        inferencer = self.trainer.get_inferencer(
            phase=MachineLearningPhase.Test, deepcopy_model=False
        )
        results = []
        return results

    @staticmethod
    def compare_model_parameters(model1: dict, model2: dict) -> bool:
        """
        Compare the parameters of two models of the same archi to check if they have the same values.

        Args:
            model1 (dict): Parameters of the first model (state_dict or parameter dictionary).
            model2 (dict): Parameters of the second model (state_dict or parameter dictionary).

        Returns:
            bool: True if all parameters are identical, False otherwise.
        """
        if model1.keys() != model2.keys():
            log_info("Model parameter keys do not match.")
            return False
        for key in model1:
            if not torch.equal(model1[key], model2[key]):
                log_info("Parameter mismatch found in key: %s ", key)
                return False
        log_info("All model parameters are identical.")
        return True

    def _compute_worker_mi(self) -> None:
        """
        Compute the mutual information (MI) between the worker's model and the global model.
        """
        log_info("Computing mutual information (MI)...")

        ############ Start MI estimation ##########
        estimated_values = np.zeros(self._num_mi_trials)
        trainer_inferencer = self.trainer.get_inferencer(
            phase=MachineLearningPhase.Validation,
            inherent_device=True,
            deepcopy_model=True,  # Ensures we copy the model separately
        )

        for i in range(self._num_mi_trials):
            ##################### Step 1: Create Worker Inferencer ##########

            # ✅ Use shallow copy for inferencer to avoid re-copying dataset collection
            self.__worker_mi_evaluator = copy.deepcopy(trainer_inferencer)

            # ✅ Deep copy only the model so that worker_mi_inferencer has its own model
            #self.__worker_mi_evaluator.running_model_evaluator.set_model(copy.deepcopy(
            #    trainer_inferencer.running_model_evaluator.model_util.model ))

            # ✅ Move the model to CUDA for evaluation if available
            if torch.cuda.is_available():
                self.__worker_mi_evaluator.running_model_evaluator.model_util.to_device("cuda")

            assert self.__worker_mi_evaluator.dataset_collection.has_dataset(MachineLearningPhase.Validation)
            model1_params = self.__worker_mi_evaluator.running_model_evaluator.model_util.get_parameters()
            ##################### Step 2: Create Global Inferencer ##########
            self.__global_mi_evaluator = copy.deepcopy(self.__worker_mi_evaluator)  # Full copy of worker inferencer

            # ✅ Load the global model from cache (ensuring only the model changes)
            global_model_state_dict = self._get_aggregated_model_from_path(self.round_index)
            self.__global_mi_evaluator.running_model_evaluator.model_util.load_parameters(global_model_state_dict)
            if torch.cuda.is_available():
                self.__global_mi_evaluator.running_model_evaluator.model_util.to_device("cuda")

            assert self.__global_mi_evaluator.dataset_collection.has_dataset(MachineLearningPhase.Validation)

            ##################### Step 3: Extract Subset for Testing ##########
            original_indices, sampled_indices = self._get_test_data_subset(0.3)

            # ✅ Assign subset to both inferencers
            self.__worker_mi_evaluator.dataset_collection.set_subset(MachineLearningPhase.Validation,
                                                                     sampled_indices)
            self.__global_mi_evaluator.dataset_collection.set_subset(MachineLearningPhase.Validation,
                                                                     sampled_indices)

            ##################### Step 4: Compare Model Parameters ##########

            model2_params = self.__global_mi_evaluator.running_model_evaluator.model_util.get_parameters()
            log_warning("Comparing worker/global inferencer params: %s",
                        self.compare_model_parameters(model1=model1_params, model2=model2_params))

            ##################### Step 5: Verify Dataset Consistency ##########
            worker_dataset = self.__worker_mi_evaluator.dataset_collection.get_dataset(
                MachineLearningPhase.Validation)
            global_dataset = self.__global_mi_evaluator.dataset_collection.get_dataset(
                MachineLearningPhase.Validation)

            log_warning("Worker dataset size: %d", len(worker_dataset))
            log_warning("Global dataset size: %d", len(global_dataset))
            #log_warning("Worker dataset indices range: %s", [item["index"] for item in worker_dataset])
            #log_warning("Global dataset indices range: %s", [item["index"] for item in global_dataset])

            ##################### Step 6: Run Inference ##########
            # Get results from worker inferencer
            worker_results = self.__worker_mi_evaluator.get_model_outputs_and_targets(EvaluationMode.Test)
            worker_outputs = worker_results["model_output"].clone().detach().cpu().numpy()
            worker_targets = worker_results["targets"].clone().detach().cpu().numpy()

            # Get results from global inferencer
            server_results = self.__global_mi_evaluator.get_model_outputs_and_targets(EvaluationMode.Test)
            server_outputs = server_results["model_output"].clone().detach().cpu().numpy()
            server_targets = server_results["targets"].clone().detach().cpu().numpy()

            assert server_targets.shape == worker_targets.shape
            assert server_outputs.shape == worker_outputs.shape

            ##################### Step 7: Compute Mutual Information ##########
            labels = self.__worker_mi_evaluator.dataset_util.get_label_names()
            estimated_values[i] = self.estimate_mutual_information(
                X=worker_outputs, Y=server_outputs, possible_values=labels
            )

            log_info("This is resulted calculated MI %s for estimation round %s ", estimated_values[i], self.round_index)

            ##################### Step 8: Reset Trainer Inferencer ##########
            trainer_inferencer.dataset_collection.set_subset(
                MachineLearningPhase.Validation, original_indices)

        self.__mi = np.mean(estimated_values)
        self._state.set_mi(self.__mi)
        log_warning("this is worker mi %s", self._state.mi)


    def estimate_mutual_information(self, X, Y, possible_values:list):

        # Convert inputs to numpy arrays if they are lists or PyTorch tensors
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

        X = to_numpy(X)  # worker results
        Y = to_numpy(Y)  # server results

        assert X.shape == Y.shape, "The shapes of X and Y must match."
        assert possible_values is not None
        possible_values = np.array(list(possible_values))

        n = possible_values.shape[0]
        P = np.zeros((n, n))
        for i, x in enumerate(possible_values):
            for j, y in enumerate(possible_values):
                P[i, j] = np.sum((X == x) * (Y == y))
        P = P / len(X)
        P_x = np.sum(P, axis=1)
        P_y = np.sum(P, axis=0)
        MI = 0
        for i, x in enumerate(possible_values):
            for j, y in enumerate(possible_values):
                if P[i, j] > 0:
                    MI += P[i, j] * np.log(P[i, j] / (P_x[i] * P_y[j]))

        return MI

    def _calculate_mutual_information(self, X, Y, log_base: int = 2, unique_values=None) -> float:
        """
        Calculate mutual information (MI) between two sets of results.
        MI(X, Y) = H(X) - H(X | Y)
        """
        # Convert inputs to numpy arrays if they are lists or PyTorch tensors
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

        X = to_numpy(X) # worker results
        #log_warning("this is X %s worker results", X)
        Y = to_numpy(Y) # server results
        #log_warning("this is Y %s server results", Y)
        assert X.shape == Y.shape, "The shapes of X and Y must match."
        assert unique_values is not None
        # Calculate H(X): Entropy of X

        X_values = np.array(list(unique_values))
        print(X_values)
        print(X_values.shape[0])
        X_counts = np.zeros(X_values.shape[0], dtype=int)
        # Count occurrences for each unique value in X_values
        for i in range(X.shape[0]):
            for j in range(X_values.shape[0]):  # Loop through all unique values
                if X[i] == X_values[j]:
                    X_counts[j] += 1  # Increment the count for the correct index
        log_warning("This is X_values %s, and X_counts %s", X_values, X_counts)
        P_X = X_counts / X_counts.sum()
        H_X = -np.sum(P_X * np.log(P_X + 1e-10) / np.log(log_base))  # Use dynamic log base

        # Calculate H(X | Y): Conditional entropy of X given Y
        unique_Y = np.unique(Y)
        H_X_given_Y = 0
        for y in unique_Y:
            X_given_Y = X[Y == y]
            X_values_given_Y, X_counts_given_Y = np.unique(X_given_Y, return_counts=True)
            P_X_given_Y = X_counts_given_Y / X_counts_given_Y.sum()
            #H_X_given_Y += -np.sum(P_X_given_Y * np.log2(P_X_given_Y + 1e-10))
            H_X_given_Y += -np.sum(P_X_given_Y * np.log(P_X_given_Y + 1e-10) / np.log(log_base))
        # Compute MI
        MI = H_X - H_X_given_Y
        return MI

    def _get_sent_data(self) -> ParameterMessageBase:
        """
            prepare the data to be sent to the server
        """
        # select the best model (& epoch) on validation if enabled
        if self.__choose_model_by_validation:
            assert self.best_model_hook is not None
            parameter = self.best_model_hook.best_model["parameter"]
            best_epoch = self.best_model_hook.best_model["epoch"]
            log_debug("use best model best_epoch %s", best_epoch)
        # otherwise use the current model
        else:
            parameter = self.trainer.model_util.get_parameters()
            best_epoch = self.trainer.hyper_parameter.epoch
            log_debug(
                "use best model best_epoch %s acc %s parameter size %s",
                best_epoch,
                self.trainer.performance_metric.get_epoch_metric(
                    best_epoch, "accuracy"
                ),
                len(parameter),
            )
        # convert the model params to the CPU
        parameter = tensor_to(parameter, device="cpu", dtype=torch.float64)
        # prepare other data
        other_data = {}
        # add training loss to other_data id necessary
        if self._send_loss:
            other_data["training_loss"] = (
                self.trainer.performance_metric.get_epoch_metric(best_epoch, "loss")
            )
            assert other_data["training_loss"] is not None
        self.__model_cache.save()
        # ADDED to handle Graphs
        #if self.config.round > self.round_index:
        #    self._compute_worker_mi()
        #log_info("******************************************************** saved parameter %s ", self.__model_cache.parameter)
        log_debug("enabled flag to communicate node_state to server along with sent data: %s", self._communicate_node_state)
        if self._communicate_node_state:
            other_data["node_state"] = (
                self._get_client_state(self.worker_id)
            )
            log_debug("worker %s node_state added to other data: %s", self.worker_id, other_data)
            #log_info("worker %s mi in graph_aggregation_worker %s", self.worker_id, self.__mi)
            assert other_data["node_state"] is not None
        # create ParameterMessage or DeltaParameterMessage
        # based on the _send_parameter_diff
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
                # old_parameter=self.__model_cache.parameter,
                # new_parameter=parameter,
                delta_parameter=self.__model_cache.get_parameter_diff(parameter),
            )
        # discard the model cache if necessary
        if not self._keep_model_cache:
            self.__model_cache.discard()
        # returned the prepared message
        return message

    def _load_result_from_server(self, result: Message) -> None:
        """
            load the result received from the server and apply it to the model
        """
        # define the path to save the model
        model_path = os.path.join(
            self.save_dir, "aggregated_model", f"round_{self.round_index}.pk"
        )
        # initialize the parameter
        parameter: ModelParameter = {}
        # check the result message type
        match result:
            case ParameterMessage():
                parameter = result.parameter
                # cache the model
                if self._keep_model_cache or self._send_parameter_diff:
                    self.__model_cache.cache_parameter(result.parameter, path=model_path)
            case DeltaParameterMessage():
                assert self.__model_cache.has_data
                self.__model_cache.add_parameter_diff(
                    result.delta_parameter, path=model_path
                )
                parameter = self.__model_cache.parameter
            case _:
                raise NotImplementedError()
        # ADDED to handle graphs
        # check if family assignment has changed
        other_data = result.other_data
        log_warning("load family assignments data from server to check changes !")
        log_info("current family: %s, for worker: %s", self.state.family, self.worker_id)
        new_family = self._load_family_assignment_from_server(other_data)
        log_info("attributed family: %s, for worker: %s", new_family, self.worker_id)
        # change it in the client state
        if new_family != 0 & new_family != self.state.family:
            log_warning("change to be made for worker %s family")
            # set the new family
            self.state.set_family(new_family)
            log_warning("new set family for worker %s is: %s",
                     self.worker_id,
                     self.state.family)
        # load params into the trainer
        load_parameters(
            trainer=self.trainer,
            parameter=parameter,
            reuse_learning_rate=self._reuse_learning_rate,
            loading_fun=self._model_loading_fun,
        )
        # stop execution if end_training is set in the result
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
