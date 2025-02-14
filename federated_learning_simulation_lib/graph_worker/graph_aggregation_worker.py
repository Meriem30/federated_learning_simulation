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
                       tensor_to, Inferencer, EvaluationMode, ModelEvaluator)
import torch
import numpy as np

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

    def _get_test_data_subset(self, ratio: float = 0.5) -> torch.utils.data.Dataset:
        """
        Extract a subset of the test dataset.
        """
        assert 0 < ratio <= 1, "Ratio must be between 0 and 1."
        if not self.trainer.dataset_collection.has_dataset(MachineLearningPhase.Validation):
            raise ValueError("Training dataset is not available.")
        test_dataset = self.trainer.dataset_collection.get_dataset(MachineLearningPhase.Validation)
        subset_size = int(ratio * len(test_dataset))
        indices = np.random.choice(len(test_dataset), size=subset_size, replace=False)
        subset = torch.utils.data.Subset(test_dataset, indices) # this ca be done differently following dataset logic
        
        log_debug(f"Extracted test data subset of size %s ", len(subset))
        return subset

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

        # Create an inferencer for the worker-specific model
        self.__worker_mi_evaluator = self.trainer.get_inferencer(
            phase=MachineLearningPhase.Validation,  # Use test phase for inference
            deepcopy_model=True,
        )
        # update the dataset used for this inference
        self.__worker_mi_evaluator.update_dataloader_kwargs(
            dataset=self._get_test_data_subset(0.4)  # Use a subset of the test dataset
        )
        # Get results(output classes & targets) on a non-seen dataset
        worker_results = self.__worker_mi_evaluator.get_model_outputs_and_targets(EvaluationMode.Test) # do not perform backend propagation
        worker_outputs = worker_results["model_output"]
        worker_targets = worker_results["targets"]
        log_debug("got worker inference results of type %s : %s ", type(worker_results["model_output"]))
        worker_outputs = np.array([v.cpu().item() for v in worker_outputs.values()])
        worker_targets = np.array([t.cpu().item() for t in worker_targets.values()])

        print(" WORKER INFERENCE OUTPUTS ", worker_outputs)
        print(" WORKER INFERENCE TARGETS ", worker_targets) # should be the same as the server ones

        # Create an inferencer for the global round-specific model
        self.__global_mi_evaluator = self.trainer.get_inferencer(
            phase=MachineLearningPhase.Validation,
            deepcopy_model=False,
        )

        # Reset the model parameters of created global inferencer to be the one extracted from cache
        global_model_state_dict = self._get_aggregated_model_from_path(self.round_index)
        #global_model = self.__global_mi_evaluator.running_model_evaluator.model.__class__()
        #global_model.load_state_dict(global_model_state_dict)  # Load saved parameters
        #global_model.to("cuda")
        #self.__global_mi_evaluator.running_model_evaluator.model.load_state_dict(global_model.items())
        self.__global_mi_evaluator.running_model_evaluator.model_util.load_parameters(global_model_state_dict) # Replace the model
        self.__global_mi_evaluator.running_model_evaluator.model_util.to_device("cuda")
        #server_results = self.__global_mi_evaluator.get_model_outputs_and_targets(ModelEvaluator.Test)

        ############ To verify
        model1 = self.__worker_mi_evaluator.running_model_evaluator.model_util.get_parameters() # or get underling model
        model2 = self.__global_mi_evaluator.running_model_evaluator.model_util.get_parameters()

        log_warning("************************************************ compare both worker/global inferencer params %s ",
        self.compare_model_parameters(model1=model1, model2=model2))
        ############

        server_results = self.__global_mi_evaluator.get_model_outputs_and_targets(EvaluationMode.Test)  # dont perform backend propagation
        server_outputs = server_results["model_output"]
        server_targets = server_results["targets"]
        server_outputs = np.array(list(server_outputs.values()))
        server_targets = np.array([t.cpu().item() for t in server_targets.values()])
        print(" SERVER INFERENCE OUTPUTS ", server_outputs)
        print(" SERVER INFERENCE TARGETS ", server_targets)  # should be the same as the server ones


        #assert set(worker_outputs.keys()) == set(server_outputs.keys()) # we already verified that we have the same indices before converting to arrays

        res_mi = self._calculate_mutual_information(worker_outputs, server_outputs)
        log_warning("this is resulted calculated mi %s" , res_mi)
        self.__mi = res_mi
        self._state.set_mi(self.__mi)
        log_error("this is worker mi %s", self._state.mi)

        log_warning("Round %s. Loaded worker model from inferencer.", self.round_index)
        # Update the dataloader to use the worker's subset of the test dataset
        self.__worker_mi_evaluator.update_dataloader_kwargs(
            dataset=self._get_test_data_subset()  # Use a subset of the test dataset
        )


    def _calculate_mutual_information(self, X, Y, log_base: int = 2) -> float:
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
        log_warning("this is X %s worker results", X)
        Y = to_numpy(Y) # server results
        log_warning("this is Y %s server results", Y)
        assert X.shape == Y.shape, "The shapes of X and Y must match."

        # Calculate H(X): Entropy of X
        X_values = np.array([0,1])
        X_counts = np.array([0,0])
        for i in range(X.shape[0]):
            if X[i]== X_values[0]:
                X_counts[0] +=1
            elif X[i] == X_values[0]:
                X_counts[1] += 1
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
        log_debug("communicate node state to server with sent data: ", self._communicate_node_state)
        if self._communicate_node_state:
            other_data["node_state"] = (
                self._get_client_state(self.worker_id)
            )
            log_info("worker %s node_state added to other data: %s", self.worker_id, other_data)
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
        log_debug("load family assignments data from server to check changes")
        new_family = self._load_family_assignment_from_server(other_data)
        # change it in the client state
        if new_family != 0 & new_family != self.state.family:
            log_info("change to be made for worker %s. old family:  %s => new family assigned: %s",
                     self.worker_id,
                     self.state.family,
                     new_family)
            # set the new family
            self.state.set_family(new_family)
            log_info("the new effective family state for worker %s is family %s",
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
        return 0
