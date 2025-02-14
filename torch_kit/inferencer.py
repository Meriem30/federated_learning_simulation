import functools

import torch
from other_libs.log import log_warning

from .executor import Executor
from .ml_type import (EvaluationMode, ExecutorHookPoint, ModelGradient,
                      StopExecutingException)


class Inferencer(Executor):
    """
        Handle inference tasks
    """
    def inference(
        self,
        evaluation_mode: EvaluationMode = EvaluationMode.Test,
    ) -> bool:
        # Running inference, with an opt param for specifying the inference phase
        # a flag to indicate if the inference was successful or not
        succ_flag: bool = False
        # a flag to determine if gradients are necessary, True unless the mode is Test
        require_grad: bool = EvaluationMode != EvaluationMode.Test
        # Context managers

        with (
            torch.set_grad_enabled(require_grad),
            # Set the device context for the model
            self.device_context,
            # Set the CUDA stream context for asynch. operations
            self.stream_context,
        ):
            try:
                # Prepare the env and the settings for the execution
                self._prepare_execution()
                # Execute one epoch for inference/evaluation
                self._execute_epoch(epoch=1, evaluation_mode=evaluation_mode)
                # Execute any hooks that should be executed after inference/evaluation
                self.exec_hooks(hook_point=ExecutorHookPoint.AFTER_EXECUTE)
                # If all done, raise a success flag
                succ_flag = True
            except StopExecutingException:
                log_warning("stop inference")
            finally:
                # Ensure that all the operations are completed before returning the succ_flag
                self.wait_stream()
            return succ_flag

    def get_gradient(self) -> ModelGradient:
        # Perform inference with gradient calculation and return the computed gradients.
        with self.hook_config:
            self.hook_config.disable_log()
            # A bool var to capture the success status of the inference
            succ: bool = self.inference(
                evaluation_mode=EvaluationMode.TestWithGrad,
            )
            assert succ
            # if the inference was successful, return the calculated grads
            return self.model_util.get_gradients()

    def get_sample_loss(self, evaluation_mode=EvaluationMode.Test) -> dict:
        """
            compute the loss for each sample in the dataset
            and return a dictionary of these losses
        """
        sample_loss: dict = {}
        with self.hook_config:
            self.hook_config.disable_log()
            hook_name = "__collect_sample_loss"
            self.append_named_hook(
                hook_point=ExecutorHookPoint.AFTER_BATCH,
                name=hook_name,
                fun=functools.partial(self.__collect_sample_loss, sample_loss),
            )
            evaluation_kwargs = {
                "reduce_loss": False,
                "need_sample_indices": True,
            }
            self.running_model_evaluator.add_evaluation_kwargs(**evaluation_kwargs)
            try:
                succ: bool = self.inference(evaluation_mode=evaluation_mode)
                assert succ
            finally:
                self.running_model_evaluator.remove_evaluation_kwargs(
                    evaluation_kwargs.keys()
                )
                self.remove_named_hook(name=hook_name)
            assert len(sample_loss) == self.dataset_size
            return sample_loss, sample_target

    def __collect_sample_loss(
        self, sample_loss: dict, result, sample_indices, **kwargs
    ) -> None:
        assert not result["is_averaged_loss"]
        if isinstance(sample_indices, torch.Tensor):
            sample_indices = sample_indices.tolist()
        sample_loss.update(zip(sample_indices, result["loss"]))

    def get_model_outputs_and_targets(self, evaluation_mode=EvaluationMode.Test) -> dict:
        """
        Compute model outputs and targets for each sample in the dataset.
        Returns a dictionary containing:
            - "model_output": {index: output_value}
            - "targets": {index: target_value}
        """
        sample_results = {
            "model_output": {},
            "targets": {}
        }

        with self.hook_config:
            self.hook_config.disable_log()
            hook_name = "__collect_model_output_and_target"

            self.append_named_hook(
                hook_point=ExecutorHookPoint.AFTER_BATCH,
                name=hook_name,
                fun=functools.partial(self.__collect_model_output_and_target, sample_results),
            )

            try:
                succ: bool = self.inference(evaluation_mode=evaluation_mode)
                assert succ
            finally:
                self.remove_named_hook(name=hook_name)

        return sample_results

    def logits_to_classes(self, output: torch.Tensor)-> torch.Tensor:
        """
        Determine whether the model is performing binary, multi-class, or multi-label classification,
        and extract the predicted classes accordingly.

        :param output: The raw model outputs (output).
        :return: Predicted classes
        """
        print("Logits shape:", output.shape)

        # Case 1: Binary classification (one output per sample)
        if output.shape[1] == 1:
            print("Detected: Binary Classification")
            probabilities = torch.sigmoid(output)  # Convert output to probabilities
            predicted_classes = (probabilities > 0.5).long()  # Thresholding at 0.5
            return predicted_classes

        # Case 2: Multi-class classification (one class per sample)
        elif output.dim() == 2 and output.shape[1] > 1:
            # Check if each sample must belong to exactly one class
            if torch.all(output.sum(dim=1) != output.sum()):  # Ensures only one label per row
                print("Detected: Multi-Class Classification")
                probabilities = torch.softmax(output, dim=1)  # Convert output to probabilities
                predicted_classes = torch.argmax(probabilities, dim=1)  # Take the index of max probability
                return predicted_classes

            # Case 3: Multi-label classification (multiple independent labels per sample)
            else:
                print("Detected: Multi-Label Classification")
                probabilities = torch.sigmoid(output)  # Convert output to probabilities
                predicted_classes = (probabilities > 0.5).long()  # Thresholding at 0.5 for multiple labels
                return predicted_classes

        else:
            raise ValueError("Unsupported output shape!")


    def __collect_model_output_and_target(
            self, sample_results: dict, result, sample_indices, **kwargs
    ) -> None:
        """
        Collects model outputs and corresponding targets for each sample index.
        """
        if isinstance(sample_indices, torch.Tensor):
            sample_indices = sample_indices.tolist()

        # Store model outputs after sigmoid/softmax the model logits
        predictions = self.logits_to_classes(output=result["model_output"])
        sample_results["model_output"].update(zip(sample_indices, predictions))
        # Store corresponding targets
        sample_results["targets"].update(zip(sample_indices, result["targets"]))
        assert sample_results["model_output"].keys() == sample_results["targets"].keys()

    def calculate_mutual_information(self, inputs, device):
        # Forward pass through both models

        worker_outputs = self.running_model_evaluator.forward_outputs_only(
            inputs=inputs,
            device=device,
            evaluation_mode=EvaluationMode.Test,
        )
        aggregated_outputs = self.aggregated_model_evaluator.forward_outputs_only(
            inputs=inputs,
            device=device,
            evaluation_mode=EvaluationMode.Test,
        )

