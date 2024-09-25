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
            return sample_loss

    def __collect_sample_loss(
        self, sample_loss: dict, result, sample_indices, **kwargs
    ) -> None:
        assert not result["is_averaged_loss"]
        if isinstance(sample_indices, torch.Tensor):
            sample_indices = sample_indices.tolist()
        sample_loss.update(zip(sample_indices, result["loss"]))