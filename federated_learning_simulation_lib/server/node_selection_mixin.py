from typing import Any
import random

import torch
from other_libs.algorithm.mapping_op import get_mapping_values_by_key_order
from torch_kit import (
    ExecutorHookPoint,
    MachineLearningPhase,
    cat_tensors_to_vector

)

from federated_learning_simulation_lib.worker.protocol import GraphAggregationWorkerProtocol
from federated_learning_simulation_lib.worker.protocol import WorkerProtocol
from torch_kit import Trainer


class NodeSelectionMixin(GraphAggregationWorkerProtocol, WorkerProtocol):
    def __init__(self):
        self.append_node_selection_hook()

    hook_name = "choose_graph_nodes"
    selection_result: dict[int, set[int]] = {}

    def remove_node_selection_hook(self) -> None:
        self.trainer.remove_named_hook(name=self.hook_name)

    def append_node_selection_hook(self) -> None:
        self.remove_node_selection_hook()
        self.trainer.append_named_hook(
            hook_point=ExecutorHookPoint.BEFORE_EPOCH,
            name=self.hook_name,
            fun=self.__hook_impl,
        )

    def __hook_impl(self, **kwargs: Any) -> None:
        self.update_nodes()

    def update_nodes(self) -> None:
        warmup_rounds: int = self.config.algorithm_kwargs.get("warmup_rounds", 0)
        if self.round_index + 1 <= warmup_rounds:
            return
        sample_indices = self._sample_nodes(self.trainer)
        input_nodes = torch.tensor(sample_indices, dtype=torch.long)

        self.trainer.update_dataloader_kwargs(
            pyg_input_nodes={MachineLearningPhase.Training: input_nodes}
        )

    def _sample_nodes(self) -> list[int]:
        # reset any previously returned node selection list
        # trainer.update_dataloader_kwargs(pyg_input_nodes={})
        # retrieve the percentage of nodes to be sampled
        sample_percent: float = self.config.algorithm_kwargs.get(
            "node_sample_percent", 1.0
        )
        # return a list of all the node indices if it's the default 100%
        if sample_percent >= 1.0:
            return list(self.training_node_indices)
        # otherwise, if random selection is enabled
        if self.config.algorithm_kwargs.get("node_random_selection", False):
            # sample nodes based on created uniform probability distribution across all nodes
            sample_prob = torch.ones(size=(self.trainer.dataset_size,))
            sample_res = torch.multinomial(
                sample_prob,
                int(sample_prob.numel() * sample_percent),
                replacement=False,
            )
            # return a sorted list of sampled nodes indices
            assert sample_res.numel() != 0
            sample_indices = sorted(self.training_node_indices)
            return [sample_indices[idx] for idx in sample_res.tolist()]
        # otherwise, if it's not, select client based on their losses
        inferencer = self.trainer.get_inferencer(
            phase=MachineLearningPhase.Training, deepcopy_model=False
        )
        if "batch_number" in self.trainer.dataloader_kwargs:
            batch_size = (
                    self.trainer.dataset_size
                    / self.trainer.dataloader_kwargs["batch_number"]
            )
            inferencer.remove_dataloader_kwargs("batch_number")
            inferencer.update_dataloader_kwargs(batch_size=batch_size)
        inferencer.update_dataloader_kwargs(ensure_batch_size_cover=True)
        # obtain a dict of the loss values for each node (worker)
        sample_loss_dict = inferencer.get_sample_loss()
        # sort their indices () only the selected nodes
        sample_indices = sorted(sample_loss_dict.keys())
        # iterate over the yielded 'sample_loss_dict' values and concatenate them into one vector 1D of all losses
        sample_loss = cat_tensors_to_vector(
            get_mapping_values_by_key_order(sample_loss_dict)
        )
        # convert the sample losses into probabilities (each loss value is divided by the sum of all losses)
        sample_prob = sample_loss / sample_loss.sum()
        # sample nodes (only a ratio)
        # based on the computed probabilities using multinomial sampling without replacement
        sample_res = torch.multinomial(
            sample_prob, int(sample_prob.numel() * sample_percent), replacement=False
        )
        assert sample_res.numel() != 0
        # return the sampled indices based on the losses
        return [sample_indices[idx] for idx in sample_res.tolist()]

    def select_workers(self) -> set[int]:
        # check to avoid redundant computation
        if self.round_index in self.selection_result:
            return self.selection_result[self.round_index]
        # determine the number of clients to be randomly selected
        random_client_number: int | None = self.config.algorithm_kwargs.get(
            "random_client_number", None
        )
        result: set[int] = set()
        # if specified
        if random_client_number is not None:
            # select randomly
            result = set(
                random.sample(list(range(self.worker_number)), k=random_client_number)
            )
        # if not, select all available workers
        else:
            result = set(range(self.worker_number))
        # store the selected subset under the current round index
        self.selection_result[self.round_index] = result
        return result
