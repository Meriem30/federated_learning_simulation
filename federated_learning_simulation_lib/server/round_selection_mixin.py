import random

from .protocol import AggregationServerProtocol


class RoundSelectionMixin(AggregationServerProtocol):
    """
        extend the AggregationServerProtocol
        manage the selection of a subset of workers
    """
    # initialize a dict to store the selection result (round_idx, set(clients))
    selection_result: dict[int, set[int]] = {}

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

    def _select_cluster_workers(self, cluster_nodes: list) -> set[int]:
        if self.round_index in self.selection_result:
            return self.selection_result[self.round_index]
        sample_percent: float = self.config.algorithm_kwargs.get(
            "node_sample_percent", 1.0
        )
        random_client_number: int | None = self.config.algorithm_kwargs.get(
            "random_client_number", None
        )
        # if all workers are to be selected, return all the cluster set
        if sample_percent >= 1.0 or random_client_number == self.worker_number:
            return set(cluster_nodes)
        assert sample_percent
        # calculate the number of workers to be selected from this cluster
        cluster_worker_number = int(sample_percent * len(cluster_nodes))
        # ensure we don't try to sample more elements than available
        cluster_worker_number = min(cluster_worker_number, len(cluster_nodes))
        # sample randomly the specified number of workers from this cluster
        result = set(random.sample(sorted(list(cluster_nodes)), k=cluster_worker_number))
        # add the resulted set of randomly selected workers to the var selection_result
        self.selection_result[self.round_index] = result

        return result












