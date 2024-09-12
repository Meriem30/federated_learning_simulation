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
