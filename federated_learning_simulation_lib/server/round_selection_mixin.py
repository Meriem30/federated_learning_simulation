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
        node_sample_percent: float = self.config.algorithm_kwargs.get(
            "node_sample_percent", 1.0
        )
        result: set[int] = set()
        # if specified
        if random_client_number is not None:
            # select randomly
            result = set(
                random.sample(list(range(self.worker_number)), k=random_client_number)
            )
        # if not, select all available workers
        elif node_sample_percent != 1.0:
            random_client_number = int(node_sample_percent * self.worker_number)
            result = set(
                random.sample(list(range(self.worker_number)), k=random_client_number)
            )
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

    def _select_workers_from_clusters(self, families: dict) -> set[int]:
        """
        Select clients from clusters (families), prioritizing centroids and
        ensuring diversity within the selection constraints.
        """
        if self.round_index == 1:
            return set(range(self.worker_number))
        if self.round_index in self.selection_result:
            return self.selection_result[self.round_index]

        # Extract config values for selection criteria
        sample_percent: float = self.config.algorithm_kwargs.get("node_sample_percent", 1.0)
        random_client_number: int | None = self.config.algorithm_kwargs.get("random_client_number", None)

        all_families = families # Dictionary {family_index: [client_indices]}
        #total_clients = sum(len(members) for members in all_families.values())
        total_clients = self.worker_number

        # Determine the number of workers to select
        if random_client_number is not None:
            num_to_select = min(random_client_number, total_clients)
        else:
            num_to_select = int(sample_percent * total_clients)

        num_to_select = max(1, num_to_select)  # Ensure at least 1 client is selected

        selected_workers = set()

        # Step 1: Prioritize centroids from each cluster
        assert all_families is not None
        family_centroids = {family_id: members[0]  for family_id, members in all_families.items() if members}
        selected_workers.update(family_centroids.values())

        # Step 2: Add more clients from clusters until we reach the required number
        remaining_clients = num_to_select - len(selected_workers)

        if remaining_clients > 0:
            # Step 3: Distribute remaining clients equitably among families
            additional_workers = set()
            family_list = list(all_families.keys())

            # Calculate fair share per family
            base_allocation = remaining_clients // len(family_list)
            extra_slots = remaining_clients % len(family_list)  # Distribute leftovers later

            for family in family_list:
                if family not in family_centroids:
                    print(f"Warning: No centroid found for family {family}, skipping selection.")
                    continue  # Skip this family if no centroid is available

                family_members = set(all_families[family]) - {family_centroids[family]}  # Exclude centroid
                num_to_select = min(base_allocation, len(family_members))  # Ensure we don't select more than available

                if num_to_select > 0:
                    additional_workers.update(random.sample(sorted(family_members), num_to_select))

            # Step 4: Assign remaining clients fairly
            remaining_clients -= len(additional_workers)
            if remaining_clients > 0:
                # Collect remaining available clients (excluding those already selected)
                available_clients = [member for family in family_list for member in all_families[family]
                                     if member not in selected_workers and member not in additional_workers]

                # Randomly pick remaining clients while ensuring fair spread
                extra_workers = random.sample(available_clients, min(remaining_clients, len(available_clients)))
                additional_workers.update(extra_workers)

            selected_workers.update(additional_workers)

        # Store selection result for the current round
        self.selection_result[self.round_index] = selected_workers

        return selected_workers












