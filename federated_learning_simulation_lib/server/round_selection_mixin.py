import random
import math

import torch
from other_libs.algorithm.mapping_op import get_mapping_values_by_key_order
from torch_kit import MachineLearningPhase, cat_tensors_to_vector


from .protocol import AggregationServerProtocol
from federated_learning_simulation_lib.worker.protocol import WorkerProtocol

class RoundSelectionMixin(AggregationServerProtocol, WorkerProtocol):
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
        loss_client_selection: bool = self.config.algorithm_kwargs.get(
            "loss_client_selection", False
        )
        # determine how many workers to select this round
        if random_client_number is not None:
            k = max(1, min(int(random_client_number), self.worker_number))
        else:
            if node_sample_percent >= 1.0:
                k = self.worker_number
            else:
                k = max(1, min(int(self.worker_number * node_sample_percent), self.worker_number))

        # loss-based selection takes precedence if enabled and not selecting all
        result: set[int] = set()
        if loss_client_selection and k < self.worker_number and getattr(self, "trainer", None) is not None:
            inferencer = self.trainer.get_inferencer(
                phase=MachineLearningPhase.Training, deepcopy_model=False
            )
            if "batch_number" in self.trainer.dataloader_kwargs:
                batch_size = (
                    self.trainer.dataset_size / self.trainer.dataloader_kwargs["batch_number"]
                )
                inferencer.remove_dataloader_kwargs("batch_number")
                inferencer.update_dataloader_kwargs(batch_size=batch_size)
            inferencer.update_dataloader_kwargs(ensure_batch_size_cover=True)

            # get per-worker losses and convert to probabilities
            sample_loss_dict = inferencer.get_sample_loss()
            sample_indices = sorted(sample_loss_dict.keys())
            sample_loss = cat_tensors_to_vector(get_mapping_values_by_key_order(sample_loss_dict))
            sample_prob = sample_loss / sample_loss.sum()

            # multinomial sampling without replacement according to loss
            k = min(k, sample_prob.numel())
            sample_res = torch.multinomial(sample_prob, k, replacement=False)
            assert sample_res.numel() != 0
            result = set(sample_indices[idx] for idx in sample_res.tolist())
        else:
            # uniform random or select all
            if k >= self.worker_number:
                result = set(range(self.worker_number))
            else:
                result = set(random.sample(list(range(self.worker_number)), k=k))
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

    def _select_workers_randomly_from_clusters(self, families: dict) -> set[int]:
        """
        Ablation B  'no-smart-selection' variant.

        Spectral clustering and family assignment run exactly as in full GRAIL-FL.
        The only thing replaced is the within-cluster selection policy:
        instead of centroid + nearest-neighbour prioritisation, every cluster
        contributes a random subset of its members, with the number of slots
        allocated proportionally to cluster size.

        This isolates the contribution of the centroid-aware selection strategy.

        Args:
            families: dict  {family_id: [worker_id, ...]}
                      Same structure passed to _select_workers_from_clusters().

        Returns:
            set[int]  Worker IDs selected for this round.
        """
        # ── 0. Early exits ────────────────────────────────────────────────────
        if self.round_index == 1:
            return set(range(self.worker_number))
        #if self.round_index in self.selection_result:
        #    return self.selection_result[self.round_index]
        # Use a dedicated cache key that survives clear_worker_data()
        cache_key = self.round_index
        if cache_key in self.selection_result:
            return self.selection_result[cache_key]


        # ── 1. Resolve selection budget (same logic as the full method) ───────
        sample_percent: float = self.config.algorithm_kwargs.get("node_sample_percent", 1.0)
        random_client_number: int | None = self.config.algorithm_kwargs.get("random_client_number", None)
        total_clients: int = self.worker_number

        if random_client_number is not None:
            budget = min(random_client_number, total_clients)
        else:
            budget = int(sample_percent * total_clients)
        budget = max(1, budget)

        # ── 2. Filter to non-empty clusters ───────────────────────────────────
        active_families = {fid: members for fid, members in families.items() if members}
        if not active_families:
            # Degenerate: no cluster has members yet  select all
            return set(range(self.worker_number))

        n_families = len(active_families)

        # ── 3. Proportional allocation across clusters ─────────────────────────
        # Each cluster gets slots proportional to its share of the total
        # population, rounded down.  Leftover slots are distributed one-by-one
        # to the clusters with the largest fractional remainders (Hamilton/
        # largest-remainder method) so that the sum always equals `budget`.
        family_sizes = {fid: len(members) for fid, members in active_families.items()}
        total_population = sum(family_sizes.values())

        # Exact (real-valued) quota for each family
        exact_quotas = {
            fid: budget * (size / total_population)
            for fid, size in family_sizes.items()
        }

        # Floor allocation  each cluster gets at least 1 slot if budget allows
        #floor_alloc = {fid: max(1, math.floor(q)) for fid, q in exact_quotas.items()}
        floor_alloc = {fid: math.floor(q) for fid, q in exact_quotas.items()}
        allocated = sum(floor_alloc.values())

        # Distribute leftover slots by largest fractional remainder
        remainders = sorted(
            active_families.keys(),
            key=lambda fid: exact_quotas[fid] - math.floor(exact_quotas[fid]),
            reverse=True,
        )
        leftover = budget - allocated
        for fid in remainders:
            if leftover <= 0:
                break
            floor_alloc[fid] += 1
            leftover -= 1

        # ── 4. Random draw from each cluster ──────────────────────────────────
        selected_workers: set[int] = set()
        for fid, members in active_families.items():
            slots = floor_alloc[fid]
            # Never ask for more than the cluster actually has
            draw_n = min(slots, len(members))
            chosen = random.sample(sorted(members), draw_n)
            selected_workers.update(chosen)

        # ── 5. Cache and return ───────────────────────────────────────────────
        # If rounding left us short (all clusters smaller than their allocation),
        # the result may be < budget  that is correct and expected.
        self.selection_result[cache_key] = selected_workers
        return selected_workers