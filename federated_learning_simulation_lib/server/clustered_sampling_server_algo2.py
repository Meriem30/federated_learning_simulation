import random
import numpy as np
import pickle
import logging
from scipy.cluster.hierarchy import linkage

from .aggregation_server import AggregationServer
from ..algorithm.clustered_sampling.algo2_utils import get_matrix_similarity, get_clusters_with_alg2

logger = logging.getLogger(__name__)


class ClusteredSamplingServerAlgo2(AggregationServer):
    """
    Server for Clustered Sampling Algorithm 2.
    Uses gradient similarity based clustering for client selection.
    """

    def __init__(self, algorithm, **kwargs):
        super().__init__(algorithm, **kwargs)
        self.selection_result = {}
        # Store last known model for each worker to compute gradients
        # We need this because we need to compute similarity between gradients
        # of *all* clients, even those not selected in the last round (using their last known update).
        self.client_models = {}
        self.global_model_param = None

    def _before_send_result(self, result):
        super()._before_send_result(result)
        # Capture the global model parameter
        if hasattr(result, 'parameter'):
            # result.parameter is a dict of tensors
            self.global_model_param = result.parameter

    def _process_worker_data(self, worker_id: int, data) -> None:
        # Intercept worker data to store their model/gradient

        # We need to ensure we have the full model parameters (restored if delta)
        # to store in our history for clustering.
        # Note: We must be careful not to consume the data if it's an iterator, but here it's a Message.

        data_to_pass = data
        if data is not None:
            # Check if it needs restoration
            if hasattr(data, 'restore') and hasattr(data, 'parameter'):
                # It might be a DeltaParameterMessage
                # We need the old parameter to restore
                old_parameter = self.get_server_cached_model.parameter
                if old_parameter is not None:
                    try:
                        # Try to restore. Note: restore typically returns a NEW message object (ParameterMessage)
                        # or modifies. Assuming it returns new or self.
                        # aggregated_server.py line 164 says: data = data.restore(old_parameter)
                        # We do the same.
                        # Check if it is indeed a Delta
                        from ..message import DeltaParameterMessage
                        if isinstance(data, DeltaParameterMessage):
                            data_to_pass = data.restore(old_parameter)
                    except Exception as e:
                        logger.warning(f"Failed to restore model for Algo 2 history: {e}")

        # Now store the parameters from the (potentially restored) message
        if data_to_pass is not None and hasattr(data_to_pass, 'parameter'):
            # Store a copy of the client's parameters
            # We need it on CPU/Numpy for the utils
            self.client_models[worker_id] = {
                k: v.detach().cpu().clone() for k, v in data_to_pass.parameter.items()
            }

        # Calls the original method to proceed with aggregation
        # We pass the original 'data' or 'data_to_pass'?
        # If we pass 'data_to_pass' (restored), AggregationServer might try to restore again?
        # AggregationServer checks `case DeltaParameterMessage()`.
        # If we converted it to ParameterMessage, AggregationServer will match `case ParameterMessage()`
        # and just complete it (line 168). This is efficient and correct.
        super()._process_worker_data(worker_id, data_to_pass)

    def select_workers(self) -> set[int]:
        """
        Select workers using Clustered Sampling Algorithm 2.
        """
        if self.round_index in self.selection_result:
            return self.selection_result[self.round_index]

        n_clients = self.worker_number

        # Initial round (no history) or insufficient clients: select ALL clients to build model cache
        # We need at least 2 clients to perform clustering (linkage).
        # Also, per requirements, Round 1 should involve all clients.
        # Initial round: select ALL clients to build model cache
        # We need at least 2 clients to perform clustering (linkage).
        # Also, per requirements, Round 1 should involve all clients.
        if self.round_index == 1:
            return set(range(n_clients))

        # Insufficient history check (fallback)
        if len(self.client_models) < 2:
            # Can't cluster with < 2 clients.
            # Return random sample or all? returning all is safer to rebuild history
            return set(range(n_clients))

        # Get n_sampled
        random_client_number = self.config.algorithm_kwargs.get("random_client_number", None)
        node_sample_percent = self.config.algorithm_kwargs.get("node_sample_percent", 1.0)

        if random_client_number is not None:
            n_sampled = max(1, min(int(random_client_number), n_clients))
        else:
            n_sampled = max(1, min(int(n_clients * node_sample_percent), n_clients))

        # We need models for ALL clients.
        # If a client hasn't participated yet, we can't cluster it effectively based on gradient.
        # Fallback: Treat missing clients as having zero gradient diff or average?
        # Or just cluster available ones and random sample others?
        # The algo implies we have gradients for all.
        # In FL, we only have recent updates from selected clients.
        # "Clustered Sampling" papers often assume we can query gradients or use stale ones.
        # We will use stale models from self.client_models.
        # For clients never seen, we might need to select them to get an update?

        # Strategy:
        # 1. Identify clients with known models.
        # 2. Identify clients without known models.
        # 3. If we have enough known models to cluster, run Algo 2 on them.
        # 4. Mix in some unknown clients?

        known_clients = list(self.client_models.keys())
        missing_clients = list(set(range(n_clients)) - set(known_clients))

        if not known_clients:
            # Should be covered by initial check, but safety fallback
            return set(random.sample(range(n_clients), k=n_sampled))

        # Prepare data for Algo 2
        # It needs global model and list of local models
        # Ensure we have global model
        if self.global_model_param is None:
            # Should have been set by _before_send_result of previous round
            # Or init.
            # If missing, fallback to random
            return set(random.sample(range(n_clients), k=n_sampled))

        # Create list of models aligned with known_clients indices
        local_models_list = [self.client_models[uid] for uid in known_clients]

        # compute similarity matrix
        distance_type = self.config.algorithm_kwargs.get("distance_type", "cosine")
        metric_matrix = get_matrix_similarity(self.global_model_param, local_models_list, distance_type)

        # compute linkage
        # linkage requires condensed distance matrix or full
        # metric_matrix is square (n, n)
        # scipy linkage works on condensed, but can handle square if we be careful?
        # Actually `scipy.cluster.hierarchy.linkage` expects condensed distance matrix usually (y)
        # or (n, m) observation matrix (X).
        # We have a distance matrix. We need to convert to condensed form `scipy.spatial.distance.squareform`
        from scipy.spatial.distance import squareform

        # metric_matrix might not be perfectly symmetric/zero-diag due to float errors, force it?
        # It's computed by get_matrix_similarity which calls get_similarity.
        # L1/L2 should be symmetric.

        # Algo 2 expects linkage matrix
        # standard linkage:
        condensed_dist = squareform(metric_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='ward')  # Default to ward or 'average'

        # weights: same as Algo 1, need sample sizes.
        # Reuse logic from Algo 1 or assume uniform
        weights = np.array([1.0 for _ in known_clients])  # simplified
        weights = weights / np.sum(weights)

        # Run Algo 2
        # n_sampled for this subset?
        # We might want to scale n_sampled if we are only looking at a subset of clients
        n_sampled_subset = min(len(known_clients), n_sampled)

        clusters = get_clusters_with_alg2(linkage_matrix, n_sampled_subset, weights)

        # Log family assignments
        # clusters matches indices of known_clients
        family_dict = {}
        for idx, client_id in enumerate(known_clients):
            # clusters[idx] is the cluster ID
            family_dict[client_id] = int(clusters[idx])

        unique_cluster_ids = set(clusters)
        logger.info(
            "[Round %s] Clustering Algo 2: Clients involved: %s, Target clusters: %s, Actual clusters found: %s",
            self.round_index, len(known_clients), n_sampled_subset, len(unique_cluster_ids))
        logger.info("[Round %s] new family assignments dict %s", self.round_index, family_dict)

        # Select 1 representative per cluster
        # If we found more clusters than n_sampled (e.g. due to criterion constraints), we must sample the clusters themselves.
        unique_clusters = list(unique_cluster_ids)
        if len(unique_clusters) > n_sampled_subset:
            chosen_clusters = random.sample(unique_clusters, k=n_sampled_subset)
        else:
            chosen_clusters = unique_clusters

        selected_subset_indices = []

        for c_id in chosen_clusters:
            # find indices of clients in this cluster
            indices_in_cluster = np.where(clusters == c_id)[0]
            # Pick one random
            chosen_idx = np.random.choice(indices_in_cluster)
            selected_subset_indices.append(chosen_idx)

        selected_workers = set([known_clients[i] for i in selected_subset_indices])

        # If we have missing clients (new ones), maybe add some of them to explore?
        # Using epsilon-greedy type approach or filling remaining slots
        if len(selected_workers) < n_sampled and missing_clients:
            needed = n_sampled - len(selected_workers)
            # select random from missing
            c = random.sample(missing_clients, k=min(needed, len(missing_clients)))
            selected_workers.update(c)

        # If still need more, pick random from known
        while len(selected_workers) < n_sampled:
            remaining = list(set(range(n_clients)) - selected_workers)
            if not remaining:
                break
            selected_workers.add(random.choice(remaining))

        self.selection_result[self.round_index] = selected_workers
        return selected_workers
