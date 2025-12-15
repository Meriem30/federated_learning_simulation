import random
import numpy as np
import pickle
import logging
from scipy.cluster.hierarchy import linkage

from federated_learning_simulation_lib import AggregationServer
from .utils import get_matrix_similarity, get_clusters_with_alg2

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
        # Calls the original method to proceed with aggregation
        super()._process_worker_data(worker_id, data)
        
        if data is not None and hasattr(data, 'parameter'):
             # Store a copy of the client's parameters
             # We need it on CPU/Numpy for the utils
             self.client_models[worker_id] = {
                 k: v.detach().cpu().clone() for k, v in data.parameter.items()
             }

    def select_workers(self) -> set[int]:
        """
        Select workers using Clustered Sampling Algorithm 2.
        """
        if self.round_index in self.selection_result:
            return self.selection_result[self.round_index]

        n_clients = self.worker_number
        
        # Initial round: random selection if no history
        if not self.client_models or len(self.client_models) < 1:
            return set(random.sample(range(n_clients), k=min(n_clients, max(1, int(n_clients * 0.1))))) # Default small sample or use config

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
        distance_type = self.config.algorithm_kwargs.get("distance_type", "L1")
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
        linkage_matrix = linkage(condensed_dist, method='ward') # Default to ward or 'average'
        
        # weights: same as Algo 1, need sample sizes.
        # Reuse logic from Algo 1 or assume uniform
        weights = np.array([1.0 for _ in known_clients]) # simplified
        weights = weights / np.sum(weights)

        # Run Algo 2
        # n_sampled for this subset?
        # We might want to scale n_sampled if we are only looking at a subset of clients
        n_sampled_subset = min(len(known_clients), n_sampled)
        
        clusters = get_clusters_with_alg2(linkage_matrix, n_sampled_subset, weights)
        
        # Select 1 representative per cluster
        unique_clusters = set(clusters) # clusters is array of cluster_ids
        selected_subset_indices = []
        
        for c_id in unique_clusters:
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
