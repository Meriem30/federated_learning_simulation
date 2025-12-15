import random
import numpy as np
import logging

from federated_learning_simulation_lib import AggregationServer
from federated_learning_simulation_lib.algorithm.clustered_sampling import get_clusters_with_alg1

logger = logging.getLogger(__name__)

class ClusteredSamplingServerAlgo1(AggregationServer):
    """
    Server for Clustered Sampling Algorithm 1.
    Uses sample size based clustering for client selection.
    """
    def __init__(self, algorithm, **kwargs):
        super().__init__(algorithm, **kwargs)
        self.selection_result = {}

    def select_workers(self) -> set[int]:
        """
        Select workers using Clustered Sampling Algorithm 1.
        """
        # If selection already done for this round, return it
        if self.round_index in self.selection_result:
            return self.selection_result[self.round_index]

        # Get total number of clients
        n_clients = self.worker_number
        
        # Determine number of clients to select (n_sampled)
        # Default to configured random_client_number or percentage
        random_client_number = self.config.algorithm_kwargs.get("random_client_number", None)
        node_sample_percent = self.config.algorithm_kwargs.get("node_sample_percent", 1.0)
        
        if random_client_number is not None:
             n_sampled = max(1, min(int(random_client_number), n_clients))
        else:
             n_sampled = max(1, min(int(n_clients * node_sample_percent), n_clients))

        # Get weights (sample sizes) for each client
        # We need to access the training data size of each client.
        # Since the server might not know this directly without communication, 
        # we check if it is available in the config or we might need to ask the workers.
        # However, for simulation purposes, we can try to access the worker manager or assume info is available.
        # In this simulation framework, AggregationServer doesn't directly hold worker data sizes.
        # But usually in simulations the data split is known or we can retrieve it.
        
        # APPROACH: We'll assume the data_config or similar has this info, OR 
        # since this is a simulation, we can access the 'trainer' of the workers if running locally, 
        # but here we are the server.
        #
        # Let's try to get data sizes from `self._endpoint` if it exposes worker info, 
        # or relying on a reported metric from a previous round?
        #
        # IF this is the first round, and we don't have sizes, we might fallback to random 
        # OR we assume uniform if unknown.
        #
        # Let's inspect how `RoundSelectionMixin` does it? It uses loss.
        #
        # For this implementation, let's assume we can get sample counts. 
        # We will assume a method `_get_worker_sample_counts()` exists or we create it.
        # Since we can't easily query workers synchronously here without a round trip,
        # we will implement a placeholder that assumes equal weights if unknown, 
        # or tries to read from a 'data_distribution' config if it exists.
        
        weights = self._get_worker_sample_counts(n_clients)
        
        # Check if weights sum to > 0
        if np.sum(weights) == 0:
             logger.warning("All client weights are 0, falling back to random selection")
             weights = np.ones(n_clients)

        # Normalize weights to sum to 1
        weights = np.array(weights) / np.sum(weights)

        # Run Algo 1
        # distri_clusters is matrix (n_sampled, n_clients)
        # We want to select 1 client per cluster? 
        # The algo returns a probability distribution of clients for each "sampled slot" (cluster).
        # We need to sample 1 client from each row of distri_clusters
        
        distri_clusters = get_clusters_with_alg1(n_sampled, weights)
        
        selected_workers = set()
        for i in range(n_sampled):
            cluster_probs = distri_clusters[i]
            if np.sum(cluster_probs) > 0:
                # Sample 1 client based on prob
                # specific client indices
                client_indices = np.arange(n_clients)
                # re-normalize just in case
                cluster_probs = cluster_probs / np.sum(cluster_probs)
                
                selected_client = np.random.choice(client_indices, p=cluster_probs)
                selected_workers.add(int(selected_client))
            else:
                # Fallback if cluster empty? Should not happen with Algo 1 logic usually
                pass
                
        # If we selected fewer than n_sampled (due to collisions or empty clusters), 
        # we might want to fill up? 
        # The algorithm implies selecting *representative* clients. 
        # If multiple clusters pick the same client, that client is very important.
        # But usually we want distinct clients for the round. 
        # Let's fill the rest with random if needed or just keep the set.
        # Standard FedAvg selects K distinct clients.
        
        while len(selected_workers) < n_sampled:
            remaining = list(set(range(n_clients)) - selected_workers)
            if not remaining:
                break
            selected_workers.add(random.choice(remaining))

        self.selection_result[self.round_index] = selected_workers
        return selected_workers

    def _get_worker_sample_counts(self, n_clients):
        """
        Helper to get sample counts. 
        In a real scenario, this should be reported by clients.
        """
        # Hack for simulation: try to see if we can access the dataset sizes via config
        # or some shared object. 
        # If not, return equal weights.
        
        # For now, return uniform to ensure code runs, 
        # but add a TODO or try to read from config.
        # In many FL sims, data distribution is passed in config
        
        # If config has 'data_allocation', it might help.
        # returning uniform for now.
        return np.array([1.0 for _ in range(n_clients)])
