# -*- coding: utf-8 -*-
import numpy as np

def get_clusters_with_alg1(n_sampled: int, weights: np.array):
    """Algorithm 1: Client sampling based on sample size (weights).
    
    Ported from: https://github.com/Accenture/Labs-Federated-Learning/blob/clustered_sampling/py_func/clustering.py
    """

    epsilon = int(10 ** 10)
    # associate each client to a cluster
    augmented_weights = np.array([w * n_sampled * epsilon for w in weights]).astype(np.int64)
    ordered_client_idx = np.flip(np.argsort(augmented_weights))

    n_clients = len(weights)
    distri_clusters = np.zeros((n_sampled, n_clients)).astype(np.int64)

    k = 0
    for client_idx in ordered_client_idx:

        while augmented_weights[client_idx] > 0:

            sum_proba_in_k = np.sum(distri_clusters[k])

            u_i = min(epsilon - sum_proba_in_k, augmented_weights[client_idx])

            distri_clusters[k, client_idx] = u_i
            augmented_weights[client_idx] += -u_i

            sum_proba_in_k = np.sum(distri_clusters[k])
            if sum_proba_in_k == 1 * epsilon:
                k += 1

    distri_clusters = distri_clusters.astype(float)
    for l in range(n_sampled):
        if np.sum(distri_clusters[l]) > 0:
             distri_clusters[l] /= np.sum(distri_clusters[l])

    return distri_clusters
