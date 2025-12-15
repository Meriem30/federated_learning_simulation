# -*- coding: utf-8 -*-
import numpy as np
import torch
from itertools import product
from scipy.cluster.hierarchy import fcluster
from copy import deepcopy

def get_similarity(grad_1, grad_2, distance_type="L1"):

    if distance_type == "L1":

        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum(np.abs(g_1 - g_2))
        return norm

    elif distance_type == "L2":
        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum((g_1 - g_2) ** 2)
        return norm

    elif distance_type == "cosine":
        norm, norm_1, norm_2 = 0, 0, 0
        for i in range(len(grad_1)):
            norm += np.sum(grad_1[i] * grad_2[i])
            norm_1 += np.sum(grad_1[i] ** 2)
            norm_2 += np.sum(grad_2[i] ** 2)

        if norm_1 == 0.0 or norm_2 == 0.0:
            return 0.0
        else:
            norm /= np.sqrt(norm_1 * norm_2)

            return np.arccos(norm)


def get_gradients(global_m, local_models):
    """return the `representative gradient` formed by the difference between
    the local work and the sent global model"""

    # Extract info from torch models
    # local_models and global_m are expected to be model dictionaries or similar structures
    # In this simulation framework, models might be passed as ParameterMessage (dicts of tensors)
    
    # helper to flattern or extract params
    def extract_params(model_param):
        # returns list of numpy arrays
        return [t.detach().cpu().numpy() for t in model_param.values()]

    local_model_params = []
    for model in local_models:
        local_model_params.append(extract_params(model))

    global_model_params = extract_params(global_m)

    local_model_grads = []
    for local_params in local_model_params:
        local_model_grads += [
            [
                local_weights - global_weights
                for local_weights, global_weights in zip(
                    local_params, global_model_params
                )
            ]
        ]

    return local_model_grads


def get_matrix_similarity(global_m, local_models, distance_type):

    n_clients = len(local_models)

    local_model_grads = get_gradients(global_m, local_models)

    metric_matrix = np.zeros((n_clients, n_clients))
    for i, j in product(range(n_clients), range(n_clients)):

        metric_matrix[i, j] = get_similarity(
            local_model_grads[i], local_model_grads[j], distance_type
        )

    return metric_matrix


def get_clusters_with_alg2(
    linkage_matrix: np.array, n_sampled: int, weights: np.array
):
    """Algorithm 2"""
    epsilon = int(10 ** 10)

    # associate each client to a cluster
    link_matrix_p = deepcopy(linkage_matrix)
    augmented_weights = deepcopy(weights)

    for i in range(len(link_matrix_p)):
        idx_1, idx_2 = int(link_matrix_p[i, 0]), int(link_matrix_p[i, 1])

        new_weight = np.array(
            [augmented_weights[idx_1] + augmented_weights[idx_2]]
        )
        augmented_weights = np.concatenate((augmented_weights, new_weight))
        link_matrix_p[i, 2] = int(new_weight * epsilon)

    clusters = fcluster(
        link_matrix_p, int(epsilon / n_sampled), criterion="distance"
    )
    
    return clusters
