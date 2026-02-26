from typing import List, Dict, Any
import numpy as np
import torch
import copy


def get_matrix_similarity(
        global_model: Dict[str, torch.Tensor],
        local_models: List[Dict[str, torch.Tensor]],
        distance_type: str = "L1"
) -> np.ndarray:
    """
    Computes the similarity matrix (distance matrix) between gradients of clients.

    Args:
        global_model: The global model parameters.
        local_models: List of local model parameters from clients.
        distance_type: Type of distance metric ("L1" or "L2" or "cosine").

    Returns:
        np.ndarray: A square symmetric distance matrix of shape (n_clients, n_clients).
    """
    n = len(local_models)

    # Pre-compute gradients (global - local)
    # We flatten them to vectors for easier distance computation
    gradients = []
    for model in local_models:
        grad_vector = []
        for key in global_model.keys():
            # Assume keys match and are tensors
            # g = w_global - w_local
            diff = global_model[key] - model[key]
            grad_vector.append(diff.view(-1))
        # Concatenate all params to single vector
        gradients.append(torch.cat(grad_vector))

    # Stack into a matrix (n_clients, n_params)
    if n == 0:
        return np.zeros((0, 0))

    grad_matrix = torch.stack(gradients).double()  # double for precision

    # Compute pairwise distances
    if distance_type == "L1":
        # |x - y|
        # pdist helps
        from torch.nn.functional import pdist
        # torch pdist returns condensed
        # we can unroll it to square or keep it
        # But for consistency let's return square matrix as numpy

        # Manually or via broadcasting?
        # pairwise distance:
        dists = torch.cdist(grad_matrix, grad_matrix, p=1)
        return dists.cpu().numpy()

    elif distance_type == "L2":
        dists = torch.cdist(grad_matrix, grad_matrix, p=2)
        return dists.cpu().numpy()

    elif distance_type == "cosine":
        # Cosine similarity is dot / (norm * norm)
        # Cosine distance = 1 - similarity
        # normalize rows
        norm_grad = torch.nn.functional.normalize(grad_matrix, p=2, dim=1)
        sim_matrix = torch.mm(norm_grad, norm_grad.t())
        dist_matrix = 1 - sim_matrix
        # Clamp for num stability
        dist_matrix = torch.clamp(dist_matrix, min=0.0)
        return dist_matrix.cpu().numpy()

    else:
        raise ValueError(f"Unknown distance type: {distance_type}")


def get_clusters_with_alg2(linkage_matrix, n_sampled, weights):
    """
    Wrapper for fcluster or cuts logic to get clusters.
    """
    from scipy.cluster.hierarchy import fcluster
    from copy import deepcopy

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

    # The `distance` criterion with weight checking often fails to merge when
    # n_sampled is high (threshold < weight of any merge).
    # We use `maxclust` to forcedly cut the tree into `n_sampled` clusters.
    # The epsilon/weight logic is kept as it modifies the linkage matrix distances
    # to account for sample sizes, which `maxclust` will still respect (it cuts based on distance).
    clusters = fcluster(
        link_matrix_p, t=n_sampled, criterion="maxclust"
    )

    return clusters

