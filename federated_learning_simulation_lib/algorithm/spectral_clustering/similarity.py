""" Similarity functions """

import enum
import numpy as np
import typing


def compute_affinity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute the affinity matrix from data.

      Note that the range of affinity is [0, 1].

      Args:
        embeddings: numpy array of shape (n_samples, n_features)

      Returns:
        affinity: numpy array of shape (n_samples, n_samples)
      """
    # Normalize the data.
    l2_norms = np.linalg.norm(embeddings, axis=1)
    embeddings_normalized = embeddings / l2_norms[:, None]
    # Compute cosine similarities. Range is [-1,1].
    cosine_similarities = np.matmul(embeddings_normalized,
                                    np.transpose(embeddings_normalized))
    # Compute the affinity. Range is [0,1].
    # Note that this step is not mentioned in the paper!
    affinity = (cosine_similarities + 1.0) / 2.0

    return affinity
