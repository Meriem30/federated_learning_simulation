""" Similarity functions """

import enum
import numpy as np
from scipy.spatial.distance import cdist
from typing import Union, Dict


class SimilarityType(enum.Enum):
    """ Different types of Similarity Matrix."""
    Gaussian = 'gaussian'
    Euclidean = 'euclidean'
    Cosine = 'cosine'
    Customized = 'customized'
    # This is a good practice to ensure the enum can be looked up from a string.
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls(name.lower())
        except ValueError:
            raise ValueError(f"Unknown similarity type: {name}. Must be one of {[e.value for e in cls]}")


def normalize_data_matrix(data_matrix: np.ndarray) -> np.ndarray:
    """Normalize the data_matrix to have zero mean and unit variance."""
    return (data_matrix - data_matrix.mean(axis=0)) / data_matrix.std(axis=0)


def set_diagonal_zero(matrix: np.ndarray) -> np.ndarray:
    """Set the diagonal elements of a matrix to zero."""
    np.fill_diagonal(matrix, 0)
    return matrix


def gaussian_similarity(data_matrix: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """ Compute the Gaussian (RBF) similarity matrix"""
    pairwise_sq_dists = cdist(data_matrix, data_matrix, 'sqeuclidean')
    similarity_matrix = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    return set_diagonal_zero(similarity_matrix)


def euclidean_similarity(data_matrix: np.ndarray) -> np.ndarray:
    """ Compute the Euclidean similarity matrix"""
    pairwise_dists = cdist(data_matrix, data_matrix, 'euclidean')
    # Normalization to convert distances to similarities in range [0, 1]
    max_dist = np.max(pairwise_dists)
    similarity_matrix = 1 - (pairwise_dists / max_dist)
    return set_diagonal_zero(similarity_matrix)


def cosine_similarity(data_matrix: np.ndarray) -> np.ndarray:
    """Compute the Cosine similarity matrix."""
    similarity_matrix = (np.dot(data_matrix, data_matrix.T) /
            (np.linalg.norm(data_matrix, axis=1).reshape(-1, 1) * np.linalg.norm(data_matrix, axis=1)))
    return set_diagonal_zero(similarity_matrix)
    # OR
    # data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
    # return np.dot(data_normalized, data_normalized.T)


def customized_similarity(data_matrix: np.ndarray, weights: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """ Compute a customized similarity matrix with weighted features"""
    def weighted_sq_euclidean_distance(matrix1, matrix2, params):
        difference = matrix1 - matrix2
        weighted_diff = params * (difference ** 2)
        return np.sum(weighted_diff, axis=-1)

    pairwise_sq_dists = cdist(data_matrix, data_matrix, lambda u, v: weighted_sq_euclidean_distance(u, v, weights))
    similarity_matrix = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    return set_diagonal_zero(similarity_matrix)
    # OR
    # weighted_X = data_matrix * weights
    # pairwise_sq_dists = cdist(weighted_X, weighted_X, 'sqeuclidean')
    # return np.exp(-pairwise_sq_dists / (2 * sigma ** 2))


def compute_similarity_matrix(data_matrix: np.ndarray, similarity_type: SimilarityType, **kwargs) -> np.ndarray:
    """ Compute the similarity matrix based on the specified similarity type """
    # should the data be normalized first
    # data_matrix = normalize_data_matrix(data_matrix)

    if similarity_type == SimilarityType.Gaussian:
        return gaussian_similarity(data_matrix, **kwargs)
    elif similarity_type == SimilarityType.Euclidean:
        return euclidean_similarity(data_matrix)
    elif similarity_type == SimilarityType.Cosine:
        return cosine_similarity(data_matrix)
    elif similarity_type == SimilarityType.Customized:
        return customized_similarity(data_matrix, **kwargs)
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")







