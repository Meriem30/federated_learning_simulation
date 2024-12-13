import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from similarity import (
    SimilarityType,
    compute_similarity_matrix,
    normalize_data_matrix,
    gaussian_similarity,
    euclidean_similarity,
    cosine_similarity,
    customized_similarity
)


class TestSimilarityFunctions(unittest.TestCase):

    def setUp(self):
        # Setup some example data matrices
        self.data_matrix = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        self.weights = np.array([1.0, 0.5, 0.2])
        self.sigma = 1.0

    @staticmethod
    def check_diagonal_zero(matrix):
        # Helper function to check if all diagonal elements are zero
        return np.all(np.diag(matrix) == 0)

    def test_gaussian_similarity(self):
        sim_matrix = gaussian_similarity(self.data_matrix, sigma=self.sigma)
        print("Gaussian Similarity Matrix:\n", sim_matrix)
        self.assertTrue(self.check_diagonal_zero(sim_matrix), "Gaussian similarity matrix diagonal is not zero")

    def test_euclidean_similarity(self):
        sim_matrix = euclidean_similarity(self.data_matrix)
        print("Euclidean Similarity Matrix:\n", sim_matrix)
        self.assertTrue(self.check_diagonal_zero(sim_matrix), "Euclidean similarity matrix diagonal is not zero")

    def test_cosine_similarity(self):
        sim_matrix = cosine_similarity(self.data_matrix)
        self.assertTrue(self.check_diagonal_zero(sim_matrix), "Cosine similarity matrix diagonal is not zero")

    def test_customized_similarity(self):
        sim_matrix = customized_similarity(self.data_matrix, weights=self.weights, sigma=self.sigma)
        print("Cosine Similarity Matrix:\n", sim_matrix)
        self.assertTrue(self.check_diagonal_zero(sim_matrix), "Customized similarity matrix diagonal is not zero")

    def test_compute_similarity_matrix(self):
        sim_matrix_gaussian = compute_similarity_matrix(self.data_matrix, SimilarityType.Gaussian, sigma=self.sigma)
        self.assertTrue(self.check_diagonal_zero(sim_matrix_gaussian),
                        "Computed Gaussian similarity matrix diagonal is not zero")

        sim_matrix_euclidean = compute_similarity_matrix(self.data_matrix, SimilarityType.Euclidean)
        self.assertTrue(self.check_diagonal_zero(sim_matrix_euclidean),
                        "Computed Euclidean similarity matrix diagonal is not zero")

        sim_matrix_cosine = compute_similarity_matrix(self.data_matrix, SimilarityType.Cosine)
        self.assertTrue(self.check_diagonal_zero(sim_matrix_cosine),
                        "Computed Cosine similarity matrix diagonal is not zero")

        sim_matrix_customized = compute_similarity_matrix(self.data_matrix, SimilarityType.Customized, weights=self.weights, sigma=self.sigma)
        self.assertTrue(self.check_diagonal_zero(sim_matrix_customized),
                        "Computed Customized similarity matrix diagonal is not zero")


if __name__ == '__main__':
    unittest.main()
