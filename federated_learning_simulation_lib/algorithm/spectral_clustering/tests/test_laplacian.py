import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from laplacian import Laplacian, LaplacianType


class TestLaplacian(unittest.TestCase):

    def setUp(self):
        """Set up a sample similarity matrix for testing."""
        self.similarity_matrix = np.array([
            [0.0, 0.2, 0.3],
            [0.2, 0.0, 0.5],
            [0.3, 0.5, 0.0]
        ])
        self.num_clusters = 2  # Example number of clusters

    def test_similarity_matrix(self):
        """Test returning the similarity matrix itself."""
        laplacian = Laplacian(self.similarity_matrix, LaplacianType.Similarity)
        result = laplacian.compute()
        np.testing.assert_array_almost_equal(result, self.similarity_matrix)

    def test_unnormalized_laplacian(self):
        """Test the unnormalized Laplacian."""
        laplacian = Laplacian(self.similarity_matrix, LaplacianType.Unnormalized)
        expected_result = np.array([
            [0.5, -0.2, -0.3],
            [-0.2, 0.7, -0.5],
            [-0.3, -0.5, 0.8]
        ])
        result = laplacian.compute()
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_random_walk_laplacian(self):
        """Test the random walk normalized Laplacian."""
        laplacian = Laplacian(self.similarity_matrix, LaplacianType.RandomWalk)
        expected_result = np.array([
            [1, -0.40000, -0.60000],
            [-0.28571, 1, -0.71429],
            [-0.37500, -0.62500, 1]
        ])
        result = laplacian.compute()
        np.testing.assert_array_almost_equal(result, expected_result, decimal=5)

    def test_graph_cut_laplacian(self): # still giving wrong results compared to the expected ones !
        """Test the graph cut normalized Laplacian."""
        laplacian = Laplacian(self.similarity_matrix, LaplacianType.GraphCut)
        expected_result = np.array([
            [1, -0.378, -0.474],
            [-0.378, 1, -0.67],
            [-0.474, -0.67, 1]
        ])
        result = laplacian.compute()
        np.testing.assert_array_almost_equal(result, expected_result, decimal=5)

    def test_spectral_decomposition(self):
        """Test the spectral decomposition."""
        laplacian = Laplacian(self.similarity_matrix, LaplacianType.GraphCut)
        eigenvectors = laplacian.spectral_decomposition(self.num_clusters)

        # Since the result of eigenvectors might have different signs,
        # we are only checking if the shapes match
        self.assertEqual(eigenvectors.shape, (self.similarity_matrix.shape[0], self.num_clusters))


if __name__ == "__main__":
    unittest.main()
