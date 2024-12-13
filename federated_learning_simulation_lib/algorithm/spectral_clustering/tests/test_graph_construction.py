import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from graph_construction import GraphConstructor, GraphType


class TestGraphConstructor(unittest.TestCase):
    def setUp(self):
        """ Set up the test case environment """
        # Example similarity matrix for testing
        self.similarity_matrix = np.array([
            [0.0, 0.1, 0.3, 0.4],
            [0.1, 0.0, 0.2, 0.3],
            [0.3, 0.2, 0.0, 0.5],
            [0.4, 0.3, 0.5, 0.0]
        ])

    def test_knn_graph(self):
        """ Test KNN graph construction """
        kwargs = {'num_neighbors': 2}  # setting the k parameter
        graph_constructor = GraphConstructor(GraphType.KNN, **kwargs)
        graph = graph_constructor.construct_graph(self.similarity_matrix)

        # Check if the graph is a sparse matrix
        self.assertIsInstance(graph, csr_matrix)

        # Ensure that the graph is not empty
        self.assertGreater(graph.nnz, 0)

        # Check if the diagonal is set to 0
        diagonal = graph.diagonal()
        self.assertTrue(np.all(diagonal == 0))

        # Test that the expected number of neighbors are selected
        num_neighbors = np.sum(graph != 0)
        expected_neighbors = kwargs['num_neighbors'] * self.similarity_matrix.shape[0]  # k neighbors for each point
        self.assertEqual(num_neighbors, expected_neighbors)

    def test_mutual_knn_graph(self):
        """ Test Mutual KNN graph construction """
        kwargs = {'num_neighbors': 2}  # Setting the k parameter
        graph_constructor = GraphConstructor(GraphType.MUTUAL_KNN, **kwargs)
        graph = graph_constructor.construct_graph(self.similarity_matrix)

        # Check if the graph is a sparse matrix
        self.assertIsInstance(graph, csr_matrix)

        # Ensure that the graph is not empty
        self.assertGreater(graph.nnz, 0)

        # Check if the diagonal is set to 0
        diagonal = graph.diagonal()
        self.assertTrue(np.all(diagonal == 0))

        # Ensure the mutual connection logic holds
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if graph[i, j] > 0:
                    self.assertGreater(graph[j, i], 0)

    def test_epsilon_neighborhood_graph(self):
        """Test Epsilon Neighborhood graph construction."""
        kwargs = {'threshold': 0.2}  # Setting the epsilon parameter
        graph_constructor = GraphConstructor(GraphType.EPSILON_NEIGHBORHOOD, **kwargs)
        graph = graph_constructor.construct_graph(self.similarity_matrix)

        # Check if the graph is a sparse matrix
        self.assertIsInstance(graph, csr_matrix)

        # Ensure that the graph is sparse
        self.assertGreater(graph.nnz, 0)

        # Check if the diagonal is set to 0
        diagonal = graph.diagonal()
        self.assertTrue(np.all(diagonal == 0))

    def test_fully_connected_graph(self):
        """Test Fully Connected graph construction."""
        graph_constructor = GraphConstructor(GraphType.FULLY_CONNECTED)
        graph = graph_constructor.construct_graph(self.similarity_matrix)

        # Check if the graph is a sparse matrix
        self.assertIsInstance(graph, csr_matrix)

        # Ensure that the graph is not empty
        self.assertGreater(graph.nnz, 0)

        # Check if the diagonal is set to 0
        diagonal = graph.diagonal()
        self.assertTrue(np.all(diagonal == 0))


if __name__ == '__main__':
    unittest.main()

