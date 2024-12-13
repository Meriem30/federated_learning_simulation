import enum
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix


class GraphType(enum.Enum):
    """ Different types of graph construction """
    EPSILON_NEIGHBORHOOD = 'epsilon-neighborhood'
    KNN = 'knn'
    MUTUAL_KNN = 'mutual_knn'
    FULLY_CONNECTED = 'fully_connected'


class GraphConstructor:
    def __init__(self, graph_type: GraphType, **kwargs):
        self.graph_type = graph_type
        self.kwargs = kwargs

    def construct_graph(self, similarity_matrix: np.ndarray) -> csr_matrix:
        """
        Construct the graph based on the similarity matrix and specified graph type.

        :param similarity_matrix: np.ndarray, precomputed similarity matrix
        :return: csr_matrix, constructed graph in sparse format
        """
        np.fill_diagonal(similarity_matrix, 0)  # Delete diagonal values

        if self.graph_type == GraphType.EPSILON_NEIGHBORHOOD:
            graph = self._construct_epsilon_neighborhood_graph(similarity_matrix)
        elif self.graph_type == GraphType.KNN:
            graph = self._construct_knn_graph(similarity_matrix)
        elif self.graph_type == GraphType.MUTUAL_KNN:
            graph = self._construct_mutual_knn_graph(similarity_matrix)
        elif self.graph_type == GraphType.FULLY_CONNECTED:
            graph = self._construct_fully_connected_graph(similarity_matrix)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
        # Convert sparse matrix to the graph is a sparse matrix
        return graph.toarray() if isinstance(graph, csr_matrix) else graph

    def _construct_epsilon_neighborhood_graph(self, similarity_matrix: np.ndarray) -> csr_matrix:
        epsilon = self.kwargs.get('threshold', 0.5)
        mask = similarity_matrix >= epsilon
        graph_matrix = csr_matrix(similarity_matrix * mask)
        return graph_matrix

    def _construct_knn_graph(self, similarity_matrix: np.ndarray) -> csr_matrix:
        k = self.kwargs.get('num_neighbors', 5)
        n_samples = similarity_matrix.shape[0]
        graph = np.zeros_like(similarity_matrix)

        for i in range(n_samples):
            neighbors = np.argsort(similarity_matrix[i])[-k:]  # get k-nearest neighbors
            graph[i, neighbors] = similarity_matrix[i, neighbors]

        return csr_matrix(graph)

    def _construct_mutual_knn_graph(self, similarity_matrix: np.ndarray) -> csr_matrix:
        k = self.kwargs.get('num_neighbors', 5)
        n_samples = similarity_matrix.shape[0]
        graph = np.zeros_like(similarity_matrix)

        for i in range(n_samples):
            neighbors = np.argsort(similarity_matrix[i])[-k:]  # get k-nearest neighbors
            for j in neighbors:
                mutual_neighbors = np.argsort(similarity_matrix[j])[-k:]  # verify mutuality
                if i in mutual_neighbors:
                    graph[i, j] = similarity_matrix[i, j]
                    graph[j, i] = similarity_matrix[j, i]

        return csr_matrix(graph)

    @staticmethod
    def _construct_fully_connected_graph(similarity_matrix: np.ndarray) -> csr_matrix:
        return csr_matrix(similarity_matrix)


