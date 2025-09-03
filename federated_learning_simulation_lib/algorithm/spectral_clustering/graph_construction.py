import enum
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt

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

    def print_graph(self, graph_matrix: csr_matrix):
        """
        Visualizes the graph from its sparse matrix representation.

        :param graph_matrix: csr_matrix, the sparse adjacency matrix of the graph.
        """
        if not isinstance(graph_matrix, csr_matrix):
            print("Input is not a CSR matrix. Please provide a sparse graph representation.")
            return
        # Convert the sparse matrix to a NetworkX graph object
        G = nx.from_scipy_sparse_array(graph_matrix)

        # Draw the graph using a spring layout
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1200, edge_color='k', font_size=12)
        plt.title(f"{self.graph_type.value.capitalize()} Graph Visualization", fontsize=16)
        plt.show()
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
        # Directly get indices where similarity is above the threshold.
        rows, cols = np.where(similarity_matrix >= epsilon)
        # Use the corresponding values and indices to create a sparse matrix.
        data = similarity_matrix[rows, cols]
        return csr_matrix((data, (rows, cols)), shape=similarity_matrix.shape)

    def _construct_knn_graph(self, similarity_matrix: np.ndarray) -> csr_matrix:
        k = self.kwargs.get('num_neighbors', 5)
        n_samples = similarity_matrix.shape[0]

        # Use argsort to get the indices of the k-largest similarities for each row.
        # more efficient : sort and slice than to loop.
        sorted_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]
        knn_indices = sorted_indices[:, :k]

        # Use advanced indexing to get the values
        rows = np.repeat(np.arange(n_samples), k)
        cols = knn_indices.flatten()
        data = similarity_matrix[rows, cols]

        # Construct the sparse matrix directly from the data, rows, and columns.
        return csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))

    def _construct_mutual_knn_graph(self, similarity_matrix: np.ndarray) -> csr_matrix:
        k = self.kwargs.get('num_neighbors', 5)
        n_samples = similarity_matrix.shape[0]

        # Find the indices of the k-nearest neighbors for all points at once
        partitioned_indices = np.argpartition(similarity_matrix, -k, axis=1)[:, -k:] # np.argpartition is faster than np.argsort for finding top-k elements

        # Create a boolean mask for KNN
        rows_knn = np.repeat(np.arange(n_samples), k)
        cols_knn = partitioned_indices.flatten()
        knn_mask = csr_matrix((np.ones(n_samples * k, dtype=bool), (rows_knn, cols_knn)), shape=(n_samples, n_samples))

        # The mutual KNN graph is the intersection of the KNN graph and its transpose.
        # This is a vectorized and highly efficient way to find mutual neighbors.
        mutual_mask = knn_mask.multiply(knn_mask.T)

        # Use the mutual mask to filter the original similarity matrix.
        return csr_matrix(similarity_matrix).multiply(mutual_mask)

    @staticmethod
    def _construct_fully_connected_graph(similarity_matrix: np.ndarray) -> csr_matrix:
        return csr_matrix(similarity_matrix)


