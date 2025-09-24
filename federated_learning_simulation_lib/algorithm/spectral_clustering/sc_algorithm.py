import os
os.environ["OMP_NUM_THREADS"] = "1" # Set the environment variable to avoid memory warnings when calling KMeans
import numpy as np
from scipy.linalg import eigh
from .laplacian import Laplacian
from .similarity import compute_similarity_matrix
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from .graph_construction import GraphType, GraphConstructor
from .similarity import SimilarityType
from .laplacian import LaplacianType

from other_libs.log import log_info


class SpectralClustering:
    """
        Performs the spectral clustering algorithm based on a specified configuration.
        This class is optimized for scalability and clean design.
        """

    def __init__(self, config: dict):
        """
        Initializes the SpectralClustering object with a configuration dictionary.

        Args:
            config: A dictionary containing all necessary parameters:
                - 'graph_type': GraphType enum.
                - 'num_neighbors': int, for KNN graphs.
                - 'threshold': float, for epsilon-neighborhood graphs.
                - 'similarity_function': SimilarityType enum.
                - 'laplacian_type': LaplacianType enum.
    """
        self.config = config
        self.adjacency_matrix = None


    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Executes the full spectral clustering pipeline on the input data.

        Args:
            data: The input data matrix, where rows are clients

        Returns:
            An array of cluster labels for each client.
        """
        # Step 1: Create similarity matrix
        similarity_matrix = compute_similarity_matrix(data, self.config['similarity_function'])

        # Step 2: Construct the graph
        log_info(f"Graph construction started with type: {self.config['graph_type'].name}")
        graph_params = {
            'num_neighbors': self.config.get('num_neighbors', 5),
            'threshold': self.config.get('threshold', 0.5),

        }
        graph_constructor = GraphConstructor(self.config['graph_type'], **graph_params)
        # Call the graph construction function the on created instance
        self.adjacency_matrix = graph_constructor.construct_graph(similarity_matrix)
        # Log concise information about the sparse graph
        if isinstance(self.adjacency_matrix, csr_matrix):
            log_info(f"Graph constructed. Shape: {self.adjacency_matrix.shape}, "
                     f"NNZ: {self.adjacency_matrix.nnz}")
        else:
            log_info(f"Graph constructed. Shape: {self.adjacency_matrix.shape}")

        # Step 3: Prepare and compute the Laplacian
        log_info(f"Laplacian computation started with type: {self.config['laplacian_type'].name}")
        laplacian = Laplacian(self.adjacency_matrix, self.config['laplacian_type'])

        # Step 4: compute the Laplacian & get the eigenvectors (use the smallest k eigenvectors)
        # Use the `spectral_decomposition` method which returns only eigenvectors.
        log_info("Performing spectral decomposition to get top eigenvectors.")
        top_k_eigenvectors = laplacian.spectral_decomposition(self.config['num_clusters'])
        log_info(f"Obtained eigenvectors matrix with shape: {top_k_eigenvectors.shape}")

        # Step 5: Apply KMeans clustering on the eigenvectors
        log_info(f"Applying KMeans with {self.config['num_clusters']} clusters.")
        kmeans = KMeans(n_clusters=self.config['num_clusters'], n_init=10)
        kmeans.fit(top_k_eigenvectors)
        labels = kmeans.labels_
        log_info("KMeans clustering complete.")

        return labels