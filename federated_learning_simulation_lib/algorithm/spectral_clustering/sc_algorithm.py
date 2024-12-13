import os
os.environ["OMP_NUM_THREADS"] = "1" # Set the environment variable to avoid memory warnings when calling KMeans
import numpy as np
from scipy.linalg import eigh
from .laplacian import Laplacian
from .similarity import compute_similarity_matrix
from sklearn.cluster import KMeans
from .graph_construction import GraphType, GraphConstructor
from .similarity import SimilarityType
from .laplacian import LaplacianType

from other_libs.log import log_info


class SpectralClustering:
    """ Perform the spectral clustering algorithm based on the specified configuration """
    def __init__(self, graph_type: GraphType, threshold: float, num_neighbors: int, laplacian_type: LaplacianType,
                 similarity_function: SimilarityType, num_clusters: int):
        self.graph_type = graph_type
        self.num_neighbors = num_neighbors
        self.threshold = threshold
        self.similarity_function = similarity_function
        self.laplacian_type = laplacian_type
        self.num_clusters = num_clusters

    def fit(self, data):
        # Step 1: create similarity matrix
        similarity_matrix = compute_similarity_matrix(data, self.similarity_function)
        log_info("************** This is the similarity matrix ***************** \n %s", similarity_matrix)
        # Step 2: prepare the kwargs to be passed to the GraphConstructor
        kwargs = {
            'num_neighbors': self.num_neighbors,
            'threshold': self.threshold,
        }
        graph_constructor = GraphConstructor(self.graph_type, **kwargs)
        log_info("************** This is the constructed graph ***************** \n %s", graph_constructor)

        # Call the graph construction function the on created instance
        adjacency_matrix = graph_constructor.construct_graph(similarity_matrix)
        log_info("************** This is the adjacency matrix ***************** \n %s", adjacency_matrix)
        # Step 3: prepare the laplacian instance (only initiate class attributes)
        laplacian = Laplacian(adjacency_matrix, self.laplacian_type)
        log_info("************** This is the Laplacian ***************** \n %s", laplacian)

        # Step 4: compute the Laplacian & get the eigenvectors (use the smallest k eigenvectors)
        _, top_k_eigenvectors = laplacian.sorted_spectral_decomposition(self.num_clusters, is_descent=False)
        log_info("************** This is the top k eigenvectors ***************** \n %s", top_k_eigenvectors)

        # Step 5: apply KMeans clustering on the eigenvectors
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(top_k_eigenvectors)

        return kmeans.labels_  # Return the cluster labels for each data point
