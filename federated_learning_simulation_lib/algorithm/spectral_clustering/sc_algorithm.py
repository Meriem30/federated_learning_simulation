import numpy as np
from scipy.linalg import eigh
from .knn import create_knn_graph
from .laplacian import compute_laplacian
from .similarity import compute_similarity
from sklearn.cluster import KMeans


class SpectralClustering:
    def __init__(self, num_neighbors=5, laplacian_type='normalized', similarity_function='gaussian', num_clusters=3):
        self.num_neighbors = num_neighbors
        self.laplacian_type = laplacian_type
        self.similarity_function = similarity_function
        self.num_clusters = num_clusters