import enum
import numpy as np
import typing

from other_libs.log import log_info

EPS = 1e-10


class LaplacianType(enum.Enum):
    """ Different types of Laplacian matrix """
    # The similarity matrix, not a Laplacian: W
    Similarity = enum.auto()

    # The unnormalized Laplacian: L = D - W
    Unnormalized = enum.auto()

    # The random walk view normalized Laplacian:  D^{-1} * L
    RandomWalk = enum.auto()

    # The graph cut view normalized Laplacian: D^{-1/2} * L * D^{-1/2}
    GraphCut = enum.auto()


class Laplacian:
    def __init__(self, adjacency_matrix: np.ndarray, laplacian_type: LaplacianType = LaplacianType.GraphCut, eps: float = EPS):
        """ Initialize the Laplacian object with a similarity matrix and chosen Laplacian type """
        self.laplacian_type = laplacian_type
        self.adjacency_matrix = adjacency_matrix
        self.eps = eps
        log_info("this is the matrix adjacency passed to the laplacian class %s", adjacency_matrix)
        self.degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        shape_verification = (self.degree_matrix.shape == adjacency_matrix.shape)


    def _calculate_degree_matrix(self, adjacency_matrix) -> np.ndarray:
        return

    def _compute_laplacian(self) -> np.ndarray:
        """Compute the Laplacian matrix based on the selected type."""
        laplacian = self.degree_matrix - self.adjacency_matrix

        if self.laplacian_type == LaplacianType.Similarity:
            return self.adjacency_matrix

        # Unnormalized version
        elif self.laplacian_type == LaplacianType.Unnormalized:
            return laplacian

        # Random walk normalized version
        elif self.laplacian_type == LaplacianType.RandomWalk:
            degree_inv = np.diag(1 / (np.diag(self.degree_matrix)))
            return degree_inv.dot(laplacian)

        # Graph cut normalized version
        elif self.laplacian_type == LaplacianType.GraphCut:
            degree_inv_sqrt = np.diag(1 / (np.sqrt(np.diag(self.degree_matrix))))
            return degree_inv_sqrt @ laplacian @ degree_inv_sqrt

        else:
            raise ValueError("Unsupported laplacian_type.")

    def compute(self) -> np.ndarray:
        """ Wrapper function to compute the laplacian """
        return self._compute_laplacian()

    def spectral_decomposition(self, num_clusters: int) -> np.ndarray:
        """     Compute the laplacian matrix &
                Perform spectral decomposition,
                return: only the eigenvectors corresponding to the smallest eigenvalues."""
        laplacian_matrix = self.compute()
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
        return eigenvectors[:, :num_clusters]

    def sorted_spectral_decomposition(self, num_clusters: int, is_descent: bool = True) -> typing.Tuple[np.ndarray, np.ndarray]:
        """     Compute the laplacian matrix &
                Perform spectral decomposition,
                return: the k first eigenvalues and eigenvectors sorted as specified"""
        laplacian_matrix = self.compute()
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        if is_descent:
            # Sort from largest to smallest
            index_array = np.argsort(-eigenvalues)
        else:
            # Sort from smallest to largest
            index_array = np.argsort(eigenvalues)
        # Re-order.
        w = eigenvalues[index_array]
        v = eigenvectors[:, index_array]
        return w[:num_clusters], v[:, :num_clusters]

    def __repr__(self) -> str:
        return (f"Laplacian(laplacian_type={self.laplacian_type}, eps={self.eps}, "
                f"adjacency_matrix_shape={self.adjacency_matrix.shape}, "
                f"adjacency_matrix =\n{self.adjacency_matrix}, "
                f"degree_matrix_shape={self.degree_matrix.shape},"
                f"degree_matrix=\n{self.degree_matrix}")


