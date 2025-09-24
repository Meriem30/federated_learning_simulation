import numpy as np
import typing
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh
import enum

EPS = 1e-10

class LaplacianType(enum.Enum):
    """Different types of Laplacian matrix, compatible with sparse operations."""
    # The similarity matrix, not a Laplacian: W
    Similarity = 'similarity'

    # The unnormalized Laplacian: L = D - W
    Unnormalized = 'unnormalized'

    # The random walk view normalized Laplacian:  D^{-1} * L
    RandomWalk = 'randomwalk'

    # The graph cut view normalized Laplacian: D^{-1/2} * L * D^{-1/2}
    GraphCut = 'graphcut'

class Laplacian:
    def __init__(self, adjacency_matrix: csr_matrix, laplacian_type: LaplacianType = LaplacianType.GraphCut,
                 eps: float = EPS, warn_if_isolated: bool = False):
        """
        Initializes the Laplacian object with a sparse adjacency matrix.
        Args:
            adjacency_matrix: A scipy.sparse.csr_matrix representing the graph.
            laplacian_type: The type of Laplacian to compute.
            eps: A small value for numerical stability.
        """
        #if not isinstance(adjacency_matrix, csr_matrix):
        #    raise TypeError("Adjacency matrix must be a scipy.sparse.csr_matrix for efficiency and scalability.")

        self.laplacian_type = laplacian_type
        self.adjacency_matrix = adjacency_matrix.astype(float)
        self.eps = eps

        # Degree vector
        self.degrees = np.sum(self.adjacency_matrix, axis=1)
        self.degrees[self.degrees == 0] = self.eps  # prevent division by zero

    def compute(self) -> np.ndarray:
        """Computes the Laplacian (dense matrix)."""
        if self.laplacian_type == LaplacianType.Similarity:
            return self.adjacency_matrix

        # Degree matrix
        D = np.diag(self.degrees)
        L = D - self.adjacency_matrix  # unnormalized Laplacian

        if self.laplacian_type == LaplacianType.Unnormalized:
            return L

        elif self.laplacian_type == LaplacianType.RandomWalk:
            # L_rw = I - D^{-1} W
            D_inv = np.diag(1.0 / self.degrees)
            return np.eye(self.adjacency_matrix.shape[0]) - D_inv @ self.adjacency_matrix

        elif self.laplacian_type == LaplacianType.GraphCut:
            # L_sym = I - D^{-1/2} W D^{-1/2}
            D_inv_sqrt = np.diag(1.0 / np.sqrt(self.degrees))
            return np.eye(self.adjacency_matrix.shape[0]) - D_inv_sqrt @ self.adjacency_matrix @ D_inv_sqrt

        else:
            raise ValueError("Unsupported Laplacian type")
    """# Normalized versions
              if self.laplacian_type == LaplacianType.RandomWalk:
                  # L_rw = I - D^{-1} * W
                  degree_inv = diags(1.0 / self.degrees)
                  return identity(n, dtype=float) - degree_inv.dot(self.adjacency_matrix)"""


    def spectral_decomposition(self, num_clusters: int) -> np.ndarray:
        """
        Performs spectral decomposition on the Laplacian matrix

        Args:
            num_clusters: The number of eigenvectors to return
        Returns:
            The eigenvectors corresponding to the smallest eigenvalues
        """
        laplacian_matrix = self.compute()
        n = laplacian_matrix.shape[0]
        k = max(1, min(num_clusters, n - 1))
        # Using `eigsh` from scipy.sparse.linalg for efficiency on sparse matrices.
        # `k` is the number of eigenvalues/eigenvectors to compute.
        # `which='SM'` finds the smallest magnitude eigenvalues.
        eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=k, which='SM')

        # Sort eigenvalues and corresponding eigenvectors
        idx = eigenvalues.argsort()
        return eigenvectors[:, idx]


    def sorted_spectral_decomposition(self, num_clusters: int, is_descent: bool = True) -> typing.Tuple[
        np.ndarray, np.ndarray]:
        """
        Performs spectral decomposition and returns sorted eigenvalues and eigenvectors

        Args:
            num_clusters: Number of eigenvectors/values to return.
            is_descent: Sort in descending order if True, ascending if False
        Returns:
            A tuple of sorted eigenvalues and eigenvectors
        """
        laplacian_matrix = self.compute()
        n = laplacian_matrix.shape[0]
        k = max(1, min(num_clusters, n - 1))
        # The `eigsh` function already sorts by magnitude, so a simple check is needed.
        eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=k, which='SM', sigma=0.0)

        # Sort based on magnitude and then direction
        if is_descent:
            eigenvalues = np.abs(eigenvalues)
            idx = eigenvalues.argsort()[::-1]
        else:
            idx = eigenvalues.argsort()

        return eigenvalues[idx], eigenvectors[:, idx]
    def __repr__(self) -> str:
        """Provides a concise string representation of the object."""
        return (f"Laplacian(laplacian_type={self.laplacian_type.value}, eps={self.eps}, "
                f"adjacency_matrix_shape={self.adjacency_matrix.shape})")