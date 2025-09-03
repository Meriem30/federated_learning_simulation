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
    RandomWalk = 'random_walk'

    # The graph cut view normalized Laplacian: D^{-1/2} * L * D^{-1/2}
    GraphCut = 'graph_cut'

class Laplacian:
    def __init__(self, adjacency_matrix: csr_matrix, laplacian_type: LaplacianType = LaplacianType.GraphCut,
                 eps: float = EPS):
        """
        Initializes the Laplacian object with a sparse adjacency matrix.
        Args:
            adjacency_matrix: A scipy.sparse.csr_matrix representing the graph.
            laplacian_type: The type of Laplacian to compute.
            eps: A small value for numerical stability.
        """
        if not isinstance(adjacency_matrix, csr_matrix):
            raise TypeError("Adjacency matrix must be a scipy.sparse.csr_matrix for efficiency and scalability.")

        self.laplacian_type = laplacian_type
        self.adjacency_matrix = adjacency_matrix.copy()
        self.eps = eps

        # Calculate degree vector for sparse matrices
        self.degrees = self.adjacency_matrix.sum(axis=1).A.flatten()
        # Add a small epsilon to zero degrees to prevent division by zero
        self.degrees[self.degrees == 0] = self.eps

    def compute(self) -> csr_matrix:
        """Computes and returns the specified Laplacian matrix in sparse format."""

        if self.laplacian_type == LaplacianType.Similarity:
            return self.adjacency_matrix

        # Unnormalized Laplacian: L = D - W
        degree_matrix_sparse = diags(self.degrees)
        laplacian = degree_matrix_sparse - self.adjacency_matrix

        if self.laplacian_type == LaplacianType.Unnormalized:
            return laplacian

        # Normalized versions
        if self.laplacian_type == LaplacianType.RandomWalk:
            # D_inv = D^{-1}
            degree_inv = diags(1.0 / self.degrees)
            # L_rw = I - D^{-1} * W
            return identity(self.adjacency_matrix.shape[0], dtype='float') - degree_inv.dot(self.adjacency_matrix)

        elif self.laplacian_type == LaplacianType.GraphCut:
            # D_inv_sqrt = D^{-1/2}
            degree_inv_sqrt = diags(1.0 / np.sqrt(self.degrees))
            # L_sym = I - D^{-1/2} * W * D^{-1/2}
            return identity(self.adjacency_matrix.shape[0], dtype='float') - degree_inv_sqrt.dot(
                self.adjacency_matrix).dot(degree_inv_sqrt)

        else:
            raise ValueError("Unsupported laplacian_type.")


    def spectral_decomposition(self, num_clusters: int) -> np.ndarray:
        """
        Performs spectral decomposition on the Laplacian matrix

        Args:
            num_clusters: The number of eigenvectors to return
        Returns:
            The eigenvectors corresponding to the smallest eigenvalues
        """
        laplacian_matrix = self.compute()

        # Using `eigsh` from scipy.sparse.linalg for efficiency on sparse matrices.
        # `k` is the number of eigenvalues/eigenvectors to compute.
        # `which='SM'` finds the smallest magnitude eigenvalues.
        eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=num_clusters, which='SM', sigma=0.0)

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
        # The `eigsh` function already sorts by magnitude, so a simple check is needed.
        eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=num_clusters, which='SM', sigma=0.0)

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

