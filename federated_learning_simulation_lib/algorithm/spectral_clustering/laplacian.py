import enum
import numpy as np

EPS = 1e-10


class LaplacianType(enum.Enum):
    """Different types of Laplacian matrix."""
    # The similarity matrix, not a Laplacian: W
    Similarity = enum.auto()

    # The unnormalized Laplacian: L = D - W
    Unnormalized = enum.auto()

    # The random walk view normalized Laplacian:  D^{-1} * L
    RandomWalk = enum.auto()

    # The graph cut view normalized Laplacian: D^{-1/2} * L * D^{-1/2}
    GraphCut = enum.auto()


def compute_laplacian(similarity: np.ndarray,
                      laplacian_type: LaplacianType = LaplacianType.GraphCut,
                      eps: float = EPS) -> np.ndarray:
    """Compute and return the Laplacian matrix """
    degree = np.diag(np.sum(similarity, axis=1))
    laplacian = degree - similarity
    if not isinstance(laplacian_type, LaplacianType):
        raise TypeError("laplacian_type must be a LaplacianType")
    elif laplacian_type == LaplacianType.Similarity:
        return similarity
    elif laplacian_type == LaplacianType.Unnormalized:
        # Unnormalized version
        return laplacian
    elif laplacian_type == LaplacianType.RandomWalk:
        # Random walk normalized version
        degree_norm = np.diag(1 / (np.diag(degree) + eps))
        laplacian_norm = degree_norm.dot(laplacian)
        return laplacian_norm
    elif laplacian_type == LaplacianType.GraphCut:
        # Graph cut normalized version
        degree_norm = np.diag(1 / (np.sqrt(np.diag(degree)) + eps))
        laplacian_norm = degree_norm.dot(laplacian).dot(degree_norm)
        return laplacian_norm
    else:
        raise ValueError("Unsupported laplacian_type.")
