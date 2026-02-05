"""
utils/gsp.py

Graph Signal Processing (GSP) functions for the Spectranorm package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import joblib
import numpy as np
import pyamg  # type: ignore[import-untyped]  # Required for AMG Preconditioning
from scipy import sparse

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "compute_symmetric_normalized_laplacian_eigenmodes",
]

MmapMode = Literal["r+", "r", "w+", "c"]


def make_csr_matrix(
    matrix: sparse.spmatrix | npt.NDArray[np.floating[Any]],
) -> sparse.csr_matrix:
    """
    Ensure the input matrix is in CSR format.
    """
    if sparse.issparse(matrix):
        # We cast to Any briefly to tell Mypy:
        # I've already verified this is sparse, let me call .tocsr()
        return cast("sparse.csr_matrix", cast("Any", matrix).tocsr())

    # If it's not sparse, it's a dense array; convert it directly.
    return sparse.csr_matrix(np.asarray(matrix))


def perform_symmetric_normalization(
    adjacency_matrix: sparse.spmatrix,
) -> sparse.csr_matrix:
    """
    Perform symmetric normalization on the adjacency matrix.

    Args:
        adjacency_matrix: sparse.spmatrix
            The adjacency matrix of the graph.

    Returns:
        sparse.spmatrix
            Symmetrically normalized adjacency matrix.
    """
    # Convert to CSR format if in a different format
    adjacency_matrix = make_csr_matrix(adjacency_matrix)
    # Compute the degree matrix
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    degree_matrix_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))

    # Perform symmetric normalization
    return sparse.csr_matrix(
        degree_matrix_inv_sqrt @ adjacency_matrix @ degree_matrix_inv_sqrt,
    )


def compute_symmetric_normalized_laplacian_eigenmodes(
    adjacency_matrix: sparse.spmatrix | npt.NDArray[np.floating[Any]],
    num_eigenvalues: int = 100,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """
    Compute the eigenvalues of the symmetric normalized Laplacian.

    Args:
        adjacency_matrix: sparse.spmatrix
            The adjacency matrix of the graph.
        num_eigenvalues: int
            Number of eigenvalues to compute.

    Returns:
        eigenvalues, eigenvectors: (np.ndarray, np.ndarray)
            Eigenvalues and eigenvectors of the symmetric normalized Laplacian.
    """
    # Convert to CSR format if in a different format
    adjacency_matrix = make_csr_matrix(adjacency_matrix)

    # Ensure the adjacency matrix is symmetric
    adjacency_matrix = sparse.csr_matrix((adjacency_matrix + adjacency_matrix.T) * 0.5)

    # Perform symmetric normalization
    normalized_matrix = perform_symmetric_normalization(adjacency_matrix)

    # Use shift invert mode to compute the eigenvalues for the
    # normalized adjacency matrix
    lambdas, vectors = sparse.linalg.eigsh(
        normalized_matrix,
        k=num_eigenvalues + 1,
        which="LA",  # type: ignore[call-overload]
    )
    # Note the largest eigenvalues of the adjacency matrix correspond to
    # the smallest eigenvalues of the Laplacian

    # Convert to Laplacian eigenvalues
    lambdas = 1 - lambdas
    lambda_idx = np.argsort(lambdas)
    lambdas = lambdas[lambda_idx]
    vectors = vectors[:, lambda_idx]

    return (lambdas, vectors.T)


def compute_symmetric_normalized_laplacian_eigenmodes_amg_lobpcg(
    adjacency_matrix: sparse.csr_matrix | npt.NDArray[np.floating[Any]],
    num_eigenvalues: int = 100,
    amg_cycles: int = 1,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """
    Compute the eigenvalues and eigenvectors of the Symmetric Normalized
    Laplacian (:math:`L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}`) using LOBPCG
    with Algebraic Multigrid (AMG) preconditioning.

    Note: this is a more advanced and efficient method for large graphs compared
    to using `scipy.sparse.linalg.eigsh`.

    Args:
        adjacency_matrix: The adjacency matrix of the graph.
        num_eigenvalues: Number of eigenvalues (modes) to compute.
        amg_cycles: Number of V-cycles to use in the AMG preconditioner solve.

    Returns:
        eigenvalues, eigenvectors, degrees: (np.ndarray, np.ndarray, np.ndarray)
            Eigenvalues and eigenvectors of L_sym, and the node degrees.
    """
    # Initialize the random number generator
    rng = np.random.default_rng(42)

    # 1. Preprocessing and Input Validation
    adjacency_matrix = make_csr_matrix(adjacency_matrix)
    n = adjacency_matrix.shape[0]

    if num_eigenvalues >= n:
        err = (
            f"num_eigenvalues ({num_eigenvalues}) must be less than "
            f"the number of nodes ({n})."
        )
        raise ValueError(err)

    # Ensure the adjacency matrix is symmetric
    adjacency_matrix = sparse.csr_matrix((adjacency_matrix + adjacency_matrix.T) * 0.5)

    # Compute degrees and handle isolated nodes
    degrees = np.asarray(adjacency_matrix.sum(axis=1)).flatten().astype(np.float64)
    if np.any(degrees == 0):
        err = "The adjacency matrix contains isolated nodes with zero degree."
        raise ValueError(err)

    # 2. Construct the Symmetric Normalized Laplacian (L_sym)
    # L_sym = I - D^(-1/2) A D^(-1/2)
    # The term D^(-1/2) A D^(-1/2) is the normalized adjacency matrix
    symmetric_normalized_laplacian = sparse.identity(
        adjacency_matrix.shape[0],
        format="csr",
    ) - perform_symmetric_normalization(adjacency_matrix)

    # 3. Construct the AMG Preconditioner (M_amg)
    # Build the Smoothed Aggregation Multigrid hierarchy on L_sym
    ml = pyamg.smoothed_aggregation_solver(symmetric_normalized_laplacian)

    # Create a LinearOperator that applies the V-cycle (the preconditioning step)
    def preconditioner_matvec(
        x: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        # Solves L_sym * z = x approximately
        z = ml.solve(x, cycle="V", maxiter=amg_cycles, tol=1e-8)
        return np.asarray(z, dtype=x.dtype)

    # M_amg is the preconditioner operator
    m_amg = sparse.linalg.LinearOperator(
        shape=tuple(symmetric_normalized_laplacian.shape),
        matvec=preconditioner_matvec,
        dtype=symmetric_normalized_laplacian.dtype,
    )  # type: ignore[call-overload]

    # 4. Solve the Eigenvalue Problem L_sym u = mu u using LOBPCG
    block_size = num_eigenvalues + 1
    # Random initial block of vectors
    initial_block = rng.random(
        (n, block_size),
        dtype=symmetric_normalized_laplacian.dtype,
    )

    # We seek the smallest eigenvalues
    # ("SA" - Smallest Algebraic), as L_sym eigenvalues are in [0, 2]
    # Finding the smallest eigenvalues is crucial for spectral clustering/analysis.
    lambdas, vectors = sparse.linalg.lobpcg(
        A=symmetric_normalized_laplacian,
        X=initial_block,
        M=m_amg,  # The AMG Preconditioner
        tol=1e-8,
        maxiter=1000,
        largest=False,  # Find the eigenvalues closest to 0
    )  # type: ignore[operator]

    # 5. Sorting and Output
    # Sort from smallest to largest L_sym eigenvalues
    lambda_idx = np.argsort(lambdas)
    lambdas = lambdas[lambda_idx]
    vectors = vectors[:, lambda_idx]

    # Return L_sym eigenvalues, L_sym eigenvectors, and degrees
    return (lambdas, vectors.T, degrees)


def convert_adjacency_to_transition_matrix(
    adjacency_matrix: sparse.spmatrix,
) -> sparse.csr_matrix:
    """
    Convert an adjacency matrix to a transition matrix.

    Args:
        adjacency_matrix: sparse.spmatrix
            The adjacency matrix of the graph.

    Returns:
        sparse.spmatrix
            Transition matrix.
    """
    # Convert to CSR format if in a different format
    adjacency_matrix = make_csr_matrix(adjacency_matrix)

    # Compute the degree matrix
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    degree_matrix_inv = sparse.diags(1.0 / degrees)

    # Create the transition matrix
    return sparse.csr_matrix(degree_matrix_inv @ adjacency_matrix)


def compute_random_walk_laplacian_eigenmodes(
    adjacency_matrix: sparse.csr_matrix | npt.NDArray[np.floating[Any]],
    num_eigenvalues: int = 100,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """
    Compute the eigenvalues of the random walk Laplacian.

    Args:
        adjacency_matrix: sparse.csr_matrix
            The adjacency matrix of the graph.
        num_eigenvalues: int
            Number of eigenvalues to compute.

    Returns:
        eigenvalues, eigenvectors: (np.ndarray, np.ndarray)
            Eigenvalues and eigenvectors of the random walk Laplacian.
    """
    # Convert to CSR format if in a different format
    adjacency_matrix = make_csr_matrix(adjacency_matrix)

    # Check if num_eigenvalues is less than number of nodes
    if num_eigenvalues >= adjacency_matrix.shape[0]:
        err = (
            f"num_eigenvalues ({num_eigenvalues}) must be less than "
            f"the number of nodes ({adjacency_matrix.shape[0]})."
        )
        raise ValueError(err)

    # Ensure the adjacency matrix is symmetric
    adjacency_matrix = sparse.csr_matrix((adjacency_matrix + adjacency_matrix.T) * 0.5)

    # Compute the degree matrix
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten().astype(np.float64)

    # Ensure no zero degrees to avoid division by zero
    if np.any(degrees == 0):
        err = "The adjacency matrix contains isolated nodes with zero degree."
        raise ValueError(err)

    # No need to compute the transition matrix
    # We can instead solve the generalized eigenvalue problem directly
    # Use shift invert mode to compute the eigenvalues for the transition matrix,
    # starting with the trivial eigenvalue 1
    initial_vector = np.ones(adjacency_matrix.shape[0]) / adjacency_matrix.shape[0]
    lambdas, vectors = sparse.linalg.eigsh(
        adjacency_matrix,
        M=sparse.diags(degrees),  # The Mass matrix in A v = lambda M v
        k=num_eigenvalues + 1,
        which="LA",  # type: ignore[call-overload]
        v0=initial_vector,
    )

    # Convert to Laplacian eigenvalues
    lambdas = 1 - lambdas
    lambda_idx = np.argsort(lambdas)
    lambdas = lambdas[lambda_idx]
    vectors = vectors[:, lambda_idx]

    return (lambdas, np.asarray(vectors).T, degrees)


@dataclass
class EigenmodeBasis:
    """
    Data class to hold eigenmode basis information.

    Attributes:
        eigenvalues: np.ndarray
            Eigenvalues of the Basis (n_modes,).
        eigenvectors: np.ndarray
            Eigenvectors corresponding to the eigenvalues (n_features, n_modes).
            This is the matrix :math:`\\Psi_{(k)} \\in \\mathbb{R}^{N_v \\times k}`
            where k is the number of modes included in the basis.
        mass_matrix: np.ndarray | sparse.csr_matrix | None
            Mass matrix associated with the eigenmodes (optional). This can be used
            in generalized eigenvalue problems (e.g. random walk Laplacian), in which
            case the eigenmodes satisfy :math:`\\Psi^T M \\Psi = I`.
    """

    eigenvalues: npt.NDArray[np.floating[Any]]
    eigenvectors: npt.NDArray[np.floating[Any]]
    mass_matrix: npt.NDArray[np.floating[Any]] | sparse.csr_matrix | None = None

    # Additional attributes
    def __post_init__(self) -> None:
        self.n_modes = self.eigenvalues.shape[0]
        self.n_features = self.eigenvectors.shape[0]

        if self.eigenvectors.shape[1] != self.n_modes:
            err = (
                f"Eigenvectors must have {self.n_modes} modes, "
                f"got {self.eigenvectors.shape[1]}."
            )
            raise ValueError(err)

    def __repr__(self) -> str:
        """
        String representation of the EigenmodeBasis.
        """
        return f"EigenmodeBasis(n_modes={self.n_modes}, n_features={self.n_features})"

    def inverted_eigenvectors(self) -> npt.NDArray[np.floating[Any]]:
        """
        Compute the inverse of the eigenvector matrix.

        .. math::
            \\Psi^{-1} = \\Psi^T M

        Returns:
            np.ndarray
                Inverse of the eigenvector matrix (n_modes, n_features).
        """
        if self.mass_matrix is None:
            # Standard eigenvalue problem
            return self.eigenvectors.T
        # Generalized eigenvalue problem (sparse @ dense is more efficient)
        return (self.mass_matrix.T @ self.eigenvectors).T

    @classmethod
    def load(cls, filepath: str, mmap_mode: MmapMode | None = "r") -> EigenmodeBasis:
        """
        Load an EigenmodeBasis instance from a joblib file.

        Args:
            filepath: str
                Path to the joblib file.
            mmap_mode: MmapMode | None
                Memory mapping mode for joblib (default: "r").
                You can set this to None to disable memory-mapping.

        Returns:
            EigenmodeBasis instance
        """
        data = joblib.load(filepath, mmap_mode=mmap_mode)
        # Expecting the saved file to contain a dict with these keys:
        # 'eigenvalues', 'eigenvectors'
        return cls(
            eigenvalues=data["eigenvalues"],
            eigenvectors=data["eigenvectors"],
            mass_matrix=data.get("mass_matrix"),
        )

    def save(
        self,
        filepath: str,
        compress: int = 0,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Save the EigenmodeBasis instance to a joblib file.

        Args:
            filepath: str
                Path to save the joblib file.
            compress: int
                Compression level for joblib (default: 0).
                Note: Compression is disabled by default to enable support for
                memory-mapped arrays. However, if you do not need memory-mapping,
                and want to reduce file size, you can enable compression by setting
                the compress parameter to a positive integer.
            overwrite: bool
                Whether to overwrite an existing file (default: False).
        """
        # Handle file overwrite
        if Path(filepath).exists():
            if not overwrite:
                err = f"File '{filepath}' already exists and overwrite is disabled."
                raise FileExistsError(err)
            Path(filepath).unlink(missing_ok=True)
        data = {
            "eigenvalues": self.eigenvalues,
            "eigenvectors": self.eigenvectors,
            "mass_matrix": self.mass_matrix,
        }
        joblib.dump(data, filepath, compress=compress)

    def reduce(self, n_modes: int, *, inplace: bool = True) -> EigenmodeBasis:
        """
        Reduce the EigenmodeBasis to only contain the first n_modes.

        This method is useful to reduce the size of the basis for efficiency (e.g.,
        to remove less important modes/degenerate modes before further processing).

        Args:
            n_modes: int
                Number of modes to retain. This must be less than or equal to the
                current number of modes.
            inplace: bool
                Whether to modify the current instance or return a new one.
                By default, this is True (modify in place, optimizing memory usage).

        Returns:
            EigenmodeBasis
                A new EigenmodeBasis instance with reduced modes.
        """
        if n_modes > self.n_modes:
            err = f"Cannot reduce to {n_modes} modes, only {self.n_modes} available."
            raise ValueError(err)
        if inplace:
            self.eigenvalues = self.eigenvalues[:n_modes]
            self.eigenvectors = self.eigenvectors[:, :n_modes]
            self.n_modes = n_modes
            return self
        # else return a new instance
        return EigenmodeBasis(
            eigenvalues=self.eigenvalues[:n_modes],
            eigenvectors=self.eigenvectors[:, :n_modes],
            mass_matrix=self.mass_matrix,
        )

    def encode(
        self,
        signals: npt.NDArray[np.floating[Any]],
        n_modes: int | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Encode a signal using the eigenmode basis.

        Given an eigenmode set :math:`\\Psi` where :math:`L = \\Psi \\Lambda \\Psi^{-1}`
        and a list of signals :math:`x`, the encoded signals :math:`\tilde{x}` are given
        by:

        .. math::
            \\tilde{x} = \\Psi^{-1} x = \\Psi^T M x

        Args:
            signals: np.ndarray
                Signals to encode (n_signals, n_features).
            n_modes: int | None
                Number of modes to use for encoding. If None, use all available modes.

        Returns:
            np.ndarray
                Encoded signal in the eigenmode basis (n_signals, n_modes).
        """
        if signals.shape[-1] != self.n_features:
            err = (
                f"Signal must have {self.n_features} features, got {signals.shape[-1]}."
            )
            raise ValueError(err)

        if n_modes is None:
            n_modes = self.n_modes

        if self.mass_matrix is not None:
            # Generalized eigenvalue problem
            return signals @ (self.mass_matrix.T @ self.eigenvectors[:, :n_modes])
        return signals @ self.eigenvectors[:, :n_modes]

    def decode(
        self,
        encoded_signals: npt.NDArray[np.floating[Any]],
        n_modes: int | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Decode encoded signals :math:`\tilde{x}` using the eigenmode basis.

        Given an eigenmode set :math:`\\Psi` where :math:`L = \\Psi \\Lambda \\Psi^{-1}`
        and a list of encoded signals :math:`\tilde{x}`, the decoded signals
        :math:`\\hat{x}` are given by:

        .. math::
            \\hat{x} = \\Psi \\tilde{x}

        Args:
            encoded_signals: np.ndarray
                Encoded signals in the eigenmode basis (n_signals, n_modes).
            n_modes: int | None
                Number of modes used for decoding. If None, use all available modes.
                This should not exceed the number of modes in the encoded_signals
                nor the number of modes in the eigenmode basis.

        Returns:
            np.ndarray
                Decoded signal in the original feature space (n_signals, n_features).
        """
        if n_modes is None:
            n_modes = self.n_modes

        if encoded_signals.shape[-1] < n_modes:
            err = (
                f"Encoded signal must have at least {n_modes} modes, "
                f"got {encoded_signals.shape[-1]}."
            )
            raise ValueError(err)

        if n_modes > self.n_modes:
            err = (
                f"Cannot decode with {n_modes} modes, "
                f"only {self.n_modes} available in the basis."
            )
            raise ValueError(err)

        return encoded_signals @ self.eigenvectors[:, :n_modes].T
