"""
utils/gsp.py

Graph Signal Processing (GSP) functions for the Spectranorm package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import joblib
import numpy as np
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
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(np.array(matrix))
    if not sparse.isspmatrix_csr(matrix):
        matrix = sparse.csr_matrix(matrix)
    return matrix


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
        which="LM",
    )
    # Note the largest eigenvalues of the adjacency matrix correspond to
    # the smallest eigenvalues of the Laplacian

    # Convert to Laplacian eigenvalues
    lambdas = 1 - lambdas
    lambda_idx = np.argsort(lambdas)
    lambdas = lambdas[lambda_idx]
    vectors = vectors[:, lambda_idx]

    return (lambdas, vectors.T)


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
    adjacency_matrix: sparse.spmatrix | npt.NDArray[np.floating[Any]],
    num_eigenvalues: int = 100,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """
    Compute the eigenvalues of the random walk Laplacian.

    Args:
        adjacency_matrix: sparse.spmatrix
            The adjacency matrix of the graph.
        num_eigenvalues: int
            Number of eigenvalues to compute.

    Returns:
        eigenvalues, eigenvectors: (np.ndarray, np.ndarray)
            Eigenvalues and eigenvectors of the random walk Laplacian.
    """
    # Convert to CSR format if in a different format
    adjacency_matrix = make_csr_matrix(adjacency_matrix)

    # Ensure the adjacency matrix is symmetric
    adjacency_matrix = sparse.csr_matrix((adjacency_matrix + adjacency_matrix.T) * 0.5)

    # Convert to transition matrix
    transition_matrix = convert_adjacency_to_transition_matrix(adjacency_matrix)

    # Use shift invert mode to compute the eigenvalues for the transition matrix,
    # starting with the trivial eigenvalue 1
    initial_vector = np.ones(adjacency_matrix.shape[0])
    lambdas, vectors = sparse.linalg.eigsh(
        transition_matrix,
        k=num_eigenvalues + 1,
        which="LM",
        v0=initial_vector,
    )

    # Convert to Laplacian eigenvalues
    lambdas = 1 - lambdas
    lambda_idx = np.argsort(lambdas)
    lambdas = lambdas[lambda_idx]
    vectors = vectors[:, lambda_idx]

    return (lambdas, np.asarray(vectors).T)


@dataclass
class EigenmodeBasis:
    """
    Data class to hold eigenmode basis information.

    Attributes:
        eigenvalues: np.ndarray
            Eigenvalues of the Basis (n_modes,).
        eigenvectors: np.ndarray
            Eigenvectors corresponding to the eigenvalues (n_modes, n_features).
    """

    eigenvalues: npt.NDArray[np.floating[Any]]
    eigenvectors: npt.NDArray[np.floating[Any]]

    # Additional attributes
    def __post_init__(self) -> None:
        self.n_modes = self.eigenvalues.shape[0]
        self.n_features = self.eigenvectors.shape[1]

        if self.eigenvectors.shape[0] != self.n_modes:
            err = (
                f"Eigenvectors must have {self.n_modes} modes, "
                f"got {self.eigenvectors.shape[0]}."
            )
            raise ValueError(err)

    def __repr__(self) -> str:
        """
        String representation of the EigenmodeBasis.
        """
        return f"EigenmodeBasis(n_modes={self.n_modes}, n_features={self.n_features})"

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
        return cls(eigenvalues=data["eigenvalues"], eigenvectors=data["eigenvectors"])

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
            self.eigenvectors = self.eigenvectors[:n_modes, :]
            self.n_modes = n_modes
            return self
        # else return a new instance
        return EigenmodeBasis(
            eigenvalues=self.eigenvalues[:n_modes],
            eigenvectors=self.eigenvectors[:n_modes, :],
        )

    def encode(
        self,
        signals: npt.NDArray[np.floating[Any]],
        n_modes: int | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Encode a signal using the eigenmode basis.

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

        return signals @ self.eigenvectors[:n_modes].T

    def decode(
        self,
        encoded_signals: npt.NDArray[np.floating[Any]],
        n_modes: int | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Decode a signal from the eigenmode basis.

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

        return encoded_signals @ self.eigenvectors[:n_modes]

    def load_and_encode_data_list(
        self,
        data_paths: list[str],
        output_path: str | None,
        n_modes: int | None = None,
        data_loader: Callable[[str], npt.NDArray[np.floating[Any]]] = np.load,
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Load and encode a list of data using the eigenmode basis.

        This function uses the data_loader to load data from the provided paths
        (for each sample), and then encodes the data using the eigenmode basis.
        Finally, it can save the encoded data to a specified output path, enabling
        efficient reuse of the encoded data in subsequent analyses.

        Args:
            data_paths: list[str]
                List of data identifier strings to load data using the data_loader.
            output_path: str | None
                Path to save the encoded data (as a .npy file). If None, the encoded
                data is not saved.
            n_modes: int | None
                Number of modes to use for encoding. If None, all modes are used.
            data_loader: Callable[[str], np.ndarray] = np.load

        data_loader: Callable[[str], np.ndarray]
            A callable function to load data from a string identifier (e.g., file path).
            Note that by default, this is set to load numpy arrays from .npy files.
            However, you may want to override this with a custom data loader
            that fits your data format and loading requirements.

            For instance, `snm.utils.nitools.compute_fslr_thickness` is an example of a
            custom data loader which can be used to load fs-LR thickness values from a
            path to a subject's FreeSurfer directory.

            If you don't want to use a custom data loader, you can alternatively convert
            all individual data files to numpy arrays and use the default data loader,
            which loads numpy arrays from .npy files.

        Returns:
            np.ndarray: (n_samples, n_modes)
                Encoded data of all individuals as a single numpy array.
        """
        if n_modes is None:
            n_modes = self.n_modes
        # Load and encode data
        encoded_data = np.nan * np.zeros(
            (len(data_paths), n_modes),
            dtype=self.eigenvectors.dtype,
        )
        for i, data_path in enumerate(data_paths):
            # Load (using data_loader) and Encode
            encoded_data[i, :] = self.encode(
                data_loader(data_path),
                n_modes=n_modes,
            )

        if output_path is not None:
            np.save(output_path, encoded_data)

        return encoded_data
