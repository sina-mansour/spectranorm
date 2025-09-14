"""
utils/nitools.py

Neuroimaging functions to deal with imaging files for spectranorm.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import nibabel as nib
import numpy as np
from scipy import sparse, spatial

from . import general

if TYPE_CHECKING:
    import numpy.typing as npt

# ruff adjustments
# ruff: noqa: PLR0913

__all__ = [
    "compute_adaptive_area_barycentric_transformation",
    "compute_barycentric_transformation",
    "compute_fslr_thickness",
    "compute_fsnative_to_fslr32k_transformation",
    "compute_total_euler_number",
    "compute_vertex_areas",
    "get_euler_number",
    "get_fslr_surface_indices_from_cifti",
    "load_freesurfer_surface",
    "load_gifti_surface",
]

# Set up logging
logger = general.get_logger(__name__)


def load_gifti_surface(
    file: Path,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.integer[Any]]]:
    """
    Load a GIfTI surface file.

    Parameters
    ----------
    file : Path
        Path to the GIfTI surface file to be loaded (.gii).

    Returns
    -------
    vertices : numpy.ndarray
        Array of shape (n_vertices, 3) containing vertex coordinates.
    triangles : numpy.ndarray
        Array of shape (n_triangles, 3) containing vertex indices for each triangular
        face.
    """
    gifti_data = cast("nib.gifti.gifti.GiftiImage", nib.loadsave.load(file))
    vertices = gifti_data.darrays[0].data
    triangles = gifti_data.darrays[1].data
    return vertices, triangles


def load_freesurfer_surface(
    file: str,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.integer[Any]]]:
    """
    Load a FreeSurfer surface file.

    Parameters
    ----------
    file : str
        Path to the FreeSurfer surface file to be loaded.

    Returns
    -------
    vertices : numpy.ndarray
        Array of shape (n_vertices, 3) containing vertex coordinates.
    triangles : numpy.ndarray
        Array of shape (n_triangles, 3) containing vertex indices for each triangular
        face.
    """
    surface = nib.freesurfer.io.read_geometry(file)  # type: ignore[no-untyped-call]
    vertices = np.asarray(surface[0], dtype=np.float32)
    triangles = np.asarray(surface[1], dtype=np.int32)
    return vertices, triangles


def get_euler_number(
    vertices: npt.NDArray[np.floating[Any]],
    triangles: npt.NDArray[np.integer[Any]],
) -> int:
    """
    Calculate the Euler number of a surface.

    Parameters
    ----------
    vertices : numpy.ndarray
        Array of shape (n_vertices, 3) containing vertex coordinates.
    triangles : numpy.ndarray
        Array of shape (n_triangles, 3) containing vertex indices for each triangular
        face.

    Returns
    -------
    euler_number : int
        The Euler number of the surface.
    """
    # Euler characteristic: V - E + F = 2 for closed surfaces
    # where V = number of vertices, E = number of edges, F = number of faces

    n_vertices = vertices.shape[0]
    n_faces = triangles.shape[0]

    # Compute number of unique edges
    # Extract all edges from triangles
    edges = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
    )
    # Sort edges so that [1,2] and [2,1] are considered the same
    edges = np.sort(edges, axis=1)
    # Count unique edges
    n_edges = len(np.unique(edges, axis=0))

    # Compute and return the Euler number
    return int(n_vertices - n_edges + n_faces)


def compute_vertex_areas(
    vertices: npt.NDArray[np.floating[Any]],
    triangles: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute the surface area of each vertex in a triangular mesh.

    Parameters
    ----------
    vertices : numpy.ndarray
        Array of shape (n_vertices, 3) containing vertex coordinates.
    triangles : numpy.ndarray
        Array of shape (n_triangles, 3) containing vertex indices for each triangular
        face.

    Returns
    -------
    vertex_areas : numpy.ndarray
        Array of shape (n_vertices,) containing the area associated with each vertex.
    """
    # Compute the area of each triangle
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    # Accumulate areas for each vertex
    # One third of each triangle's area is attributed to each of its vertices
    vertex_areas = np.zeros(vertices.shape[0])
    np.add.at(vertex_areas, triangles[:, 0], triangle_areas / 3)
    np.add.at(vertex_areas, triangles[:, 1], triangle_areas / 3)
    np.add.at(vertex_areas, triangles[:, 2], triangle_areas / 3)

    return vertex_areas


def compute_barycentric_transformation(
    source_vertices: npt.NDArray[np.floating[Any]],
    source_triangles: npt.NDArray[np.integer[Any]],
    target_vertices: npt.NDArray[np.floating[Any]],
    k_candidates: int = 10,
    tol: float = 1e-10,
) -> sparse.coo_matrix:
    """
    Compute sparse barycentric transformation matrix from target_vertices to
    source_vertices.

    Parameters
    ----------
    source_vertices : numpy.ndarray
        Array of shape (N1, 3) containing the coordinates of the source vertices.
    source_triangles : numpy.ndarray
        Array of shape (M1, 3) containing the indices of the source triangles.
    target_vertices : numpy.ndarray
        Array of shape (N2, 3) containing the coordinates of the target vertices.
    k_candidates : int
        Number of candidate triangles to consider for each target vertex. (default: 10)
    tol : float
        Tolerance for inside-triangle check. (default: 1e-10)

    Returns
    -------
    barycentric_transformation : scipy.sparse.csr_matrix
        Sparse matrix of shape (N2, N1) representing the barycentric transformation.
    """
    # Compute source triangle centroids
    source_triangle_centroids = np.mean(source_vertices[source_triangles], axis=1)

    # Build KD-tree on triangle centroids to get a quick nearest triangle search
    triangle_tree = spatial.cKDTree(source_triangle_centroids)

    # Find the K nearest source triangles for each target vertex
    _, candidate_triangles = triangle_tree.query(target_vertices, k=k_candidates)

    # Reshape candidate_triangles to ensure it's 2D
    candidate_triangles = candidate_triangles.reshape(-1, k_candidates)

    # Compute coordinates of all vertices in the candidate triangles
    a = source_vertices[source_triangles][candidate_triangles, 0, :]
    b = source_vertices[source_triangles][candidate_triangles, 1, :]
    c = source_vertices[source_triangles][candidate_triangles, 2, :]

    # Compute vectors for barycentric coordinates
    # v0 = b - a, v1 = c - a, v2 = target - a
    v0 = b - a
    v1 = c - a
    v2 = target_vertices[:, None, :] - a

    # Compute vector dot products
    d00 = np.einsum("ijk,ijk->ij", v0, v0)
    d01 = np.einsum("ijk,ijk->ij", v0, v1)
    d11 = np.einsum("ijk,ijk->ij", v1, v1)
    d20 = np.einsum("ijk,ijk->ij", v2, v0)
    d21 = np.einsum("ijk,ijk->ij", v2, v1)

    # Compute barycentric coordinates
    with np.errstate(divide="ignore", invalid="ignore"):
        den = d00 * d11 - d01 * d01
        beta = (d11 * d20 - d01 * d21) / den
        gamma = (d00 * d21 - d01 * d20) / den
        alpha = 1 - beta - gamma

    # Check if target is inside any of the candidate triangles
    is_inside = (alpha >= -tol) & (beta >= -tol) & (gamma >= -tol) & (np.abs(den) > tol)
    found_valid = is_inside.any(axis=1)

    # Choose the optimal candidate
    selected_candidate = np.argmax(is_inside, axis=1)
    selected_candidate[~found_valid] = -1  # Mark cases with no valid candidates

    # Start preparing the sparse transformation matrix
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    valid_idx = np.where(selected_candidate >= 0)[0]
    if valid_idx.size > 0:
        selected_triangles = candidate_triangles[
            valid_idx,
            selected_candidate[valid_idx],
        ]
        selected_vertex_indices = source_triangles[selected_triangles]
        alpha_values = alpha[valid_idx, selected_candidate[valid_idx]]
        beta_values = beta[valid_idx, selected_candidate[valid_idx]]
        gamma_values = gamma[valid_idx, selected_candidate[valid_idx]]

        rows.extend(np.repeat(valid_idx, 3))
        cols.extend(selected_vertex_indices.flatten())
        bary_coords = np.stack([alpha_values, beta_values, gamma_values], axis=1)
        data.extend(bary_coords.flatten())

    # Fallback: nearest vertex for points without valid candidates
    invalid_idx = np.where(selected_candidate < 0)[0]
    if invalid_idx.size > 0:
        vertex_tree = spatial.cKDTree(source_vertices)

        _, nearest_vertices = vertex_tree.query(target_vertices[invalid_idx], k=1)
        rows.extend(invalid_idx)
        cols.extend(nearest_vertices.reshape(-1))
        data.extend(np.ones(invalid_idx.size, dtype=float))

    # Build and return the barycentric transformation matrix
    return sparse.coo_matrix(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=(target_vertices.shape[0], source_vertices.shape[0]),
    )


def compute_adaptive_area_barycentric_transformation(
    source_vertices: npt.NDArray[np.floating[Any]],  # (n_source, 3)
    source_triangles: npt.NDArray[np.integer[Any]],  # (n_tri_source, 3)
    source_vertex_areas: npt.NDArray[np.floating[Any]],  # (n_source,)
    target_vertices: npt.NDArray[np.floating[Any]],  # (n_target, 3)
    target_triangles: npt.NDArray[np.integer[Any]],  # (n_tri_target, 3)
    target_vertex_areas: npt.NDArray[np.floating[Any]],  # (n_target,)
    source_vertex_mask: npt.NDArray[np.bool_] | None = None,  # (n_source,)
    k_candidates: int = 10,  # number of candidate triangles to check
    tol: float = 1e-10,  # tolerance for barycentric coords
) -> sparse.csr_matrix:
    """
    Computes an adaptive area barycentric transformation matrix from the source mesh to
    the target mesh (similar to workbench command's ADAP_BARY_AREA option).

    Parameters
    ----------
    source_vertices : np.ndarray
        Array of shape (N1, 3) containing the coordinates of the source vertices.
    source_triangles : np.ndarray
        Array of shape (M1, 3) containing the indices of the source triangles.
    source_vertex_areas : np.ndarray
        Array of shape (N1,) containing the areas of the source vertices.
    target_vertices : np.ndarray
        Array of shape (N2, 3) containing the coordinates of the target vertices.
    target_triangles : np.ndarray
        Array of shape (M2, 3) containing the indices of the target triangles.
    target_vertex_areas : np.ndarray
        Array of shape (N2,) containing the areas of the target vertices.
    source_vertex_mask : np.ndarray | None
        (N1,) Optional mask for source vertices.
    k_candidates : int
        Number of candidate triangles to consider for each target vertex.
    tol : float
        Tolerance for barycentric coordinate computation.

    Returns
    -------
    sparse.csr_matrix
        Sparse matrix of shape (N2, N1) representing the transformation.
    """
    # --- Step 1: Compute forward and reverse barycentric transformations ---
    forward_transform = compute_barycentric_transformation(
        source_vertices=source_vertices,
        source_triangles=source_triangles,
        target_vertices=target_vertices,
        k_candidates=k_candidates,
        tol=tol,
    ).tocsr()

    reverse_transform = compute_barycentric_transformation(
        source_vertices=target_vertices,
        source_triangles=target_triangles,
        target_vertices=source_vertices,
        k_candidates=k_candidates,
        tol=tol,
    ).tocsr()

    # Convert reverse scatter to gather (transpose)
    reverse_gather_transform = reverse_transform.T.tocsr()

    # --- Step 2: Decide forward vs reverse for each target vertex
    # Choose the mapping with higher number of sources involved
    # Count non-zero entries in each row of the forward and reverse matrices
    # Note: Since we have CSR format, we can use indptr for efficiency
    forward_sources = np.diff(forward_transform.indptr)
    reverse_sources = np.diff(reverse_gather_transform.indptr)

    # Pick forward if it has >= as many sources as reverse
    use_forward = forward_sources >= reverse_sources

    # --- Step 3: Build the adaptive gather matrix
    adap_gather = (
        sparse.diags((use_forward).astype(float)) @ forward_transform
        + sparse.diags((~use_forward).astype(float)) @ reverse_gather_transform
    )

    # --- Step 4: Multiply each row by target vertex area (area contributions)
    adap_gather = adap_gather.tocsr()
    adap_gather = sparse.csr_matrix(sparse.diags(target_vertex_areas) @ adap_gather)

    # --- Step 5: Correction sum (sum over target vertices for each source vertex)
    correction_sum = np.asarray(adap_gather.sum(axis=0)).ravel()

    # --- Step 6: Scale by source_areas / correction_sum (scatter normalization)
    scale = np.zeros_like(correction_sum)
    valid = correction_sum > 0
    scale[valid] = source_vertex_areas[valid] / correction_sum[valid]
    adap_gather = sparse.csr_matrix(adap_gather @ sparse.diags(scale))

    # --- Step 7: ROI masking
    # (remove columns corresponding to masked-out source vertices)
    if source_vertex_mask is not None:
        adap_gather = sparse.csr_matrix(
            adap_gather.multiply(
                source_vertex_mask[None, :],
            ),
        )

    # --- Step 8: Normalize each row to sum to 1
    row_sums = np.asarray(adap_gather.sum(axis=1)).ravel()
    row_scale = np.ones_like(row_sums)
    valid_rows = row_sums > 0
    row_scale[valid_rows] = 1.0 / row_sums[valid_rows]
    adap_gather = sparse.diags(row_scale) @ adap_gather

    return sparse.csr_matrix(adap_gather)


def compute_fsnative_to_fslr32k_transformation(
    subject_freesurfer_directory: str,
) -> sparse.csr_matrix:
    """
    Compute the transformation matrix from FreeSurfer native space to fs_LR-32k
    space using an adaptive area barycentric transformation.

    Note: This function combines the vertices across left and right hemispheres
    to create a unified transformation matrix (left first, then right).

    Parameters
    ----------
    subject_freesurfer_directory : str
        Path to the FreeSurfer subject directory.

    Returns
    -------
    sparse.csr_matrix
        Sparse matrix representing the transformation from fsnative space to fs_LR-32k.
    """
    # Load fs_LR 32k files
    left_fslr_sphere_vertices, left_fslr_triangles = load_gifti_surface(
        Path(
            str(
                resources.files("spectranorm.data.templates.fs_LR_32k")
                / "fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii",
            ),
        ),
    )
    left_fslr_midthickness_vertices, _ = load_gifti_surface(
        Path(
            str(
                resources.files("spectranorm.data.templates.HCP")
                / "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii",
            ),
        ),
    )
    left_fslr_vertex_areas = compute_vertex_areas(
        left_fslr_midthickness_vertices,
        left_fslr_triangles,
    )
    right_fslr_sphere_vertices, right_fslr_triangles = load_gifti_surface(
        Path(
            str(
                resources.files("spectranorm.data.templates.fs_LR_32k")
                / "fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii",
            ),
        ),
    )
    right_fslr_midthickness_vertices, _ = load_gifti_surface(
        Path(
            str(
                resources.files("spectranorm.data.templates.HCP")
                / "S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii",
            ),
        ),
    )
    right_fslr_vertex_areas = compute_vertex_areas(
        right_fslr_midthickness_vertices,
        right_fslr_triangles,
    )

    # Load subject native files
    left_fsnative_sphere_vertices, left_fsnative_triangles = load_freesurfer_surface(
        f"{subject_freesurfer_directory}/surf/lh.sphere.reg",
    )
    left_fsnative_white_vertices, _ = load_freesurfer_surface(
        f"{subject_freesurfer_directory}/surf/lh.white",
    )
    left_fsnative_pial_vertices, _ = load_freesurfer_surface(
        f"{subject_freesurfer_directory}/surf/lh.pial",
    )
    left_fsnative_midthickness_vertices = np.asarray(
        (left_fsnative_white_vertices + left_fsnative_pial_vertices) * 0.5,
    )
    left_fsnative_vertex_areas = compute_vertex_areas(
        left_fsnative_midthickness_vertices,
        left_fsnative_triangles,
    )
    right_fsnative_sphere_vertices, right_fsnative_triangles = load_freesurfer_surface(
        f"{subject_freesurfer_directory}/surf/rh.sphere.reg",
    )
    right_fsnative_white_vertices, _ = load_freesurfer_surface(
        f"{subject_freesurfer_directory}/surf/rh.white",
    )
    right_fsnative_pial_vertices, _ = load_freesurfer_surface(
        f"{subject_freesurfer_directory}/surf/rh.pial",
    )
    right_fsnative_midthickness_vertices = np.asarray(
        (right_fsnative_white_vertices + right_fsnative_pial_vertices) * 0.5,
    )
    right_fsnative_vertex_areas = compute_vertex_areas(
        right_fsnative_midthickness_vertices,
        right_fsnative_triangles,
    )

    # Compute transformations
    left_surface_transformation = compute_adaptive_area_barycentric_transformation(
        source_vertices=left_fsnative_sphere_vertices,
        source_triangles=left_fsnative_triangles,
        source_vertex_areas=left_fsnative_vertex_areas,
        target_vertices=left_fslr_sphere_vertices,
        target_triangles=left_fslr_triangles,
        target_vertex_areas=left_fslr_vertex_areas,
    )
    right_surface_transformation = compute_adaptive_area_barycentric_transformation(
        source_vertices=right_fsnative_sphere_vertices,
        source_triangles=right_fsnative_triangles,
        source_vertex_areas=right_fsnative_vertex_areas,
        target_vertices=right_fslr_sphere_vertices,
        target_triangles=right_fslr_triangles,
        target_vertex_areas=right_fslr_vertex_areas,
    )

    # Combine the transformations and return
    return sparse.csr_matrix(
        sparse.block_diag(
            [
                left_surface_transformation,
                right_surface_transformation,
            ],
        ),
    )


def get_fslr_surface_indices_from_cifti(
    cifti_file: Path | None = None,
) -> npt.NDArray[np.integer[Any]]:
    """
    Get the fs_LR surface indices from a CIFTI file (excluding the medial wall).

    Parameters
    ----------
    cifti_file : str
        Path to the CIFTI file. By default a ones.dscalar.nii template will be used.

    Returns
    -------
    np.ndarray
        The indices indicating of vertices present in the CIFTI format.
    """
    # Use default ones.dscalar.nii if no file is provided
    if cifti_file is None:
        cifti_file = Path(
            str(
                resources.files("spectranorm.data.templates.CIFTI")
                / "ones.dscalar.nii",
            ),
        )
    cifti = nib.loadsave.load(str(cifti_file))
    if not isinstance(cifti, nib.cifti2.cifti2.Cifti2Image):
        err = f"File {cifti_file} is not a valid CIFTI file."
        raise TypeError(err)

    # Extract the brain models for left and right cortical surfaces
    brain_models = list(cifti.header.get_index_map(1).brain_models)  # type: ignore[no-untyped-call]
    left_surface_model, right_surface_model = brain_models[0], brain_models[1]

    # Return the indices for left and right surfaces
    return np.concatenate(
        [
            np.asarray(left_surface_model.vertex_indices),
            (
                np.asarray(right_surface_model.vertex_indices)
                + left_surface_model.surface_number_of_vertices
            ),
        ],
    )


def compute_fslr_thickness(
    subject_freesurfer_directory: str,
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute the fs_LR thickness from FreeSurfer output directory.

    This function can be passed to a `snm.utils.gsp.EigenmodeBasis` as a dataloader
    for the `load_and_encode_data_list` method.

    Note: Assuming FreeSurfer's recon-all is completed, this function automatically
    maps thickness estimates from subject's native cortical surface space onto the
    standard fs_LR space.

    Parameters
    ----------
    subject_freesurfer_directory : str
        Path to the FreeSurfer subject directory.

    Returns
    -------
    np.ndarray
        Array of shape (59412,) containing the thickness values for fs_LR vertices
        (excluding the medial wall).
    """
    cifti_mask = get_fslr_surface_indices_from_cifti()
    adap_area_barycentric_transformation = compute_fsnative_to_fslr32k_transformation(
        subject_freesurfer_directory,
    )[cifti_mask, :]
    fsnative_thickness = np.concatenate(
        [
            nib.freesurfer.io.read_morph_data(  # type: ignore[no-untyped-call]
                f"{subject_freesurfer_directory}/surf/lh.thickness",
            ),
            nib.freesurfer.io.read_morph_data(  # type: ignore[no-untyped-call]
                f"{subject_freesurfer_directory}/surf/rh.thickness",
            ),
        ],
    )
    return np.asarray(adap_area_barycentric_transformation @ fsnative_thickness)


def compute_total_euler_number(subject_freesurfer_directory: str) -> int:
    """
    Compute the total Euler number from FreeSurfer output directory.
    Note: This function assumes FreeSurfer's recon-all is completed.

    Parameters
    ----------
    subject_freesurfer_directory : str
        Path to the FreeSurfer subject directory.

    Returns
    -------
    int
        The total Euler number (sum over left and right surfaces).
    """
    left_surface = f"{subject_freesurfer_directory}/surf/lh.orig.nofix"
    right_surface = f"{subject_freesurfer_directory}/surf/rh.orig.nofix"

    # Compute the Euler characteristic for each surface
    left_euler = get_euler_number(*load_freesurfer_surface(left_surface))
    right_euler = get_euler_number(*load_freesurfer_surface(right_surface))

    # Return the total Euler number
    return left_euler + right_euler
