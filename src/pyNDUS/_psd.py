"""Shared positive-semidefinite covariance checks."""

import warnings

import numpy as np
import scipy.linalg as spla


class PSDCheckCostWarning(UserWarning):
    """Warn that a requested PSD eigenvalue check may be expensive."""

    pass


def _check_psd_matrix(matrix, rtol=1e-10, atol=1e-12, return_eigenpairs=False,
                      dim_warning_threshold=1000):
    """Check one symmetric covariance matrix using eigenvalues and optional eigenvectors."""
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("PSD checks require a square two-dimensional matrix.")

    for name, value in (("rtol", rtol), ("atol", atol)):

        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be >=0")

        if not np.isfinite(value) or value < 0:
            raise ValueError(f"'{name}' must be >=0")

        if name == "rtol":
            rtol = value
        else:
            atol = value

    if dim_warning_threshold is not None:

        if not isinstance(dim_warning_threshold, int) or dim_warning_threshold <= 0:
            raise ValueError("'dim_warning_threshold' must be a positive integer or None")

        if matrix.shape[0] > dim_warning_threshold:
            warnings.warn(
                "PSD checking requires an eigenvalue calculation for a "
                f"{matrix.shape[0]}x{matrix.shape[0]} covariance matrix; this "
                "may be computationally expensive.",
                PSDCheckCostWarning,
                stacklevel=3,)

    symmetric = 0.5 * (matrix + matrix.T)
    dimension = matrix.shape[0]

    if dimension == 0:
        eigenvalues = np.array([], dtype=float) if return_eigenpairs else None
        eigenvectors = np.empty((0, 0), dtype=float) if return_eigenpairs else None
        minimum_eigenvalue = 0.0
        maximum_eigenvalue = 0.0
    elif return_eigenpairs:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        minimum_eigenvalue = float(eigenvalues[0])
        maximum_eigenvalue = float(eigenvalues[-1])
    else:
        eigenvalues = None
        eigenvectors = None
        checked_eigenvalues = spla.eigh(symmetric, eigvals_only=True)
        minimum_eigenvalue = float(checked_eigenvalues[0])
        maximum_eigenvalue = float(checked_eigenvalues[-1])

    spectral_radius = max(abs(minimum_eigenvalue), abs(maximum_eigenvalue))
    tolerance = atol + rtol * spectral_radius
    is_psd = minimum_eigenvalue >= -tolerance

    if eigenvalues is None:
        negative_eigenvalues = 0 if is_psd else None
    else:
        negative_eigenvalues = int(np.count_nonzero(eigenvalues < -tolerance))

    return {
        "dimension": dimension,
        "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_eigenvalue": maximum_eigenvalue,
        "spectral_radius": spectral_radius,
        "tolerance": tolerance,
        "is_psd": bool(is_psd),
        "negative_eigenvalues": negative_eigenvalues,
        "asymmetry_norm": float(np.linalg.norm(matrix - matrix.T)),
        "symmetric": symmetric,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }
