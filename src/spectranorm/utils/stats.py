"""
utils/stats.py

Statistical utility functions for the Spectranorm package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "compute_censored_log_likelihood",
    "compute_centiles_from_z_scores",
    "compute_correlation_significance_by_fisher_z",
    "compute_log_likelihood",
]


def compute_centiles_from_z_scores(
    z_scores: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Convert z-scores to percentiles.

    Args:
        z_scores: np.ndarray
            Array of z-scores.

    Returns:
        np.ndarray
            Array of percentiles corresponding to the z-scores.
    """
    # Convert z-scores to percentiles using the cumulative distribution function (CDF)
    return np.asarray(stats.norm.cdf(z_scores) * 100)


def compute_log_likelihood(
    observations: npt.NDArray[np.floating[Any]],
    predicted_mus: npt.NDArray[np.floating[Any]],
    predicted_sigmas: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute the log likelihood of observations given predicted means and standard
    deviations.

    Args:
        observations: np.ndarray
            Observed data points.
        predicted_mus: np.ndarray
            Predicted means for the observations.
        predicted_sigmas: np.ndarray
            Predicted standard deviations for the observations.

    Returns:
        np.ndarray
            Log likelihood of each observation.
    """
    return np.asarray(
        stats.norm.logpdf(observations, loc=predicted_mus, scale=predicted_sigmas),
    )


def compute_censored_log_likelihood(
    observations: npt.NDArray[np.floating[Any]],
    predicted_mus: npt.NDArray[np.floating[Any]],
    predicted_sigmas: npt.NDArray[np.floating[Any]],
    censored_quantile: float = 0.01,
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute censored log likelihood, replacing extreme low likelihoods with a censoring
    threshold.

    Args:
        observations: np.ndarray
            Observed data points (N,).
        predicted_mus: np.ndarray
            Predicted means for each observation (N,).
        predicted_sigmas: np.ndarray
            Predicted standard deviations for each observation (N,).
        censored_quantile: float (default=0.01)
            Quantile below which log-likelihoods are censored.

    Returns:
        np.ndarray:
            The censored log likelihood of all observations.
    """
    # Compute log likelihoods
    log_likelihoods = compute_log_likelihood(
        observations,
        predicted_mus,
        predicted_sigmas,
    )

    # Compute censoring threshold based on standard normal
    censor_threshold = stats.norm.logpdf(
        stats.norm.ppf(censored_quantile),
        loc=0,
        scale=1,
    )

    # Apply censoring (two-sided) and return
    return np.where(
        log_likelihoods < censor_threshold,
        np.log(2 * censored_quantile),
        log_likelihoods,
    )


def compute_correlation_significance_by_fisher_z(
    correlation_matrix: npt.NDArray[np.floating[Any]],
    n_samples: int,
    correlation_threshold: float = 0.0,
) -> npt.NDArray[np.bool_]:
    """
    Compute the significance of correlations between variables in the data matrix,
    thresholded by a specified correlation value, using Fisher's z-transformation.

    Args:
        correlation_matrix: np.ndarray
            A 2D array of pairwise correlation values.
        n_samples: int
            The number of samples used to compute correlation.
        correlation_threshold: float (default=0.0)
            The correlation threshold above which correlations are to be considered
            significant.

    Returns:
        np.ndarray
            A boolean matrix indicating significantly large correlations between
            variables. The True values indicate correlations that are significantly
            larger than the threshold.
    """
    # Fisher's z-transformation
    fisher_z = np.arctanh(correlation_matrix)
    z_threshold = np.arctanh(correlation_threshold)
    # Standard error of the Fisher z
    se = 1 / np.sqrt(n_samples - 3)
    # Apply the correlation threshold and return significance matrix
    # 95% confidence interval
    return np.asarray(np.abs(fisher_z) > (z_threshold + (se * 1.96)))
