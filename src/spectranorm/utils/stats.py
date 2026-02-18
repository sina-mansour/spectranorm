"""
utils/stats.py

Statistical utility functions for the Spectranorm package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "compute_censored_log_likelihood",
    "compute_centiles_from_z_scores",
    "compute_correlation_significance",
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

    if censored_quantile <= 0:
        return log_likelihoods

    # Standardized residuals
    z_scores = (observations - predicted_mus) / predicted_sigmas

    # Two-sided z threshold based on standard normal
    z_threshold = stats.norm.ppf(1 - censored_quantile)

    # Replace extreme z's with constant tail mass
    censored_value = np.log(2 * censored_quantile)

    # Apply censoring (two-sided) and return
    return np.where(
        np.abs(z_scores) > z_threshold,
        censored_value,
        log_likelihoods,
    )


def compute_correlation_significance(
    correlation_matrix: npt.NDArray[np.floating[Any]],
    n_samples: int,
    correlation_threshold: float = 0.0,
    correction_method: str = "fdr_bh",
) -> npt.NDArray[np.floating[Any]]:
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
        correction_method: str (default='fdr_bh')
            Method for multiple testing correction. Options include 'bonferroni',
            'holm', 'fdr_bh', etc. See statsmodels.stats.multitest.multipletests for
            more details.

    Returns:
        np.ndarray
            A matrix of p-values indicating the significance of each correlation.
            (Testing if the correlation is significantly greater than the threshold.)
    """
    # set the diagonal to zero to avoid NaNs in arctanh
    np.fill_diagonal(correlation_matrix, 0)

    # Fisher's z-transformation
    fisher_z = np.arctanh(correlation_matrix)
    z_threshold = np.arctanh(correlation_threshold)

    # Standard error of the Fisher z
    se = 1 / np.sqrt(n_samples - 3)

    # Compute the test statistic
    z_score = (np.abs(fisher_z) - z_threshold) / se

    # compute uncorrected p-values for one-tailed test
    p_values = 1 - stats.norm.cdf(z_score)

    # Take upper triangle of the p-value matrix (excluding diagonal)
    triu_indices = np.triu_indices_from(p_values, k=1)
    p_values_triu = p_values[triu_indices]

    # Apply multiple testing correction
    _, corrected_p_values, _, _ = multipletests(
        p_values_triu,
        method=correction_method,
    )

    # Reconstruct the full p-value matrix
    p_values_corrected = np.full_like(p_values, fill_value=np.nan, dtype=np.float64)
    p_values_corrected[triu_indices] = corrected_p_values
    p_values_corrected[(triu_indices[1], triu_indices[0])] = corrected_p_values

    return p_values_corrected
