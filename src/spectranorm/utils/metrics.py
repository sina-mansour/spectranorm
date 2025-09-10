"""
utils/metrics.py

Utility functions for computing various model metrics in the Spectranorm package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "compute_bic",
    "compute_expv",
    "compute_mae",
    "compute_mape",
    "compute_mse",
    "compute_msll",
    "compute_r2",
    "compute_rmse",
]


def compute_mae(
    y: npt.NDArray[np.floating[Any]],
    y_pred: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y: np.ndarray
            True values. (n_samples) or (n_samples, n_outputs) if multiple outputs
            are being assessed.
        y_pred: np.ndarray
            Predicted values. (n_samples) or (n_samples, n_outputs) same shape as y.

    Returns:
        np.ndarray
            Mean Absolute Error. (n_outputs,) if multiple outputs are assessed,
            otherwise a scalar.
    """
    return np.asarray(np.mean(np.abs(y - y_pred), axis=0))


def compute_mse(
    y: npt.NDArray[np.floating[Any]],
    y_pred: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute Mean Squared Error (MSE) between true and predicted values.

    Args:
        y: np.ndarray
            True values. (n_samples) or (n_samples, n_outputs) if multiple outputs
            are being assessed.
        y_pred: np.ndarray
            Predicted values. (n_samples) or (n_samples, n_outputs) same shape as y.

    Returns:
        np.ndarray
            Mean Squared Error. (n_outputs,) if multiple outputs are assessed,
            otherwise a scalar.
    """
    return np.asarray(np.mean((y - y_pred) ** 2, axis=0))


def compute_rmse(
    y: npt.NDArray[np.floating[Any]],
    y_pred: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y: np.ndarray
            True values. (n_samples) or (n_samples, n_outputs) if multiple outputs
            are being assessed.
        y_pred: np.ndarray
            Predicted values. (n_samples) or (n_samples, n_outputs) same shape as y.

    Returns:
        np.ndarray
            Root Mean Squared Error. (n_outputs,) if multiple outputs are assessed,
            otherwise a scalar.
    """
    return np.asarray(np.sqrt(compute_mse(y, y_pred)))


def compute_mape(
    y: npt.NDArray[np.floating[Any]],
    y_pred: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Args:
        y: np.ndarray
            True values. (n_samples) or (n_samples, n_outputs) if multiple outputs
            are being assessed.
        y_pred: np.ndarray
            Predicted values. (n_samples) or (n_samples, n_outputs) same shape as y.

    Returns:
        np.ndarray
            Mean Absolute Percentage Error. (n_outputs,) if multiple outputs are
            assessed, otherwise a scalar.
    """
    return np.asarray(np.mean(np.abs((y - y_pred) / y), axis=0) * 100)


def compute_r2(
    y: npt.NDArray[np.floating[Any]],
    y_pred: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute R-squared (coefficient of determination) between true and predicted values.

    Args:
        y: np.ndarray
            True values. (n_samples) or (n_samples, n_outputs) if multiple outputs
            are being assessed.
        y_pred: np.ndarray
            Predicted values. (n_samples) or (n_samples, n_outputs) same shape as y.

    Returns:
        np.ndarray
            R-squared value. (n_outputs,) if multiple outputs are assessed,
            otherwise a scalar.
    """
    ss_res = np.sum((y - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    return np.asarray(1 - (ss_res / ss_tot))


def compute_expv(
    y: npt.NDArray[np.floating[Any]],
    y_pred: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute Explained Variance (EXPV) between true and predicted values.

    Args:
        y: np.ndarray
            True values. (n_samples) or (n_samples, n_outputs) if multiple outputs
            are being assessed.
        y_pred: np.ndarray
            Predicted values. (n_samples) or (n_samples, n_outputs) same shape as y.

    Returns:
        np.ndarray
            Explained Variance. (n_outputs,) if multiple outputs are assessed,
            otherwise a scalar.
    """
    return np.asarray(1 - np.var(y - y_pred, axis=0) / np.var(y, axis=0))


def compute_msll(
    model_log_likelihoods: npt.NDArray[np.floating[Any]],
    baseline_log_likelihoods: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute Mean Standardized Log Loss (MSLL) based on model and baseline log
    likelihoods.

    Args:
        model_log_likelihoods: np.ndarray
            Log likelihoods from the model. (n_samples) or (n_samples, n_outputs)
            if multiple outputs are being assessed.
        baseline_log_likelihoods: np.ndarray
            Log likelihoods from the baseline model (a trivial model). Same shape as
            model_log_likelihoods.

    Returns:
        np.ndarray
            Mean Standardized Log Loss. (n_outputs,) if multiple outputs are assessed,
            otherwise a scalar.
    """
    return np.asarray(
        -(
            np.mean(model_log_likelihoods, axis=0)
            - np.mean(baseline_log_likelihoods, axis=0)
        ),
    )


def compute_bic(
    model_log_likelihoods: npt.NDArray[np.floating[Any]],
    n_params: int,
    n_samples: int,
) -> npt.NDArray[np.floating[Any]]:
    """
    Compute Bayesian Information Criterion (BIC) for model evaluation.

    Args:
        model_log_likelihoods: np.ndarray
            Log likelihoods from the model. (n_samples) or (n_samples, n_outputs)
            if multiple outputs are being assessed.
        n_params: int
            Number of parameters in the model.
        n_samples: int
            Number of samples used in the model.

    Returns:
        np.ndarray
            Bayesian Information Criterion. (n_outputs,) if multiple outputs are
            assessed, otherwise a scalar.
    """
    return np.asarray(
        -2 * np.mean(model_log_likelihoods, axis=0) + n_params * np.log(n_samples),
    )
