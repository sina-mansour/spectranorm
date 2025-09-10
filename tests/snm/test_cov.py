"""
tests/test_cov.py

Tests for the CovarianceNormativeModel class in the Spectranorm package.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from spectranorm import snm


@pytest.fixture
def mock_dataframe() -> Callable[..., pd.DataFrame]:
    """
    Fixture to create a mock DataFrame for testing.
    """

    def _make_df(
        n_rows: int = 100,
        n_categorical: int = 2,
        n_numerical: int = 4,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Create a mock DataFrame with specified number of rows, categorical and
        numerical columns.
        """
        # Random number generator for reproducibility
        rng = np.random.default_rng(random_state)

        # Create categorical data
        categorical_data = {}
        for i in range(n_categorical):
            n_categories = rng.integers(2, 5)
            categories = [f"cat_{j}" for j in range(n_categories)]
            categorical_data[f"cat_col_{i}"] = rng.choice(categories, size=n_rows)

        # Create numerical data
        numerical_data = {
            f"num_col_{i}": rng.random(n_rows) for i in range(n_numerical)
        }

        return pd.DataFrame({**categorical_data, **numerical_data})

    return _make_df


def test_model(mock_dataframe: Callable[..., pd.DataFrame]) -> None:
    # Generate a mock dataframe
    df = mock_dataframe(n_rows=100, n_categorical=2, n_numerical=2)
    # Ensure the dataframe is as expected
    assert df.shape == (100, 4)
    assert all(col.startswith(("num_col_", "cat_col_")) for col in df.columns)

    # Add two variables of interest
    df["voi_1"] = (
        df["num_col_0"] * 2
        + df["num_col_1"] * 3
        + (df["cat_col_0"] == "cat_0").astype(float) * 0.5
    )
    df["voi_2"] = (
        df["num_col_0"] * 4
        + df["num_col_1"] * 2
        + (df["cat_col_1"] == "cat_0").astype(float) * 0.5
    )

    # Instantiate the DirectNormativeModel
    model_1 = snm.DirectNormativeModel.from_dataframe(
        model_type="HBR",
        dataframe=df,
        variable_of_interest="voi_1",
        numerical_covariates=["num_col_0", "num_col_1"],
        categorical_covariates=["cat_col_0", "cat_col_1"],
        batch_covariates=["cat_col_1"],
        nonlinear_covariates=["num_col_0"],
        influencing_mean=["num_col_0", "num_col_1", "cat_col_0", "cat_col_1"],
        influencing_variance=["num_col_1"],
    )
    model_2 = snm.DirectNormativeModel.from_dataframe(
        model_type="HBR",
        dataframe=df,
        variable_of_interest="voi_2",
        numerical_covariates=["num_col_0", "num_col_1"],
        categorical_covariates=["cat_col_0", "cat_col_1"],
        batch_covariates=["cat_col_1"],
        nonlinear_covariates=["num_col_0"],
        influencing_mean=["num_col_0", "num_col_1", "cat_col_0", "cat_col_1"],
        influencing_variance=["num_col_1"],
    )

    # Reduce number of iterations for testing purposes
    model_1.defaults["advi_iterations"] = 100
    model_2.defaults["advi_iterations"] = 100

    # Fit the direct models
    model_1.fit(df)
    model_2.fit(df)

    # Extract direct predictions
    df[
        [
            "voi_1_mu_estimate",
            "voi_1_std_estimate",
        ]
    ] = model_1.predict(df).to_array().T
    df[
        [
            "voi_2_mu_estimate",
            "voi_2_std_estimate",
        ]
    ] = model_2.predict(df).to_array().T

    # Instantiate the CovarianceNormativeModel
    cov_model = snm.CovarianceNormativeModel.from_direct_model(
        model_1,
        variable_of_interest_1="voi_1",
        variable_of_interest_2="voi_2",
    )

    # Make sure the model is initialized correctly
    assert isinstance(cov_model, snm.CovarianceNormativeModel)

    # Fit the model
    cov_model.fit(df)

    # Check that the model has fitted attributes
    assert hasattr(cov_model, "model_params")
