"""
tests/test_direct.py

Tests for the DirectNormativeModel class in the Spectranorm package.
"""

from collections.abc import Callable
from pathlib import Path

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


def test_model(mock_dataframe: Callable[..., pd.DataFrame], tmp_path: Path) -> None:
    # Generate a mock dataframe
    df = mock_dataframe(n_rows=100, n_categorical=2, n_numerical=2)
    # Ensure the dataframe is as expected
    assert df.shape == (100, 4)
    assert all(col.startswith(("num_col_", "cat_col_")) for col in df.columns)

    # Add a variable of interest
    df["voi"] = (
        df["num_col_0"] * 2
        + df["num_col_1"] * 3
        + (df["cat_col_0"] == "cat_0").astype(float) * 0.5
    )

    # Instantiate the DirectNormativeModel
    model = snm.DirectNormativeModel.from_dataframe(
        model_type="HBR",
        dataframe=df,
        variable_of_interest="voi",
        numerical_covariates=["num_col_0", "num_col_1"],
        categorical_covariates=["cat_col_0", "cat_col_1"],
        batch_covariates=["cat_col_1"],
        nonlinear_covariates=["num_col_0"],
        influencing_mean=["num_col_0", "num_col_1", "cat_col_0", "cat_col_1"],
        influencing_variance=["num_col_1"],
    )

    # Reduce number of iterations for testing purposes
    model.defaults["advi_iterations"] = 100

    # Make sure the model is initialized correctly
    assert isinstance(model, snm.DirectNormativeModel)

    # Fit the model
    model.fit(df)

    # Check that the model has fitted attributes
    assert hasattr(model, "model_params")

    # Predict on the same data
    predictions = model.predict(df, extended=True)

    # Check that predictions have the correct type
    assert isinstance(predictions, snm.NormativePredictions)

    # Check that predictions can be converted to DataFrame
    pred_df = predictions.to_dataframe()
    assert isinstance(pred_df, pd.DataFrame)

    # Run evaluations
    predictions.evaluate_predictions(
        variable_of_interest=df["voi"].to_numpy(),
        train_mean=np.asarray(df["voi"].mean()),
        train_std=np.asarray(df["voi"].std()),
        n_params=len(model.model_params),
    )

    # Check that evaluation metrics are computed
    assert hasattr(predictions, "evaluations")

    # Define a path to save the model
    model_path = tmp_path / "saved_direct_model"
    snm.utils.general.ensure_dir(model_path / "saved_model")
    assert model_path.exists()

    # Check saving the model
    model.save_model(model_path)
    assert (model_path / "saved_model/model_dict.joblib").exists()

    # Check loading the model
    loaded_model = snm.DirectNormativeModel.load_model(model_path)
    assert isinstance(loaded_model, snm.DirectNormativeModel)
