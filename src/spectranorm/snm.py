"""
snm.py

Core implementation of spectral normative modeling (SNM).

This module provides the code base for using spectral normative models. It can
be used to fit direct and spectral normative models to data and also to
predict normative centiles using pre-trained models.

See full documentation at:
https://sina-mansour.github.io/spectranorm
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import arviz as az
import joblib
import numpy as np
import pandas as pd
import patsy
import pymc as pm  # type: ignore[import-untyped]
import pytensor.tensor as pt
from pytensor.compile.sharedvalue import shared
from scipy import sparse

from . import utils

if TYPE_CHECKING:
    import numpy.typing as npt

    TensorVariable = Any  # mypy-safe stand-in for PyTensor objects

__all__ = ["utils"]

# ruff adjustments
# ruff: noqa: PLR0913

# Type aliases
CovariateType = Literal["categorical", "numerical"]
NumericalEffect = Literal["linear", "spline"]
ModelType = Literal["HBR", "BLR"]
MmapMode = Literal["r+", "r", "w+", "c"]

# Constants
DEFAULT_SPLINE_DF: int = 5
DEFAULT_SPLINE_DEGREE: int = 3
DEFAULT_SPLINE_EXTRAPOLATION_FACTOR: float = 0.1
DEFAULT_ADVI_ITERATIONS: int = 20_000
DEFAULT_ADVI_CONVERGENCE_TOLERANCE: float = 1e-3
DEFAULT_RANDOM_SEED: int = 12345
DEFAULT_ADAM_LEARNING_RATE: float = 0.1
DEFAULT_ADAM_LEARNING_RATE_DECAY: float = 0.9995

# Set up logging
logger = utils.general.get_logger(__name__)


@dataclass
class NormativePredictions:
    """
    Container for the results of model.predict() function.

    Attributes:
        predictions: dict
            Dictionary containing the model's predictions, including
            - Predictions of mean (mu_estimate).
            - Predictions of standard deviation (std_estimate).
            - [Optional] The observed variable of interest (the name of which is
              provided in the function argument).
            - [Optional] Additional evaluation metrics for the predictions.
    """

    predictions: dict[str, npt.NDArray[np.floating[Any]]]
    evaluations: dict[str, npt.NDArray[np.floating[Any]] | float] = field(
        default_factory=dict,
    )

    def extend_predictions(
        self,
        variable_of_interest: npt.NDArray[np.floating[Any]],
    ) -> NormativePredictions:
        """
        Extend the NormativePredictions (predictions dictionary) with additional
        statistics.

        Args:
            variable_of_interest: np.ndarray
                The observed values for the variable(s) of interest.

        Returns:
            NormativePredictions
                Extended NormativePredictions with additional statistics.
        """
        self.predictions["z-score"] = (
            variable_of_interest - self.predictions["mu_estimate"]
        ) / self.predictions["std_estimate"]
        self.predictions["log-likelihood"] = (
            utils.stats.compute_censored_log_likelihood(
                variable_of_interest,
                self.predictions["mu_estimate"],
                self.predictions["std_estimate"],
            )
        )
        self.predictions["centiles"] = utils.stats.compute_centiles_from_z_scores(
            self.predictions["z-score"],
        )

        self.predictions["variable_of_interest"] = variable_of_interest

        return self

    def evaluate_predictions(
        self,
        variable_of_interest: npt.NDArray[np.floating[Any]],
        train_mean: npt.NDArray[np.floating[Any]],
        train_std: npt.NDArray[np.floating[Any]],
        n_params: int | None = None,
        msll_censored_quantile: float = 0.01,
    ) -> NormativePredictions:
        """
        Evaluate the predictions against the observed variable of interest.

        This function computes a battery of evaluation metrics implemented
        in `snm.utils.metrics`. Namely the evaluations include:
            - Mean Absolute Error (MAE)
            - Mean Squared Error (MSE)
            - Root Mean Squared Error (RMSE)
            - Mean Absolute Percentage Error (MAPE)
            - R-squared
            - Explained Variance Score
            - Mean Standardized Log Loss (MSLL)

        Args:
            variable_of_interest: np.ndarray
                The observed values for the variable(s) of interest.
            train_mean: np.ndarray
                Mean(s) of the variable(s) of interest from the training data.
            train_std: np.ndarray
                Standard deviation(s) of the variable(s) of interest from the training
                data.
            n_params: int
                Number of free parameters in the model.
            msll_censored_quantile: float (default=0.02)
                Quantile below which log-likelihoods are censored for MSLL.

        Returns:
            NormativePredictions
                Object containing the evaluation results.
        """
        self.extend_predictions(variable_of_interest)
        # Mean Absolute Error (MAE)
        self.evaluations["MAE"] = utils.metrics.compute_mae(
            y=self.predictions["variable_of_interest"],
            y_pred=self.predictions["mu_estimate"],
        )
        # Mean Squared Error (MSE)
        self.evaluations["MSE"] = utils.metrics.compute_mse(
            y=self.predictions["variable_of_interest"],
            y_pred=self.predictions["mu_estimate"],
        )
        # Root Mean Squared Error (RMSE)
        self.evaluations["RMSE"] = utils.metrics.compute_rmse(
            y=self.predictions["variable_of_interest"],
            y_pred=self.predictions["mu_estimate"],
        )
        # Mean Absolute Percentage Error (MAPE)
        self.evaluations["MAPE"] = utils.metrics.compute_mape(
            y=self.predictions["variable_of_interest"],
            y_pred=self.predictions["mu_estimate"],
        )
        # R-squared
        self.evaluations["R-squared"] = utils.metrics.compute_r2(
            y=self.predictions["variable_of_interest"],
            y_pred=self.predictions["mu_estimate"],
        )
        # Explained Variance Score
        self.evaluations["Explained Variance"] = utils.metrics.compute_expv(
            y=self.predictions["variable_of_interest"],
            y_pred=self.predictions["mu_estimate"],
        )
        # Mean Standardized Log Loss (MSLL)
        self.evaluations["MSLL"] = utils.metrics.compute_msll(
            model_log_likelihoods=self.predictions["log-likelihood"],
            baseline_log_likelihoods=utils.stats.compute_censored_log_likelihood(
                self.predictions["variable_of_interest"],
                train_mean,
                train_std,
                censored_quantile=msll_censored_quantile,
            ),
        )
        _ = n_params  # keep for future use (e.g. information criteria calculations)

        return self

    def to_array(self, keys: list[str] | None = None) -> npt.NDArray[np.floating[Any]]:
        """
        Return prediction results as a list of NumPy arrays.

        Args:
            keys: list[str]
                Optional list of keys to return.
                Defaults to ["mu_estimate", "std_estimate"].

        Returns:
            list[np.ndarray]
                NumPy arrays for the requested predictions
        """
        keys = keys or ["mu_estimate", "std_estimate"]
        return np.array([self.predictions[key] for key in keys])

    def to_dataframe(
        self,
        index: pd.Index[Any] | list[Any] | None = None,
    ) -> pd.DataFrame:
        """
        Return prediction results as a DataFrame.

        Args:
            index: pd.Index | list | None
                Optional index for the DataFrame (defaults to None)

        Returns:
            pd.DataFrame
                DataFrame containing the predictions
        """
        predictions = self.predictions.copy()
        # Flatten the predictions dictionary if multiple queries are predicted
        for key in predictions:
            if predictions[key].ndim > 1:
                if predictions[key].shape[1] == 1:
                    predictions[key] = predictions[key].flatten()
                else:
                    for i in range(predictions[key].shape[1]):
                        predictions[f"{key}_{i + 1}"] = predictions[key][:, i]
                    # delete the key
                    del predictions[key]

        # Make a new DataFrame for the predictions dictionary
        return pd.DataFrame(predictions, index=index)


@dataclass
class SplineSpec:
    """
    Specification for spline basis construction.

    Attributes:
        df: int
            Degrees of freedom (number of basis functions).
        degree: int
            Degree of the spline (e.g., 3 for cubic splines).
        lower_bound: float
            Lower boundary for the spline domain.
        upper_bound: float
            Upper boundary for the spline domain.
        knots: Optional[List[float]]
            Optional list of internal knot locations within the spline domain
            (excluding the boundary knots). Must be strictly increasing and
            contain exactly `df - degree - 1` values. If unspecified, then
            equally spaced quantiles of the input data are used.
    """

    lower_bound: float
    upper_bound: float
    df: int = DEFAULT_SPLINE_DF
    degree: int = DEFAULT_SPLINE_DEGREE
    knots: list[float] | None = None

    # Validation checks for the spline specification.
    def __post_init__(self) -> None:
        # Check that df (degrees of freedom) is greater than degree
        if self.df <= self.degree:
            err = "df (degrees of freedom) must be greater than degree."
            raise ValueError(err)
        # Check that degree is at least 1
        if self.degree < 1:
            err = "degree must be at least 1."
            raise ValueError(err)
        # Check that lower_bound and upper_bound are numeric
        if self.lower_bound >= self.upper_bound:
            err = "lower_bound must be less than upper_bound."
            raise ValueError(err)
        if self.knots is not None:
            if not all(isinstance(k, (int, float)) for k in self.knots):
                err = "All knots must be numeric (int or float)."
                raise TypeError(err)
            # Check if knots are strictly increasing
            if not all(x < y for x, y in zip(self.knots, self.knots[1:])):
                err = "Knots must be strictly increasing."
                raise ValueError(err)
            # Check if knots are within bounds
            if any(k < self.lower_bound or k > self.upper_bound for k in self.knots):
                err = (
                    "All knots must be within the bounds defined by "
                    "lower_bound and upper_bound."
                )
                raise ValueError(err)
            # Check if the number of knots is correct
            if len(self.knots) != (self.df - self.degree - 1):
                err = (
                    f"knots must contain exactly {self.df - self.degree - 1} "
                    f"values, got {len(self.knots)}."
                )
                raise ValueError(err)

    @classmethod
    def create_spline_spec(
        cls,
        values: pd.Series[float],
        df: int = DEFAULT_SPLINE_DF,
        degree: int = DEFAULT_SPLINE_DEGREE,
        knots: list[float] | None = None,
        extrapolation_factor: float = DEFAULT_SPLINE_EXTRAPOLATION_FACTOR,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> SplineSpec:
        """
        Create a spline specification from a pandas Series.

        Args:
            values: pd.Series
                The list of input values to make the spline.
            df: int
                Degrees of freedom for the spline (default is 5).
            degree: int
                Degree of the spline (default is 3).
            knots: list[float] | None
                [Optional] List of internal knot locations within the spline domain.
                If None, equally spaced quantiles of the input data are used.
            extrapolation_factor: float, positive, default is 0.1
                [Optional] Factor to extend the lower and upper bounds of the spline
                domain.
            lower_bound: float | None
                [Optional] Lower boundary for the spline domain. If None, it is set to
                `values.min() - extrapolation_factor * (values.max() - values.min())`.
            upper_bound: float | None
                [Optional] Upper boundary for the spline domain. If None, it is set to
                `values.max() + extrapolation_factor * (values.max() - values.min())`.

        Returns:
            SplineSpec
                The created spline specification.
        """
        extrapolation = extrapolation_factor * (values.max() - values.min())
        if lower_bound is None:
            lower_bound = values.min() - extrapolation
        if upper_bound is None:
            upper_bound = values.max() + extrapolation
        if knots is None:
            # Use equally spaced quantiles as knots
            knots = np.linspace(lower_bound, upper_bound, df - degree + 1)[
                1:-1
            ].tolist()
        return cls(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            df=df,
            degree=degree,
            knots=knots,
        )


@dataclass
class CovariateSpec:
    """
    Specification of a single covariate and how it should be modeled.

    Attributes:
        name: str
            Name of the covariate (e.g., 'age', 'site').
        cov_type: str
            Type of the covariate ('numerical' or 'categorical').
        effect: str
            For numerical covariates, how the effect is modeled ('linear'
            or 'spline').
        categories: np.ndarray | None
            For categorical covariates, the category labels stored as a NumPy array.
        hierarchical: bool
            For categorical covariates, whether to model with a
            hierarchical structure.
        spline_spec: SplineSpec | None
            Optional SplineSpec instance for spline modeling;
            required if effect is 'spline'.

    Validation:
        - Numerical covariates must specify 'effect'.
        - If 'effect' is 'spline', 'spline_spec' must be provided.
        - Categorical covariates must specify 'hierarchical'.
        - Categorical covariates cannot have 'effect' or 'spline_spec'.
        - Categorical covariates must have categories listed.
    """

    name: str
    cov_type: CovariateType  # "categorical" or "numerical"
    effect: NumericalEffect | None = None  # Only if numerical
    categories: npt.NDArray[np.str_] | None = None  # Only if categorical
    hierarchical: bool | None = None  # Only if categorical
    spline_spec: SplineSpec | None = None  # Only for spline modeling
    moments: tuple[float, float] | None = None  # Only for linear effects

    def __repr__(self) -> str:
        """
        String representation of the CovariateSpec instance.
        """
        representation = f"CovariateSpec(name={self.name}, cov_type={self.cov_type}"
        if self.cov_type == "numerical":
            representation += f", effect={self.effect}"
        elif self.cov_type == "categorical":
            representation += f", hierarchical={self.hierarchical}"
            if self.categories is not None:
                representation += f", n_categories={len(self.categories.tolist())}"
        representation += ")"
        return representation

    # Validation checks for the covariate specification.
    def validate_numerical(self) -> None:
        if self.effect not in {"linear", "spline"}:
            err = (
                f"Numerical covariate '{self.name}' must specify effect as "
                "'linear' or 'spline'."
            )
            raise ValueError(err)
        if self.hierarchical is not None:
            err = (
                f"Numerical covariate '{self.name}' should not specify 'hierarchical'."
            )
            raise ValueError(err)
        if self.categories is not None:
            err = f"Numerical covariate '{self.name}' should not specify 'categories'."
            raise ValueError(err)
        if self.effect == "spline":
            if self.spline_spec is None:
                err = (
                    f"Numerical covariate '{self.name}' must have spline "
                    "specification if effect is 'spline'."
                )
                raise ValueError(err)
            if self.moments is not None:
                err = (
                    f"Numerical covariate '{self.name}' should not specify "
                    "moments if effect is 'spline'."
                )
                raise ValueError(err)
        if self.effect == "linear":
            if self.spline_spec is not None:
                err = (
                    f"Numerical covariate '{self.name}' should not have spline "
                    "specification unless effect is 'spline'."
                )
                raise ValueError(err)
            if self.moments is None:
                err = (
                    f"Numerical covariate '{self.name}' must specify moments "
                    "(mean and standard deviation) for linear effects."
                )
                raise ValueError(err)

    def validate_categorical(self) -> None:
        if self.effect is not None:
            err = (
                f"Categorical covariate '{self.name}' should not have a "
                "numerical effect type."
            )
            raise ValueError(err)
        if self.spline_spec is not None:
            err = (
                f"Categorical covariate '{self.name}' should not have spline "
                "specification."
            )
            raise ValueError(err)
        if self.hierarchical is None:
            err = (
                f"Categorical covariate '{self.name}' must specify whether "
                "it is hierarchical."
            )
            raise ValueError(err)
        if self.categories is None:
            err = f"Categorical covariate '{self.name}' must specify categories."
            raise ValueError(err)
        if not isinstance(self.categories, np.ndarray):
            err = (
                f"Categorical covariate '{self.name}' must specify categories "
                "as a NumPy array."
            )
            raise TypeError(err)

    def __post_init__(self) -> None:
        if self.cov_type == "numerical":
            self.validate_numerical()
        elif self.cov_type == "categorical":
            self.validate_categorical()
        else:
            err = f"Invalid covariate type '{self.cov_type}' for '{self.name}'."
            raise ValueError(err)

    def make_spline_bases(
        self,
        values: npt.NDArray[np.floating[Any]],
        *,
        include_intercept: bool = True,
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Create B-spline basis expansion functions for a given covariate.

        Args:
            values (np.ndarray): The values to create the spline basis functions for.

        Returns:
            np.ndarray: The B-spline basis function expansion.
        """
        if self.effect != "spline" or self.spline_spec is None:
            err = f"Covariate '{self.name}' is not a spline covariate."
            raise ValueError(err)

        # Create B-spline basis functions
        return np.array(
            patsy.bs(  # pyright: ignore[reportAttributeAccessIssue]
                values,
                knots=self.spline_spec.knots,
                df=self.spline_spec.df,
                degree=self.spline_spec.degree,
                lower_bound=self.spline_spec.lower_bound,
                upper_bound=self.spline_spec.upper_bound,
                include_intercept=include_intercept,
            ),
        )

    def factorize_categories(
        self,
        values: npt.NDArray[np.str_],
    ) -> npt.NDArray[np.int_]:
        """
        Factorize categorical covariate values into numerical indices.

        Args:
            values (np.ndarray): The values to factorize.

        Returns:
            np.ndarray: The factorized numerical indices for the categories.
        """
        if self.cov_type != "categorical":
            err = (
                f"Covariate '{self.name}' is not a categorical "
                "covariate to be factorized."
            )
            raise ValueError(err)

        # Create a mapping from category values to indices
        if self.categories is None:  # to satisfy type checker
            err = f"Covariate '{self.name}' does not have categories defined."
            raise ValueError(err)
        category_mapping: dict[str, int] = {
            category: idx for idx, category in enumerate(self.categories)
        }
        # Factorize the values using the mapping
        return np.array([category_mapping[val] for val in values], dtype=int)

    def extend_categories(
        self,
        new_categories: npt.NDArray[np.str_],
    ) -> None:
        """
        Extend the categories of a categorical covariate with new categories.

        Args:
            new_categories (np.ndarray): The new categories to add.

        Returns:
            None
        """
        if self.cov_type != "categorical":
            err = f"Covariate '{self.name}' is not a categorical covariate to extend."
            raise ValueError(err)
        if self.categories is None:  # to satisfy type checker
            err = f"Covariate '{self.name}' does not have categories defined."
            raise ValueError(err)

        # Make sure categories are unique and new
        unique_new_categories = np.setdiff1d(new_categories, self.categories)
        if unique_new_categories.size < new_categories.size:
            err = (
                f"Some new categories are already present in the "
                f"covariate '{self.name}'."
            )
            raise ValueError(err)

        # Extend the categories array
        self.categories = np.concatenate((self.categories, unique_new_categories))


@dataclass
class NormativeModelSpec:
    """
    General specification of a normative model.

    Attributes:
        variable_of_interest: str
            Name of the target variable to model (e.g., "thickness").
        covariates: list[CovariateSpec]
            Listing all model covariates and specifying how each covariate is modeled.
        influencing_mean: list[str]
            List of covariate names that influence the mean of the variable of interest.
        influencing_variance: list[str]
            List of covariate names that influence the variance of the variable of
            interest.
    """

    variable_of_interest: str
    covariates: list[CovariateSpec]
    influencing_mean: list[str]
    influencing_variance: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.variable_of_interest, str):
            err = "variable_of_interest must be a string."
            raise TypeError(err)
        if not isinstance(self.covariates, list):
            err = "covariates must be a list of CovariateSpec instances."
            raise TypeError(err)
        if not all(isinstance(cov, CovariateSpec) for cov in self.covariates):
            err = "All items in covariates must be CovariateSpec instances."
            raise TypeError(err)
        if not isinstance(self.influencing_mean, list):
            err = "influencing_mean must be a list of covariate names."
            raise TypeError(err)
        if not isinstance(self.influencing_variance, list):
            err = "influencing_variance must be a list of covariate names."
            raise TypeError(err)


@dataclass
class CovarianceModelSpec:
    """
    General specification of a normative model for covariance.

    This model aims to learn the relationships between two variables of interest for
    both of which a normative model is specified. This can capture patterns where the
    normative trends in two variables are related.

    Attributes:
        variable_of_interest_1: str
            Name of the first variable of interest.
        variable_of_interest_2: str
            Name of the second variable of interest.
        covariates: list[CovariateSpec]
            Listing all model covariates and specifying how each covariate is modeled.
        influencing_covariance: list[str]
            List of covariate names that influence the covariance between the two
            variables of interest.
    """

    variable_of_interest_1: str
    variable_of_interest_2: str
    covariates: list[CovariateSpec]
    influencing_covariance: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.variable_of_interest_1, str):
            err = "variable_of_interest_1 must be a string."
            raise TypeError(err)
        if not isinstance(self.variable_of_interest_2, str):
            err = "variable_of_interest_2 must be a string."
            raise TypeError(err)
        if not isinstance(self.covariates, list):
            err = "covariates must be a list of CovariateSpec instances."
            raise TypeError(err)
        if not all(isinstance(cov, CovariateSpec) for cov in self.covariates):
            err = "All items in covariates must be CovariateSpec instances."
            raise TypeError(err)
        if not isinstance(self.influencing_covariance, list):
            err = "influencing_covariance must be a list of covariate names."
            raise TypeError(err)


@dataclass
class DirectNormativeModel:
    """
    Direct normative model implementation.

    This class implements the direct normative modeling approach, which
    directly models the variable of interest using the specified covariates.
    It can be used to fit a model to data and predict normative centiles.

    Attributes:
        spec: NormativeModelSpec
            Specification of the normative model including variable of interest,
            covariates, and data source.
        defaults: dict
            Default parameters for the model, including spline specifications,
            ADVI iterations, convergence tolerance, random seed, and Adam optimizer
            learning rates.
    """

    spec: NormativeModelSpec
    defaults: dict[str, Any] = field(
        default_factory=lambda: {
            "spline_df": DEFAULT_SPLINE_DF,
            "spline_degree": DEFAULT_SPLINE_DEGREE,
            "spline_extrapolation_factor": DEFAULT_SPLINE_EXTRAPOLATION_FACTOR,
            "advi_iterations": DEFAULT_ADVI_ITERATIONS,
            "advi_convergence_tolerance": DEFAULT_ADVI_CONVERGENCE_TOLERANCE,
            "random_seed": DEFAULT_RANDOM_SEED,
            "adam_learning_rate": DEFAULT_ADAM_LEARNING_RATE,
            "adam_learning_rate_decay": DEFAULT_ADAM_LEARNING_RATE_DECAY,
        },
    )

    def __repr__(self) -> str:
        """
        String representation of the DirectNormativeModel instance.
        """
        return f"DirectNormativeModel(spec={self.spec})"

    @staticmethod
    def _validate_init_args(
        model_type: ModelType,
        variable_of_interest: str,
        numerical_covariates: list[str],
        categorical_covariates: list[str],
        batch_covariates: list[str],
        nonlinear_covariates: list[str],
    ) -> None:
        # Validity checks for input parameters
        if model_type not in {"HBR", "BLR"}:
            err = f"Invalid model type '{model_type}'. Must be 'HBR' or 'BLR'."
            raise ValueError(err)
        for list_name, covariate_list in [
            ("numerical", numerical_covariates),
            ("categorical", categorical_covariates),
            ("batch", batch_covariates),
            ("nonlinear", nonlinear_covariates),
        ]:
            if not all(isinstance(item, str) for item in covariate_list):
                err = f"All covariate names must be strings: {list_name} covariates."
                raise TypeError(err)
        if not isinstance(variable_of_interest, str):
            err = "Variable of interest must be a string."
            raise TypeError(err)
        if not all(col in categorical_covariates for col in batch_covariates):
            err = "All batch covariates must be included in categorical covariates."
            raise ValueError(err)
        if not all(col in numerical_covariates for col in nonlinear_covariates):
            err = "All nonlinear covariates must be included in numerical covariates."
            raise ValueError(err)

    @classmethod
    def from_dataframe(
        cls,
        model_type: ModelType,
        dataframe: pd.DataFrame,
        variable_of_interest: str,
        numerical_covariates: list[str] | None = None,
        categorical_covariates: list[str] | None = None,
        batch_covariates: list[str] | None = None,
        nonlinear_covariates: list[str] | None = None,
        influencing_mean: list[str] | None = None,
        influencing_variance: list[str] | None = None,
        spline_kwargs: dict[str, Any] | None = None,
    ) -> DirectNormativeModel:
        """
        Initialize a normative model from a pandas DataFrame.

        Args:
            model_type: ModelType
                Type of the model to create, either "HBR" (Hierarchical Bayesian
                Regression) or "BLR" (Bayesian Linear Regression).
            dataframe: pd.DataFrame
                DataFrame containing the data.
            variable_of_interest: str
                Name of the target variable to model.
            numerical_covariates: list[str] | None
                List of numerical covariate names.
            categorical_covariates: list[str] | None
                List of categorical covariate names.
            batch_covariates: list[str] | None
                List of batch covariate names which should also be included in
                categorical_covariates.
            nonlinear_covariates: list[str] | None
                List of covariate names to be modeled as nonlinear effects.
                These should also be included in numerical_covariates.
            influencing_mean: list[str] | None
                List of covariate names that influence the mean of the variable
                of interest. These should be included in either numerical_covariates
                or categorical_covariates.
            influencing_variance: list[str] | None
                List of covariate names that influence the variance of the variable
                of interest. These should be included in either numerical_covariates
                or categorical_covariates.
            spline_kwargs: dict
                Additional keyword arguments for spline specification, such as
                `df`, `degree`, and `knots`. These are passed to the
                `create_spline_spec` method to create spline specifications for
                nonlinear covariates.

        Returns:
            DirectNormativeModel
                An instance of DirectNormativeModel initialized with the provided data.
        """
        # Set default values for optional parameters
        numerical_covariates = numerical_covariates or []
        categorical_covariates = categorical_covariates or []
        batch_covariates = batch_covariates or []
        nonlinear_covariates = nonlinear_covariates or []
        influencing_mean = influencing_mean or []
        influencing_variance = influencing_variance or []
        spline_kwargs = spline_kwargs or {}

        # Validity checks for input parameters
        cls._validate_init_args(
            model_type,
            variable_of_interest,
            numerical_covariates,
            categorical_covariates,
            batch_covariates,
            nonlinear_covariates,
        )
        utils.general.validate_dataframe(
            dataframe,
            [variable_of_interest, *numerical_covariates, *categorical_covariates],
        )

        # Create an instance of the class
        self = cls(
            spec=NormativeModelSpec(
                variable_of_interest=variable_of_interest,
                covariates=[],
                influencing_mean=influencing_mean,
                influencing_variance=influencing_variance,
            ),
        )

        # Populate the spline_kwargs with defaults if not provided
        spline_kwargs["df"] = spline_kwargs.get("df", self.defaults["spline_df"])
        spline_kwargs["degree"] = spline_kwargs.get(
            "degree",
            self.defaults["spline_degree"],
        )
        spline_kwargs["extrapolation_factor"] = spline_kwargs.get(
            "extrapolation_factor",
            self.defaults["spline_extrapolation_factor"],
        )

        # Start building the model specification
        # Add categorical covariates
        for cov_name in categorical_covariates:
            hierarchical = False
            if cov_name in batch_covariates and model_type == "HBR":
                hierarchical = True
            self.spec.covariates.append(
                CovariateSpec(
                    name=cov_name,
                    cov_type="categorical",
                    categories=dataframe[cov_name].unique(),
                    hierarchical=hierarchical,
                ),
            )
        for cov_name in numerical_covariates:
            if cov_name not in nonlinear_covariates:
                self.spec.covariates.append(
                    CovariateSpec(
                        name=cov_name,
                        cov_type="numerical",
                        effect="linear",
                        moments=(
                            dataframe[cov_name].mean(),
                            dataframe[cov_name].std(),
                        ),
                    ),
                )
            else:
                self.spec.covariates.append(
                    CovariateSpec(
                        name=cov_name,
                        cov_type="numerical",
                        effect="spline",
                        spline_spec=SplineSpec.create_spline_spec(
                            dataframe[cov_name],
                            **spline_kwargs,
                        ),
                    ),
                )
        return self

    def _validate_model(self) -> None:
        """
        Validate the model instance.

        This method checks if the model instance is complete and valid.
        It raises errors if any required fields are missing or if there are
        inconsistencies in the model specification.
        """
        if self.spec is None:
            err = (
                "Model specification is not set. "
                "Please initialize the model, e.g., with 'from_dataframe'."
            )
            raise ValueError(err)
        if len(self.spec.covariates) == 0:
            err = (
                "No covariates specified in the model. "
                "Please add covariates to the specification."
            )
            raise ValueError(err)
        if (len(self.spec.influencing_mean) == 0) and (
            len(self.spec.influencing_variance) == 0
        ):
            err = (
                "No covariates specified to influence the mean or "
                "variance of the variable of interest."
            )
            raise ValueError(err)

    def save_model(self, directory: Path, *, save_posterior: bool = False) -> None:
        """
        Save the fitted model and it's posterior to a directory.
        The model will be saved in a subdirectory named 'saved_model'.
        If this directory is not empty, an error is raised.

        Args:
            directory: Path
                Path to a directory to save the model.
            save_posterior: bool (default=False)
                If True, save the model's posterior trace inference data.
        """
        # Prepare the save directory
        directory = Path(directory)
        saved_model_dir = utils.general.prepare_save_directory(directory, "saved_model")

        model_dict = {
            "spec": self.spec,
            "defaults": self.defaults,
        }
        if hasattr(self, "model_params"):
            model_dict["model_params"] = self.model_params
            if hasattr(self, "model_inference_data") and save_posterior:
                self.model_inference_data.to_netcdf(
                    saved_model_dir / "model_inference_data.nc",
                )
        joblib.dump(model_dict, saved_model_dir / "model_dict.joblib")

    @classmethod
    def load_model(
        cls,
        directory: Path,
        *,
        load_posterior: bool = False,
    ) -> DirectNormativeModel:
        """
        Load the model and its posterior from a directory.
        The model will be loaded from a subdirectory named 'saved_model'.

        Args:
            directory: Path
                Path to the directory containing the model.
            load_posterior: bool (default=False)
                If True, load the model's posterior trace from the saved inference data.
        """
        # Validate the load directory
        directory = Path(directory)
        saved_model_dir = utils.general.validate_load_directory(
            directory,
            "saved_model",
        )

        # Load the saved model dict
        model_dict = joblib.load(saved_model_dir / "model_dict.joblib")

        # Create an instance of the class
        instance = cls(
            spec=model_dict["spec"],
        )

        # Set the attributes from the loaded model dictionary
        instance.defaults.update(model_dict["defaults"])
        if "model_params" in model_dict:
            instance.model_params = model_dict["model_params"]
            if load_posterior:
                instance.model_inference_data = az.from_netcdf(  # type: ignore[no-untyped-call]
                    saved_model_dir / "model_inference_data.nc",
                )

        return instance

    def _validate_dataframe_for_fitting(self, train_data: pd.DataFrame) -> None:
        """
        Validate the training DataFrame for fitting.
        """
        utils.general.validate_dataframe(
            train_data,
            (
                [cov.name for cov in self.spec.covariates]
                + [self.spec.variable_of_interest]
            ),
        )

    def _build_model_coordinates(
        self,
        observations: npt.NDArray[np.integer[Any]],
    ) -> dict[str, Any]:
        """
        Build the model coordinates for the training DataFrame.
        """
        # Data coordinates
        model_coords = {"observations": observations, "scalar": [0]}

        # Additional coordinates for covariates
        for cov in self.spec.covariates:
            if cov.cov_type == "numerical":
                if cov.effect == "spline":
                    if cov.spline_spec is not None:  # to satisfy type checker
                        model_coords[f"{cov.name}_splines"] = np.arange(
                            cov.spline_spec.df,
                        )
                elif cov.effect == "linear":
                    model_coords[f"{cov.name}_linear"] = np.arange(1)
            elif cov.cov_type == "categorical":
                model_coords[cov.name] = cov.categories
            else:
                err = f"Invalid covariate type '{cov.cov_type}' for '{cov.name}'."
                raise ValueError(err)
        return model_coords

    def _model_linear_mean_effect(
        self,
        train_data: pd.DataFrame,
        cov: CovariateSpec,
        effects_list: list[TensorVariable],
        sigma_prior: float = 10,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Model a linear effect for a numerical covariate on the mean estimate.
        """
        # Linear effect
        if adapt is None:  # Model fitting
            linear_beta = pm.Normal(
                f"linear_beta_{cov.name}",
                mu=0,
                sigma=sigma_prior,
                size=1,
                dims=(f"{cov.name}_linear",),
            )
            # Increment parameter count for linear effect
            self.model_params["n_params"] += 1
        else:  # Freeze during adaptation/fine-tuning
            linear_beta = pm.Deterministic(
                f"linear_beta_{cov.name}",
                pt.as_tensor_variable(
                    adapt["pretrained_model_params"]["posterior_means"][
                        f"linear_beta_{cov.name}"
                    ],
                ),
                dims=(f"{cov.name}_linear",),
            )
        if cov.moments is not None:  # to satisfy type checker
            effects_list.append(
                (
                    cast("npt.NDArray[Any]", train_data[cov.name].to_numpy())
                    - cov.moments[0]
                )
                / cov.moments[1]
                * linear_beta,
            )

    def _model_spline_mean_effect(
        self,
        train_data: pd.DataFrame,
        cov: CovariateSpec,
        effects_list: list[TensorVariable],
        spline_bases: dict[str, npt.NDArray[np.floating[Any]]],
        sigma_prior: float = 10,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Model a spline effect for a numerical covariate on the mean estimate.
        """
        # Spline effect
        spline_bases[cov.name] = spline_bases.get(
            cov.name,
            cov.make_spline_bases(
                cast("npt.NDArray[Any]", train_data[cov.name].to_numpy()),
            ),
        )
        if adapt is None:  # Model fitting
            # ZeroSumNormal imposes a centering constraint ensuring identifiability
            spline_betas = pm.ZeroSumNormal(
                f"spline_betas_{cov.name}",
                sigma=sigma_prior,
                shape=spline_bases[cov.name].shape[1],
                dims=(f"{cov.name}_splines",),
            )
            # Increment parameter count for spline effects
            if cov.spline_spec is not None:  # to satisfy type checker
                self.model_params["n_params"] += cov.spline_spec.df - 1
        else:  # Freeze during adaptation/fine-tuning
            spline_betas = pm.Deterministic(
                f"spline_betas_{cov.name}",
                pt.as_tensor_variable(
                    adapt["pretrained_model_params"]["posterior_means"][
                        f"spline_betas_{cov.name}"
                    ],
                ),
                dims=(f"{cov.name}_splines",),
            )
        effects_list.append(pt.dot(spline_bases[cov.name], spline_betas.T))

    def _model_categorical_mean_effect(
        self,
        train_data: pd.DataFrame,
        cov: CovariateSpec,
        effects_list: list[TensorVariable],
        category_indices: dict[str, npt.NDArray[np.integer[Any]]],
        sigma_prior: float = 10,
        hierarchical_sigma_prior: float = 1,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Model the effect of a categorical covariate on the mean estimate.
        """
        # Factorize categories
        category_indices[cov.name] = category_indices.get(
            cov.name,
            cov.factorize_categories(
                cast("npt.NDArray[Any]", train_data[cov.name].to_numpy()),
            ),
        )
        if adapt is None:  # Model fitting
            if cov.hierarchical:
                # Hierarchical categorical effect
                # Hyperpriors for category (Bayesian equivalent of random effects)
                sigma_intercept_category = pm.HalfNormal(
                    f"sigma_intercept_{cov.name}",
                    sigma=sigma_prior,
                    dims=("scalar",),
                )

                # Hierarchical intercepts for each category (using reparameterized form)
                categorical_intercept_offset = pm.ZeroSumNormal(
                    f"intercept_offset_{cov.name}",
                    sigma=hierarchical_sigma_prior,
                    dims=(cov.name,),
                )
                # Note ZeroSumNormal imposes a centering constraint
                # (ensuring identifiability)
                categorical_intercept = pm.Deterministic(
                    f"intercept_{cov.name}",
                    (
                        categorical_intercept_offset
                        * pt.reshape(sigma_intercept_category, (1,))
                    ),
                    dims=(cov.name,),
                )

                # Increment parameter count for hierarchical intercept
                self.model_params["n_params"] += 1

            else:
                # Non-hierarchical (linear) categorical effect
                categorical_intercept = pm.ZeroSumNormal(
                    f"intercept_{cov.name}",
                    sigma=sigma_prior,
                    dims=(cov.name,),
                )
                # Note ZeroSumNormal imposes a centering constraint
                # (ensuring identifiability)
            # Increment parameter count for categorical effects
            if cov.categories is not None:  # to satisfy type checker
                self.model_params["n_params"] += len(cov.categories) - 1
        elif cov.name != adapt["covariate_to_adapt"]:
            # Freeze during adaptation/fine-tuning
            if cov.hierarchical:
                # Hierarchical categorical effect
                # Hyperpriors for category (Bayesian equivalent of random effects)
                sigma_intercept_category = pm.Deterministic(
                    f"sigma_intercept_{cov.name}",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"sigma_intercept_{cov.name}"
                        ],
                    ),
                    dims=("scalar",),
                )
                # Hierarchical intercepts for each category (using reparameterized form)
                categorical_intercept_offset = pm.Deterministic(
                    f"intercept_offset_{cov.name}",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"intercept_offset_{cov.name}"
                        ],
                    ),
                    dims=(cov.name,),
                )
                categorical_intercept = pm.Deterministic(
                    f"intercept_{cov.name}",
                    (
                        categorical_intercept_offset
                        * pt.reshape(sigma_intercept_category, (1,))
                    ),
                    dims=(cov.name,),
                )
            else:
                categorical_intercept = pm.Deterministic(
                    f"intercept_{cov.name}",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"intercept_{cov.name}"
                        ],
                    ),
                    dims=(cov.name,),
                )
        else:  # Partial freezing (fit parameters for the new site only)
            if cov.hierarchical:
                # Hierarchical categorical effect
                # Hyperpriors for category (Bayesian equivalent of random effects)
                # Hyperpriors are fixed during adaptation
                sigma_intercept_category = pm.Deterministic(
                    f"sigma_intercept_{cov.name}",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"sigma_intercept_{cov.name}"
                        ],
                    ),
                    dims=("scalar",),
                )
                # Hierarchical intercepts for each category (using reparameterized form)
                # New categories get new parameters, old categories are fixed
                # Freeze old category parameters during adaptation
                fixed_categorical_intercept_offset = pm.Deterministic(
                    f"intercept_offset_{cov.name}_fixed",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"intercept_offset_{cov.name}"
                        ],
                    ),
                )
                # Create new parameters for new categories
                new_category_count = len(adapt["new_category_names"])
                pretrain_sigma_prior = adapt["pretrained_model_params"][
                    "posterior_means"
                ][f"variance_intercept_offset_{cov.name}"].std()
                new_categorical_intercept_offset = pm.Normal(
                    f"intercept_offset_{cov.name}_adapt",
                    mu=0,
                    sigma=pretrain_sigma_prior,
                    size=new_category_count,
                )
                # Combine fixed and new offsets
                categorical_intercept_offset = pm.Deterministic(
                    f"intercept_offset_{cov.name}",
                    pt.concatenate(
                        [
                            fixed_categorical_intercept_offset,
                            new_categorical_intercept_offset,
                        ],
                    ),
                    dims=(cov.name,),
                )
                categorical_intercept = pm.Deterministic(
                    f"intercept_{cov.name}",
                    (
                        categorical_intercept_offset
                        * pt.reshape(sigma_intercept_category, (1,))
                    ),
                    dims=(cov.name,),
                )
            else:
                # Non-hierarchical (linear) categorical effect
                # New categories get new parameters, old categories are fixed
                # Freeze old category parameters during adaptation
                fixed_categorical_intercept = pm.Deterministic(
                    f"intercept_{cov.name}_fixed",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"intercept_{cov.name}"
                        ],
                    ),
                )
                # Create new parameters for new categories
                new_category_count = len(adapt["new_category_names"])
                new_categorical_intercept = pm.Normal(
                    f"intercept_{cov.name}_adapt",
                    mu=0,
                    sigma=sigma_prior,
                    size=new_category_count,
                )
                # Combine fixed and new offsets
                categorical_intercept = pm.Deterministic(
                    f"intercept_{cov.name}",
                    pt.concatenate(
                        [
                            fixed_categorical_intercept,
                            new_categorical_intercept,
                        ],
                    ),
                    dims=(cov.name,),
                )
            self.model_params["n_params"] += new_category_count
        effects_list.append(
            categorical_intercept[category_indices[cov.name]],
        )

    def _model_all_mean_effects(
        self,
        train_data: pd.DataFrame,
        spline_bases: dict[str, npt.NDArray[np.floating[Any]]],
        category_indices: dict[str, npt.NDArray[np.integer[Any]]],
        adapt: dict[str, Any] | None = None,
    ) -> list[TensorVariable]:
        """
        Model all covariate mean effects.
        """
        mean_effects = []
        # Model the global intercept
        if adapt is None:  # Model fitting
            global_intercept = pm.Normal(
                "global_intercept",
                mu=0,
                sigma=5,
                dims=("scalar",),
            )
            # Increment parameter count for global intercept
            self.model_params["n_params"] += 1
        else:  # Freeze during adaptation/fine-tuning
            global_intercept = pm.Deterministic(
                "global_intercept",
                pt.as_tensor_variable(
                    adapt["pretrained_model_params"]["posterior_means"][
                        "global_intercept"
                    ],
                ),
                dims=("scalar",),
            )
        mean_effects.append(global_intercept)
        # Model additional covariate effects on the mean
        for cov in self.spec.covariates:
            if cov.name in self.spec.influencing_mean:
                if cov.cov_type == "numerical":
                    if cov.effect == "linear":
                        self._model_linear_mean_effect(
                            train_data,
                            cov,
                            mean_effects,
                            sigma_prior=5,
                            adapt=adapt,
                        )
                    elif cov.effect == "spline":
                        self._model_spline_mean_effect(
                            train_data,
                            cov,
                            mean_effects,
                            spline_bases,
                            sigma_prior=5,
                            adapt=adapt,
                        )
                elif cov.cov_type == "categorical":
                    self._model_categorical_mean_effect(
                        train_data,
                        cov,
                        mean_effects,
                        category_indices,
                        sigma_prior=1,
                        hierarchical_sigma_prior=5,
                        adapt=adapt,
                    )
                else:
                    err = f"Invalid covariate type '{cov.cov_type}' for '{cov.name}'."
                    raise ValueError(err)
        return mean_effects

    def _model_linear_variance_effect(
        self,
        train_data: pd.DataFrame,
        cov: CovariateSpec,
        effects_list: list[TensorVariable],
        sigma_prior: float = 0.1,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Model a linear effect for a numerical covariate on the variance estimate.
        """
        # Linear effect
        if adapt is None:  # Model fitting
            linear_beta = pm.Normal(
                f"variance_linear_beta_{cov.name}",
                mu=0,
                sigma=sigma_prior,
                size=1,
                dims=(f"{cov.name}_linear",),
            )
            # Increment parameter count for linear effect
            self.model_params["n_params"] += 1
        else:  # Freeze during adaptation/fine-tuning
            linear_beta = pm.Deterministic(
                f"variance_linear_beta_{cov.name}",
                pt.as_tensor_variable(
                    adapt["pretrained_model_params"]["posterior_means"][
                        f"variance_linear_beta_{cov.name}"
                    ],
                ),
                dims=(f"{cov.name}_linear",),
            )
        if cov.moments is not None:  # to satisfy type checker
            effects_list.append(
                (
                    cast("npt.NDArray[Any]", train_data[cov.name].to_numpy())
                    - cov.moments[0]
                )
                / cov.moments[1]
                * linear_beta,
            )

    def _model_spline_variance_effect(
        self,
        train_data: pd.DataFrame,
        cov: CovariateSpec,
        effects_list: list[TensorVariable],
        spline_bases: dict[str, npt.NDArray[np.floating[Any]]],
        sigma_prior: float = 0.1,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Model a spline effect for a numerical covariate on the variance estimate.
        """
        # Spline effect
        spline_bases[cov.name] = spline_bases.get(
            cov.name,
            cov.make_spline_bases(
                cast("npt.NDArray[Any]", train_data[cov.name].to_numpy()),
            ),
        )
        if adapt is None:  # Model fitting
            spline_betas = pm.ZeroSumNormal(
                f"variance_spline_betas_{cov.name}",
                sigma=sigma_prior,
                shape=spline_bases[cov.name].shape[1],
                dims=(f"{cov.name}_splines",),
            )
            # Note ZeroSumNormal imposes a centering constraint
            # (ensuring identifiability)
            # Increment parameter count for spline effects
            if cov.spline_spec is not None:  # to satisfy type checker
                self.model_params["n_params"] += cov.spline_spec.df - 1
        else:  # Freeze during adaptation/fine-tuning
            spline_betas = pm.Deterministic(
                f"variance_spline_betas_{cov.name}",
                pt.as_tensor_variable(
                    adapt["pretrained_model_params"]["posterior_means"][
                        f"variance_spline_betas_{cov.name}"
                    ],
                ),
                dims=(f"{cov.name}_splines",),
            )
        effects_list.append(pt.dot(spline_bases[cov.name], spline_betas.T))

    def _model_categorical_variance_effect(
        self,
        train_data: pd.DataFrame,
        cov: CovariateSpec,
        effects_list: list[TensorVariable],
        category_indices: dict[str, npt.NDArray[np.integer[Any]]],
        sigma_prior: float = 0.1,
        hierarchical_sigma_prior: float = 0.1,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Model the effect of a categorical covariate on the variance estimate.
        """
        # Factorize categories
        category_indices[cov.name] = category_indices.get(
            cov.name,
            cov.factorize_categories(
                cast("npt.NDArray[Any]", train_data[cov.name].to_numpy()),
            ),
        )
        if adapt is None:  # Model fitting
            if cov.hierarchical:
                # Hierarchical categorical effect
                # Hyperpriors for category (Bayesian equivalent of random effects)
                sigma_intercept_category = pm.HalfNormal(
                    f"variance_sigma_intercept_{cov.name}",
                    sigma=sigma_prior,
                    dims=("scalar",),
                )

                # Hierarchical intercepts for each category (using reparameterized form)
                categorical_intercept_offset = pm.ZeroSumNormal(
                    f"variance_intercept_offset_{cov.name}",
                    sigma=hierarchical_sigma_prior,
                    dims=(cov.name,),
                )
                # Note ZeroSumNormal imposes a centering constraint
                # (ensuring identifiability)
                categorical_intercept = pm.Deterministic(
                    f"variance_intercept_{cov.name}",
                    (
                        categorical_intercept_offset
                        * pt.reshape(sigma_intercept_category, (1,))
                    ),
                    dims=(cov.name,),
                )

                # Increment parameter count for hierarchical intercept
                self.model_params["n_params"] += 1

            else:
                # Non-hierarchical (linear) categorical effect
                categorical_intercept = pm.ZeroSumNormal(
                    f"variance_intercept_{cov.name}",
                    sigma=sigma_prior,
                    dims=(cov.name,),
                )
                # Note ZeroSumNormal imposes a centering constraint
                # (ensuring identifiability)
            # Increment parameter count for categorical effects
            if cov.categories is not None:  # to satisfy type checker
                self.model_params["n_params"] += len(cov.categories) - 1
        elif cov.name != adapt["covariate_to_adapt"]:
            # Freeze during adaptation/fine-tuning
            if cov.hierarchical:
                # Hierarchical categorical effect
                # Hyperpriors for category (Bayesian equivalent of random effects)
                sigma_intercept_category = pm.Deterministic(
                    f"variance_sigma_intercept_{cov.name}",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"variance_sigma_intercept_{cov.name}"
                        ],
                    ),
                    dims=("scalar",),
                )
                # Hierarchical intercepts for each category (using reparameterized form)
                categorical_intercept_offset = pm.Deterministic(
                    f"variance_intercept_offset_{cov.name}",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"variance_intercept_offset_{cov.name}"
                        ],
                    ),
                    dims=(cov.name,),
                )
                categorical_intercept = pm.Deterministic(
                    f"variance_intercept_{cov.name}",
                    (
                        categorical_intercept_offset
                        * pt.reshape(sigma_intercept_category, (1,))
                    ),
                    dims=(cov.name,),
                )
            else:
                categorical_intercept = pm.Deterministic(
                    f"variance_intercept_{cov.name}",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"variance_intercept_{cov.name}"
                        ],
                    ),
                    dims=(cov.name,),
                )
        else:  # Partial freezing (fit parameters for the new site only)
            if cov.hierarchical:
                # Hierarchical categorical effect
                # Hyperpriors for category (Bayesian equivalent of random effects)
                # Hyperpriors are fixed during adaptation
                sigma_intercept_category = pm.Deterministic(
                    f"variance_sigma_intercept_{cov.name}",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"variance_sigma_intercept_{cov.name}"
                        ],
                    ),
                    dims=("scalar",),
                )
                # Hierarchical intercepts for each category (using reparameterized form)
                # New categories get new parameters, old categories are fixed
                # Freeze old category parameters during adaptation
                fixed_categorical_intercept_offset = pm.Deterministic(
                    f"variance_intercept_offset_{cov.name}_fixed",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"variance_intercept_offset_{cov.name}"
                        ],
                    ),
                )
                # Create new parameters for new categories
                new_category_count = len(adapt["new_category_names"])
                pretrain_sigma_prior = adapt["pretrained_model_params"][
                    "posterior_means"
                ][f"variance_intercept_offset_{cov.name}"].std()
                new_categorical_intercept_offset = pm.Normal(
                    f"variance_intercept_offset_{cov.name}_adapt",
                    mu=0,
                    sigma=pretrain_sigma_prior,
                    size=new_category_count,
                )
                # Combine fixed and new offsets
                categorical_intercept_offset = pm.Deterministic(
                    f"variance_intercept_offset_{cov.name}",
                    pt.concatenate(
                        [
                            fixed_categorical_intercept_offset,
                            new_categorical_intercept_offset,
                        ],
                    ),
                    dims=(cov.name,),
                )
                categorical_intercept = pm.Deterministic(
                    f"variance_intercept_{cov.name}",
                    (
                        categorical_intercept_offset
                        * pt.reshape(sigma_intercept_category, (1,))
                    ),
                    dims=(cov.name,),
                )
            else:
                # Non-hierarchical (linear) categorical effect
                # New categories get new parameters, old categories are fixed
                # Freeze old category parameters during adaptation
                fixed_categorical_intercept = pm.Deterministic(
                    f"variance_intercept_{cov.name}_fixed",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"variance_intercept_{cov.name}"
                        ],
                    ),
                )
                # Create new parameters for new categories
                new_category_count = len(adapt["new_category_names"])
                new_categorical_intercept = pm.Normal(
                    f"variance_intercept_{cov.name}_adapt",
                    mu=0,
                    sigma=sigma_prior,
                    size=new_category_count,
                )
                # Combine fixed and new offsets
                categorical_intercept = pm.Deterministic(
                    f"variance_intercept_{cov.name}",
                    pt.concatenate(
                        [
                            fixed_categorical_intercept,
                            new_categorical_intercept,
                        ],
                    ),
                    dims=(cov.name,),
                )
            self.model_params["n_params"] += new_category_count
        effects_list.append(
            categorical_intercept[category_indices[cov.name]],
        )

    def _model_all_variance_effects(
        self,
        train_data: pd.DataFrame,
        spline_bases: dict[str, npt.NDArray[np.floating[Any]]],
        category_indices: dict[str, npt.NDArray[np.integer[Any]]],
        adapt: dict[str, Any] | None = None,
    ) -> list[TensorVariable]:
        """
        Model all covariate variance effects.
        """
        variance_effects = []
        # Model the global variance
        if adapt is None:
            global_variance_baseline = pm.Normal(
                "global_variance_baseline",
                mu=-0.0,
                sigma=0.5,
                dims=("scalar",),
            )
            # Increment parameter count for global variance
            self.model_params["n_params"] += 1
        else:
            global_variance_baseline = pm.Deterministic(
                "global_variance_baseline",
                pt.as_tensor_variable(
                    adapt["pretrained_model_params"]["posterior_means"][
                        "global_variance_baseline"
                    ],
                ),
                dims=("scalar",),
            )
        variance_effects.append(global_variance_baseline)
        # Model additional covariate effects on the variance
        for cov in self.spec.covariates:
            if cov.name in self.spec.influencing_variance:
                if cov.cov_type == "numerical":
                    if cov.effect == "linear":
                        self._model_linear_variance_effect(
                            train_data,
                            cov,
                            variance_effects,
                            sigma_prior=0.1,
                            adapt=adapt,
                        )
                    elif cov.effect == "spline":
                        self._model_spline_variance_effect(
                            train_data,
                            cov,
                            variance_effects,
                            spline_bases,
                            sigma_prior=0.1,
                            adapt=adapt,
                        )
                elif cov.cov_type == "categorical":
                    self._model_categorical_variance_effect(
                        train_data,
                        cov,
                        variance_effects,
                        category_indices,
                        sigma_prior=0.1,
                        hierarchical_sigma_prior=0.1,
                        adapt=adapt,
                    )
                else:
                    err = f"Invalid covariate type '{cov.cov_type}' for '{cov.name}'."
                    raise ValueError(err)
        return variance_effects

    def _combine_all_effects(
        self,
        mean_effects: list[TensorVariable],
        variance_effects: list[TensorVariable],
        standardized_voi: npt.NDArray[np.floating[Any]],
    ) -> None:
        """
        Combine all effects to model the observed data likelihood.
        """
        # Combine all mean and variance effects
        mu_estimate = sum(mean_effects)
        log_sigma_estimate = sum(variance_effects)
        sigma_estimate = pt.exp(log_sigma_estimate)

        effective_sample_size = self.model_params["sample_size"]

        # Model likelihood of the variable of interest
        _likelihood = pm.Normal(
            f"likelihood_{self.spec.variable_of_interest}",
            mu=mu_estimate,
            sigma=sigma_estimate,
            observed=standardized_voi,
            total_size=effective_sample_size,
        )

    def _fit_model_with_advi(self, *, progress_bar: bool = True) -> None:
        """
        Fit the model using Automatic Differentiation Variational Inference (ADVI).
        """
        base_lr = self.defaults["adam_learning_rate"]
        decay = self.defaults["adam_learning_rate_decay"]
        lr = shared(base_lr)
        optimizer = pm.adam(learning_rate=cast("float", lr))

        # Adaptive learning rate schedule callback
        def update_learning_rate(_approx: Any, _loss: Any, iteration: int) -> None:
            lr.set_value(base_lr * (decay**iteration))

        # Run automatic differential variational inference to fit the model
        self._trace = pm.fit(
            method="advi",
            n=self.defaults["advi_iterations"],
            random_seed=self.defaults["random_seed"],  # For reproducibility
            obj_optimizer=optimizer,
            callbacks=[
                update_learning_rate,
                pm.callbacks.CheckParametersConvergence(
                    tolerance=self.defaults["advi_convergence_tolerance"],
                    diff="relative",
                ),
            ],
            progressbar=progress_bar,
        )

        # Sample from the posterior distribution and store the results
        self.model_inference_data = self._trace.sample(
            2000,
            random_seed=self.defaults["random_seed"],
        )

        # Compute posterior means and standard deviations
        posterior_means = self.model_inference_data.posterior.mean(
            dim=("chain", "draw"),
        )
        posterior_stds = self.model_inference_data.posterior.std(dim=("chain", "draw"))

        # Store posterior means and stds as a dictionary in model parameters
        self.model_params["posterior_means"] = {
            x: posterior_means.data_vars[x].to_numpy()
            for x in posterior_means.data_vars
        }
        self.model_params["posterior_stds"] = {
            x: posterior_stds.data_vars[x].to_numpy() for x in posterior_stds.data_vars
        }

    def fit(
        self,
        train_data: pd.DataFrame,
        *,
        save_directory: Path | None = None,
        progress_bar: bool = True,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Fit the normative model to the training data.

        This method implements the fitting logic for the normative model
        based on the provided training data and model specification.

        Args:
            train_data: pd.DataFrame
                DataFrame containing the training data. It must include the variable
                of interest and all specified covariates.
            save_directory: Path | None
                A path to a directory to save the model. If provided, the fitted model
                will be saved to this path.
            progress_bar: bool
                If True, display a progress bar during fitting. Defaults to True.
            adapt: dict[str, Any] | None
                If provided, adapt a pre-trained model to a new covariate.
                Note: We recommended using the `adapt_fit` method, and not directly
                changing this argument, unless you know what you are doing.
        """
        # Validation checks
        self._validate_model()
        self._validate_dataframe_for_fitting(train_data)

        # Extract the variable of interest
        variable_of_interest = train_data[self.spec.variable_of_interest].to_numpy()

        # A dictionary to hold the model parameters after fitting
        if adapt is None:
            self.model_params = {}
            self.model_params["mean_VOI"] = variable_of_interest.mean()
            self.model_params["std_VOI"] = variable_of_interest.std()
            self.model_params["sample_size"] = variable_of_interest.shape[0]
            # Initialize parameter count
            self.model_params["n_params"] = 0
        else:
            # Update the pretrained model parameters
            if not hasattr(self, "model_params") or self.model_params is None:
                self.model_params = copy.deepcopy(adapt["pretrained_model_params"])
            self.model_params["sample_size"] += variable_of_interest.shape[0]

        # Data preparation
        model_coords = self._build_model_coordinates(
            observations=np.arange(train_data.shape[0]),
        )

        # Fitting logic
        with pm.Model(coords=model_coords) as self._model:
            # Standardize the variable of interest, and store mean and std
            # This ensures that the model is not sensitive to the scale of the variable
            standardized_voi = (
                variable_of_interest - self.model_params["mean_VOI"]
            ) / self.model_params["std_VOI"]

            # A dictionary for precomputed bspline basis functions
            spline_bases: dict[str, npt.NDArray[np.floating[Any]]] = {}

            # A dictionary for factorized categories
            category_indices: dict[str, npt.NDArray[np.integer[Any]]] = {}

            # Model the mean of the variable of interest
            mean_effects = self._model_all_mean_effects(
                train_data,
                spline_bases,
                category_indices,
                adapt=adapt,
            )

            # Model the variance of the variable of interest
            variance_effects = self._model_all_variance_effects(
                train_data,
                spline_bases,
                category_indices,
                adapt=adapt,
            )

            # Combine all mean and variance effects
            self._combine_all_effects(
                mean_effects,
                variance_effects,
                standardized_voi,
            )

            # Fit the model using ADVI
            self._fit_model_with_advi(progress_bar=progress_bar)

        # Save the model if a save path is provided
        if save_directory is not None:
            self.save_model(Path(save_directory))

    def adapt_fit(
        self,
        covariate_to_adapt: str,
        new_category_names: npt.NDArray[np.str_],
        train_data: pd.DataFrame,
        *,
        pretrained_model_params: dict[str, Any] | None = None,
        save_directory: Path | None = None,
        progress_bar: bool = True,
    ) -> None:
        """
        Using a previously fitted model, adapt the model to a new batch.
        This method enables adaptation of the model to data from a new
        batch/site by freezing all fitted parameters, and only estimating
        new parameters for the new batch/site category.

        Args:
            covariate_to_adapt: str
                Name of the categorical covariate representing the batch/site
                to which the model should be adapted.
                Note: This covariate must have been specified in the original
                model.
            new_category_names: list[str]
                Names of the new categories in the covariate_to_adapt representing
                the new batch/site labels (e.g. names of the new site).
                Note: These names must not have been present in the original
                fitted model.
            train_data: pd.DataFrame
                DataFrame containing the training data for adaptation.
                It must include the variable of interest and all specified covariates.
                Note: The covariate_to_adapt column must only contain the
                new_category_names (no new data from previously trained batches).
            pretrained_model_params: dict[str, Any] | None
                The model parameters from a previously fitted model to adapt.
                If None, the model parameters from the current instance will be used
                (assuming fitting was done).
            save_directory: Path | None
                A path to a directory to save the adapted model. If provided,
                the fitted model will be saved to this path.
            progress_bar: bool
                If True, display a progress bar during fitting. Defaults to True.
        """
        # Validation checks
        self._validate_model()
        self._validate_dataframe_for_fitting(train_data)

        # Locate the covariate to adapt
        cov_to_adapt_index = [cov.name for cov in self.spec.covariates].index(
            covariate_to_adapt,
        )

        # Extend the covariate categories to include the new categories
        self.spec.covariates[cov_to_adapt_index].extend_categories(new_category_names)

        # Extract the pre-trained model parameters
        if pretrained_model_params is None:
            if not self.model_params:
                err = (
                    "No pretrained model parameters found. "
                    "Please provide pretrained_model_params or fit the model first."
                )
                raise ValueError(err)
            pretrained_model_params = copy.deepcopy(self.model_params)

        # Fit the adapted model
        self.fit(
            train_data,
            save_directory=save_directory,
            progress_bar=progress_bar,
            adapt={
                "covariate_to_adapt": covariate_to_adapt,
                "new_category_names": new_category_names,
                "pretrained_model_params": pretrained_model_params,
            },
        )

    def _predict_mu(
        self,
        test_covariates: pd.DataFrame,
        model_params: dict[str, Any],
        predict_without: list[str],
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Internal method to predict the mean of the variable of interest.
        """
        # Calculate mean effect
        mu_estimate = np.full(
            test_covariates.shape[0],
            model_params["posterior_means"]["global_intercept"].item(),
        )

        for cov in self.spec.covariates:
            if (cov.name in self.spec.influencing_mean) and (
                cov.name not in predict_without
            ):
                if cov.cov_type == "numerical":
                    if cov.effect == "linear":
                        if cov.moments is not None:  # to satisfy type checker
                            mu_estimate += (
                                (
                                    cast(
                                        "npt.NDArray[Any]",
                                        test_covariates[cov.name].to_numpy(),
                                    )
                                    - cov.moments[0]
                                )
                                / cov.moments[1]
                            ) * model_params["posterior_means"][
                                f"linear_beta_{cov.name}"
                            ]
                    elif cov.effect == "spline":
                        spline_bases = cov.make_spline_bases(
                            cast(
                                "npt.NDArray[Any]",
                                test_covariates[cov.name].to_numpy(),
                            ),
                        )
                        spline_betas = model_params["posterior_means"][
                            f"spline_betas_{cov.name}"
                        ]
                        mu_estimate += np.dot(spline_bases, spline_betas)
                elif cov.cov_type == "categorical":
                    category_indices = cov.factorize_categories(
                        cast("npt.NDArray[Any]", test_covariates[cov.name].to_numpy()),
                    )
                    if cov.hierarchical:
                        categorical_intercept = (
                            model_params["posterior_means"][
                                f"intercept_offset_{cov.name}"
                            ]
                            * model_params["posterior_means"][
                                f"sigma_intercept_{cov.name}"
                            ]
                        )
                    else:
                        categorical_intercept = model_params["posterior_means"][
                            f"intercept_{cov.name}"
                        ]
                    mu_estimate += categorical_intercept[category_indices]

        return np.array(
            mu_estimate * model_params["std_VOI"] + model_params["mean_VOI"],
        )

    def _predict_std(
        self,
        test_covariates: pd.DataFrame,
        model_params: dict[str, Any],
        predict_without: list[str],
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Internal method to predict the standard deviation of the variable of interest.
        """
        # Calculate deviation effect
        log_sigma_estimate = np.full(
            test_covariates.shape[0],
            model_params["posterior_means"]["global_variance_baseline"].item(),
        )

        for cov in self.spec.covariates:
            if (
                cov.name in self.spec.influencing_variance
                and cov.name not in predict_without
            ):
                if cov.cov_type == "numerical":
                    if cov.effect == "linear":
                        if cov.moments is not None:  # to satisfy type checker
                            log_sigma_estimate += (
                                (
                                    cast(
                                        "npt.NDArray[Any]",
                                        test_covariates[cov.name].to_numpy(),
                                    )
                                    - cov.moments[0]
                                )
                                / cov.moments[1]
                            ) * model_params["posterior_means"][
                                f"variance_linear_beta_{cov.name}"
                            ]
                    elif cov.effect == "spline":
                        spline_bases = cov.make_spline_bases(
                            cast(
                                "npt.NDArray[Any]",
                                test_covariates[cov.name].to_numpy(),
                            ),
                        )
                        variance_spline_betas = model_params["posterior_means"][
                            f"variance_spline_betas_{cov.name}"
                        ]
                        log_sigma_estimate += spline_bases @ variance_spline_betas
                elif cov.cov_type == "categorical":
                    category_indices = cov.factorize_categories(
                        cast("npt.NDArray[Any]", test_covariates[cov.name].to_numpy()),
                    )
                    if cov.hierarchical:
                        categorical_variance_intercept = (
                            model_params["posterior_means"][
                                f"variance_intercept_offset_{cov.name}"
                            ]
                            * model_params["posterior_means"][
                                f"variance_sigma_intercept_{cov.name}"
                            ]
                        )
                    else:
                        categorical_variance_intercept = model_params[
                            "posterior_means"
                        ][f"variance_intercept_{cov.name}"]
                    log_sigma_estimate += categorical_variance_intercept[
                        category_indices
                    ]

        return np.array(np.exp(log_sigma_estimate) * model_params["std_VOI"])

    def predict(
        self,
        test_covariates: pd.DataFrame,
        *,
        extended: bool = False,
        model_params: dict[str, Any] | None = None,
        predict_without: list[str] | None = None,
    ) -> NormativePredictions:
        """
        Predict normative moments (mean, std) for new data using the fitted model.

        Args:
            test_covariates: pd.DataFrame
                DataFrame containing the new covariate data to predict.
                This must include all specified covariates.
                Note: covariates listed in predict_without will be ignored and are
                hence not required.
            extended: bool
                If True, return additional stats such as log-likelihood, centiles, etc.
                Note that extended predictions require variable_of_interest to be
                provided in the test_covariates DataFrame.
            model_params: dict | None
                Optional dictionary of model parameters to use. If not provided,
                the stored parameters from model.fit() will be used.
            predict_without: list[str] | None
                Optional list of covariate names to ignore during prediction.
                This can be used to check the effect of removing certain covariates
                from the model.

        Returns:
            NormativePredictions: Object containing the predicted moments (mean, std)
                for the variable of interest.
        """
        # Validate the new data
        validation_columns = [
            cov.name
            for cov in self.spec.covariates
            if cov.name not in (predict_without or [])
        ]
        if extended:
            validation_columns.append(self.spec.variable_of_interest)
        utils.general.validate_dataframe(test_covariates, validation_columns)

        # Parameters
        model_params = model_params or self.model_params
        if model_params is None:
            err = "No model parameters found. Please provide model_params."
            raise ValueError(err)

        # Calculate mean and variance effects and store in the predictions object
        predictions = NormativePredictions(
            {
                "mu_estimate": self._predict_mu(
                    test_covariates,
                    model_params,
                    (predict_without or []),
                ),
                "std_estimate": self._predict_std(
                    test_covariates,
                    model_params,
                    (predict_without or []),
                ),
            },
        )

        # Check if extended predictions are requested
        if extended:
            # Add extended statistics to predictions (e.g. centiles, log loss, etc.)
            predictions.extend_predictions(
                variable_of_interest=(
                    cast(
                        "npt.NDArray[Any]",
                        test_covariates[self.spec.variable_of_interest].to_numpy(),
                    )
                ),
            )

        return predictions

    def evaluate(self, new_data: pd.DataFrame) -> NormativePredictions:
        """
        Evaluate the model on new data and return predictions.

        Args:
            new_data: pd.DataFrame
                DataFrame containing the new data to evaluate.
                It must include all specified covariates and the variable of interest.

        Returns:
            NormativePredictions: Object containing the predictions and evaluation
            metrics.
        """
        # Run extended predictions
        return self.predict(test_covariates=new_data).evaluate_predictions(
            variable_of_interest=(
                cast(
                    "npt.NDArray[Any]",
                    new_data[self.spec.variable_of_interest].to_numpy(),
                )
            ),
            train_mean=self.model_params["mean_VOI"],
            train_std=self.model_params["std_VOI"],
            n_params=self.model_params["n_params"],
        )

    def harmonize(
        self,
        data: pd.DataFrame,
        covariates_to_harmonize: list[str],
        *,
        model_params: dict[str, Any] | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Harmonize the variable of interest in the data to remove effects of
        certain covariates (e.g. batch).

        Args:
            data: pd.DataFrame
                DataFrame containing the data to harmonize.
                It must include all specified covariates and the variable of interest.
            covariates_to_harmonize: list[str]
                List of covariate names to harmonize.
                The partial effects of these covariates will be removed from the
                variable of interest, and the harmonized values will be returned.
            model_params: dict | None
                Optional dictionary of model parameters to use. If not provided,
                the stored parameters from model.fit() will be used.

        Returns:
            npt.NDArray[np.floating[Any]]: Array of harmonized values for the
                variable of interest.
        """
        # Validate the new data
        validation_columns = [cov.name for cov in self.spec.covariates]
        validation_columns.append(self.spec.variable_of_interest)
        utils.general.validate_dataframe(data, validation_columns)

        # Parameters
        if model_params is None:
            model_params = self.model_params

        # Predict the mean and std with all covariates
        full_predictions = self.predict(
            test_covariates=data,
            model_params=model_params,
            predict_without=[],
        )

        # Predict the mean and std without the covariates to harmonize
        reduced_predictions = self.predict(
            test_covariates=data,
            model_params=model_params,
            predict_without=covariates_to_harmonize,
        )

        # First standardize the variable of interest based on the full model
        voi_standardized = (
            (cast("npt.NDArray[Any]", data[self.spec.variable_of_interest].to_numpy()))
            - full_predictions.predictions["mu_estimate"]
        ) / full_predictions.predictions["std_estimate"]

        # Then return the harmonized values based on the reduced model
        return np.asarray(
            (
                voi_standardized * reduced_predictions.predictions["std_estimate"]
                + reduced_predictions.predictions["mu_estimate"]
            ),
            dtype=np.float64,
        )


@dataclass
class CovarianceNormativeModel:
    """
    Covariance normative model implementation.

    This class implements covariance normative modeling, which models the covariance
    structure between a pair of variables as a normative random variable.

    Attributes:
        spec: CovarianceModelSpec
            Specification of the covariance model including variables of interest,
            and list of covariates.
        defaults: dict
            Default parameters for the model, including spline specifications,
            ADVI iterations, convergence tolerance, random seed, and Adam optimizer
            learning rates.
    """

    spec: CovarianceModelSpec
    defaults: dict[str, Any] = field(
        default_factory=lambda: {
            "spline_df": DEFAULT_SPLINE_DF,
            "spline_degree": DEFAULT_SPLINE_DEGREE,
            "spline_extrapolation_factor": DEFAULT_SPLINE_EXTRAPOLATION_FACTOR,
            "advi_iterations": DEFAULT_ADVI_ITERATIONS,
            "advi_convergence_tolerance": DEFAULT_ADVI_CONVERGENCE_TOLERANCE,
            "random_seed": DEFAULT_RANDOM_SEED,
            "adam_learning_rate": DEFAULT_ADAM_LEARNING_RATE,
            "adam_learning_rate_decay": DEFAULT_ADAM_LEARNING_RATE_DECAY,
        },
    )

    def __repr__(self) -> str:
        """
        String representation of the CovarianceNormativeModel instance.
        """
        return f"CovarianceNormativeModel(\n\tspec={self.spec}\n)"

    @classmethod
    def from_direct_model(
        cls,
        direct_model: DirectNormativeModel,
        variable_of_interest_1: str,
        variable_of_interest_2: str,
        influencing_covariance: list[str] | None = None,
        defaults_overwrite: dict[str, Any] | None = None,
    ) -> CovarianceNormativeModel:
        """
        Initialize the model from a direct model instance, and two variable names.

        Args:
            direct_model: DirectNormativeModel
                This model will be used to instantiate a similar covariance model.
            variable_of_interest_1: str
                Name of the first target variable to model.
            variable_of_interest_2: str
                Name of the second target variable to model.
            influencing_covariance: list[str] | None
                List of covariates that influence the covariance structure. If not
                provided, this will be copied from the direct model's
                `influencing_variance`.

        Returns:
            CovarianceNormativeModel
                An instance of CovarianceNormativeModel initialized with the provided
                data.
        """
        # Validity checks for input parameters
        if not isinstance(direct_model, DirectNormativeModel):
            err = "direct_model must be an instance of DirectNormativeModel."
            raise TypeError(err)
        if not (
            isinstance(variable_of_interest_1, str)
            and isinstance(variable_of_interest_2, str)
        ):
            err = "Variables of interest must be strings."
            raise TypeError(err)

        # Substitute influencing_covariance if not provided
        if influencing_covariance is None:
            influencing_covariance = direct_model.spec.influencing_variance

        # Use the same setup as the direct model
        model = cls(
            spec=CovarianceModelSpec(
                variable_of_interest_1=variable_of_interest_1,
                variable_of_interest_2=variable_of_interest_2,
                covariates=direct_model.spec.covariates,
                influencing_covariance=influencing_covariance,
            ),
        )

        # update defaults
        model.defaults.update(direct_model.defaults)
        model.defaults.update(defaults_overwrite or {})

        return model

    def _validate_model(self) -> None:
        """
        Validate the covariance model instance.

        This method checks if the model instance is complete and valid.
        It raises errors if any required fields are missing or if there are
        inconsistencies in the model instance.
        """
        if self.spec is None:
            err = (
                "Model specification is not set. Please initialize the model,"
                " e.g., with 'from_dataframe'."
            )
            raise ValueError(err)
        if len(self.spec.covariates) == 0:
            err = (
                "No covariates specified in the model. "
                "Please add covariates to the specification."
            )
            raise ValueError(err)
        if len(self.spec.influencing_covariance) == 0:
            err = (
                "No covariates specified to influence the covariance "
                "between the variables of interest."
            )
            raise ValueError(err)

    def save_model(self, directory: Path, *, save_posterior: bool = False) -> None:
        """
        Save the fitted model and it's posterior to a directory.
        The model will be saved in a subdirectory named 'saved_model'.
        If this directory is not empty, an error is raised.

        Args:
            directory: Path
                Path to a directory to save the model.
            save_posterior: bool (default=False)
                If True, save the model's posterior trace inference data.
        """
        # Prepare the save directory
        directory = Path(directory)
        saved_model_dir = utils.general.prepare_save_directory(directory, "saved_model")

        model_dict = {
            "spec": self.spec,
            "defaults": self.defaults,
        }
        if hasattr(self, "model_params"):
            model_dict["model_params"] = self.model_params
            if hasattr(self, "model_inference_data") and save_posterior:
                self.model_inference_data.to_netcdf(
                    saved_model_dir / "model_inference_data.nc",
                )
        joblib.dump(model_dict, saved_model_dir / "model_dict.joblib")

    @classmethod
    def load_model(
        cls,
        directory: Path,
        *,
        load_posterior: bool = False,
    ) -> CovarianceNormativeModel:
        """
        Load the model and its posterior from a directory.
        The model will be loaded from a subdirectory named 'saved_model'.

        Args:
            directory: Path
                Path to the directory containing the model.
            load_posterior: bool (default=False)
                If True, load the model's posterior trace from the saved inference data.
        """
        # Validate the load directory
        directory = Path(directory)
        saved_model_dir = utils.general.validate_load_directory(
            directory,
            "saved_model",
        )

        # Load the saved model dict
        model_dict = joblib.load(saved_model_dir / "model_dict.joblib")

        # Create an instance of the class
        instance = cls(
            spec=model_dict["spec"],
        )

        # Set the attributes from the loaded model dictionary
        instance.defaults.update(model_dict["defaults"])
        if "model_params" in model_dict:
            instance.model_params = model_dict["model_params"]
            if load_posterior:
                instance.model_inference_data = az.from_netcdf(  # type: ignore[no-untyped-call]
                    saved_model_dir / "model_inference_data.nc",
                )

        return instance

    def _validate_dataframe_for_fitting(self, train_data: pd.DataFrame) -> None:
        """
        Validate the training DataFrame for fitting.
        """
        utils.general.validate_dataframe(
            train_data,
            (
                [cov.name for cov in self.spec.covariates]
                + [
                    self.spec.variable_of_interest_1,
                    self.spec.variable_of_interest_2,
                    f"{self.spec.variable_of_interest_1}_mu_estimate",
                    f"{self.spec.variable_of_interest_2}_mu_estimate",
                    f"{self.spec.variable_of_interest_1}_std_estimate",
                    f"{self.spec.variable_of_interest_2}_std_estimate",
                ]
            ),
        )

    def _build_model_coordinates(
        self,
        observations: npt.NDArray[np.integer[Any]],
    ) -> dict[str, Any]:
        """
        Build the model coordinates for the training DataFrame.
        """
        # Data coordinates
        model_coords = {"observations": observations, "scalar": [0]}

        # Additional coordinates for covariates
        for cov in self.spec.covariates:
            if cov.cov_type == "numerical":
                if cov.effect == "spline":
                    if cov.spline_spec is not None:  # to satisfy type checker
                        model_coords[f"{cov.name}_splines"] = np.arange(
                            cov.spline_spec.df,
                        )
                elif cov.effect == "linear":
                    model_coords[f"{cov.name}_linear"] = np.arange(1)
            elif cov.cov_type == "categorical":
                model_coords[cov.name] = cov.categories
            else:
                err = f"Invalid covariate type '{cov.cov_type}' for '{cov.name}'."
                raise ValueError(err)
        return model_coords

    def _model_linear_correlation_effect(
        self,
        train_data: pd.DataFrame,
        cov: CovariateSpec,
        effects_list: list[TensorVariable],
        sigma_prior: float = 0.1,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Model a linear effect for a numerical covariate.
        """
        # Linear effect
        if adapt is None:  # Model fitting
            linear_beta = pm.Normal(
                f"linear_beta_{cov.name}",
                mu=0,
                sigma=sigma_prior,
                size=1,
                dims=(f"{cov.name}_linear",),
            )
            # Increment parameter count for linear effect
            self.model_params["n_params"] += 1
        else:  # Freeze during adaptation/fine-tuning
            linear_beta = pm.Deterministic(
                f"linear_beta_{cov.name}",
                pt.as_tensor_variable(
                    adapt["pretrained_model_params"]["posterior_means"][
                        f"linear_beta_{cov.name}"
                    ],
                ),
                dims=(f"{cov.name}_linear",),
            )
        if cov.moments is not None:  # to satisfy type checker
            effects_list.append(
                (
                    (
                        (cast("npt.NDArray[Any]", train_data[cov.name].to_numpy()))
                        - cov.moments[0]
                    )
                    / cov.moments[1]
                )
                * linear_beta,
            )

    def _model_spline_correlation_effect(
        self,
        train_data: pd.DataFrame,
        cov: CovariateSpec,
        effects_list: list[TensorVariable],
        spline_bases: dict[str, npt.NDArray[np.floating[Any]]],
        sigma_prior: float = 1,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Model a spline effect for a numerical covariate.
        """
        # Spline effect
        spline_bases[cov.name] = spline_bases.get(
            cov.name,
            cov.make_spline_bases(
                cast("npt.NDArray[Any]", train_data[cov.name].to_numpy()),
            ),
        )
        if adapt is None:  # Model fitting
            spline_betas = pm.ZeroSumNormal(
                f"spline_betas_{cov.name}",
                sigma=sigma_prior,
                shape=spline_bases[cov.name].shape[1],
                dims=(f"{cov.name}_splines",),
            )
            # Note ZeroSumNormal imposes a centering constraint
            # (ensuring identifiability)
            # Increment parameter count for spline effects
            if cov.spline_spec is not None:  # to satisfy type checker
                self.model_params["n_params"] += cov.spline_spec.df - 1
        else:  # Freeze during adaptation/fine-tuning
            spline_betas = pm.Deterministic(
                f"spline_betas_{cov.name}",
                pt.as_tensor_variable(
                    adapt["pretrained_model_params"]["posterior_means"][
                        f"spline_betas_{cov.name}"
                    ],
                ),
                dims=(f"{cov.name}_splines",),
            )
        effects_list.append(pt.dot(spline_bases[cov.name], spline_betas.T))

    def _model_categorical_correlation_effect(
        self,
        train_data: pd.DataFrame,
        cov: CovariateSpec,
        effects_list: list[TensorVariable],
        category_indices: dict[str, npt.NDArray[np.integer[Any]]],
        sigma_prior: float = 1,
        hierarchical_sigma_prior: float = 0.1,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Model the effect of a categorical covariate.
        """
        # Factorize categories
        category_indices[cov.name] = category_indices.get(
            cov.name,
            cov.factorize_categories(
                cast("npt.NDArray[Any]", train_data[cov.name].to_numpy()),
            ),
        )
        if adapt is None:  # Model fitting
            if cov.hierarchical:
                # Hierarchical categorical effect
                # Hyperpriors for category (Bayesian equivalent of random effects)
                sigma_intercept_category = pm.HalfNormal(
                    f"sigma_intercept_{cov.name}",
                    sigma=sigma_prior,
                    dims=("scalar",),
                )

                # Hierarchical intercepts for each category (using reparameterized form)
                categorical_intercept_offset = pm.ZeroSumNormal(
                    f"intercept_offset_{cov.name}",
                    sigma=hierarchical_sigma_prior,
                    dims=(cov.name,),
                )
                # Note ZeroSumNormal imposes a centering constraint
                # (ensuring identifiability)
                categorical_intercept = pm.Deterministic(
                    f"intercept_{cov.name}",
                    (
                        categorical_intercept_offset
                        * pt.reshape(sigma_intercept_category, (1,))
                    ),
                    dims=(cov.name,),
                )

                # Increment parameter count for hierarchical intercept
                self.model_params["n_params"] += 1

            else:
                # Non-hierarchical (linear) categorical effect
                categorical_intercept = pm.ZeroSumNormal(
                    f"intercept_{cov.name}",
                    sigma=sigma_prior,
                    dims=(cov.name,),
                )
                # Note ZeroSumNormal imposes a centering constraint
                # (ensuring identifiability)
            # Increment parameter count for categorical effects
            if cov.categories is not None:  # to satisfy type checker
                self.model_params["n_params"] += len(cov.categories) - 1
        elif cov.name != adapt["covariate_to_adapt"]:
            # Freeze during adaptation/fine-tuning
            if cov.hierarchical:
                # Hierarchical categorical effect
                # Hyperpriors for category (Bayesian equivalent of random effects)
                sigma_intercept_category = pm.Deterministic(
                    f"sigma_intercept_{cov.name}",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"sigma_intercept_{cov.name}"
                        ],
                    ),
                    dims=("scalar",),
                )
                # Hierarchical intercepts for each category (using reparameterized form)
                categorical_intercept_offset = pm.Deterministic(
                    f"intercept_offset_{cov.name}",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"intercept_offset_{cov.name}"
                        ],
                    ),
                    dims=(cov.name,),
                )
                categorical_intercept = pm.Deterministic(
                    f"intercept_{cov.name}",
                    (
                        categorical_intercept_offset
                        * pt.reshape(sigma_intercept_category, (1,))
                    ),
                    dims=(cov.name,),
                )
            else:
                # Non-hierarchical (linear) categorical effect
                categorical_intercept = pm.Deterministic(
                    f"intercept_{cov.name}",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"intercept_{cov.name}"
                        ],
                    ),
                    dims=(cov.name,),
                )
        else:
            if cov.hierarchical:
                # Hierarchical categorical effect
                # Hyperpriors for category (Bayesian equivalent of random effects)
                # Hyperpriors are fixed during adaptation
                sigma_intercept_category = pm.Deterministic(
                    f"sigma_intercept_{cov.name}",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"sigma_intercept_{cov.name}"
                        ],
                    ),
                    dims=("scalar",),
                )
                # Hierarchical intercepts for each category (using reparameterized form)
                # New categories get new parameters, old categories are fixed
                # Freeze old category parameters during adaptation
                fixed_categorical_intercept_offset = pm.Deterministic(
                    f"intercept_offset_{cov.name}_fixed",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"intercept_offset_{cov.name}"
                        ],
                    ),
                )
                # Create new parameters for new categories
                new_category_count = len(adapt["new_category_names"])
                new_categorical_intercept_offset = pm.Normal(
                    f"intercept_offset_{cov.name}_adapt",
                    mu=0,
                    sigma=hierarchical_sigma_prior,
                    size=new_category_count,
                )
                # Combine fixed and new offsets
                categorical_intercept_offset = pm.Deterministic(
                    f"intercept_offset_{cov.name}",
                    pt.concatenate(
                        [
                            fixed_categorical_intercept_offset,
                            new_categorical_intercept_offset,
                        ],
                    ),
                    dims=(cov.name,),
                )
                categorical_intercept = pm.Deterministic(
                    f"intercept_{cov.name}",
                    (
                        categorical_intercept_offset
                        * pt.reshape(sigma_intercept_category, (1,))
                    ),
                    dims=(cov.name,),
                )
            else:
                # Non-hierarchical (linear) categorical effect
                # New categories get new parameters, old categories are fixed
                # Freeze old category parameters during adaptation
                fixed_categorical_intercept = pm.Deterministic(
                    f"intercept_{cov.name}_fixed",
                    pt.as_tensor_variable(
                        adapt["pretrained_model_params"]["posterior_means"][
                            f"intercept_{cov.name}"
                        ],
                    ),
                )
                # Create new parameters for new categories
                new_category_count = len(adapt["new_category_names"])
                new_categorical_intercept = pm.Normal(
                    f"intercept_{cov.name}_adapt",
                    mu=0,
                    sigma=sigma_prior,
                    size=new_category_count,
                )
                # Combine fixed and new intercepts
                categorical_intercept = pm.Deterministic(
                    f"intercept_{cov.name}",
                    pt.concatenate(
                        [
                            fixed_categorical_intercept,
                            new_categorical_intercept,
                        ],
                    ),
                    dims=(cov.name,),
                )
            self.model_params["n_params"] += new_category_count
        effects_list.append(
            categorical_intercept[category_indices[cov.name]],
        )

    def _model_all_correlation_effects(
        self,
        train_data: pd.DataFrame,
        spline_bases: dict[str, npt.NDArray[np.floating[Any]]],
        category_indices: dict[str, npt.NDArray[np.integer[Any]]],
        adapt: dict[str, Any] | None = None,
    ) -> list[TensorVariable]:
        """
        Model all covariate correlation effects.
        """
        # Create a list to contain the effects of covariates on
        # the z-transformed correlation
        z_transformed_correlation_effects = []

        # Model the z-transformed correlation between the variables of interest
        # Model the global intercept for z
        if adapt is None:
            global_intercept_z = pm.Normal(
                "global_intercept_z",
                mu=0,
                sigma=5,
                dims=("scalar",),
            )
            # Increment parameter count for global intercept
            self.model_params["n_params"] += 1
        else:
            # Use the pretrained global intercept
            global_intercept_z = pm.Deterministic(
                "global_intercept_z",
                pt.as_tensor_variable(
                    adapt["pretrained_model_params"]["posterior_means"][
                        "global_intercept_z"
                    ],
                ),
                dims=("scalar",),
            )
        z_transformed_correlation_effects.append(global_intercept_z)
        # Model additional covariate effects on the z estimate
        for cov in self.spec.covariates:
            if cov.name in self.spec.influencing_covariance:
                if cov.cov_type == "numerical":
                    if cov.effect == "linear":
                        self._model_linear_correlation_effect(
                            train_data=train_data,
                            cov=cov,
                            effects_list=z_transformed_correlation_effects,
                            adapt=adapt,
                        )
                    elif cov.effect == "spline":
                        self._model_spline_correlation_effect(
                            train_data=train_data,
                            cov=cov,
                            effects_list=z_transformed_correlation_effects,
                            spline_bases=spline_bases,
                            adapt=adapt,
                        )
                elif cov.cov_type == "categorical":
                    self._model_categorical_correlation_effect(
                        train_data=train_data,
                        cov=cov,
                        effects_list=z_transformed_correlation_effects,
                        category_indices=category_indices,
                        adapt=adapt,
                    )

                else:
                    err = f"Invalid covariate type '{cov.cov_type}' for '{cov.name}'."
                    raise ValueError(err)
        return z_transformed_correlation_effects

    def _combine_all_correlation_effects(
        self,
        z_transformed_correlation_effects: list[TensorVariable],
        combination_indices: npt.NDArray[np.integer[Any]],
        combination_weights: npt.NDArray[np.floating[Any]],
        standardized_vois: npt.NDArray[np.floating[Any]],
        standardized_vois_mu_estimate: npt.NDArray[np.floating[Any]],
        standardized_vois_std_estimate: npt.NDArray[np.floating[Any]],
    ) -> None:
        """
        Combine all effects to model the observed data likelihood from the list of
        correlation effects.
        """
        # Combine all covariance effects
        z_transformed_correlation_estimate = sum(z_transformed_correlation_effects)

        # Convert z-transformed score to correlation
        correlation_estimate = pt.tanh(z_transformed_correlation_estimate)

        # Now apply the random combinations to get final distribution estimates
        # Apply combination weights for mu estimate
        combined_mu_estimate = pt.sum(
            pt.mul(
                standardized_vois_mu_estimate[combination_indices, :],
                combination_weights,
            ),
            axis=1,
        )
        # Apply combination weights for sigma estimate
        combined_sigma_estimate = pt.mul(
            standardized_vois_std_estimate[combination_indices, :],
            combination_weights,
        )
        # Apply combination weights for correlation estimate
        combined_correlation_estimate = correlation_estimate[combination_indices]  # pyright: ignore[reportOptionalSubscript]
        # Now build the std estimate
        combined_std_estimate = pt.sqrt(
            combined_sigma_estimate[:, 0] ** 2  # pyright: ignore[reportOptionalSubscript]
            + combined_sigma_estimate[:, 1] ** 2  # pyright: ignore[reportOptionalSubscript]
            + (
                2
                * combined_sigma_estimate[:, 0]  # pyright: ignore[reportOptionalSubscript]
                * combined_sigma_estimate[:, 1]  # pyright: ignore[reportOptionalSubscript]
                * combined_correlation_estimate
            ),
        )

        # Apply combination to the variables of interest
        combined_variable_of_interest = pt.sum(
            pt.mul(
                standardized_vois[combination_indices, :],
                combination_weights,
            ),
            axis=1,
        )

        effective_sample_size = self.model_params["sample_size"]

        # Model likelihood estimation for covariance model
        _likelihood = pm.Normal(
            f"likelihood_cov_{self.spec.variable_of_interest_1}_{self.spec.variable_of_interest_2}",
            mu=combined_mu_estimate,
            sigma=combined_std_estimate,
            observed=combined_variable_of_interest,
            total_size=(effective_sample_size),
        )

    def _fit_model_with_advi(self, *, progress_bar: bool = True) -> None:
        """
        Fit the model using Automatic Differentiation Variational Inference (ADVI).
        """
        base_lr = self.defaults["adam_learning_rate"]
        decay = self.defaults["adam_learning_rate_decay"]
        lr = shared(base_lr)
        optimizer = pm.adam(learning_rate=cast("float", lr))

        # Adaptive learning rate schedule callback
        def update_learning_rate(_approx: Any, _loss: Any, iteration: int) -> None:
            lr.set_value(base_lr * (decay**iteration))

        # Run automatic differential variational inference to fit the model
        self._trace = pm.fit(
            method="advi",
            n=self.defaults["advi_iterations"],
            random_seed=self.defaults["random_seed"],  # For reproducibility
            obj_optimizer=optimizer,
            callbacks=[
                update_learning_rate,
                pm.callbacks.CheckParametersConvergence(
                    tolerance=self.defaults["advi_convergence_tolerance"],
                    diff="relative",
                ),
            ],
            progressbar=progress_bar,
        )

        # Sample from the posterior distribution and store the results
        self.model_inference_data = self._trace.sample(
            2000,
            random_seed=self.defaults["random_seed"],
        )

        # Compute posterior means and standard deviations
        posterior_means = self.model_inference_data.posterior.mean(
            dim=("chain", "draw"),
        )
        posterior_stds = self.model_inference_data.posterior.std(
            dim=("chain", "draw"),
        )

        # Store posterior means and stds as a dictionary in model parameters
        self.model_params["posterior_means"] = {
            x: posterior_means.data_vars[x].to_numpy()
            for x in posterior_means.data_vars
        }
        self.model_params["posterior_stds"] = {
            x: posterior_stds.data_vars[x].to_numpy() for x in posterior_stds.data_vars
        }

    def fit(
        self,
        train_data: pd.DataFrame,
        *,
        save_directory: Path | None = None,
        progress_bar: bool = True,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Fit the normative model to the training data.

        This method implements the fitting logic for the normative model
        based on the provided training data and model specification.

        Args:
            train_data: pd.DataFrame
                DataFrame containing the training data. It must include the variable
                of interest, their predicted moments, and all specified covariates.
            save_directory: Path | None
                A path to a directory to save the model. If provided, the fitted model
                will be saved to this path.
            progress_bar: bool
                If True, display a progress bar during fitting. Defaults to True.
            adapt: dict[str, Any] | None
                If provided, adapt a pre-trained model to a new covariate.
                Note: We recommended using the `adapt_fit` method, and not directly
                changing this argument, unless you know what you are doing.
        """
        # Validation checks
        self._validate_model()
        self._validate_dataframe_for_fitting(train_data)

        # Extract the variables of interest
        variables_of_interest = train_data[
            [self.spec.variable_of_interest_1, self.spec.variable_of_interest_2]
        ].to_numpy()

        # A dictionary to hold the model parameters after fitting
        if adapt is None:
            self.model_params = {}
            self.model_params["mean_vois"] = variables_of_interest.mean(axis=0)
            self.model_params["std_vois"] = variables_of_interest.std(axis=0)
            self.model_params["sample_size"] = variables_of_interest.shape[0]
            # Initialize parameter count
            self.model_params["n_params"] = 0
        else:
            # Update the pretrained model parameters
            if not hasattr(self, "model_params") or self.model_params is None:
                self.model_params = copy.deepcopy(adapt["pretrained_model_params"])
            self.model_params["sample_size"] += variables_of_interest.shape[0]

        # Data preparation
        # Combination weights
        combination_weights = np.ones(shape=(train_data.shape[0], 2))

        # Data coordinates
        combination_indices = np.arange(train_data.shape[0])
        model_coords = self._build_model_coordinates(
            observations=combination_indices,
        )

        # Fitting logic
        with pm.Model(coords=model_coords) as self._model:
            # Standardize the variable of interest, and store mean and std
            # This is done to ensure that the model is not sensitive to
            # the scale of the variable
            standardized_vois = (
                variables_of_interest - self.model_params["mean_vois"]
            ) / self.model_params["std_vois"]
            variables_of_interest_mu_estimate = train_data[
                [
                    f"{self.spec.variable_of_interest_1}_mu_estimate",
                    f"{self.spec.variable_of_interest_2}_mu_estimate",
                ]
            ].to_numpy()
            standardized_vois_mu_estimate = (
                variables_of_interest_mu_estimate - self.model_params["mean_vois"]
            ) / self.model_params["std_vois"]
            variables_of_interest_std_estimate = train_data[
                [
                    f"{self.spec.variable_of_interest_1}_std_estimate",
                    f"{self.spec.variable_of_interest_2}_std_estimate",
                ]
            ].to_numpy()
            standardized_vois_std_estimate = (
                variables_of_interest_std_estimate / self.model_params["std_vois"]
            )

            # A dictionary for precomputed bspline basis functions
            spline_bases: dict[str, npt.NDArray[np.floating[Any]]] = {}

            # A dictionary for factorized categories
            category_indices: dict[str, npt.NDArray[np.integer[Any]]] = {}

            # Model the covariance between the variables of interest
            z_transformed_correlation_effects = self._model_all_correlation_effects(
                train_data,
                spline_bases,
                category_indices,
                adapt=adapt,
            )

            # Combine all covariance effects
            self._combine_all_correlation_effects(
                z_transformed_correlation_effects=z_transformed_correlation_effects,
                combination_indices=combination_indices,
                combination_weights=combination_weights,
                standardized_vois=standardized_vois,
                standardized_vois_mu_estimate=standardized_vois_mu_estimate,
                standardized_vois_std_estimate=standardized_vois_std_estimate,
            )

            # Fit the model using ADVI
            self._fit_model_with_advi(progress_bar=progress_bar)

        # Save the model if a save path is provided
        if save_directory is not None:
            self.save_model(Path(save_directory))

    def adapt_fit(
        self,
        covariate_to_adapt: str,
        new_category_names: npt.NDArray[np.str_],
        train_data: pd.DataFrame,
        *,
        pretrained_model_params: dict[str, Any] | None = None,
        save_directory: Path | None = None,
        progress_bar: bool = True,
    ) -> None:
        """
        Using a previously fitted model, adapt the model to a new batch.
        This method enables adaptation of the model to data from a new
        batch/site by freezing all fitted parameters, and only estimating
        new parameters for the new batch/site category.

        Args:
            covariate_to_adapt: str
                Name of the categorical covariate representing the batch/site
                to which the model should be adapted.
                Note: This covariate must have been specified in the original
                model.
            new_category_names: list[str]
                Names of the new categories in the covariate_to_adapt representing
                the new batch/site labels (e.g. names of the new site).
                Note: These names must not have been present in the original
                fitted model.
            train_data: pd.DataFrame
                DataFrame containing the training data for adaptation.
                It must include the variable of interest and all specified covariates.
                Note: The covariate_to_adapt column must only contain the
                new_category_names (no new data from previously trained batches).
            pretrained_model_params: dict[str, Any] | None
                The model parameters from a previously fitted model to adapt.
                If None, the model parameters from the current instance will be used
                (assuming fitting was done).
            save_directory: Path | None
                A path to a directory to save the adapted model. If provided,
                the fitted model will be saved to this path.
            progress_bar: bool
                If True, display a progress bar during fitting. Defaults to True.
        """
        # Validation checks
        self._validate_model()
        self._validate_dataframe_for_fitting(train_data)

        # Locate the covariate to adapt
        cov_to_adapt_index = [cov.name for cov in self.spec.covariates].index(
            covariate_to_adapt,
        )

        # Extend the covariate categories to include the new categories
        self.spec.covariates[cov_to_adapt_index].extend_categories(new_category_names)

        # Extract the pre-trained model parameters
        if pretrained_model_params is None:
            if not hasattr(self, "model_params") or self.model_params is None:
                err = (
                    "No pretrained model parameters found. "
                    "Please provide pretrained_model_params or fit the model first."
                )
                raise ValueError(err)
            pretrained_model_params = copy.deepcopy(self.model_params)

        # Fit the adapted model
        self.fit(
            train_data,
            save_directory=save_directory,
            progress_bar=progress_bar,
            adapt={
                "covariate_to_adapt": covariate_to_adapt,
                "new_category_names": new_category_names,
                "pretrained_model_params": pretrained_model_params,
            },
        )

    def predict(
        self,
        test_covariates: pd.DataFrame,
        model_params: dict[str, Any] | None = None,
        predict_without: list[str] | None = None,
    ) -> NormativePredictions:
        """
        Predict correlation for new data (from covariates) using the fitted model.

        Args:
            test_covariates: pd.DataFrame
                DataFrame containing the new covariate data to predict.
                This must include all specified covariates.
                Note: covariates listed in predict_without will be ignored and are
                hence not required.
            model_params: dict | None
                Optional dictionary of model parameters to use. If not provided,
                the stored parameters from model.fit() will be used.
            predict_without: list[str] | None
                Optional list of covariate names to ignore during prediction.
                This can be used to check the effect of removing certain covariates
                from the model.

        Returns:
            NormativePredictions: Object containing the predicted pairwise correlations
                for the variables of interest.
        """
        # Validate the new data
        validation_columns = [
            cov.name
            for cov in self.spec.covariates
            if cov.name not in (predict_without or [])
        ]
        utils.general.validate_dataframe(test_covariates, validation_columns)

        # Parameters
        model_params = model_params or self.model_params
        if model_params is None:
            err = "No model parameters found. Please provide model_params."
            raise ValueError(err)

        # Posterior means
        posterior_means = model_params["posterior_means"]

        # Calculate mean and variance effects
        z_transformed_correlation_estimate = np.zeros(test_covariates.shape[0]) + float(
            posterior_means["global_intercept_z"],
        )

        for cov in self.spec.covariates:
            if (cov.name in self.spec.influencing_covariance) and (
                cov.name not in (predict_without or [])
            ):
                if cov.cov_type == "numerical":
                    if cov.effect == "linear":
                        if cov.moments is None:
                            err = (
                                f"Covariate '{cov.name}' is missing moments for"
                                " standardization."
                            )
                            raise ValueError(err)
                        z_transformed_correlation_estimate += (
                            (
                                cast(
                                    "npt.NDArray[Any]",
                                    test_covariates[cov.name].to_numpy(),
                                )
                                - cov.moments[0]
                            )
                            / cov.moments[1]
                        ) * posterior_means[f"linear_beta_{cov.name}"]
                    elif cov.effect == "spline":
                        spline_bases = cov.make_spline_bases(
                            cast(
                                "npt.NDArray[Any]",
                                test_covariates[cov.name].to_numpy(),
                            ),
                        )
                        spline_betas = posterior_means[f"spline_betas_{cov.name}"]
                        z_transformed_correlation_estimate += (
                            spline_bases @ spline_betas
                        )
                elif cov.cov_type == "categorical":
                    category_indices = cov.factorize_categories(
                        cast("npt.NDArray[Any]", test_covariates[cov.name].to_numpy()),
                    )
                    categorical_intercept = None
                    if cov.hierarchical:
                        categorical_intercept = (
                            posterior_means[f"intercept_offset_{cov.name}"]
                            * posterior_means[f"sigma_intercept_{cov.name}"]
                        )
                    else:
                        categorical_intercept = posterior_means[f"intercept_{cov.name}"]
                    z_transformed_correlation_estimate += categorical_intercept[
                        category_indices
                    ]

        # Convert z-transformed score to correlation
        correlation_estimate = np.tanh(z_transformed_correlation_estimate)

        # Create a the predictions object and return
        return NormativePredictions({"correlation_estimate": correlation_estimate})


@dataclass
class SpectralNormativeModel:
    """
    Spectral normative model implementation.

    This class implements the spectral normative modeling approach, which
    utilizes a base direct model to generalize normative modeling to any
    arbitrary variable of interest reconstructed from a graph spectral
    embedding. It can be used to fit a normative model to high-dimensional
    data and predict normative centiles for arbitrary variables of interest.

    Attributes:
        eigenmode_basis: utils.gsp.EigenmodeBasis
            The eigenmode basis used for spectral normative modeling. This should be an
            instance of utils.gsp.EigenmodeBasis.
        base_model: DirectNormativeModel
            The base (direct) normative model used for spectral normative modeling.
    """

    eigenmode_basis: utils.gsp.EigenmodeBasis
    base_model: DirectNormativeModel

    @classmethod
    def build_from_dataframe(
        cls,
        eigenmode_basis: utils.gsp.EigenmodeBasis,
        model_type: ModelType,
        covariates_dataframe: pd.DataFrame,
        numerical_covariates: list[str] | None = None,
        categorical_covariates: list[str] | None = None,
        batch_covariates: list[str] | None = None,
        nonlinear_covariates: list[str] | None = None,
        influencing_mean: list[str] | None = None,
        influencing_variance: list[str] | None = None,
        spline_kwargs: dict[str, Any] | None = None,
    ) -> SpectralNormativeModel:
        """
        Initialize SNM with an eigenmode basis and a base direct model built from a
        pandas DataFrame containing all covariates.

        This uses the from_dataframe method of the DirectNormativeModel class
        to populate the direct model specification of SNM. Given that SNM does not
        require a fixed variable of interest, this method assigns a dummy name
        to the variable_of_interest parameter of the DirectNormativeModel. As such,
        the provided dataframe should not contain a column with "dummy_VOI" as name.

        Essentially, the provided dataframe should contain all covariates as columns.

        Args:
            eigenmode_basis: utils.gsp.EigenmodeBasis
                The eigenmode basis to be used for spectral normative modeling.
            model_type: ModelType
                Type of the model to create, either "HBR" (Hierarchical Bayesian
                Regression) or "BLR" (Bayesian Linear Regression).
            covariates_dataframe: pd.DataFrame
                DataFrame containing the data for all covariates and all samples.
            numerical_covariates: list[str] | None
                List of numerical covariate names.
            categorical_covariates: list[str] | None
                List of categorical covariate names.
            batch_covariates: list[str] | None
                List of batch covariate names which should also be included in
                categorical_covariates.
            nonlinear_covariates: list[str] | None
                List of covariate names to be modeled as nonlinear effects.
                These should also be included in numerical_covariates.
            influencing_mean: list[str] | None
                List of covariate names that influence the mean of the variable
                of interest. These should be included in either numerical_covariates
                or categorical_covariates.
            influencing_variance: list[str] | None
                List of covariate names that influence the variance of the variable
                of interest. These should be included in either numerical_covariates
                or categorical_covariates.
            spline_kwargs: dict
                Additional keyword arguments for spline specification, such as
                `df`, `degree`, and `knots`. These are passed to the
                `create_spline_spec` method to create spline specifications for
                nonlinear covariates.

        Returns:
            SpectralNormativeModel
                An instance of SpectralNormativeModel with base model specs initialized
                based on the provided data.
        """
        # Add a dummy variable of interest to the covariates_dataframe
        covariates_dataframe = covariates_dataframe.copy()
        covariates_dataframe["dummy_VOI"] = 0.0  # Dummy variable of interest
        # Specify the base model from the dataframe
        return cls(
            eigenmode_basis=eigenmode_basis,
            base_model=DirectNormativeModel.from_dataframe(
                model_type=model_type,
                dataframe=covariates_dataframe,
                variable_of_interest="dummy_VOI",  # Dummy variable of interest
                numerical_covariates=(numerical_covariates or []),
                categorical_covariates=(categorical_covariates or []),
                batch_covariates=(batch_covariates or []),
                nonlinear_covariates=(nonlinear_covariates or []),
                influencing_mean=(influencing_mean or []),
                influencing_variance=(influencing_variance or []),
                spline_kwargs=(spline_kwargs or {}),
            ),
        )

    def save_model(self, directory: Path) -> None:
        """
        Save the fitted spectral normative model to the specified directory.

        Args:
            directory: Path
                Directory to save the fitted model. A subdirectory named
                "spectral_normative_model" will be created within this directory.
        """
        # Prepare the save directory
        directory = Path(directory)
        saved_model_dir = utils.general.prepare_save_directory(
            directory,
            "spectral_normative_model",
        )

        # Save the eigenmode basis separately
        self.eigenmode_basis.save(str(saved_model_dir / "eigenmode_basis.joblib"))

        # Save the model
        model_dict = {
            "spec": self.base_model.spec,
            "defaults": self.base_model.defaults,
        }
        if hasattr(self, "model_params"):
            model_dict["model_params"] = self.model_params
        joblib.dump(model_dict, saved_model_dir / "spectral_model_dict.joblib")

    @classmethod
    def load_model(
        cls,
        directory: Path,
        mmap_mode: MmapMode | None = "r",
    ) -> SpectralNormativeModel:
        """
        Load a spectral normative model instance from the specified save directory.

        Args:
            directory: Path
                Directory to load the fitted model from. A subdirectory named
                "spectral_normative_model" will be searched within this directory.
            mmap_mode: MmapMode | None
                Memory mapping mode for joblib (default: "r").
                You can set this to None to disable memory-mapping.
        """
        # Validate the load directory
        directory = Path(directory)
        saved_model_dir = utils.general.validate_load_directory(
            directory,
            "spectral_normative_model",
        )

        # Check if the pickled joblib file exists in this directory
        for filename in ["spectral_model_dict.joblib", "eigenmode_basis.joblib"]:
            if not (saved_model_dir / filename).exists():
                err = f"Model Load Error: Required file '{filename}' does not exist."
                raise FileNotFoundError(err)

        # Load the pickled model dictionary
        model_dict = joblib.load(saved_model_dir / "spectral_model_dict.joblib")

        # Load the eigenmode basis
        eigenmode_basis = utils.gsp.EigenmodeBasis.load(
            str(saved_model_dir / "eigenmode_basis.joblib"),
            mmap_mode=mmap_mode,
        )

        # Create an instance of the class
        instance = cls(
            eigenmode_basis=eigenmode_basis,
            base_model=DirectNormativeModel(
                spec=model_dict["spec"],
                defaults=model_dict["defaults"],
            ),
        )

        if "model_params" in model_dict:
            instance.model_params = model_dict["model_params"]

        return instance

    def _validate_fit_input(
        self,
        encoded_train_data: npt.NDArray[np.floating[Any]],
        n_modes: int,
    ) -> None:
        """
        Internal method to validate input data for fitting the spectral normative model.
        """
        # Validate the input data
        if not isinstance(encoded_train_data, np.ndarray):
            err = "encoded_train_data must be a numpy array."
            raise TypeError(err)
        if encoded_train_data.shape[1] < n_modes:
            err = f"encoded_train_data must have at least {n_modes} columns (n_modes)."
            raise ValueError(err)
        if self.eigenmode_basis.n_modes < n_modes:
            err = (
                f"Eigenmode basis has only {self.eigenmode_basis.n_modes}"
                f" modes, while {n_modes} were requested."
            )
            raise ValueError(err)

    def identify_sparse_covariance_structure(
        self,
        data: npt.NDArray[np.floating[Any]],
        sparsity_threshold: float = 1,
    ) -> npt.NDArray[np.integer[Any]]:
        """
        Identify the sparse cross-basis covariance structure in the phenotype.
        This method analyzes the encoded phenotype to determine the covariance
        pairs that need to be modeled.

        Note: if the batches become too small, this estimate can become less stable
        in which case it is recommended to provide the sparse covariance structure
        to the model instead.

        Args:
            data: np.ndarray
                The encoded training data representing the phenotype in the graph
                frequency domain.
            sparsity_threshold: float
                Number of strongest correlations to keep (proportional to the number
                of modes). Defaults to 1, meaning that the number of sparse covariance
                pairs will be equal to the number of modes. If set to a lower value,
                fewer covariance pairs will be retained.

        Returns:
            np.ndarray:
                A (N, 2) array: the rows and columns of the
                identified sparse covariance structure.
        """
        # Start with correlation structure across the whole sample
        correlations = np.corrcoef(data.T)

        # Remove self-correlations
        np.fill_diagonal(correlations, 0)

        # Extract the upper triangle of the correlation matrix
        upper_triangle_indices = np.triu_indices(correlations.shape[0], k=1)

        # Determine the number of correlations to keep
        n_correlations_to_keep = int(
            sparsity_threshold * correlations.shape[0],
        )

        # Find the cutoff value for the top correlations
        if n_correlations_to_keep < len(upper_triangle_indices[0]):
            cutoff_value = np.partition(
                np.abs(correlations[upper_triangle_indices]),
                -n_correlations_to_keep,
            )[-n_correlations_to_keep]
        else:
            cutoff_value = 0
            # Warn the user if they are keeping all correlations
            logger.warning(
                "Sparsity threshold is high, keeping all correlations.",
            )

        # Now compute the sparsity structure based on the resulting matrix
        rows, cols = np.where(np.abs(correlations) > cutoff_value)

        # Remove redundant and duplicate pairs
        rows_lim = rows[rows < cols]
        cols_lim = cols[rows < cols]

        return np.array([rows_lim, cols_lim]).T

    @staticmethod
    def _is_valid_covariance_structure(
        covariance_structure: npt.NDArray[np.integer[Any]] | float,
    ) -> bool:
        """
        Verify the validity of the sparse covariance structure.
        """
        # Check it's a 2D array with two columns
        expected_ndims = 2
        expected_ncols = 2
        if not (
            isinstance(covariance_structure, np.ndarray)
            and covariance_structure.ndim == expected_ndims
            and covariance_structure.shape[1] == expected_ncols
        ):
            return False
        return np.issubdtype(covariance_structure.dtype, np.integer)

    def fit_single_direct(
        self,
        variable_of_interest: npt.NDArray[np.floating[Any]],
        covariates_dataframe: pd.DataFrame,
        *,
        save_directory: Path | None = None,
        return_model_params: bool = True,
        adapt: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Fit a direct normative model for a single spectral eigenmode.
        This method fits the base direct model to the provided variable of interest
        and covariates dataframe, allowing for the model to be trained on a specific
        eigenmode of the spectral embedding.

        Args:
            variable_of_interest: np.ndarray
                The loading vector capturing the variance within training data that
                corresponds to a single eigenmode.
            covariates_dataframe: pd.DataFrame
                DataFrame containing the covariates for the samples.
            save_directory: Path | None
                Directory to save the fitted model. If None, the model is not saved.
            return_model_params: bool
                If True, return the fitted model parameters.
            adapt: dict[str, Any] | None
                Adaptation parameters from a previously fitted model. If provided,
                the model will be adapted using these parameters during fitting.

        Returns:
            dict:
                If `return_model_params` is True, return the fitted model parameters
                in a dictionary.
        """
        # Prepare the data for fitting
        train_data = covariates_dataframe.copy()
        # Add the mode loading as the variable of interest
        train_data["VOI"] = variable_of_interest

        # Instantiate a direct normative model from the base model
        direct_model = DirectNormativeModel(
            spec=NormativeModelSpec(
                variable_of_interest="VOI",  # Use the added VOI column
                covariates=self.base_model.spec.covariates,
                influencing_mean=self.base_model.spec.influencing_mean,
                influencing_variance=self.base_model.spec.influencing_variance,
            ),
            defaults=self.base_model.defaults,
        )

        # Fit the model silently
        with utils.general.suppress_output():
            direct_model.fit(
                train_data=train_data,
                save_directory=save_directory,
                progress_bar=False,
                adapt=adapt,
            )

        # Return the fitted model parameters if requested
        if return_model_params:
            return direct_model.model_params

        # If not returning model parameters, return None
        return None

    def fit_single_covariance(
        self,
        variable_of_interest_1: npt.NDArray[np.floating[Any]],
        variable_of_interest_2: npt.NDArray[np.floating[Any]],
        direct_model_params_1: dict[str, Any],
        direct_model_params_2: dict[str, Any],
        covariates_dataframe: pd.DataFrame,
        *,
        save_directory: Path | None = None,
        return_model_params: bool = True,
        defaults_overwrite: dict[str, Any] | None = None,
        adapt: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Fit a covariance normative model between a single pair of eigenmodes.
        This method fits a covariance model to the provided pair of variables
        and covariates dataframe, considering the direct model fits for each
        eigenmode, while allowing for the cross-eigenmode covariance to vary
        normatively.

        Args:
            variable_of_interest_1: np.ndarray
                The loading vector capturing the variance within training data that
                corresponds to a single eigenmode.
            variable_of_interest_2: np.ndarray
                The loading vector capturing the variance within training data that
                corresponds to a second eigenmode.
            direct_model_params_1: dict
                The parameters of the direct model fitted to the first eigenmode.
            direct_model_params_2: dict
                The parameters of the direct model fitted to the second eigenmode.
            covariates_dataframe: pd.DataFrame
                DataFrame containing the covariates for the samples.
            save_directory: Path | None
                Directory to save the fitted model. If None, the model is not saved.
            return_model_params: bool
                If True, return the fitted model parameters.
            defaults_overwrite: dict (default={})
                Dictionary of default values to overwrite in the model fitting process.
            adapt: dict[str, Any] | None = None
                Adaptation parameters from a previously fitted model. If provided,
                the model will be adapted using these parameters during fitting.

        Returns:
            dict:
                If `return_model_params` is True, return the fitted model parameters
                in a dictionary.
        """
        # Prepare the data for fitting
        train_data = covariates_dataframe.copy()
        # Add the respective mode loadings as the variables of interest
        train_data["VOI_1"] = variable_of_interest_1
        train_data["VOI_2"] = variable_of_interest_2
        train_data[["VOI_1_mu_estimate", "VOI_1_std_estimate"]] = (
            self.base_model.predict(
                train_data,
                model_params=direct_model_params_1,
            )
            .to_array()
            .T
        )  # Add the direct model predictions
        train_data[["VOI_2_mu_estimate", "VOI_2_std_estimate"]] = (
            self.base_model.predict(
                train_data,
                model_params=direct_model_params_2,
            )
            .to_array()
            .T
        )  # Add the direct model predictions

        # Instantiate a covariance normative model from the base model
        covariance_model = CovarianceNormativeModel.from_direct_model(
            self.base_model,
            variable_of_interest_1="VOI_1",
            variable_of_interest_2="VOI_2",
            defaults_overwrite=(defaults_overwrite or {}),
        )

        # Fit the model silently
        with utils.general.suppress_output():
            covariance_model.fit(
                train_data=train_data,
                save_directory=save_directory,
                progress_bar=False,
                adapt=adapt,
            )

        # Return the fitted model parameters if requested
        if return_model_params:
            return covariance_model.model_params

        # If not returning model parameters, return None
        return None

    def fit_all_direct(
        self,
        encoded_train_data: npt.NDArray[np.floating[Any]],
        covariates_dataframe: pd.DataFrame,
        *,
        n_modes: int = -1,
        n_jobs: int = -1,
        save_directory: Path | None = None,
        save_separate: bool = False,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Fit the direct models for all specified eigenmodes.

        Args:
            encoded_train_data: np.ndarray
                Encoded training data as a numpy array (n_samples, n_modes).
            covariates_dataframe: pd.DataFrame
                DataFrame containing the covariates for the samples.
                It must include all specified covariates in the model specification.
            n_modes: int (default=-1)
                Number of eigenmodes to fit the model to. If -1, all modes are
                used. If a positive integer, only the first n_modes are used.
                Note that the encoded_train_data and the eigenmode basis should have
                at least n_modes columns/eigenvectors.
            n_jobs: int (default=-1)
                Number of parallel jobs to use for fitting the model. If -1, all
                available CPU cores are used. If 1, no parallelization is used.
            save_directory: Path | None
                Directory to save the fitted model. If None, the model is not saved.
                A subdirectory named "spectral_normative_model" will be created
                within the specified save_directory.
            save_separate: bool (default=False)
                Whether to save the fitted direct model parameters separately for each
                eigenmode as individual files. This is only applicable if
                `save_directory` is provided.
            adapt: dict[str, Any] | None
                Adaptation parameters from a previously fitted model. If provided,
                the model will be adapted using these parameters during fitting.
        """
        # Setup the save directory if needed
        if save_directory is not None:
            save_directory = Path(save_directory)

        # Evaluate the number of modes to fit
        if n_modes == -1:
            n_modes = self.eigenmode_basis.n_modes

        # Fit the base direct model for each eigenmode using parallel processing
        tasks = (
            joblib.delayed(self.fit_single_direct)(
                variable_of_interest=encoded_train_data[:, i],
                covariates_dataframe=covariates_dataframe,
                save_directory=(
                    utils.general.ensure_dir(
                        save_directory
                        / "spectral_normative_model"
                        / "direct_models"
                        / f"mode_{i + 1}",
                    )
                    if save_directory is not None and save_separate
                    else None
                ),
                adapt=(
                    None
                    if adapt is None
                    else {
                        "covariate_to_adapt": adapt["covariate_to_adapt"],
                        "new_category_names": adapt["new_category_names"],
                        "pretrained_model_params": adapt["pretrained_model_params"][
                            "direct_model_params"
                        ][i],
                    }
                ),
            )
            for i in range(n_modes)
        )
        self.direct_model_params = list(
            utils.parallel.ParallelTqdm(
                n_jobs=n_jobs,
                total_tasks=n_modes,
                desc="Fitting direct models",
            )(tasks),  # pyright: ignore[reportCallIssue]
        )

    def identify_covariance_structure(
        self,
        encoded_train_data: npt.NDArray[np.floating[Any]],
        covariates_dataframe: pd.DataFrame,
        n_modes: int,
        covariance_structure: npt.NDArray[np.floating[Any]] | float = 0.5,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Identify and set the sparse covariance structure for the spectral normative
        model based on the provided training data and covariance structure input.

        Args:
            encoded_train_data: np.ndarray
                Encoded training data as a numpy array (n_samples, n_modes).
            covariates_dataframe: pd.DataFrame
                DataFrame containing the covariates for the samples.
            n_modes: int
                Number of eigenmodes to consider.
            covariance_structure: np.ndarray | float
                Sparse covariance structure to use for the model fitting. If a
                (2, n_pairs) array of row and column indices are provided, the model
                will use this structure. If float, the model will estimate the
                covariance structure based on the training data and the float value
                will be used as the sparsity threshold for the number of covariance
                pairs to keep proportional to the number of modes. Defaults to 0.5,
                meaning that the number of modeled sparse covariance pairs will be
                half the number of modes.
            adapt: dict[str, Any] | None
                Adaptation parameters from a previously fitted model. If provided,
                the sparse covariance structure from the pretrained model parameters
                will be used instead of estimating a new one.
        """
        if adapt is not None:
            covariance_structure = adapt["pretrained_model_params"][
                "sparse_covariance_structure"
            ]

        # Identify sparse covariance structure if a float value is given
        if isinstance(covariance_structure, float):
            # Use trained models to compute z-scores
            encoded_train_z_scores = np.array(
                [
                    self.base_model.predict(
                        test_covariates=covariates_dataframe,
                        model_params=self.direct_model_params[x],
                    )
                    .extend_predictions(
                        variable_of_interest=encoded_train_data[:, x],
                    )
                    .predictions["z-score"]
                    for x in range(n_modes)
                ],
            ).T

            self.sparse_covariance_structure = (
                self.identify_sparse_covariance_structure(
                    encoded_train_z_scores,
                    covariance_structure,
                )
            )
        else:
            self.sparse_covariance_structure = np.array(covariance_structure)

        # Verify that the covariance structure is valid
        if not self._is_valid_covariance_structure(self.sparse_covariance_structure):
            err = "Invalid sparse covariance structure."
            raise ValueError(err)

    def fit_all_covariance(
        self,
        encoded_train_data: npt.NDArray[np.floating[Any]],
        covariates_dataframe: pd.DataFrame,
        *,
        n_jobs: int = -1,
        save_directory: Path | None = None,
        save_separate: bool = False,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Fit the direct models for all specified eigenmodes.

        Args:
            encoded_train_data: np.ndarray
                Encoded training data as a numpy array (n_samples, n_modes).
            covariates_dataframe: pd.DataFrame
                DataFrame containing the covariates for the samples.
                It must include all specified covariates in the model specification.
            n_jobs: int (default=-1)
                Number of parallel jobs to use for fitting the model. If -1, all
                available CPU cores are used. If 1, no parallelization is used.
            save_directory: Path | None
                Directory to save the fitted model. If None, the model is not saved.
                A subdirectory named "spectral_normative_model" will be created
                within the specified save_directory.
            save_separate: bool (default=False)
                Whether to save the fitted direct model parameters separately for each
                eigenmode as individual files. This is only applicable if
                `save_directory` is provided.
            adapt: dict[str, Any] | None
                Adaptation parameters from a previously fitted model. If provided,
                the model will be adapted using these parameters during fitting.
        """
        # Setup the save directory if needed
        if save_directory is not None:
            save_directory = Path(save_directory)

        # Fit the base covariance models for selected eigenmode pairs in parallel
        tasks = (
            joblib.delayed(self.fit_single_covariance)(
                variable_of_interest_1=encoded_train_data[
                    :,
                    self.sparse_covariance_structure[i, 0],
                ],
                variable_of_interest_2=encoded_train_data[
                    :,
                    self.sparse_covariance_structure[i, 1],
                ],
                direct_model_params_1=self.direct_model_params[
                    self.sparse_covariance_structure[i, 0]
                ],
                direct_model_params_2=self.direct_model_params[
                    self.sparse_covariance_structure[i, 1]
                ],
                covariates_dataframe=covariates_dataframe,
                save_directory=(
                    utils.general.ensure_dir(
                        save_directory
                        / "spectral_normative_model"
                        / "covariance_models"
                        / (
                            f"mode_{self.sparse_covariance_structure[i, 0] + 1},"
                            f"mode_{self.sparse_covariance_structure[i, 1] + 1}"
                        ),
                    )
                    if save_directory is not None and save_separate
                    else None
                ),
                adapt=(
                    None
                    if adapt is None
                    else {
                        "covariate_to_adapt": adapt["covariate_to_adapt"],
                        "new_category_names": adapt["new_category_names"],
                        "pretrained_model_params": adapt["pretrained_model_params"][
                            "covariance_model_params"
                        ][i],
                    }
                ),
            )
            for i in range(self.sparse_covariance_structure.shape[0])
        )
        self.covariance_model_params = utils.parallel.ParallelTqdm(
            n_jobs=n_jobs,
            total_tasks=self.sparse_covariance_structure.shape[0],
            desc="Fitting covariance models",
        )(tasks)  # pyright: ignore[reportCallIssue]

    def fit(
        self,
        encoded_train_data: npt.NDArray[np.floating[Any]],
        covariates_dataframe: pd.DataFrame,
        *,
        n_modes: int = -1,
        n_jobs: int = -1,
        save_directory: Path | None = None,
        save_separate: bool = False,
        covariance_structure: npt.NDArray[np.floating[Any]] | float = 0.5,
        adapt: dict[str, Any] | None = None,
    ) -> None:
        """
        Fit the spectral normative model to the provided encoded training data.

        Args:
            encoded_train_data: np.ndarray
                Encoded training data as a numpy array (n_samples, n_modes).
            covariates_dataframe: pd.DataFrame
                DataFrame containing the covariates for the samples.
                It must include all specified covariates in the model specification.
            n_modes: int (default=-1)
                Number of eigenmodes to fit the model to. If -1, all modes are
                used. If a positive integer, only the first n_modes are used.
                Note that the encoded_train_data and the eigenmode basis should have
                at least n_modes columns/eigenvectors.
            n_jobs: int (default=-1)
                Number of parallel jobs to use for fitting the model. If -1, all
                available CPU cores are used. If 1, no parallelization is used.
            save_directory: Path | None
                Directory to save the fitted model. If None, the model is not saved.
                A subdirectory named "spectral_normative_model" will be created
                within the specified save_directory.
            save_separate: bool (default=False)
                Whether to save the fitted direct model parameters separately for each
                eigenmode as individual files. This is only applicable if
                `save_directory` is provided.
            covariance_structure: np.ndarray | float
                Sparse covariance structure to use for the model fitting. If a
                (2, n_pairs) array of row and column indices are provided, the model
                will use this structure. If float, the model will estimate the
                covariance structure based on the training data and the float value
                will be used as the sparsity threshold for the number of covariance
                pairs to keep proportional to the number of modes. Defaults to 0.5,
                meaning that the number of modeled sparse covariance pairs will be
                half the number of modes.
            adapt: dict[str, Any] | None (default=None)
                If provided, adapt a pre-trained model to a new covariate.
                Note: We recommended using the `adapt_fit` method, and not directly
                changing this argument, unless you know what you are doing.
        """
        logger.info("Starting SNM model fitting:")
        # Evaluate the number of modes to fit
        if n_modes == -1:
            n_modes = self.eigenmode_basis.n_modes
        # Validate the input data
        if not isinstance(encoded_train_data, np.ndarray):
            err = "encoded_train_data must be a numpy array."
            raise TypeError(err)
        if encoded_train_data.shape[1] < n_modes:
            err = f"encoded_train_data must have at least {n_modes} columns (n_modes)."
            raise ValueError(err)
        if self.eigenmode_basis.n_modes < n_modes:
            err = (
                f"Eigenmode basis has only {self.eigenmode_basis.n_modes}"
                f" modes, while {n_modes} were requested."
            )
            raise ValueError(err)

        # Setup the save directory if needed
        if save_directory is not None:
            # Prepare the save directory
            save_directory = Path(save_directory)
            utils.general.prepare_save_directory(
                save_directory,
                "spectral_normative_model",
            )

        logger.info("Step 1; direct models for each eigenmode (%s modes)", n_modes)

        self.fit_all_direct(
            encoded_train_data=encoded_train_data,
            covariates_dataframe=covariates_dataframe,
            n_modes=n_modes,
            n_jobs=n_jobs,
            save_directory=save_directory,
            save_separate=save_separate,
            adapt=adapt,
        )

        logger.info("Step 2; identify sparse covariance structure")

        self.identify_covariance_structure(
            encoded_train_data=encoded_train_data,
            covariates_dataframe=covariates_dataframe,
            n_modes=n_modes,
            covariance_structure=covariance_structure,
            adapt=adapt,
        )

        # Verify that the covariance structure is valid
        if not self._is_valid_covariance_structure(self.sparse_covariance_structure):
            err = "Invalid sparse covariance structure."
            raise ValueError(err)

        # Model cross basis sparse covariance structure
        logger.info(
            "Step 3; cross-eigenmode dependency modeling (%s pairs)",
            self.sparse_covariance_structure.shape[0],
        )

        self.fit_all_covariance(
            encoded_train_data=encoded_train_data,
            covariates_dataframe=covariates_dataframe,
            n_jobs=n_jobs,
            save_directory=save_directory,
            save_separate=save_separate,
            adapt=adapt,
        )

        # Save SNM model parameters
        sample_size = encoded_train_data.shape[0]
        if adapt is not None:
            sample_size += adapt["pretrained_model_params"]["sample_size"]
        self.model_params = {
            "n_modes": n_modes,
            "sample_size": sample_size,
            "direct_model_params": self.direct_model_params,
            "sparse_covariance_structure": self.sparse_covariance_structure,
            "covariance_model_params": self.covariance_model_params,
        }
        if (self.direct_model_params[0] is not None) and (
            "n_params" in self.direct_model_params[0]
        ):
            self.model_params["n_params"] = self.direct_model_params[0]["n_params"]
        else:
            err = "Direct model parameters are not valid."
            raise ValueError(err)

        # Save the model if a save path is provided
        if save_directory is not None:
            self.save_model(save_directory)

    def adapt_fit(
        self,
        covariate_to_adapt: str,
        new_category_names: npt.NDArray[np.str_],
        encoded_train_data: npt.NDArray[np.floating[Any]],
        covariates_dataframe: pd.DataFrame,
        *,
        pretrained_model_params: dict[str, Any] | None = None,
        n_jobs: int = -1,
        save_directory: Path | None = None,
        save_separate: bool = False,
    ) -> None:
        """
        Using a previously fitted spectral normative model, adapt to a new
        batch.
        This method enables adaptation (fine-tuning) of the model to data
        from a new batch/site by freezing all fitted parameters, and only
        estimating new parameters for the new batch/site category.

        Args:
            covariate_to_adapt: str
                Name of the categorical covariate representing the batch/site
                to which the model should be adapted.
                Note: This covariate must have been specified in the original
                model.
            new_category_names: list[str]
                Names of the new categories in the covariate_to_adapt representing
                the new batch/site labels (e.g. names of the new site).
                Note: These names must not have been present in the original
                fitted model.
            encoded_train_data: np.ndarray
                Encoded training data as a numpy array (n_samples, n_modes).
            covariates_dataframe: pd.DataFrame
                DataFrame containing the covariates for the samples.
                It must include all specified covariates in the model specification.
                Note: The covariate_to_adapt column must only contain the
                new_category_names (no new data from previously trained batches).
            pretrained_model_params: dict[str, Any] | None
                The model parameters from a previously fitted model to adapt.
                If None, the model parameters from the current instance will be used
                (assuming fitting was done).
            n_jobs: int (default=-1)
                Number of parallel jobs to use for fitting the model. If -1, all
                available CPU cores are used. If 1, no parallelization is used.
            save_directory: Path | None
                A path to a directory to save the adapted model. If provided,
                the fitted model will be saved to this path.
            save_separate: bool (default=False)
                Whether to save the fitted direct model parameters separately for each
                eigenmode as individual files. This is only applicable if
                `save_directory` is provided.
        """
        # Locate the covariate to adapt
        cov_to_adapt_index = [
            cov.name for cov in self.base_model.spec.covariates
        ].index(covariate_to_adapt)

        # Extend the covariate categories to include the new categories
        self.base_model.spec.covariates[cov_to_adapt_index].extend_categories(
            new_category_names,
        )

        # Extract the pre-trained model parameters
        if pretrained_model_params is None:
            if not hasattr(self, "model_params") or self.model_params is None:
                err = (
                    "No pretrained model parameters found. "
                    "Please provide pretrained_model_params or fit the model first."
                )
                raise ValueError(err)
            pretrained_model_params = copy.deepcopy(self.model_params)

        # Fit the adapted model
        self.fit(
            encoded_train_data,
            covariates_dataframe,
            n_modes=pretrained_model_params["n_modes"],
            n_jobs=n_jobs,
            save_directory=save_directory,
            save_separate=save_separate,
            covariance_structure=pretrained_model_params["sparse_covariance_structure"],
            adapt={
                "covariate_to_adapt": covariate_to_adapt,
                "new_category_names": new_category_names,
                "pretrained_model_params": pretrained_model_params,
            },
        )

    @staticmethod
    def _compute_single_std_estimate_from_spectral_estimates(
        encoded_query: npt.NDArray[np.floating[Any]],
        eigenmode_std_estimates: npt.NDArray[np.floating[Any]],
        rho_estimates: npt.NDArray[np.floating[Any]],
        row_indices: npt.NDArray[np.integer[Any]],
        col_indices: npt.NDArray[np.integer[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Internal method to compute a standard deviation estimate for a single sample.
        """
        # Build sparse correlation matrix
        sparse_correlations = sparse.coo_matrix(
            (
                rho_estimates,  # sparse data values
                (row_indices, col_indices),  # row, column indices
            ),
            shape=(eigenmode_std_estimates.shape[0], eigenmode_std_estimates.shape[0]),
        ).tocsr()
        # Make it symmetric
        sparse_correlations = sparse_correlations + sparse_correlations.T
        # Set diagonal to 1
        sparse_correlations.setdiag(np.array(1))
        # Weight mode stds by encoding
        weighted_mode_stds = (
            np.asarray(eigenmode_std_estimates).reshape(-1, 1) * encoded_query
        )
        # Compute the variance estimate
        return np.asarray(
            np.sqrt(
                np.sum(
                    weighted_mode_stds * (sparse_correlations @ weighted_mode_stds),
                    axis=0,
                ),
            ),
        )

    def _predict_from_spectral_estimates(
        self,
        encoded_query: npt.NDArray[np.floating[Any]],
        eigenmode_mu_estimates: npt.NDArray[np.floating[Any]],
        eigenmode_std_estimates: npt.NDArray[np.floating[Any]],
        rho_estimates: npt.NDArray[np.floating[Any]],
        model_params: dict[str, Any],
        n_modes: int,
    ) -> NormativePredictions:
        """
        Internal method to predict only the mean and sigma for new data using the fitted
        spectral moments.
        """
        # Constrain mu and std estimates to the number of modes
        eigenmode_mu_estimates = eigenmode_mu_estimates[:, :n_modes]
        eigenmode_std_estimates = eigenmode_std_estimates[:, :n_modes]

        # Prepare the predictions
        predictions_dict = {}
        predictions_dict["mu_estimate"] = eigenmode_mu_estimates @ encoded_query

        # Load sparse covariance structure
        row_indices = model_params["sparse_covariance_structure"][:, 0]
        col_indices = model_params["sparse_covariance_structure"][:, 1]
        # Select indices that are both within n_modes
        corr_index_valid = (row_indices < n_modes) & (col_indices < n_modes)

        # Estimate query variance for each sample
        tasks = (
            joblib.delayed(self._compute_single_std_estimate_from_spectral_estimates)(
                encoded_query,
                eigenmode_std_estimates[sample_idx],
                rho_estimates[sample_idx, corr_index_valid],
                row_indices[corr_index_valid],
                col_indices[corr_index_valid],
            )
            for sample_idx in range(eigenmode_mu_estimates.shape[0])
        )
        results = list(
            utils.parallel.ParallelTqdm(
                n_jobs=-1,
                total_tasks=eigenmode_mu_estimates.shape[0],
                desc="Computing std estimates",
            )(tasks),  # pyright: ignore[reportCallIssue]
        )
        predictions_dict["std_estimate"] = np.array(results)

        # Create a the predictions object
        return NormativePredictions(predictions=predictions_dict)

    @staticmethod
    def _predict_single_mode_estimates(
        direct_model_spec: NormativeModelSpec,
        direct_model_defaults: dict[str, Any],
        test_covariates: pd.DataFrame,
        model_params: dict[str, Any],
        predict_without: list[str] | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Internal method to predict single mode estimates for new data using the fitted
        spectral normative model.
        """
        # Instantiate a direct normative model from the base model
        direct_model = DirectNormativeModel(
            spec=NormativeModelSpec(
                variable_of_interest="VOI",  # Use the added VOI column
                covariates=direct_model_spec.covariates,
                influencing_mean=direct_model_spec.influencing_mean,
                influencing_variance=direct_model_spec.influencing_variance,
            ),
            defaults=direct_model_defaults,
        )
        return (
            direct_model.predict(
                test_covariates,
                model_params=model_params,
                predict_without=predict_without,
            )
            .to_array(["mu_estimate", "std_estimate"])
            .T
        )

    def _predict_all_mode_estimates(
        self,
        test_covariates: pd.DataFrame,
        model_params: dict[str, Any],
        n_modes: int,
        n_jobs: int = -1,
        predict_without: list[str] | None = None,
    ) -> tuple[
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]],
    ]:
        """
        Internal method to predict all direct estimates for new data using the fitted
        spectral normative model.
        """
        # direct normative predictions for each eigenmode
        tasks = (
            joblib.delayed(self._predict_single_mode_estimates)(
                self.base_model.spec,
                self.base_model.defaults,
                test_covariates,
                model_params=direct_model_params,
                predict_without=predict_without,
            )
            for direct_model_params in model_params["direct_model_params"][:n_modes]
        )

        results = list(
            utils.parallel.ParallelTqdm(
                n_jobs=n_jobs,
                total_tasks=n_modes,
                desc="Computing direct eigenmode estimates",
            )(tasks),  # pyright: ignore[reportCallIssue]
        )

        # Unpack results, estimates have a shape of (n_samples, n_modes)
        eigenmode_mu_estimates, eigenmode_std_estimates = np.array(results).T

        return eigenmode_mu_estimates, eigenmode_std_estimates

    @staticmethod
    def _predict_single_covariance_estimates(
        covariance_model_spec: CovarianceModelSpec,
        covariance_model_defaults: dict[str, Any],
        test_covariates: pd.DataFrame,
        model_params: dict[str, Any],
        predict_without: list[str] | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Internal method to predict single covariance estimates for new data using the
        fitted spectral normative model.
        """
        # create a dummy covariance model
        covariance_model = CovarianceNormativeModel(
            spec=CovarianceModelSpec(
                variable_of_interest_1="VOI_1",
                variable_of_interest_2="VOI_2",
                covariates=covariance_model_spec.covariates,
                influencing_covariance=covariance_model_spec.influencing_covariance,
            ),
            defaults=covariance_model_defaults,
        )
        return (
            covariance_model.predict(
                test_covariates,
                model_params=model_params,
                predict_without=predict_without,
            )
            .to_array(["correlation_estimate"])
            .T
        )

    def _predict_all_covariance_estimates(
        self,
        test_covariates: pd.DataFrame,
        model_params: dict[str, Any],
        n_modes: int,
        n_jobs: int = -1,
        predict_without: list[str] | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Internal method to predict all covariance estimates for new data using the
        fitted spectral normative model.
        """
        # create a dummy covariance model
        covariance_model = CovarianceNormativeModel.from_direct_model(
            self.base_model,
            variable_of_interest_1="dummy_VOI_1",  # Dummy variable of interest
            variable_of_interest_2="dummy_VOI_2",  # Dummy variable of interest
        )

        # Check sparse covariance structure
        row_indices = model_params["sparse_covariance_structure"][:, 0]
        col_indices = model_params["sparse_covariance_structure"][:, 1]
        # Select indices that are within n_modes
        corr_index_valid = (row_indices < n_modes) & (col_indices < n_modes)

        # cross-mode dependence structure for valid pairs
        tasks = (
            joblib.delayed(self._predict_single_covariance_estimates)(
                covariance_model.spec,
                covariance_model.defaults,
                test_covariates,
                model_params=covariance_model_params,
                predict_without=predict_without,
            )
            for i, covariance_model_params in enumerate(
                model_params["covariance_model_params"],
            )
            if corr_index_valid[i]
        )

        results = list(
            utils.parallel.ParallelTqdm(
                n_jobs=n_jobs,
                total_tasks=np.sum(corr_index_valid),
                desc="Computing cross-mode dependence estimates",
            )(tasks),  # pyright: ignore[reportCallIssue]
        )

        # Unpack results, (n_samples, n_valid_covariance_pairs)
        valid_rho_estimates = np.array(results).T[0]

        # Now fill in the full set of rho estimates with NaNs for the invalid pairs
        rho_estimates = np.full(
            (
                test_covariates.shape[0],
                model_params["sparse_covariance_structure"].shape[0],
            ),
            np.nan,
        )
        rho_estimates[:, corr_index_valid] = valid_rho_estimates
        # final estimates have a shape of (n_samples, n_covariance_pairs)

        return rho_estimates

    def compute_spectral_predictions(
        self,
        test_covariates: pd.DataFrame,
        *,
        model_params: dict[str, Any] | None = None,
        n_modes: int | None = None,
        n_jobs: int = -1,
        predict_without: list[str] | None = None,
    ) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """
        Predict normative moments (mean, std) of the eigenmode basis for new data
        using the fitted spectral normative model.

        This function requires a dataframe of covariates (test_covariates) to compute
        a set of spectral predictions that can subsequently be combined to efficiently
        estimate normative predictions for any query(ies).

        Args:
            test_covariates: pd.DataFrame
                DataFrame containing the new covariate data to predict.
                This must include all specified covariates.
                Note: covariates listed in predict_without will be ignored and are
                hence not required.
            model_params: dict | None
                Optional dictionary of model parameters to use. If not provided,
                the stored parameters from model.fit() will be used.
            n_modes: int | None
                Optional number of modes to use for the prediction. If not provided,
                the number of modes from model_params will be used.
            n_jobs: int (default=-1)
                Number of parallel jobs to utilize. If -1, all available CPU cores are
                used. If 1, no parallelization is used.
            predict_without: list[str] | None
                Optional list of covariate names to ignore during prediction.
                This can be used to check the effect of removing certain covariates
                from the model.

        Returns:
            dict:
                A dictionary containing:
                - 'eigenmode_mu_estimates': np.ndarray (n_samples, n_modes)
                - 'eigenmode_std_estimates': np.ndarray (n_samples, n_modes)
                - 'rho_estimates': np.ndarray (n_samples, n_covariance_pairs)
        """
        # Parameters
        if model_params is None:
            model_params = self.model_params

        # Find n_modes
        if n_modes is None:
            n_modes = int(model_params["n_modes"])

        if self.base_model.spec is None:
            err = "The base model is not specified. Cannot predict new data."
            raise ValueError(err)

        # Validate the covariate data
        validation_columns = [
            cov.name
            for cov in self.base_model.spec.covariates
            if cov.name not in (predict_without or [])
        ]
        utils.general.validate_dataframe(test_covariates, validation_columns)

        # direct normative predictions for each eigenmode
        (
            eigenmode_mu_estimates,
            eigenmode_std_estimates,
        ) = self._predict_all_mode_estimates(
            test_covariates,
            model_params,
            n_modes,
            n_jobs=n_jobs,
            predict_without=predict_without,
        )  # estimates have a shape of (n_samples, n_modes)

        # cross-mode dependence structure
        rho_estimates = self._predict_all_covariance_estimates(
            test_covariates,
            model_params,
            n_modes,
            n_jobs=n_jobs,
            predict_without=predict_without,
        )  # estimates have a shape of (n_samples, n_covariance_pairs)

        return {
            "eigenmode_mu_estimates": eigenmode_mu_estimates,
            "eigenmode_std_estimates": eigenmode_std_estimates,
            "rho_estimates": rho_estimates,
        }

    def _validate_spectral_predictions(
        self,
        spectral_predictions: dict[str, npt.NDArray[np.floating[Any]]],
    ) -> None:
        """
        Internal method to validate the spectral predictions dictionary.
        """
        required_keys = [
            "eigenmode_mu_estimates",
            "eigenmode_std_estimates",
            "rho_estimates",
        ]
        if not all(key in spectral_predictions for key in required_keys):
            err = (
                "spectral_predictions must contain 'eigenmode_mu_estimates',"
                " 'eigenmode_std_estimates', and 'rho_estimates'."
            )
            raise ValueError(err)

    def predict(
        self,
        encoded_query: npt.NDArray[np.floating[Any]],
        *,
        spectral_predictions: dict[str, npt.NDArray[np.floating[Any]]] | None = None,
        test_covariates: pd.DataFrame | None = None,
        extended: bool = False,
        model_params: dict[str, Any] | None = None,
        encoded_test_data: npt.NDArray[np.floating[Any]] | None = None,
        n_modes: int | None = None,
        predict_without: list[str] | None = None,
    ) -> NormativePredictions:
        """
        Predict normative moments (mean, std) for new data using the fitted spectral
        normative model.
        Spectral normative modeling can estimate the normative distribution of any
        variable of interest defined as a spatial query encoded in the latent low-pass
        graph spectral space.

        As such, the predict method requires:
            - The encoded query(ies) defining the variable(s) of interest.

        In addition, the method requires either:
            - A dataframe of covariates (test_covariates) to be used for prediction
              of a set of spectral predictions that will subsequently be combined to
              apply the normative predictions for the encoded query(ies).
            OR
            - A dictionary of precomputed spectral predictions (spectral_predictions)
              to be used for efficient normative predictions of the encoded query(ies).

        The precomputed spectral predictions can be obtained using the
        'compute_spectral_predictions' function. This is particularly useful when
        predicting multiple queries or when the same covariate set is used for
        multiple predictions, as it avoids redundant computations.

        Args:
            encoded_query: np.ndarray
                Encoded query data defining the normative variable of interest.
                Can be provided as:
                - shape = (n_modes) for a single query vector
                - shape = (n_modes, n_queries) for multiple queries predicted at once
            spectral_predictions: dict | None
                Optional dictionary of precomputed spectral predictions to use for
                the prediction. If not provided, test_covariates must be provided
                instead to compute the spectral predictions.
                The dictionary should contain:
                - 'eigenmode_mu_estimates': np.ndarray (n_samples, n_modes)
                - 'eigenmode_std_estimates': np.ndarray (n_samples, n_modes)
                - 'rho_estimates': np.ndarray (n_samples, n_covariance_pairs)
                This can be obtained using the 'compute_spectral_predictions' method.
            test_covariates: pd.DataFrame | None
                DataFrame containing the new covariate data to predict.
                This must include all specified covariates.
                Note: covariates listed in predict_without will be ignored and are
                hence not required.
            extended: bool (default: False)
                If True, return additional stats such as log-likelihood, centiles, etc.
                Note that extended predictions require encoded_test_data to be
                provided in addition to the covariates.
            model_params: dict | None
                Optional dictionary of model parameters to use. If not provided,
                the stored parameters from model.fit() will be used.
            encoded_test_data: np.ndarray | None
                Optional encoded test data for the phenotype being modeled (only
                required for extended predictions).
                Expects a numpy array (n_samples, n_modes)
            n_modes: int | None
                Optional number of modes to use for the prediction. If not provided,
                the number of modes from model_params will be used.
            predict_without: list[str] | None
                Optional list of covariate names to ignore during prediction.
                This can be used to check the effect of removing certain covariates
                from the model.

        Returns:
            pd.DataFrame: DataFrame containing the predicted moments (mean, std) for
                the variable of interest defined by the encoded query.
        """
        # Parameters
        if model_params is None:
            model_params = self.model_params

        # Find n_modes
        if n_modes is None:
            n_modes = int(model_params["n_modes"])

        if self.base_model.spec is None:
            err = "The base model is not specified. Cannot predict new data."
            raise ValueError(err)

        if spectral_predictions is None:
            if test_covariates is None:
                err = "Either test_covariates or spectral_predictions must be provided."
                raise ValueError(err)

            # Compute spectral predictions if not provided
            spectral_predictions = self.compute_spectral_predictions(
                test_covariates=test_covariates,
                model_params=model_params,
                n_modes=n_modes,
                predict_without=predict_without,
            )
        elif test_covariates is not None:
            logger.warning(
                "Both test_covariates and spectral_predictions are provided."
                " Ignoring test_covariates and using spectral_predictions.",
            )

        # Unpack spectral predictions
        self._validate_spectral_predictions(spectral_predictions)
        eigenmode_mu_estimates = spectral_predictions["eigenmode_mu_estimates"]
        eigenmode_std_estimates = spectral_predictions["eigenmode_std_estimates"]
        rho_estimates = spectral_predictions["rho_estimates"]

        # reformat encoded queries
        encoded_query = np.asarray(encoded_query[:n_modes]).reshape(n_modes, -1)

        # Compute the predictions
        predictions = self._predict_from_spectral_estimates(
            encoded_query=encoded_query,
            eigenmode_mu_estimates=eigenmode_mu_estimates,
            eigenmode_std_estimates=eigenmode_std_estimates,
            rho_estimates=rho_estimates,
            model_params=model_params,
            n_modes=n_modes,
        )

        # Check if extended predictions are requested
        if extended:
            if encoded_test_data is None:
                err = "Extended predictions require encoded_test_data to be provided."
                raise ValueError(err)
            # Add extended statistics to predictions (e.g. centiles, log-loss, etc.)
            predictions.extend_predictions(
                variable_of_interest=encoded_test_data @ encoded_query,
            )

        return predictions

    def evaluate(
        self,
        encoded_query: npt.NDArray[np.floating[Any]],
        test_covariates: pd.DataFrame,
        encoded_test_data: npt.NDArray[np.floating[Any]],
        query_train_moments: npt.NDArray[np.floating[Any]] | None = None,
        model_params: dict[str, Any] | None = None,
        n_modes: int | None = None,
    ) -> NormativePredictions:
        """
        Evaluate the model on new data and return predictions along with evaluation
        metrics.

        Args:
            encoded_query: np.ndarray
                Encoded query data defining the normative variable of interest.
                Can be provided as:
                - shape = (n_modes) for a single query vector
                - shape = (n_modes, n_queries) for multiple queries predicted at once
            test_covariates: pd.DataFrame
                DataFrame containing the new covariate data to predict.
                This must include all specified covariates.
            encoded_test_data: np.ndarray | None
                Encoded test data for the phenotype being modeled. Expects a numpy array
                of shape: (n_test, n_modes).
            query_train_moments: np.ndarray | None
                A (2, n_queries) array containing the query moments (mean, std) directly
                measured in the training data. While optional, providing these moments
                is strongly recommended for accurate evaluation of the model's MSLL.
                If not provided, the model will use the test data moments as an
                approximation, which may lead to overestimating MSLL. This is made
                optional to allow evaluating MSLL when the training data is not
                accessible (e.g. using a pre-trained model).
            model_params: dict | None
                Optional dictionary of model parameters to use. If not provided,
                the stored parameters from model.fit() will be used.
            n_modes: int | None
                Optional number of modes to use for the prediction. If not provided,
                the stored number of modes from model.fit() will be used.

        Returns:
            NormativePredictions:
                Object containing the predicted moments (mean, std) for
                the variable of interest defined by the encoded query, along with
                evaluation metrics.
        """
        # Find n_modes
        if n_modes is None:
            n_modes = int(self.model_params["n_modes"])

        # Parameters
        if model_params is None:
            model_params = self.model_params

        # reformat encoded queries
        encoded_query = np.asarray(encoded_query[:n_modes]).reshape(n_modes, -1)

        # Run extended predictions
        predictions = self.predict(
            encoded_query=encoded_query,
            test_covariates=test_covariates,
            extended=True,
            model_params=model_params,
            encoded_test_data=encoded_test_data,
            n_modes=n_modes,
        )
        if query_train_moments is None:
            logger.warning(
                "Query moments not provided. Using test data moments as an"
                " approximation, which may lead to overestimating MSLL.",
            )
            query_train_moments = np.array(
                [
                    np.mean(encoded_test_data @ encoded_query, axis=0),
                    np.std(encoded_test_data @ encoded_query, axis=0, ddof=1),
                ],
            )
        return predictions.evaluate_predictions(
            variable_of_interest=encoded_test_data @ encoded_query,
            train_mean=query_train_moments[0],
            train_std=query_train_moments[1],
            n_params=model_params["n_params"],
        )

    def harmonize(
        self,
        encoded_query: npt.NDArray[np.floating[Any]],
        covariates_dataframe: pd.DataFrame,
        encoded_data: npt.NDArray[np.floating[Any]],
        covariates_to_harmonize: list[str],
        *,
        model_params: dict[str, Any] | None = None,
        n_modes: int | None = None,
    ) -> npt.NDArray[np.floating[Any]]:
        """
        Harmonize the variables of interest in the data to remove effects of
        certain covariates (e.g. batch). This method uses the spectral normative model
        to harmonize one or several variables of interest defined by the encoded query.

        Args:
            encoded_query: np.ndarray
                Encoded query data defining the normative variable of interest.
                Can be provided as:
                - shape = (n_modes) for a single query vector
                - shape = (n_modes, n_queries) for multiple queries predicted at once
            covariates_to_harmonize: list[str]
                List of covariate names to harmonize.
                The partial effects of these covariates will be removed from the
                variable of interest, and the harmonized values will be returned.
            covariates_dataframe: pd.DataFrame
                DataFrame containing covariate information for the data to harmonize.
                This must include all specified covariates. The dataframe is expected
                to have all covariates as columns and samples as rows.
            encoded_data: np.ndarray | None
                Encoded data for the variable(s) of interest being modeled. Expects a
                numpy array of shape: (n_samples, n_modes).
            model_params: dict | None
                Optional dictionary of model parameters to use. If not provided,
                the stored parameters from model.fit() will be used.
            n_modes: int | None
                Optional number of modes to use for the prediction. If not provided,
                the stored number of modes from model.fit() will be used.

        Returns:
            npt.NDArray[np.floating[Any]]: Array of harmonized values for the
                variable of interest.
        """
        # Validate the new data
        validation_columns = [cov.name for cov in self.base_model.spec.covariates]
        utils.general.validate_dataframe(covariates_dataframe, validation_columns)

        # Find n_modes
        if n_modes is None:
            n_modes = int(self.model_params["n_modes"])

        # Parameters
        if model_params is None:
            model_params = self.model_params

        # Predict the mean and std with all covariates
        full_predictions = self.predict(
            encoded_query=encoded_query,
            test_covariates=covariates_dataframe,
            model_params=model_params,
            n_modes=n_modes,
            predict_without=[],
        )

        # Predict the mean and std without the covariates to harmonize
        reduced_predictions = self.predict(
            encoded_query=encoded_query,
            test_covariates=covariates_dataframe,
            model_params=model_params,
            n_modes=n_modes,
            predict_without=covariates_to_harmonize,
        )

        # First standardize the variable of interest based on the full model
        vois_standardized = (
            encoded_data - full_predictions.predictions["mu_estimate"]
        ) / full_predictions.predictions["std_estimate"]

        # Then return the harmonized values based on the reduced model
        return np.asarray(
            (
                vois_standardized * reduced_predictions.predictions["std_estimate"]
                + reduced_predictions.predictions["mu_estimate"]
            ),
            dtype=np.float64,
        )
