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

from dataclasses import dataclass
from typing import Literal

# Type aliases
CovariateType = Literal["categorical", "numerical"]
NumericalEffect = Literal["linear", "spline"]


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
    df: int = 5
    degree: int = 3
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


@dataclass
class CovariateSpec:
    """
    Specification of a single covariate and how it should be modeled.

    Attributes:
        name: Name of the covariate (e.g., 'age', 'site').
        cov_type: Type of the covariate ('numerical' or 'categorical').
        effect: For numerical covariates, how the effect is modeled ('linear'
            or 'spline').
        hierarchical: For categorical covariates, whether to model with a
            hierarchical structure.
        spline_spec: Optional SplineSpec instance for spline modeling;
            required if effect is 'spline'.

    Validation:
        - Numerical covariates must specify 'effect'.
        - If 'effect' is 'spline', 'spline_spec' must be provided.
        - Categorical covariates must specify 'hierarchical'.
        - Categorical covariates cannot have 'effect' or 'spline_spec'.
    """

    name: str
    cov_type: CovariateType  # "categorical" or "numerical"
    effect: NumericalEffect | None = None  # Only if numerical
    hierarchical: bool | None = None  # Only if categorical
    spline_spec: SplineSpec | None = None  # Only for spline modeling

    # Validation checks for the covariate specification.
    def __post_init__(self) -> None:
        if self.cov_type == "numerical":
            if self.effect not in {"linear", "spline"}:
                err = (
                    f"Numerical covariate '{self.name}' must specify effect as "
                    "'linear' or 'spline'."
                )
                raise ValueError(err)
            if self.hierarchical is not None:
                err = (
                    f"Numerical covariate '{self.name}' should not specify "
                    "'hierarchical'."
                )
                raise ValueError(err)
            if self.spline_spec is not None and self.effect != "spline":
                err = (
                    f"Numerical covariate '{self.name}' should not have spline "
                    "specification unless effect is 'spline'."
                )
                raise ValueError(err)
            if self.spline_spec is None and self.effect == "spline":
                err = (
                    f"Numerical covariate '{self.name}' must have spline "
                    "specification if effect is 'spline'."
                )
                raise ValueError(err)
        elif self.cov_type == "categorical":
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
        else:
            err = f"Invalid covariate type '{self.cov_type}' for '{self.name}'."
            raise ValueError(err)


@dataclass
class NormativeModelSpec:
    """
    General specification of a normative model.

    Attributes:
        variable_of_interest: str
            Name of the target variable to model (e.g., "thickness").
        covariates: list[CovariateSpec]
            Listing all model covariates and specifying how each covariate is modeled.
        data: str | list[str]
            Input data source, which can be:
            - A single CSV file path where each row is a subject and
                columns are features (including the variable of interest).
            - A list of multiple CSV file paths if the data is split across files.
    """

    variable_of_interest: str
    covariates: list[CovariateSpec]
    data: str | list[str]
