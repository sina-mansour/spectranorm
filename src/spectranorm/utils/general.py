"""
utils/general.py

General utility functions for spectranorm.
"""

from __future__ import annotations

import datetime
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

__all__ = [
    "ensure_dir",
    "get_logger",
    "prepare_save_directory",
    "report_time",
    "suppress_output",
    "validate_dataframe",
    "validate_load_directory",
]


def ensure_dir(file_name: Path) -> Path:
    """
    Ensure that the directory for the given file name exists.

    Parameters
    ----------
    file_name : Path
        The file name for which to ensure the directory exists.

    Returns
    -------
    Path
        The original file name.
    """
    file_name.parent.mkdir(parents=True, exist_ok=True)
    return file_name


def report_time(
    *,
    relative_to: float | None = None,
    absolute: bool = False,
) -> str | float:
    """
    Report the current time or the time elapsed since a given reference point.

    Parameters
    ----------
    relative_to : float | None
        The reference time in seconds since the epoch. If None, report the current time.
    absolute : bool
        If True, report the absolute time format (float). If False, report the time in a
        human-readable format (str).

    Returns
    -------
    str | float
        The current time or the elapsed time since the reference point.
    """
    if relative_to is not None:
        elapsed = time.time() - relative_to
        return elapsed if absolute else str(datetime.timedelta(seconds=elapsed))

    now = time.time()
    return (
        now
        if absolute
        else datetime.datetime.fromtimestamp(
            now,
            tz=datetime.datetime.now().astimezone().tzinfo,
        ).strftime("%Y-%m-%d %H:%M:%S")
    )


class ReportTimeFormatter(logging.Formatter):
    """Custom log formatter that injects general.report_time()."""

    def format(self, record: logging.LogRecord) -> str:
        record.report_time = report_time()
        return super().format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with standardized formatting and time reporting.

    Args:
        name: Logger name (usually __name__).
        level: Logging level, default INFO.

    Returns:
        logging.Logger with a custom ReportTimeFormatter.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = ReportTimeFormatter(
            fmt="%(report_time)s : [%(levelname)s] - %(name)s - %(message)s",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def validate_dataframe(dataframe: pd.DataFrame, column_names: list[str]) -> None:
    """
    Validate the input DataFrame to ensure all required columns are available.

    Args:
        dataframe: pd.DataFrame
            The DataFrame to validate.

    Raises:
        TypeError: If the DataFrame is not valid.
    """
    if not isinstance(dataframe, pd.DataFrame):
        err = "Input data must be a pandas DataFrame."
        raise TypeError(err)
    # report if a column is missing
    missing_columns = [col for col in column_names if col not in dataframe.columns]
    if missing_columns:
        err = f"Missing columns in DataFrame: {', '.join(missing_columns)}"
        raise ValueError(err)


def prepare_save_directory(directory: Path, subdirectory: str = "saved_model") -> Path:
    """
    Prepare a directory to save a model.

    A subdirectory named 'saved_model' will be created if it does not exist.
    If this directory exists, but is not empty, an error is raised.

    Args:
        directory (Path): Path to a directory to save the model.
        subdirectory (str): Name of the subdirectory to create (default: "saved_model").

    Returns
    -------
    Path
        The path to the created subdirectory.
    """
    # Check if the directory exists
    if not directory.exists():
        err = f"Model Save Error: Directory '{directory}' does not exist."
        raise FileNotFoundError(err)
    # Check if the subdirectory exists and is empty
    saved_model_dir = directory / subdirectory
    if saved_model_dir.exists():
        if any(saved_model_dir.iterdir()):
            err = f"Model Save Error: Directory '{saved_model_dir}' is not empty."
            raise ValueError(err)
    else:
        saved_model_dir.mkdir(parents=True, exist_ok=True)

    return saved_model_dir


def validate_load_directory(directory: Path, subdirectory: str = "saved_model") -> Path:
    """
    Validate the directory structure for loading a model.

    Args:
        directory (Path): Path to the main directory.
        subdirectory (str): Name of the subdirectory to check (default: "saved_model").

    Returns
    -------
    Path
        The path to the validated subdirectory.
    """
    # Check if the directory exists
    if not directory.exists():
        err = f"Model Load Error: Directory '{directory}' does not exist."
        raise FileNotFoundError(err)
    # Check if the subdirectory exists
    saved_model_dir = directory / subdirectory
    if not saved_model_dir.exists():
        err = f"Model Load Error: Directory '{saved_model_dir}' does not exist."
        raise FileNotFoundError(err)

    return saved_model_dir


@contextmanager
def suppress_output() -> Iterator[None]:
    """
    Context manager to suppress stdout and stderr output.
    """
    with Path(os.devnull).open("w") as fnull:
        old_stdout, old_stderr = os.dup(1), os.dup(2)
        try:
            os.dup2(fnull.fileno(), 1)  # Redirect stdout
            os.dup2(fnull.fileno(), 2)  # Redirect stderr
            yield
        finally:
            os.dup2(old_stdout, 1)  # Restore
            os.dup2(old_stderr, 2)  # Restore
            os.close(old_stdout)
            os.close(old_stderr)
