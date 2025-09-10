"""
utils/parallel.py

Utility functions for spectranorm's parallel execution (e.g. via Joblib).
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Iterable, Sized

import tqdm.auto as tqdm
from joblib import Parallel

__all__ = [
    "ParallelTqdm",
]


class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar

    Additional parameters:
    ----------------------
    total_tasks: int, default: None
        the number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.

    desc: str, default: None
        the description used in the tqdm progressbar.

    disable_progressbar: bool, default: False
        If True, a tqdm progressbar is not used.

    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.

    Removed parameters:
    -------------------
    verbose: will be ignored


    Usage:
    ------
    >>> from joblib import delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(.1) for _ in range(10)])
    80%|████████  | 8/10 [00:02<00:00,  3.12tasks/s]

    """

    _original_iterator: Iterable[Any] | None  # mimic joblib internal attribute
    n_dispatched_tasks: int
    n_completed_tasks: int

    def __init__(
        self,
        *,
        total_tasks: int | None = None,
        desc: str | None = None,
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        **kwargs: Any,
    ) -> None:
        if "verbose" in kwargs:
            err = (
                "verbose is not supported. "
                "Use disable_progressbar and show_joblib_header instead."
            )
            raise ValueError(err)
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.progress_bar: tqdm.tqdm[Any] | None = None

    def __call__(self, iterable: Iterable[Any]) -> Any:
        try:
            if self.total_tasks is None and isinstance(iterable, Sized):
                # try to infer total_tasks from the length of the called iterator
                with suppress(TypeError, AttributeError):
                    self.total_tasks = len(iterable)
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator: Iterable[Any]) -> Any:
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm.tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self) -> None:
        """Display the process of the parallel execution using tqdm"""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.total_tasks is None and self._original_iterator is None:
            self.total_tasks = self.n_dispatched_tasks
            if self.progress_bar is not None:
                self.progress_bar.total = self.total_tasks
                self.progress_bar.refresh()
        # update progressbar
        if self.progress_bar is not None:
            self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)
