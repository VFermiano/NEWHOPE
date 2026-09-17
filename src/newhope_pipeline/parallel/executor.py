"""Thin wrapper around ProcessPoolExecutor with per-task timeout handling.

This generalizes the timeout/hung-process handling originally built for
the scamp wrapper so every PipelineStage gets it for free, instead of
each stage reimplementing its own parallel-execution logic.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Callable, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    func: Callable[[T], R],
    items: list[T],
    n_workers: int = 1,
    timeout: float | None = None,
) -> list[Union[R, Exception]]:
    """Run func(item) for each item, in parallel if n_workers > 1.

    Returns a list aligned with `items`: either the result of func(item)
    or the Exception it raised (including a timeout), so the caller can
    decide what to do with individual failures without aborting the
    whole batch.

    Note: func and every item must be picklable (ProcessPoolExecutor
    requirement) -- keep stage `process()` methods free of unpicklable
    state like open file handles.
    """
    if n_workers <= 1:
        return [_safe_call(func, item) for item in items]

    results: list[Union[R, Exception, None]] = [None] * len(items)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {executor.submit(func, item): i for i, item in enumerate(items)}
        for future, i in future_to_index.items():
            try:
                results[i] = future.result(timeout=timeout)
            except FutureTimeoutError as exc:
                logger.warning("Task %d timed out after %ss", i, timeout)
                results[i] = exc
            except Exception as exc:  # noqa: BLE001 -- reported per-item, not raised
                results[i] = exc

    return results  # type: ignore[return-value]


def _safe_call(func: Callable[[T], R], item: T) -> Union[R, Exception]:
    try:
        return func(item)
    except Exception as exc:  # noqa: BLE001
        return exc
