"""Thin wrapper around ProcessPoolExecutor with per-task timeout handling.

This generalizes the timeout/hung-process handling originally built for
the scamp wrapper so every PipelineStage gets it for free, instead of
each stage reimplementing its own parallel-execution logic.
"""
from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures import TimeoutError as FutureTimeoutError
from time import monotonic
from typing import Callable, TypeVar, Union

from ..core.exceptions import detach_traceback

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

_WAIT_LOG_INTERVAL_SECONDS = 30.0


def run_parallel(
    func: Callable[[T], R],
    items: list[T],
    n_workers: int = 1,
    timeout: float | None = None,
    max_in_flight: int | None = None,
    task_labels: list[str] | None = None,
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
    total = len(items)
    if task_labels is not None and len(task_labels) != total:
        raise ValueError("task_labels must have one label per task")

    def label(index: int) -> str:
        return task_labels[index] if task_labels is not None else f"task {index + 1}/{total}"

    max_in_flight = max(1, max_in_flight or max(1, n_workers * 2))
    if n_workers <= 1 or max_in_flight <= 1:
        logger.info("Executing %d task(s) sequentially", total)
        outcomes = []
        for index, item in enumerate(items, start=1):
            logger.debug("Starting task %d/%d (%s)", index, total, label(index - 1))
            outcome = _safe_call(func, item)
            if isinstance(outcome, Exception):
                details = getattr(outcome, "_newhope_traceback", None)
                if details:
                    logger.error("Task %d/%d (%s) failed: %s\n%s", index, total, label(index - 1), outcome, details)
                else:
                    logger.error("Task %d/%d (%s) failed: %s", index, total, label(index - 1), outcome)
            else:
                logger.info("Task progress: %d/%d completed (%s)", index, total, label(index - 1))
            outcomes.append(outcome)
        return outcomes

    results: list[Union[R, Exception, None]] = [None] * total
    flight_limit = min(total, max_in_flight)
    logger.info(
        "Submitting %d task(s) to %d worker process(es), at most %d in flight",
        total,
        n_workers,
        flight_limit,
    )
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {}
        submitted_at = {}
        next_index = 0
        while next_index < total and len(future_to_index) < flight_limit:
            future = executor.submit(func, items[next_index])
            future_to_index[future] = next_index
            submitted_at[future] = monotonic()
            next_index += 1
        logger.info(
            "Submitted %d/%d task(s); waiting for worker results (in flight: %s)",
            next_index,
            total,
            ", ".join(label(index) for index in future_to_index.values()),
        )
        collected = 0
        while future_to_index:
            now = monotonic()
            deadlines = (
                [submitted_at[future] + timeout for future in future_to_index]
                if timeout is not None
                else []
            )
            wait_timeout = _WAIT_LOG_INTERVAL_SECONDS
            if deadlines:
                wait_timeout = max(0.0, min(wait_timeout, min(deadlines) - now))
            done, _ = wait(future_to_index, timeout=wait_timeout, return_when=FIRST_COMPLETED)
            if not done:
                now = monotonic()
                expired = [
                    future
                    for future in future_to_index
                    if timeout is not None and now - submitted_at[future] >= timeout
                ]
                for future in expired:
                    index = future_to_index.pop(future)
                    submitted_at.pop(future, None)
                    future.cancel()
                    results[index] = FutureTimeoutError(f"task exceeded {timeout}s timeout")
                    collected += 1
                    logger.warning(
                        "Task %d/%d (%s) timed out after %ss", index + 1, total, label(index), timeout
                    )
                if not expired:
                    logger.info(
                        "Still waiting for worker task(s) %s; %d/%d in flight, %d/%d completed",
                        ", ".join(label(index) for index in future_to_index.values()),
                        len(future_to_index),
                        total,
                        collected,
                        total,
                    )
            for future in done:
                if future not in future_to_index:
                    continue
                index = future_to_index.pop(future)
                submitted_at.pop(future, None)
                try:
                    results[index] = future.result()
                except Exception as exc:  # noqa: BLE001 -- reported per-item, not raised
                    logger.exception(
                        "Task %d/%d (%s) raised %s: %s",
                        index + 1,
                        total,
                        label(index),
                        type(exc).__name__,
                        exc,
                    )
                    results[index] = detach_traceback(exc)
                collected += 1
                logger.info(
                    "Worker progress: %d/%d result(s) collected (%s)",
                    collected,
                    total,
                    label(index),
                )

            while next_index < total and len(future_to_index) < flight_limit:
                future = executor.submit(func, items[next_index])
                future_to_index[future] = next_index
                submitted_at[future] = monotonic()
                next_index += 1
            if next_index > collected:
                logger.debug("Submitted task progress: %d/%d", next_index, total)

    return results  # type: ignore[return-value]


def _safe_call(func: Callable[[T], R], item: T) -> Union[R, Exception]:
    try:
        return func(item)
    except Exception as exc:  # noqa: BLE001
        return detach_traceback(exc)
