"""Thin wrapper around ProcessPoolExecutor with per-task timeout handling.

This generalizes the timeout/hung-process handling originally built for
the scamp wrapper so every PipelineStage gets it for free, instead of
each stage reimplementing its own parallel-execution logic.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from time import monotonic
from typing import Callable, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

_WAIT_LOG_INTERVAL_SECONDS = 30.0


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
    total = len(items)
    if n_workers <= 1:
        logger.info("Executing %d task(s) sequentially", total)
        outcomes = []
        for index, item in enumerate(items, start=1):
            logger.debug("Starting task %d/%d", index, total)
            outcome = _safe_call(func, item)
            if isinstance(outcome, Exception):
                logger.error(
                    "Task %d/%d failed: %s",
                    index,
                    total,
                    outcome,
                    exc_info=(type(outcome), outcome, outcome.__traceback__),
                )
            else:
                logger.info("Task progress: %d/%d completed", index, total)
            outcomes.append(outcome)
        return outcomes

    results: list[Union[R, Exception, None]] = [None] * total
    logger.info("Submitting %d task(s) to %d worker process(es)", total, n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {executor.submit(func, item): i for i, item in enumerate(items)}
        logger.info("Submitted %d task(s); waiting for worker results", total)
        collected = 0
        for future, i in future_to_index.items():
            wait_started = monotonic()
            while True:
                wait_interval = _WAIT_LOG_INTERVAL_SECONDS
                if timeout is not None:
                    remaining = timeout - (monotonic() - wait_started)
                    wait_interval = max(0.0, min(wait_interval, remaining))
                try:
                    results[i] = future.result(timeout=wait_interval)
                    collected += 1
                    logger.info("Worker progress: %d/%d result(s) collected", collected, total)
                    break
                except FutureTimeoutError as exc:
                    elapsed = monotonic() - wait_started
                    if timeout is not None and elapsed >= timeout:
                        logger.warning("Task %d/%d timed out after %ss", i + 1, total, timeout)
                        results[i] = exc
                        collected += 1
                        break
                    finished = sum(task.done() for task in future_to_index)
                    logger.info(
                        "Still waiting for task %d/%d; %d/%d worker task(s) have finished "
                        "(%d/%d result(s) collected)",
                        i + 1,
                        total,
                        finished,
                        total,
                        collected,
                        total,
                    )
                except Exception as exc:  # noqa: BLE001 -- reported per-item, not raised
                    logger.exception(
                        "Task %d/%d raised %s: %s", i + 1, total, type(exc).__name__, exc
                    )
                    results[i] = exc
                    collected += 1
                    break

    return results  # type: ignore[return-value]


def _safe_call(func: Callable[[T], R], item: T) -> Union[R, Exception]:
    try:
        return func(item)
    except Exception as exc:  # noqa: BLE001
        return exc
