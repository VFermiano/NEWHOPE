"""PipelineStage base class.

Subclasses implement `process(frame) -> Frame` only. Checkpointing,
parallel execution, timeout handling, and per-frame error capture are
handled uniformly here so no individual stage reimplements them.

Persistence: if `stage.output_dir` is set (the CLI sets it to
``<run>/processed/<stage name>``), every successful result is written there
as a MEF file and recorded in the manifest. On a resumed run a frame that is
marked done is reloaded from that file, so the next stage receives the
*processed* data and not the raw frame. With `output_dir = None` (default,
e.g. in unit tests) stages stay purely in memory and cached frames pass
through unchanged.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from time import monotonic
from typing import Optional

from ..parallel.executor import run_parallel
from .frame import Frame, FrameCollection
from .state import PipelineState

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    name: str = "unnamed_stage"
    output_dir: Optional[Path] = None

    def __init__(self, config, n_workers: int = 1, timeout: float | None = None) -> None:
        self.config = config
        self.n_workers = n_workers
        self.timeout = timeout

    @abstractmethod
    def process(self, frame: Frame) -> Frame:
        """Process a single frame and return the resulting Frame.

        Must not mutate `frame` in place -- use `frame.with_extension_data(...)` / `frame.map_extensions(...)`
        so checkpointing and debugging stay simple. Must be picklable
        along with anything it captures, since it may run in a worker
        process.
        """

    # ------------------------------------------------------------ persistence
    def _output_path(self, frame: Frame) -> Optional[Path]:
        return None if self.output_dir is None else Path(self.output_dir) / frame.path.name

    def _persist(self, frame: Frame) -> Optional[str]:
        """Write the stage output to disk (if configured); return the path for the manifest."""
        out = self._output_path(frame)
        if out is None:
            return None
        frame.write(out)
        return str(out)

    def _restore_cached(self, frame: Frame, state: PipelineState) -> Optional[Frame]:
        """Return this stage's stored output for `frame`, or None if it must be (re)processed."""
        if not state.is_done(frame.frame_id, self.name):
            return None
        out = self._output_path(frame)
        if out is None:
            return frame  # in-memory mode: pass through unchanged
        if not out.is_file():
            logger.warning("%s: %s is marked done but %s is missing; reprocessing", self.name, frame.frame_id, out)
            return None
        logger.debug("%s: restoring cached result for %s from %s", self.name, frame.frame_id, out)
        restored = Frame.from_fits(out)
        restored.frame_id = frame.frame_id  # identity must survive the move to processed/
        return restored

    def _record(self, frame_id: str, outcome, state: PipelineState, output: FrameCollection) -> None:
        """Persist + checkpoint one result (or failure) and add successes to `output`."""
        if isinstance(outcome, Exception):
            logger.error(
                "%s failed on %s: %s",
                self.name,
                frame_id,
                outcome,
                exc_info=(type(outcome), outcome, outcome.__traceback__),
            )
            state.mark(frame_id, self.name, status="failed", error=str(outcome))
            return
        try:
            out_path = self._persist(outcome)
        except Exception as exc:  # noqa: BLE001 -- a failed write fails the frame, not the run
            logger.exception("%s could not write output for %s: %s", self.name, frame_id, exc)
            state.mark(frame_id, self.name, status="failed", error=f"could not write output: {exc}")
            return
        state.mark(frame_id, self.name, status="done", output_path=out_path)
        output.add(outcome)

    # -------------------------------------------------------------------- run
    def run(self, frames: FrameCollection, state: PipelineState) -> FrameCollection:
        started = monotonic()
        pending, cached = [], []
        for f in frames:
            restored = self._restore_cached(f, state)
            if restored is None:
                pending.append(f)
            else:
                cached.append(restored)

        if cached:
            logger.info("%s: skipping %d already-completed frames", self.name, len(cached))
        logger.info(
            "%s: processing started; input=%d, pending=%d, cached=%d, workers=%d, output_dir=%s",
            self.name,
            len(frames),
            len(pending),
            len(cached),
            self.n_workers,
            self.output_dir or "<in-memory>",
        )

        outcomes = run_parallel(self.process, pending, n_workers=self.n_workers, timeout=self.timeout)

        output = FrameCollection()
        for index, (frame, outcome) in enumerate(zip(pending, outcomes), start=1):
            self._record(frame.frame_id, outcome, state, output)
            progress_interval = max(1, (len(pending) + 99) // 100)
            if index % progress_interval == 0 or index == len(pending):
                logger.info(
                    "%s: result handling progress %d/%d (%.1f%%)",
                    self.name,
                    index,
                    len(pending),
                    100.0 * index / len(pending),
                )
        for frame in cached:
            output.add(frame)
        failed = len(frames) - len(output)
        logger.info(
            "%s: processing complete; input=%d, successful=%d, failed=%d, resumed=%d, elapsed=%.1fs",
            self.name,
            len(frames),
            len(output),
            failed,
            len(cached),
            monotonic() - started,
        )
        return output
