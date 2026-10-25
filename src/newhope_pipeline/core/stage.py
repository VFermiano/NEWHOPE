"""PipelineStage base class.

Subclasses implement `process(frame) -> Frame` only. Checkpointing,
parallel execution, timeout handling, and per-frame error capture are
handled uniformly here so no individual stage reimplements them.

Persistence: if `stage.output_dir` is set (the CLI sets it to
``<run>/processed/<stage name>``), every successful result is written there
as a MEF file and recorded in the manifest. Reference inputs are loaded in
the worker and their output is written there before only metadata is returned,
so the next stage reads the processed product without retaining previous
arrays. On resume, a valid cached product is checked from its headers and
reused without loading the raw image. With `output_dir = None`, direct stage
use stays in memory; `Pipeline` assigns persistent stage directories when its
input is a collection of references.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from time import monotonic
from typing import Optional

from ..parallel.executor import run_parallel
from .frame import Frame, FrameCollection, FrameLike, FrameReference
from .state import PipelineState

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    name: str = "unnamed_stage"
    output_dir: Optional[Path] = None

    def __init__(
        self,
        config,
        n_workers: int = 1,
        timeout: float | None = None,
        batch_size: int = 1,
    ) -> None:
        self.config = config
        self.n_workers = n_workers
        self.timeout = timeout
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = batch_size

    @abstractmethod
    def process(self, frame: Frame) -> Frame:
        """Process a single frame and return the resulting Frame.

        Must not mutate `frame` in place -- use `frame.with_extension_data(...)` / `frame.map_extensions(...)`
        so checkpointing and debugging stay simple. Must be picklable
        along with anything it captures, since it may run in a worker
        process.
        """

    # ------------------------------------------------------------ persistence
    def _output_path(self, frame: FrameLike) -> Optional[Path]:
        return None if self.output_dir is None else Path(self.output_dir) / frame.path.name

    def _ensure_reference_persistence(
        self, frames: FrameCollection, state: PipelineState
    ) -> None:
        """Use disk intermediates whenever lazy references enter a stage."""
        if self.output_dir is None and any(isinstance(frame, FrameReference) for frame in frames):
            self.output_dir = state.manifest_path.parent / "processed" / self.name
            logger.info(
                "%s: persisting lazy frame results under %s",
                self.name,
                self.output_dir,
            )

    def _persist(self, frame: Frame) -> Optional[str]:
        """Write the stage output to disk (if configured); return the path for the manifest."""
        out = self._output_path(frame)
        if out is None:
            return None
        frame.write(out)
        return str(out)

    def _restore_cached(self, frame: FrameLike, state: PipelineState) -> Optional[FrameLike]:
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
        try:
            if isinstance(frame, FrameReference):
                restored = FrameReference.from_fits(out, expected_extensions=frame.n_ext)
                if restored.shapes != frame.shapes:
                    raise ValueError(f"cached extension shapes {restored.shapes} do not match {frame.shapes}")
                restored.frame_id = frame.frame_id
                return restored
            restored_frame = Frame.from_fits(out, expected_extensions=frame.n_ext)
            if restored_frame.shapes != frame.shapes:
                raise ValueError(f"cached extension shapes {restored_frame.shapes} do not match {frame.shapes}")
            restored_frame.frame_id = frame.frame_id
            return restored_frame
        except Exception:
            logger.exception("%s: cached output for %s is invalid; reprocessing", self.name, frame.frame_id)
            return None

    def _record(
        self,
        frame: FrameLike,
        outcome,
        state: PipelineState,
        output: FrameCollection,
    ) -> None:
        """Persist + checkpoint one result (or failure) and add successes to `output`."""
        if isinstance(outcome, Exception):
            details = getattr(outcome, "_newhope_traceback", None)
            if details:
                logger.error(
                    "%s failed on %s (%s): %s\n%s",
                    self.name,
                    frame.path.name,
                    frame.frame_id,
                    outcome,
                    details,
                )
            else:
                logger.error("%s failed on %s (%s): %s", self.name, frame.path.name, frame.frame_id, outcome)
            state.mark(frame.frame_id, self.name, status="failed", error=str(outcome))
            return
        if isinstance(outcome, FrameReference):
            state.mark(
                frame.frame_id,
                self.name,
                status="done",
                output_path=str(outcome.path),
            )
            output.add(outcome)
            return
        try:
            out_path = self._persist(outcome)
        except Exception as exc:  # noqa: BLE001 -- a failed write fails the frame, not the run
            logger.exception("%s could not write output for %s: %s", self.name, frame.frame_id, exc)
            state.mark(frame.frame_id, self.name, status="failed", error=f"could not write output: {exc}")
            return
        reference = None
        if out_path is not None and isinstance(frame, FrameReference):
            try:
                reference = FrameReference.from_fits(out_path, expected_extensions=frame.n_ext)
                if reference.shapes != frame.shapes:
                    raise ValueError(
                        f"persisted output for {frame.path.name} has shapes {reference.shapes}, expected {frame.shapes}"
                    )
                reference.frame_id = frame.frame_id
            except Exception as exc:  # noqa: BLE001 -- invalid products are recorded as failures
                logger.exception("%s wrote an invalid output for %s", self.name, frame.frame_id)
                state.mark(frame.frame_id, self.name, status="failed", error=f"invalid output: {exc}")
                return
        state.mark(frame.frame_id, self.name, status="done", output_path=out_path)
        if reference is not None:
            output.add(reference)
        else:
            output.add(outcome)

    def _process_item(self, item: FrameLike) -> Frame | FrameReference:
        """Load references inside the worker, write results there, then return metadata only."""
        if isinstance(item, FrameReference):
            loaded = item.load()
            result = self.process(loaded)
            out_path = self._output_path(item)
            if out_path is None:
                return result
            result.write(out_path)
            written = FrameReference.from_fits(out_path, expected_extensions=item.n_ext)
            if written.shapes != item.shapes:
                raise ValueError(
                    f"stage output for {item.path.name} has shapes {written.shapes}, expected {item.shapes}"
                )
            written.frame_id = item.frame_id
            written.status = dict(item.status)
            return written
        return self.process(item)

    # -------------------------------------------------------------------- run
    def run(self, frames: FrameCollection, state: PipelineState) -> FrameCollection:
        started = monotonic()
        self._ensure_reference_persistence(frames, state)
        pending: list[FrameLike] = []
        cached: dict[str, FrameLike] = {}
        for f in frames:
            restored = self._restore_cached(f, state)
            if restored is None:
                pending.append(f)
            else:
                cached[f.frame_id] = restored

        if cached:
            logger.info("%s: skipping %d already-completed frames", self.name, len(cached))
        logger.info(
            "%s: processing started; input=%d, pending=%d, cached=%d, batch_size=%d, workers=%d, max_in_flight=%d, output_dir=%s",
            self.name,
            len(frames),
            len(pending),
            len(cached),
            self.batch_size,
            self.n_workers,
            min(self.batch_size, self.n_workers, len(pending)) if pending else 0,
            self.output_dir or "<in-memory>",
        )

        outcomes = run_parallel(
            self._process_item,
            pending,
            n_workers=self.n_workers,
            timeout=self.timeout,
            max_in_flight=self.batch_size,
            task_labels=[frame.path.name for frame in pending],
        )
        outcomes_by_id = {frame.frame_id: outcome for frame, outcome in zip(pending, outcomes)}

        output = FrameCollection()
        failed = 0
        progress_interval = max(1, (len(frames) + 99) // 100)
        for index, frame in enumerate(frames, start=1):
            if frame.frame_id in cached:
                output.add(cached[frame.frame_id])
            else:
                outcome = outcomes_by_id[frame.frame_id]
                before = len(output)
                self._record(frame, outcome, state, output)
                if len(output) == before:
                    failed += 1
            if index % progress_interval == 0 or index == len(frames):
                logger.info(
                    "%s: completed=%d/%d, skipped=%d, failed=%d",
                    self.name,
                    index,
                    len(frames),
                    sum(1 for f in frames if f.frame_id in cached),
                    failed,
                )
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
