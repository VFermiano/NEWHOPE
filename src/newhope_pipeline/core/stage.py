"""PipelineStage base class.

Subclasses implement `process(frame) -> Frame` only. Checkpointing,
parallel execution, timeout handling, and per-frame error capture are
handled uniformly here so no individual stage reimplements them.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ..parallel.executor import run_parallel
from .frame import Frame, FrameCollection
from .state import PipelineState

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    name: str = "unnamed_stage"

    def __init__(self, config, n_workers: int = 1, timeout: float | None = None) -> None:
        self.config = config
        self.n_workers = n_workers
        self.timeout = timeout

    @abstractmethod
    def process(self, frame: Frame) -> Frame:
        """Process a single frame and return the resulting Frame.

        Must not mutate `frame` in place -- use `frame.with_data(...)`
        so checkpointing and debugging stay simple. Must be picklable
        along with anything it captures, since it may run in a worker
        process.
        """

    def run(self, frames: FrameCollection, state: PipelineState) -> FrameCollection:
        pending, cached = [], []
        for f in frames:
            (cached if state.is_done(f.frame_id, self.name) else pending).append(f)

        if cached:
            logger.info("%s: skipping %d already-completed frames", self.name, len(cached))
        logger.info("%s: processing %d frames (n_workers=%d)", self.name, len(pending), self.n_workers)

        outcomes = run_parallel(self.process, pending, n_workers=self.n_workers, timeout=self.timeout)

        output = FrameCollection()
        for frame, outcome in zip(pending, outcomes):
            if isinstance(outcome, Exception):
                logger.error("%s failed on %s: %s", self.name, frame.frame_id, outcome)
                state.mark(frame.frame_id, self.name, status="failed", error=str(outcome))
            else:
                state.mark(frame.frame_id, self.name, status="done")
                output.add(outcome)

        # TODO: cached frames currently pass through with their pre-stage data
        # unchanged, since PipelineState only tracks status, not the output
        # itself. Once stages have a concrete on-disk output convention (e.g.
        # config.output_dir / stage.name / frame.path.name), reload the
        # processed data here via Frame.from_fits(output_path) instead.
        for frame in cached:
            output.add(frame)

        return output
