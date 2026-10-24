"""Pipeline orchestrator: runs an ordered list of stages over a FrameCollection."""
from __future__ import annotations

import logging
from pathlib import Path
from time import monotonic

from .frame import FrameCollection
from .stage import PipelineStage
from .state import PipelineState

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, stages: list[PipelineStage], manifest_path: Path) -> None:
        self.stages = stages
        self.state = PipelineState(manifest_path=manifest_path)

    def run(self, frames: FrameCollection) -> FrameCollection:
        started = monotonic()
        initial_count = len(frames)
        current = frames
        logger.info(
            "Pipeline started: %d frame(s), %d stage(s), manifest=%s",
            initial_count,
            len(self.stages),
            self.state.manifest_path,
        )
        for stage in self.stages:
            stage_started = monotonic()
            input_count = len(current)
            logger.info(
                "Stage %s started: input=%d, workers=%d, output_dir=%s",
                stage.name,
                input_count,
                stage.n_workers,
                stage.output_dir or "<in-memory>",
            )
            try:
                current = stage.run(current, self.state)
            except Exception:
                logger.exception("Stage %s aborted after %.1fs", stage.name, monotonic() - stage_started)
                raise
            failed_count = input_count - len(current)
            logger.info(
                "Stage %s completed: input=%d, successful=%d, failed=%d, elapsed=%.1fs",
                stage.name,
                input_count,
                len(current),
                failed_count,
                monotonic() - stage_started,
            )
        logger.info(
            "Pipeline complete: successful=%d/%d, failed=%d, elapsed=%.1fs",
            len(current),
            initial_count,
            initial_count - len(current),
            monotonic() - started,
        )
        return current
