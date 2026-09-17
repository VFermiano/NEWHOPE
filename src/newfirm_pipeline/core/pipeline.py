"""Pipeline orchestrator: runs an ordered list of stages over a FrameCollection."""
from __future__ import annotations

import logging
from pathlib import Path

from .frame import FrameCollection
from .stage import PipelineStage
from .state import PipelineState

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, stages: list[PipelineStage], manifest_path: Path) -> None:
        self.stages = stages
        self.state = PipelineState(manifest_path=manifest_path)

    def run(self, frames: FrameCollection) -> FrameCollection:
        current = frames
        for stage in self.stages:
            logger.info("=== Running stage: %s (%d frames in) ===", stage.name, len(current))
            current = stage.run(current, self.state)
            logger.info("=== %s done: %d frames out ===", stage.name, len(current))
        return current
