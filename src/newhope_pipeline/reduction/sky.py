"""Sky subtraction stage.

TODO: port the sky-frame construction (dithered stack median, or
whatever method the existing script uses) and subtraction logic here.
Runs after FlatFielding in the pipeline order.
"""
from __future__ import annotations

from ..core.frame import Frame
from ..core.stage import PipelineStage


class SkySubtraction(PipelineStage):
    name = "sky_subtraction"

    def process(self, frame: Frame) -> Frame:
        sky = self._estimate_sky(frame)
        corrected = frame.data - sky
        return frame.with_data(corrected)

    def _estimate_sky(self, frame: Frame):
        # TODO: port the sky estimation method from the existing script
        # (e.g. median of dithered frames in the same filter/sequence).
        raise NotImplementedError("port sky estimation from the existing script")
