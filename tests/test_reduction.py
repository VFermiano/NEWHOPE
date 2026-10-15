import numpy as np
import pytest

from newhope_pipeline.core.exceptions import ReductionError
from newhope_pipeline.core.frame import Frame, FrameCollection
from newhope_pipeline.core.pipeline import Pipeline
from newhope_pipeline.reduction.dark import DarkSubtraction
from newhope_pipeline.reduction.flat import FlatFielding

from .conftest import SHAPE


class FakeDark(DarkSubtraction):
    def _load_master_dark(self, exptime):
        return {k: np.full(SHAPE, 10.0 * k, dtype=np.float32) for k in range(1, 5)}


class FakeFlat(FlatFielding):
    def _load_master_flat(self, filt):
        return {k: np.full(SHAPE, 2.0, dtype=np.float32) for k in range(1, 5)}


def test_dark_uses_each_detectors_own_master(synthetic_frame_file):
    frame = Frame.from_fits(synthetic_frame_file)
    out = FakeDark(config=None).process(frame)
    for k in range(1, 5):
        assert np.all(out[k].data == 100.0 * k - 10.0 * k)


def test_dark_missing_detector_master_fails(synthetic_frame_file):
    class Partial(DarkSubtraction):
        def _load_master_dark(self, exptime):
            return {1: np.zeros(SHAPE, dtype=np.float32)}

    frame = Frame.from_fits(synthetic_frame_file)
    with pytest.raises(ReductionError, match=r"\[2, 3, 4\]"):
        Partial(config=None).process(frame)


def test_dark_shape_mismatch_reports_extension(synthetic_frame_file):
    class Bad(DarkSubtraction):
        def _load_master_dark(self, exptime):
            m = {k: np.zeros(SHAPE, dtype=np.float32) for k in range(1, 5)}
            m[3] = np.zeros((8, 8), dtype=np.float32)
            return m

    frame = Frame.from_fits(synthetic_frame_file)
    with pytest.raises(ReductionError, match="extension 3"):
        Bad(config=None).process(frame)


def test_dark_then_flat_through_pipeline(synthetic_frame_file, tmp_path):
    frames = FrameCollection([Frame.from_fits(synthetic_frame_file)])
    pipe = Pipeline([FakeDark(None), FakeFlat(None)], manifest_path=tmp_path / "state.json")
    out = list(pipe.run(frames))[0]
    for k in range(1, 5):
        assert np.all(out[k].data == (100.0 * k - 10.0 * k) / 2.0)
