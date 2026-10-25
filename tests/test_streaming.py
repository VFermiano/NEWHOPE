import gc
import weakref

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyUserWarning

from newhope_pipeline.core.frame import Frame, FrameCollection, FrameReference
from newhope_pipeline.core.pipeline import Pipeline
from newhope_pipeline.core.stage import PipelineStage
from newhope_pipeline.core.state import PipelineState
from newhope_pipeline.reduction.dark import DarkSubtraction
from newhope_pipeline.reduction.flat import FlatFielding

from .conftest import SHAPE, make_mef


class AddOne(PipelineStage):
    name = "add_one"

    def __init__(self, **kwargs):
        super().__init__(config=None, **kwargs)
        self.input_array = None
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        self.input_array = weakref.ref(frame[1].data)
        return frame.map_extensions(lambda ext: ext.data + 1.0)


class SometimesFails(AddOne):
    name = "sometimes_fails"

    def process(self, frame):
        if frame.path.name == "b.fits":
            raise ValueError("synthetic worker failure")
        return super().process(frame)


class FailsAfterLoad(AddOne):
    name = "fails_after_load"

    def process(self, frame):
        self.input_array = weakref.ref(frame[1].data)
        raise ValueError("failure after loading the image")


class FakeDark(DarkSubtraction):
    def _load_master_dark(self, frame):
        return {index: np.full(SHAPE, 10.0 * index, dtype=np.float32) for index in range(1, 5)}


class FakeFlat(FlatFielding):
    def _load_master_flat(self, filt):
        return {index: np.full(SHAPE, 2.0, dtype=np.float32) for index in range(1, 5)}


def _references(directory, n=3):
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(n):
        make_mef(directory / f"{chr(ord('a') + index)}.fits")
    return FrameCollection.from_directory(directory, expected_extensions=4)


def test_reference_pipeline_persists_metadata_and_releases_input_arrays(tmp_path):
    references = _references(tmp_path / "raw", n=1)
    manifest = tmp_path / "pipeline_state.json"
    stage = AddOne(n_workers=1, batch_size=1)

    result = Pipeline([stage], manifest_path=manifest).run(references)
    gc.collect()

    assert len(result) == 1
    output = next(iter(result))
    assert isinstance(output, FrameReference)
    assert output.path == tmp_path / "processed" / stage.name / "a.fits"
    assert output.load()[1].data[0, 0] == 101.0
    assert stage.input_array() is None
    assert PipelineState(manifest).is_done(next(iter(references)).frame_id, stage.name)


def test_reference_pipeline_resumes_without_loading_raw_frames(tmp_path, monkeypatch):
    references = _references(tmp_path / "raw", n=1)
    manifest = tmp_path / "pipeline_state.json"
    first = AddOne(n_workers=1)
    Pipeline([first], manifest_path=manifest).run(references)
    assert first.calls == 1

    def raw_load_must_not_happen(self, extension_indices=None):
        raise AssertionError("a valid completed output should be reused")

    monkeypatch.setattr(FrameReference, "load", raw_load_must_not_happen)
    resumed = AddOne(n_workers=1)
    result = Pipeline([resumed], manifest_path=manifest).run(references)

    assert resumed.calls == 0
    assert isinstance(next(iter(result)), FrameReference)


def test_reference_pipeline_reprocesses_truncated_completed_output(tmp_path):
    references = _references(tmp_path / "raw", n=1)
    manifest = tmp_path / "pipeline_state.json"
    first = AddOne(n_workers=1)
    Pipeline([first], manifest_path=manifest).run(references)
    output_path = tmp_path / "processed" / first.name / "a.fits"
    output_path.write_bytes(output_path.read_bytes()[:-2880])

    retry = AddOne(n_workers=1)
    with pytest.warns(AstropyUserWarning, match="File may have been truncated"):
        result = Pipeline([retry], manifest_path=manifest).run(references)

    assert retry.calls == 1
    assert next(iter(result)).load()[1].data[0, 0] == 101.0


def test_reference_pipeline_records_partial_failures_and_recovers(tmp_path):
    references = _references(tmp_path / "raw", n=3)
    manifest = tmp_path / "pipeline_state.json"
    stage = SometimesFails(n_workers=1, batch_size=2)
    result = Pipeline([stage], manifest_path=manifest).run(references)

    assert len(result) == 2
    state = PipelineState(manifest)
    assert len(state.failures(stage.name)) == 1
    assert (tmp_path / "processed" / stage.name / "a.fits").is_file()
    assert not (tmp_path / "processed" / stage.name / "b.fits").exists()
    assert (tmp_path / "processed" / stage.name / "c.fits").is_file()


def test_failed_task_releases_loaded_arrays(tmp_path):
    references = _references(tmp_path / "raw", n=1)
    stage = FailsAfterLoad(n_workers=1)
    Pipeline([stage], manifest_path=tmp_path / "pipeline_state.json").run(references)
    gc.collect()

    assert stage.input_array() is None


def test_parallel_reference_pipeline_limits_in_flight_frames(tmp_path, caplog):
    import logging

    references = _references(tmp_path / "raw", n=3)
    stage = AddOne(n_workers=2, batch_size=2)
    with caplog.at_level(logging.INFO, logger="newhope_pipeline.parallel.executor"):
        result = Pipeline([stage], manifest_path=tmp_path / "pipeline_state.json").run(references)

    assert len(result) == 3
    assert all(isinstance(frame, FrameReference) for frame in result)
    assert any("at most 2 in flight" in record.getMessage() for record in caplog.records)
    for reference in result:
        assert reference.load()[1].data[0, 0] == 101.0


def test_lazy_dark_and_flat_match_loaded_frame_pixels(tmp_path):
    raw = tmp_path / "raw"
    references = _references(raw, n=1)
    original = Frame.from_fits(raw / "a.fits")
    expected = FakeFlat(None).process(FakeDark(None).process(original))

    dark, flat = FakeDark(None, n_workers=1), FakeFlat(None, n_workers=1)
    result = Pipeline([dark, flat], manifest_path=tmp_path / "pipeline_state.json").run(references)
    actual = next(iter(result)).load()

    for index in range(1, 5):
        np.testing.assert_array_equal(actual[index].data, expected[index].data)
        assert actual[index].header == expected[index].header
    assert (tmp_path / "processed" / "dark_subtraction" / "a.fits").is_file()
    assert (tmp_path / "processed" / "flat_fielding" / "a.fits").is_file()
