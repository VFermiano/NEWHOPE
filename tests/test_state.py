from newfirm_pipeline.core.state import PipelineState


def test_state_roundtrip(tmp_path):
    manifest = tmp_path / "state.json"
    state = PipelineState(manifest_path=manifest)

    assert not state.is_done("frame_1", "dark_subtraction")

    state.mark("frame_1", "dark_subtraction", status="done")
    assert state.is_done("frame_1", "dark_subtraction")
    assert not state.is_done("frame_1", "flat_fielding")


def test_state_persists_and_reloads(tmp_path):
    manifest = tmp_path / "state.json"
    state = PipelineState(manifest_path=manifest)
    state.mark("frame_1", "dark_subtraction", status="done")

    reloaded = PipelineState(manifest_path=manifest)
    assert reloaded.is_done("frame_1", "dark_subtraction")


def test_state_tracks_failures(tmp_path):
    manifest = tmp_path / "state.json"
    state = PipelineState(manifest_path=manifest)
    state.mark("frame_1", "astrometric_calibration", status="failed", error="scamp timeout")

    failures = state.failures("astrometric_calibration")
    assert "frame_1" in failures
    assert failures["frame_1"]["error"] == "scamp timeout"
