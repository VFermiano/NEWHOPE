import numpy as np

from newfirm_pipeline.core.frame import Frame, FrameCollection


def test_frame_from_fits(synthetic_frame_file):
    frame = Frame.from_fits(synthetic_frame_file)
    assert frame.filt == "HX"
    assert frame.exptime == 30.0
    assert frame.data.shape == (32, 32)
    assert frame.frame_id  # non-empty


def test_frame_id_stable_for_same_path(synthetic_frame_file):
    a = Frame.from_fits(synthetic_frame_file)
    b = Frame.from_fits(synthetic_frame_file)
    assert a.frame_id == b.frame_id


def test_with_data_preserves_identity_and_metadata(synthetic_frame_file):
    frame = Frame.from_fits(synthetic_frame_file)
    new_frame = frame.with_data(frame.data * 2)

    assert new_frame.frame_id == frame.frame_id
    assert new_frame.filt == frame.filt
    assert new_frame.exptime == frame.exptime
    assert np.array_equal(new_frame.data, frame.data * 2)
    # original untouched
    assert np.array_equal(frame.data, np.full((32, 32), 100.0, dtype=np.float32))


def test_frame_collection_by_filter(synthetic_frame_file):
    frame = Frame.from_fits(synthetic_frame_file)
    collection = FrameCollection([frame])

    assert len(collection.by_filter("HX")) == 1
    assert len(collection.by_filter("JX")) == 0
