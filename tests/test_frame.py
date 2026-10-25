import numpy as np
import pytest
from astropy.io import fits

from newhope_pipeline.core.exceptions import PipelineError
from newhope_pipeline.core.frame import Frame, FrameCollection, FrameReference

from .conftest import SHAPE, make_mef


def test_frame_from_fits_loads_all_extensions(synthetic_frame_file):
    frame = Frame.from_fits(synthetic_frame_file, expected_extensions=4)
    assert frame.filt == "HX"
    assert frame.exptime == 30.0
    assert frame.n_ext == 4
    assert [e.index for e in frame] == [1, 2, 3, 4]
    assert [e.name for e in frame] == ["IM1", "IM2", "IM3", "IM4"]
    for k in range(1, 5):
        assert frame[k].data.shape == SHAPE
        assert frame[k].data.dtype == np.float32
        assert np.all(frame[k].data == 100.0 * k)
    assert frame.frame_id  # non-empty


def test_wrong_extension_count_is_an_error(synthetic_frame_file):
    with pytest.raises(PipelineError, match="expected 3"):
        Frame.from_fits(synthetic_frame_file, expected_extensions=3)


def test_data_in_primary_is_rejected(tmp_path):
    path = tmp_path / "simple.fits"
    fits.writeto(path, np.zeros(SHAPE, dtype=np.float32))
    with pytest.raises(PipelineError, match="primary"):
        Frame.from_fits(path)


def test_frame_id_stable_for_same_path(synthetic_frame_file):
    a = Frame.from_fits(synthetic_frame_file)
    b = Frame.from_fits(synthetic_frame_file)
    assert a.frame_id == b.frame_id


def test_with_extension_data_preserves_identity_and_metadata(synthetic_frame_file):
    frame = Frame.from_fits(synthetic_frame_file)
    new_frame = frame.with_extension_data([e.data * 2 for e in frame])

    assert new_frame.frame_id == frame.frame_id
    assert new_frame.filt == frame.filt
    assert new_frame.exptime == frame.exptime
    assert [e.name for e in new_frame] == [e.name for e in frame]
    for k in range(1, 5):
        assert np.all(new_frame[k].data == 200.0 * k)
        assert np.all(frame[k].data == 100.0 * k)  # original untouched


def test_with_extension_data_checks_count(synthetic_frame_file):
    frame = Frame.from_fits(synthetic_frame_file)
    with pytest.raises(PipelineError):
        frame.with_extension_data([frame[1].data])


def test_map_extensions_is_per_detector(synthetic_frame_file):
    frame = Frame.from_fits(synthetic_frame_file)
    out = frame.map_extensions(lambda ext: ext.data - 10.0 * ext.index)
    for k in range(1, 5):
        assert np.all(out[k].data == 90.0 * k)


def test_write_roundtrip_is_mef_with_headers(synthetic_frame_file, tmp_path):
    frame = Frame.from_fits(synthetic_frame_file)
    out = tmp_path / "out" / "roundtrip.fits"
    frame.map_extensions(lambda e: e.data + 1).write(out)

    with fits.open(out) as hdul:
        assert len(hdul) == 5
        assert hdul[0].header["FILTER"] == "HX"
        assert [h.name for h in hdul[1:]] == ["IM1", "IM2", "IM3", "IM4"]
        assert hdul[3].header["DETSEC"] == "chip3"
        assert np.all(hdul[3].data == 301.0)

    again = Frame.from_fits(out, expected_extensions=4)
    assert np.all(again[4].data == 401.0)


def test_scaled_integer_extensions_are_read_as_physical_values(tmp_path):
    path = tmp_path / "scaled.fits"
    primary = fits.PrimaryHDU()
    hdus = [primary]
    for k in range(1, 5):
        raw = np.full(SHAPE, 1000 * k, dtype=np.uint16)  # stored as int16 + BZERO
        hdus.append(fits.ImageHDU(raw, name=f"im{k}"))
    fits.HDUList(hdus).writeto(path)

    frame = Frame.from_fits(path)
    assert np.all(frame[2].data == 2000)
    assert "BZERO" not in frame[2].header
    out = tmp_path / "o.fits"
    frame.write(out)
    assert np.all(Frame.from_fits(out)[2].data == 2000)


def test_compressed_extensions_are_supported(tmp_path):
    path = tmp_path / "comp.fits.fz"
    hdus = [fits.PrimaryHDU()]
    hdus[0].header["FILTER"] = "JX"
    for k in range(1, 5):
        hdus.append(fits.CompImageHDU(np.full(SHAPE, 7.0 * k, dtype=np.float32), name=f"im{k}"))
    fits.HDUList(hdus).writeto(path)

    frame = Frame.from_fits(path, expected_extensions=4)
    assert frame.filt == "JX"
    assert np.allclose(frame[3].data, 21.0)
    reference = FrameReference.from_fits(path, expected_extensions=4)
    assert reference.shapes == {k: SHAPE for k in range(1, 5)}
    assert np.all(reference.load(extension_indices=[3])[3].data == 21.0)


def test_frame_collection_by_filter(synthetic_frame_file):
    frame = Frame.from_fits(synthetic_frame_file)
    collection = FrameCollection([frame])

    assert len(collection.by_filter("HX")) == 1
    assert len(collection.by_filter("JX")) == 0


def test_from_directory_passes_expected_extensions(tmp_path):
    make_mef(tmp_path / "a.fits")
    make_mef(tmp_path / "b.fits", n_ext=2)
    with pytest.raises(PipelineError):
        FrameCollection.from_directory(tmp_path, expected_extensions=4)
    assert len(FrameCollection.from_directory(tmp_path)) == 2


def test_directory_discovery_returns_header_only_references(tmp_path, monkeypatch):
    path = make_mef(tmp_path / "a.fits")

    def eager_load_was_called(*args, **kwargs):
        raise AssertionError("discovery must not load frame arrays")

    monkeypatch.setattr(Frame, "from_fits", eager_load_was_called)
    frames = FrameCollection.from_directory(tmp_path, expected_extensions=4)

    assert len(frames) == 1
    reference = next(iter(frames))
    assert isinstance(reference, FrameReference)
    assert reference.path == path
    assert reference.filt == "HX" and reference.exptime == 30.0
    assert reference.shapes == {1: SHAPE, 2: SHAPE, 3: SHAPE, 4: SHAPE}
    assert not any(hasattr(extension, "data") for extension in reference.extensions)


def test_reference_loads_only_requested_detector(synthetic_frame_file):
    reference = FrameReference.from_fits(synthetic_frame_file, expected_extensions=4)
    frame = reference.load(extension_indices=[3])

    assert frame.n_ext == 1
    assert [extension.index for extension in frame] == [3]
    assert np.all(frame[3].data == 300.0)
