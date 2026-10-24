import numpy as np
from astropy.io import fits

from newhope_pipeline.core.frame import Frame
from newhope_pipeline.io.decompress import uncompress_directory

SHAPE = (32, 32)


def _make_fz(path, n_ext=4, value=1.0):
    hdus = [fits.PrimaryHDU(header=fits.Header({"FILTER": "HX", "EXPTIME": 30.0}))]
    for k in range(1, n_ext + 1):
        hdus.append(fits.CompImageHDU(np.full(SHAPE, value * k, dtype=np.float32), name=f"im{k}"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def test_uncompress_keeps_mef_layout(tmp_path):
    _make_fz(tmp_path / "a.fits.fz", value=3.0)
    res = uncompress_directory(tmp_path)
    assert res.decompressed == 1 and not res.failed
    with fits.open(tmp_path / "a.fits") as h:
        assert h[0].data is None
        assert len(h) == 5
        assert "ZIMAGE" not in h[3].header
    frame = Frame.from_fits(tmp_path / "a.fits", expected_extensions=4)
    assert np.allclose(frame[3].data, 9.0) and frame.filt == "HX"
    assert (tmp_path / "a.fits.fz").exists()


def test_skips_existing_and_noop(tmp_path):
    assert uncompress_directory(tmp_path).found == 0
    _make_fz(tmp_path / "a.fits.fz")
    uncompress_directory(tmp_path)
    again = uncompress_directory(tmp_path)
    assert again.skipped == 1 and again.decompressed == 0


def test_delete_original(tmp_path):
    _make_fz(tmp_path / "a.fits.fz")
    uncompress_directory(tmp_path, delete_original=True)
    assert not (tmp_path / "a.fits.fz").exists()
