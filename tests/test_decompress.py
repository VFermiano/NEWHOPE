import numpy as np
from astropy.io import fits

from newhope_pipeline.io.decompress import uncompress_directory


def _make_fz(path, value=1.0):
    data = np.full((32, 32), value, dtype=np.float32)
    hdul = fits.HDUList([fits.PrimaryHDU(header=fits.Header({"FILTER": "Ks"})), fits.CompImageHDU(data)])
    hdul.writeto(path, overwrite=True)


def test_uncompress_promotes_image(tmp_path):
    _make_fz(tmp_path / "a.fits.fz", 3.0)
    res = uncompress_directory(tmp_path)
    assert res.decompressed == 1 and not res.failed
    with fits.open(tmp_path / "a.fits") as h:
        assert h[0].data.shape == (32, 32)
        assert float(h[0].data[0, 0]) == 3.0
        assert h[0].header["FILTER"] == "Ks"
        assert "ZIMAGE" not in h[0].header
    assert (tmp_path / "a.fits.fz").exists()


def test_skips_existing_and_noop(tmp_path):
    assert uncompress_directory(tmp_path).found == 0  # nothing to do
    _make_fz(tmp_path / "a.fits.fz")
    uncompress_directory(tmp_path)
    again = uncompress_directory(tmp_path)
    assert again.skipped == 1 and again.decompressed == 0


def test_delete_original(tmp_path):
    _make_fz(tmp_path / "a.fits.fz")
    uncompress_directory(tmp_path, delete_original=True)
    assert not (tmp_path / "a.fits.fz").exists()
