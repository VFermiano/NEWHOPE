import numpy as np
import pytest
from astropy.io import fits


N_EXT = 4
SHAPE = (32, 32)


def make_mef(path, fill=lambda k: 100.0 * k, n_ext=N_EXT, filt="HX", exptime=30.0):
    """Synthetic NEWFIRM-like MEF: empty primary + n_ext image extensions.

    Extension k is filled with fill(k) (default 100*k) so tests can tell the
    detectors apart and catch any mixing between them.
    """
    primary = fits.PrimaryHDU()
    primary.header["FILTER"] = filt
    primary.header["EXPTIME"] = exptime
    hdus = [primary]
    for k in range(1, n_ext + 1):
        data = np.full(SHAPE, fill(k), dtype=np.float32)
        hdu = fits.ImageHDU(data=data, name=f"im{k}")
        hdu.header["DETSEC"] = f"chip{k}"
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path)
    return path


@pytest.fixture
def synthetic_frame_file(tmp_path):
    """A tiny synthetic 4-extension FITS file with known per-detector values."""
    return make_mef(tmp_path / "synthetic.fits")

