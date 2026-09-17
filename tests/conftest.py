import numpy as np
import pytest
from astropy.io import fits


@pytest.fixture
def synthetic_frame_file(tmp_path):
    """A tiny synthetic FITS file with known values, for exact-value assertions."""
    data = np.full((32, 32), 100.0, dtype=np.float32)
    header = fits.Header()
    header["FILTER"] = "HX"
    header["EXPTIME"] = 30.0
    path = tmp_path / "synthetic.fits"
    fits.writeto(path, data, header)
    return path
