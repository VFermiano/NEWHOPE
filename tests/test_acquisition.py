from unittest.mock import MagicMock

import pytest

from newhope_pipeline.acquisition.noirlab import NoirlabClient, download_night
from newhope_pipeline.config import AcquisitionConfig
from newhope_pipeline.core.exceptions import AcquisitionError


def _mock_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = text
    return resp


def test_search_calibrations_filters_by_obs_type():
    client = NoirlabClient()
    client.session = MagicMock()
    rows = [
        {"obs_type": "dflat", "archive_filename": "a.fits"},
        {"obs_type": "object", "archive_filename": "b.fits"},
        {"obs_type": "dark", "archive_filename": "c.fits"},
    ]
    client.session.post.return_value = _mock_response(json_data=[{"meta": True}, *rows])

    result = client.search_calibrations("newfirm", "ct4m", "2019-06-07", "raw", 5000)

    assert {r["archive_filename"] for r in result} == {"a.fits", "c.fits"}


def test_search_before_login_raises_acquisition_error():
    client = NoirlabClient()
    with pytest.raises(AcquisitionError):
        client.search_science("newfirm", "ct4m", "2019A-0305", "2019-06-07", "raw", 5000)


def test_search_http_error_raises_acquisition_error():
    client = NoirlabClient()
    client.session = MagicMock()
    client.session.post.return_value = _mock_response(status_code=500, text="server error")

    with pytest.raises(AcquisitionError):
        client.search_science("newfirm", "ct4m", "2019A-0305", "2019-06-07", "raw", 5000)


def test_search_temporarily_removes_authorization_header():
    client = NoirlabClient()
    client.session = MagicMock()
    client.session.headers = {"Authorization": "Bearer sometoken", "Referer": "https://astroarchive.noirlab.edu"}
    client.session.post.return_value = _mock_response(json_data=[{"meta": True}])

    client.search_science("newfirm", "ct4m", "2019A-0305", "2019-06-07", "raw", 5000)

    assert client.session.post.call_args.kwargs["json"]["search"][0] == ["instrument", "newfirm"]
    assert client.session.headers["Authorization"] == "Bearer sometoken"


def test_download_file_skips_when_checksum_matches(tmp_path):
    client = NoirlabClient()
    client.session = MagicMock()

    dest_dir = tmp_path
    existing = dest_dir / "frame.fits"
    existing.write_bytes(b"hello")
    import hashlib

    md5sum = hashlib.md5(b"hello").hexdigest()

    row = {"archive_filename": "frame.fits", "md5sum": md5sum}
    ok = client.download_file(row, dest_dir)

    assert ok is True
    client.session.get.assert_not_called()


def test_download_file_flips_auth_scheme_on_401(tmp_path):
    client = NoirlabClient()
    client.session = MagicMock()
    client.session.headers = {"Authorization": "Bearer sometoken"}

    unauthorized = _mock_response(status_code=401, text="unauthorized")
    success = _mock_response(status_code=200)
    success.iter_content.return_value = [b"data"]
    client.session.get.side_effect = [unauthorized, success]

    row = {"archive_filename": "frame.fits", "md5sum": None}
    ok = client.download_file(row, tmp_path)

    assert ok is True
    assert client.session.headers["Authorization"].startswith("Token")


def test_download_file_uses_stored_token(tmp_path):
    client = NoirlabClient()
    client.session = MagicMock()
    client.session.headers = {}
    client.token = "sometoken"
    success = _mock_response(status_code=200)
    success.iter_content.return_value = [b"data"]
    client.session.get.return_value = success

    row = {"archive_filename": "frame.fits", "md5sum": None}
    ok = client.download_file(row, tmp_path)

    assert ok is True
    assert client.session.headers["Authorization"] == "Bearer sometoken"


def test_download_night_requires_proposal_and_night():
    cfg = AcquisitionConfig(proposal=None, night=None)
    with pytest.raises(AcquisitionError):
        download_night(cfg, outdir=None, dry_run=True)


def test_acquisition_config_defaults():
    cfg = AcquisitionConfig(proposal="2019A-0305", night="2019-06-07")
    assert cfg.instrument == "newfirm"
    assert cfg.telescope == "ct4m"
    assert cfg.proctype == "raw"
    assert cfg.skip_calibrations is False
