from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from newhope_pipeline.core.exceptions import ReductionError
from newhope_pipeline.core.frame import FrameCollection, FrameReference
from newhope_pipeline.core.state import PipelineState
from newhope_pipeline.reduction.dark import DarkSubtraction


SHAPE = (8, 8)


def make_mef(
    path: Path,
    obstype: str,
    *,
    expcoadd=5,
    coadds=1,
    fsample=1,
    value_base=0,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    primary.header["OBSTYPE"] = obstype
    for key, value in (
        ("EXPCOADD", expcoadd),
        ("COADDS", coadds),
        ("FSAMPLE", fsample),
    ):
        if value is not None:
            primary.header[key] = value
    hdus = [primary]
    for detector in range(1, 5):
        data = np.full(SHAPE, value_base + detector, dtype=np.float32)
        hdus.append(fits.ImageHDU(data=data, name=f"DET{detector}"))
    fits.HDUList(hdus).writeto(path)
    return path


def reference(path):
    return FrameReference.from_fits(path, expected_extensions=4)


def test_calibrations_are_classified_from_obstype_not_filename(tmp_path):
    make_mef(tmp_path / "arbitrary_name_1.fits", "dark")
    make_mef(tmp_path / "dark_object.fits", "object")
    make_mef(tmp_path / "flat_from_archive.fits", "dflat")
    frames = FrameCollection.from_directory(tmp_path, expected_extensions=4)

    assert len(frames.by_obstype("DARK")) == 1
    assert next(iter(frames.by_obstype("dark"))).path.name == "arbitrary_name_1.fits"
    assert len(frames.by_obstype("dflat")) == 1
    assert len(frames.by_predicate(lambda frame: frame.obstype not in {"dark", "dflat"})) == 1


@pytest.mark.parametrize(
    "expcoadd,coadds,fsample",
    [
        (3, 1, 1),
        (3.0, 1.0, 1.0),
        ("3.0", "1.0", "1.0"),
    ],
)
def test_dark_key_normalizes_integer_float_and_string_header_values(
    tmp_path, expcoadd, coadds, fsample
):
    path = make_mef(
        tmp_path / "science.fits",
        "object",
        expcoadd=expcoadd,
        coadds=coadds,
        fsample=fsample,
    )
    assert DarkSubtraction._frame_dark_key(reference(path)) == (3.0, 1, 1)


@pytest.mark.parametrize("missing", ["EXPCOADD", "COADDS", "FSAMPLE"])
def test_missing_science_dark_metadata_names_required_keyword(tmp_path, missing):
    fields = {"EXPCOADD": 3, "COADDS": 1, "FSAMPLE": 1}
    fields[missing] = None
    path = make_mef(tmp_path / "science.fits", "object", **{k.lower(): v for k, v in fields.items()})
    with pytest.raises(ReductionError, match=f"{missing} is missing"):
        DarkSubtraction._frame_dark_key(reference(path))


def test_non_integral_coadds_is_reported_as_malformed(tmp_path):
    path = make_mef(tmp_path / "science.fits", "object", expcoadd=3, coadds=1.5, fsample=1)
    with pytest.raises(ReductionError, match="COADDS must be a positive integer"):
        DarkSubtraction._frame_dark_key(reference(path))


@pytest.mark.parametrize(
    "keyword,value,message",
    [
        ("EXPCOADD", "bad-value", "EXPCOADD has non-numeric value"),
        ("FSAMPLE", "bad-value", "FSAMPLE has non-numeric value"),
    ],
)
def test_non_numeric_dark_metadata_is_reported(tmp_path, keyword, value, message):
    fields = {"expcoadd": 3, "coadds": 1, "fsample": 1}
    fields[keyword.lower()] = value
    path = make_mef(tmp_path / "science.fits", "object", **fields)
    with pytest.raises(ReductionError, match=message):
        DarkSubtraction._frame_dark_key(reference(path))


def test_raw_darks_are_mean_combined_stored_and_retrieved_with_compatible_scaling(tmp_path):
    raw_dir = tmp_path / "raw"
    dark_refs = [
        reference(make_mef(raw_dir / f"random_cal_{i}.fits", "dark", expcoadd=5, coadds=2, value_base=10 * i))
        for i in (1, 2)
    ]
    science_ref = reference(
        make_mef(
            raw_dir / "science.fits",
            "object",
            expcoadd=2,
            coadds=2,
            fsample=1,
            value_base=100,
        )
    )
    stage = DarkSubtraction(
        SimpleNamespace(dark_dir=raw_dir),
        calibration_frames=dark_refs,
    )
    stage.output_dir = tmp_path / "processed" / "dark_subtraction"
    state = PipelineState(tmp_path / "pipeline_state.json")

    output = stage.run(FrameCollection([science_ref]), state)

    assert len(output) == 1
    corrected = FrameReference.from_fits(
        stage.output_dir / "science.fits", expected_extensions=4
    ).load()
    for detector in range(1, 5):
        # Raw means are 15 + detector, then EXPCOADD scaling is 2 / 5.
        expected = 100 + detector - (15 + detector) * 0.4
        assert np.allclose(corrected[detector].data, expected)

    master_path = stage.output_dir / "master_darks" / "dark_master_5s_2c_1f.fits"
    assert master_path.is_file()
    with fits.open(master_path) as hdul:
        assert hdul[0].header["OBSTYPE"] == "dark"
        assert hdul[0].header["PROCTYPE"] == "MasterCal"
        assert hdul[0].header["NCOMBINE"] == 2
        assert hdul[0].header["CALHASH"]
        assert np.allclose(hdul[1].data, 15 + 1)

    # A fresh stage retrieves the stored MEF master without raw calibration refs.
    reload_stage = DarkSubtraction(SimpleNamespace(dark_dir=raw_dir))
    reload_stage.output_dir = stage.output_dir
    loaded = reload_stage._load_master_dark(science_ref.load())
    assert np.allclose(loaded[1], (15 + 1) * 0.4)


@pytest.mark.parametrize(
    "dark_key",
    [
        (5, 1, 1),  # COADDS mismatch
        (5, 2, 2),  # FSAMPLE mismatch
    ],
)
def test_no_compatible_dark_does_not_fall_back_across_coadds_or_fsample(tmp_path, dark_key):
    raw_dir = tmp_path / "raw"
    dark = reference(
        make_mef(
            raw_dir / "dark_input.fits",
            "dark",
            expcoadd=dark_key[0],
            coadds=dark_key[1],
            fsample=dark_key[2],
        )
    )
    science = reference(
        make_mef(raw_dir / "science.fits", "object", expcoadd=2, coadds=2, fsample=1)
    ).load()
    stage = DarkSubtraction(SimpleNamespace(dark_dir=raw_dir), calibration_frames=[dark])

    with pytest.raises(ReductionError, match="no compatible master dark") as error:
        stage._load_master_dark(science)
    assert "available keys" in str(error.value)


def test_missing_master_dark_has_distinct_diagnostic(tmp_path):
    science = reference(
        make_mef(tmp_path / "science.fits", "object", expcoadd=2, coadds=2, fsample=1)
    ).load()
    stage = DarkSubtraction(SimpleNamespace(dark_dir=tmp_path))

    with pytest.raises(ReductionError, match="no master dark calibration available"):
        stage._load_master_dark(science)


def test_malformed_raw_dark_header_is_reported(tmp_path):
    raw = reference(
        make_mef(
            tmp_path / "raw_dark.fits",
            "dark",
            expcoadd=5,
            coadds=2,
            fsample=None,
        )
    )
    science = reference(
        make_mef(tmp_path / "science.fits", "object", expcoadd=2, coadds=2, fsample=1)
    ).load()
    stage = DarkSubtraction(SimpleNamespace(dark_dir=tmp_path), calibration_frames=[raw])

    with pytest.raises(ReductionError, match="FSAMPLE is missing"):
        stage._load_master_dark(science)
