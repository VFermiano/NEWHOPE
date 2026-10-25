from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from newhope_pipeline.core.exceptions import ReductionError
from newhope_pipeline.core.frame import FrameCollection, FrameReference
from newhope_pipeline.core.pipeline import Pipeline
from newhope_pipeline.core.state import PipelineState
from newhope_pipeline.reduction.dark import DarkSubtraction
from newhope_pipeline.reduction.flat import FlatFielding


SHAPE = (8, 8)
PATTERN = np.full(SHAPE, 2.0, dtype=np.float32)
PATTERN[0, 0] = 1.0
DETECTOR_NAMES = {1: "SCA022", 2: "SCA011", 3: "SCA013", 4: "SCA019"}


def make_frame(
    path: Path,
    obstype: str,
    *,
    filt: str = "HX",
    data_by_detector=None,
    lamp=None,
    expcoadd=2,
    coadds=1,
    fsample=1,
    missing_filter=False,
    missing_lamp=False,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    primary.header["OBSTYPE"] = obstype
    if not missing_filter:
        primary.header["FILTER"] = filt
    if lamp is not None and not missing_lamp:
        primary.header["DFLAMPVA"] = lamp
    primary.header["EXPCOADD"] = expcoadd
    primary.header["COADDS"] = coadds
    primary.header["FSAMPLE"] = fsample
    hdus = [primary]
    for detector in range(1, 5):
        if data_by_detector is None:
            data = np.full(SHAPE, 100.0 * detector, dtype=np.float32)
        else:
            data = np.asarray(data_by_detector(detector), dtype=np.float32)
        hdu = fits.ImageHDU(data=data, name="COMPRESSED_IMAGE")
        hdu.header["DETECTOR"] = DETECTOR_NAMES[detector]
        hdu.header["DETSEC"] = f"chip{detector}"
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path)
    return path


def reference(path):
    return FrameReference.from_fits(path, expected_extensions=4)


def flat_refs(raw_dir: Path, filt: str = "HX"):
    refs = []
    # Two lamp levels verify the legacy per-exposure mode scaling before the
    # median combine. The lamp-off group has a uniform detector background.
    for index, level in enumerate((1.0, 2.0), start=1):
        refs.append(
            reference(
                make_frame(
                    raw_dir / f"archive_uuid_on_{index}.fits",
                    "dflat",
                    filt=filt,
                    lamp=100.0,
                    data_by_detector=lambda detector, level=level: (
                        np.full(SHAPE, 10.0 * detector * level, dtype=np.float32)
                        + PATTERN * detector * level
                    ),
                )
            )
        )
    for index, level in enumerate((1.0, 3.0), start=1):
        refs.append(
            reference(
                make_frame(
                    raw_dir / f"archive_uuid_off_{index}.fits",
                    "dflat",
                    filt=filt,
                    lamp=0.0,
                    data_by_detector=lambda detector, level=level: np.full(
                        SHAPE, 10.0 * detector * level, dtype=np.float32
                    ),
                )
            )
        )
    return refs


@pytest.mark.parametrize("filt", ["HX", "JX", "KXs"])
def test_raw_dflat_discovery_builds_per_extension_products_and_fielding(tmp_path, filt):
    raw_dir = tmp_path / "raw"
    flats = flat_refs(raw_dir, filt)
    science_ref = reference(
        make_frame(
            raw_dir / "random_archive_filename.fits",
            "object",
            filt=filt,
            data_by_detector=lambda detector: np.full(SHAPE, 100.0 * detector),
        )
    )
    discovered = FrameCollection.from_directory(raw_dir, expected_extensions=4)
    calibrations = discovered.by_obstype("dflat")
    science = discovered.by_predicate(lambda frame: frame.obstype not in {"dark", "dflat"})
    assert len(calibrations) == 4
    assert [frame.path.name for frame in science] == ["random_archive_filename.fits"]
    assert all(frame.obstype == "dflat" for frame in calibrations)

    # Use the classification result, as the CLI does, rather than filenames.
    stage = FlatFielding(SimpleNamespace(flat_dir=raw_dir), calibration_frames=calibrations)
    stage.output_dir = tmp_path / "processed" / "flat_fielding"
    output = stage.run(science, PipelineState(tmp_path / "pipeline_state.json"))

    assert len(output) == 1
    corrected = next(iter(output)).load()
    shared_median = 4.0  # median over all restored detector pixels, including PATTERN's low pixel
    for detector in range(1, 5):
        product = stage.output_dir / "master_flats" / f"nflat_{filt}_{detector}.fits"
        assert product.is_file()
        with fits.open(product) as hdul:
            assert hdul[0].header["FILTER"] == filt
            assert hdul[0].header["DETINDEX"] == detector
            assert hdul[0].header["DETECTOR"] == DETECTOR_NAMES[detector]
            assert hdul[0].header["FLATNORM"] == pytest.approx(2.0 * detector)
            assert hdul[0].data.shape == SHAPE
        expected_master = PATTERN * detector / shared_median
        np.testing.assert_allclose(corrected[detector].data, 100.0 * detector / expected_master)


def test_flat_products_are_reused_and_filtered_sets_do_not_mix(tmp_path):
    raw_dir = tmp_path / "raw"
    refs = flat_refs(raw_dir, "KXs")
    science = reference(make_frame(raw_dir / "science.fits", "object", filt="KXs"))
    stage = FlatFielding(SimpleNamespace(flat_dir=raw_dir), calibration_frames=refs)
    stage.output_dir = tmp_path / "processed" / "flat_fielding"
    state = PipelineState(tmp_path / "pipeline_state.json")
    stage.run(FrameCollection([science]), state)

    # A new stage can load the products without raw calibration references.
    reloaded = FlatFielding(SimpleNamespace(flat_dir=raw_dir))
    reloaded.output_dir = stage.output_dir
    loaded = reloaded._load_master_flat("kXs")
    assert set(loaded) == {1, 2, 3, 4}
    assert (stage.output_dir / "master_flats" / "nflat_KXs_4.fits").is_file()
    product_names = {path.name for path in (stage.output_dir / "master_flats").glob("*.fits")}
    assert "nflat_KXs_4.fits" in product_names
    assert len(product_names) == 4


def test_flat_mode_ignores_untrimmed_zero_padding(tmp_path):
    def data(detector):
        values = np.arange(1, SHAPE[0] * SHAPE[1] + 1, dtype=np.float32)
        if detector == 3:
            values[:12] = 0.0
        return values.reshape(SHAPE)

    raw = reference(
        make_frame(
            tmp_path / "zero_padded_dflat.fits",
            "dflat",
            lamp=100.0,
            data_by_detector=data,
        )
    )

    assert FlatFielding._flat_mode(raw, 3) == 13.0


def test_flat_mode_uses_active_width_from_detsec(tmp_path):
    def data(detector):
        values = np.zeros(SHAPE, dtype=np.float32)
        values[:, :4] = 100.0 + detector
        return values

    path = make_frame(
        tmp_path / "overscan_dflat.fits",
        "dflat",
        lamp=100.0,
        data_by_detector=data,
    )
    with fits.open(path, mode="update") as hdul:
        hdul[3].header["DETSEC"] = "[1:4,1:8]"
    raw = reference(path)

    assert FlatFielding._flat_mode(raw, 3) == 103.0


@pytest.mark.parametrize("layout", ["root", "detector_directories"])
def test_complete_precomputed_flat_set_remains_supported(tmp_path, layout):
    flat_dir = tmp_path / "calibrations"
    for detector in range(1, 5):
        directory = flat_dir if layout == "root" else flat_dir / str(detector)
        directory.mkdir(parents=True, exist_ok=True)
        header = fits.Header()
        header["FILTER"] = "HX"
        header["FLATNORM"] = float(detector)
        fits.PrimaryHDU(
            data=np.full(SHAPE, float(detector), dtype=np.float32), header=header
        ).writeto(directory / f"nflat_HX_{detector}.fits")

    stage = FlatFielding(SimpleNamespace(flat_dir=flat_dir))
    masters = stage._load_master_flat("hx")

    for detector in range(1, 5):
        np.testing.assert_allclose(masters[detector], detector**2 / 6.5)


def test_missing_all_flat_sources_lists_expected_products(tmp_path):
    stage = FlatFielding(SimpleNamespace(flat_dir=tmp_path / "calibrations"))

    with pytest.raises(ReductionError, match="no raw OBSTYPE=dflat frames were supplied") as error:
        stage._load_master_flat("HX")

    assert "nflat_HX_1.fits" in str(error.value)


def test_failed_master_construction_is_distinguished_from_missing_calibrations(tmp_path):
    raw_dir = tmp_path / "raw"
    on = reference(
        make_frame(
            raw_dir / "lamp_on.fits",
            "dflat",
            lamp=100.0,
            data_by_detector=lambda detector: np.full(SHAPE, 10.0),
        )
    )
    off = reference(
        make_frame(
            raw_dir / "lamp_off.fits",
            "dflat",
            lamp=0.0,
            data_by_detector=lambda detector: np.full(SHAPE, 20.0),
        )
    )
    science = reference(make_frame(raw_dir / "science.fits", "object"))
    stage = FlatFielding(SimpleNamespace(flat_dir=raw_dir), calibration_frames=[on, off])
    stage.output_dir = tmp_path / "processed" / "flat_fielding"
    state = PipelineState(tmp_path / "pipeline_state.json")
    stage._prepare_raw_master_flats(FrameCollection([science]), state)

    with pytest.raises(ReductionError, match="master-flat construction failed") as error:
        stage._load_master_flat("HX")
    assert "invalid median" in str(error.value)


def test_missing_flat_groups_report_on_and_off_counts_and_expected_products(tmp_path):
    raw_dir = tmp_path / "raw"
    only_on = reference(
        make_frame(raw_dir / "uuid.fits", "dflat", filt="JX", lamp=1.0)
    )
    science = reference(make_frame(raw_dir / "science.fits", "object", filt="JX"))
    stage = FlatFielding(SimpleNamespace(flat_dir=raw_dir), calibration_frames=[only_on])
    stage.output_dir = tmp_path / "processed" / "flat_fielding"
    stage.run(FrameCollection([science]), PipelineState(tmp_path / "state.json"))

    with pytest.raises(ReductionError, match="lamp-on=1, lamp-off=0") as error:
        stage._load_master_flat("JX")
    assert "nflat_JX_1.fits" in str(error.value)


@pytest.mark.parametrize(
    "missing,expected",
    [("filter", "FILTER is missing"), ("lamp", "DFLAMPVA is missing")],
)
def test_missing_flat_header_values_are_reported(tmp_path, missing, expected):
    raw_dir = tmp_path / "raw"
    bad = reference(
        make_frame(
            raw_dir / "arbitrary_name.fits",
            "dflat",
            lamp=1.0,
            missing_filter=missing == "filter",
            missing_lamp=missing == "lamp",
        )
    )
    science = reference(make_frame(raw_dir / "science.fits", "object", filt="HX"))
    stage = FlatFielding(SimpleNamespace(flat_dir=raw_dir), calibration_frames=[bad])
    stage.output_dir = tmp_path / "processed" / "flat_fielding"
    stage.run(FrameCollection([science]), PipelineState(tmp_path / "state.json"))

    with pytest.raises(ReductionError, match=expected):
        stage._load_master_flat("HX")


def test_pipeline_builds_darks_then_flats_before_fielding_science(tmp_path):
    raw_dir = tmp_path / "raw"
    dark = reference(
        make_frame(
            raw_dir / "calibration_a.fits",
            "dark",
            data_by_detector=lambda detector: np.full(SHAPE, float(detector)),
        )
    )
    flats = flat_refs(raw_dir)
    science = reference(
        make_frame(
            raw_dir / "science_uuid.fits",
            "object",
            data_by_detector=lambda detector: np.full(SHAPE, 100.0 * detector),
        )
    )
    dark_stage = DarkSubtraction(SimpleNamespace(dark_dir=raw_dir), calibration_frames=[dark])
    flat_stage = FlatFielding(SimpleNamespace(flat_dir=raw_dir), calibration_frames=flats)
    dark_stage.output_dir = tmp_path / "processed" / dark_stage.name
    flat_stage.output_dir = tmp_path / "processed" / flat_stage.name

    result = Pipeline(
        [dark_stage, flat_stage], manifest_path=tmp_path / "pipeline_state.json"
    ).run(FrameCollection([science]))

    assert len(result) == 1
    assert (dark_stage.output_dir / "master_darks" / "dark_master_2s_1c_1f.fits").is_file()
    assert (flat_stage.output_dir / "master_flats" / "nflat_HX_1.fits").is_file()
    final = next(iter(result)).load()
    for detector in range(1, 5):
        assert np.all(np.isfinite(final[detector].data))

