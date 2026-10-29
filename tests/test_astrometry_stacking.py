from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from typer.testing import CliRunner

from newhope_pipeline.astrometry import (
    calibrate,
    scamp_wrapper,
    source_extractor,
)
from newhope_pipeline.astrometry import (
    stack as stack_module,
)
from newhope_pipeline.astrometry.calibrate import AstrometricCalibration
from newhope_pipeline.astrometry.stack import ImageStacking
from newhope_pipeline.cli import app
from newhope_pipeline.config import AstrometryConfig, PipelineConfig, StackingConfig
from newhope_pipeline.core.exceptions import AstrometryError, ScampTimeoutError
from newhope_pipeline.core.frame import FrameCollection, FrameReference
from newhope_pipeline.core.state import PipelineState

DETECTORS = ("SCA022", "SCA011", "SCA013", "SCA019")


def _wcs_header(shape=(16, 20), ra=150.0):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [(shape[1] + 1) / 2, (shape[0] + 1) / 2]
    wcs.wcs.crval = [ra, -30.0]
    wcs.wcs.cdelt = [-0.0001, 0.0001]
    return wcs.to_header(relax=True)


def _make_sky_mef(
    path: Path,
    *,
    filt="HX",
    target="Trumpler19",
    ra=150.0,
    expcoadd=3,
    coadds=1,
    fsample=1,
    shape=(16, 20),
):
    primary = fits.Header()
    primary["OBSTYPE"] = "object"
    primary["SKYSUB"] = True
    primary["ASTROMET"] = True
    primary["FILTER"] = filt
    primary["OBJECT"] = target
    primary["EXPTIME"] = 24.0
    primary["EXPCOADD"] = expcoadd
    primary["COADDS"] = coadds
    primary["FSAMPLE"] = fsample
    hdus = [fits.PrimaryHDU(header=primary)]
    for index, detector in enumerate(DETECTORS, start=1):
        header = _wcs_header(shape, ra + (index - 1) * 0.001)
        header["DETECTOR"] = detector
        header["EXTNAME"] = detector
        hdus.append(
            fits.ImageHDU(
                np.full(shape, index, dtype=np.float32), header=header, name=detector
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def _refs(paths):
    return FrameCollection(
        [FrameReference.from_fits(path, expected_extensions=4) for path in paths]
    )


def _stack_config(tmp_path, **overrides):
    config_path = tmp_path / "swarp.conf"
    config_path.write_text("# test SWarp config\n")
    values = {"swarp_config_path": config_path, **overrides}
    return StackingConfig(**values)


def _fake_swarp(monkeypatch, calls):
    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert "-NTHREADS" in command and command[command.index("-NTHREADS") + 1] == "1"
        assert (
            "-COMBINE_TYPE" in command
            and command[command.index("-COMBINE_TYPE") + 1] == "MEDIAN"
        )
        assert (
            "-PIXELSCALE_TYPE" in command
            and command[command.index("-PIXELSCALE_TYPE") + 1] == "MEDIAN"
        )
        imageout = Path(command[command.index("-IMAGEOUT_NAME") + 1])
        input_paths = [Path(value) for value in command[1 : command.index("-c")]]
        assert input_paths
        for image in input_paths:
            with fits.open(image, memmap=False) as hdul:
                assert len(hdul) == 1
                assert hdul[0].data is not None and hdul[0].data.ndim == 2
                assert hdul[0].header["DETECTOR"]
        with fits.open(input_paths[0], memmap=False) as hdul:
            header = hdul[0].header.copy()
            shape = hdul[0].data.shape
        fits.PrimaryHDU(np.ones(shape, dtype=np.float32), header=header).writeto(
            imageout
        )
        return subprocess.CompletedProcess(command, 0, "", "mock swarp ok")

    monkeypatch.setattr(stack_module.subprocess, "run", run)
    monkeypatch.setattr(stack_module, "resolve_executable", lambda *_: "/mock/swarp")


def test_grouping_uses_filter_target_detector_and_four_hdu_mapping(tmp_path):
    paths = (
        [
            _make_sky_mef(tmp_path / f"hx-{i}.fits", filt="HX", ra=150 + i * 0.001)
            for i in range(2)
        ]
        + [
            _make_sky_mef(tmp_path / f"jx-{i}.fits", filt="JX", ra=151 + i * 0.001)
            for i in range(2)
        ]
        + [
            _make_sky_mef(
                tmp_path / f"other-{i}.fits",
                filt="HX",
                target="Other field",
                ra=152 + i * 0.001,
            )
            for i in range(2)
        ]
    )
    refs = _refs(paths)
    assert all(frame.header["NAXIS"] == 0 and frame.n_ext == 4 for frame in refs)
    stage = ImageStacking(_stack_config(tmp_path))
    stage.output_dir = tmp_path / "products"
    groups, errors = stage._groups(refs)

    assert not errors
    assert len(groups) == 12
    assert {group.filt for group in groups} == {"HX", "JX"}
    assert {group.target for group in groups} == {"Trumpler19", "Other field"}
    assert {group.detector for group in groups} == set(DETECTORS)
    assert {group.extension_index for group in groups} == {1, 2, 3, 4}
    assert all(len(group.inputs) == 2 for group in groups)


def test_numeric_acquisition_values_are_normalized_and_incompatible_setup_is_separate(
    tmp_path,
):
    paths = [
        _make_sky_mef(tmp_path / "int.fits", expcoadd=3, coadds=1, fsample=1),
        _make_sky_mef(tmp_path / "float.fits", expcoadd=3.0, coadds=1.0, fsample=1.0),
        _make_sky_mef(
            tmp_path / "string.fits", expcoadd="3.0", coadds="1.0", fsample="1.0"
        ),
        _make_sky_mef(tmp_path / "different.fits", expcoadd=5, coadds=1, fsample=1),
    ]
    stage = ImageStacking(_stack_config(tmp_path))
    stage.output_dir = tmp_path / "products"
    groups, errors = stage._groups(_refs(paths))

    assert not errors
    assert len(groups) == 8
    assert sorted(len(group.inputs) for group in groups) == [1] * 4 + [3] * 4
    assert {dict(group.acquisition)["EXPCOADD"] for group in groups} == {"3", "5"}


def test_missing_required_metadata_is_reported(tmp_path):
    path = _make_sky_mef(tmp_path / "missing.fits")
    with fits.open(path, mode="update") as hdul:
        del hdul[0].header["OBJECT"]
        del hdul[0].header["FSAMPLE"]
    stage = ImageStacking(_stack_config(tmp_path))
    stage.output_dir = tmp_path / "products"
    groups, errors = stage._groups(_refs([path]))

    assert not groups
    reason = next(iter(errors.values()))
    assert "science target metadata" in reason


def test_missing_acquisition_setting_is_reported(tmp_path):
    path = _make_sky_mef(tmp_path / "missing-setting.fits")
    with fits.open(path, mode="update") as hdul:
        del hdul[0].header["FSAMPLE"]
    stage = ImageStacking(_stack_config(tmp_path))
    stage.output_dir = tmp_path / "products"
    groups, errors = stage._groups(_refs([path]))

    assert not groups
    assert "missing FSAMPLE" in next(iter(errors.values()))


def test_sextractor_command_uses_ldac_config_and_checks_output(tmp_path, monkeypatch):
    image = tmp_path / "one.fits"
    fits.PrimaryHDU(np.ones((10, 10), dtype=np.float32)).writeto(image)
    config = tmp_path / "default.sex"
    params = tmp_path / "default.param"
    filt = tmp_path / "default.conv"
    for path in (config, params, filt):
        path.write_text("test\n")
    catalog = tmp_path / "one.cat"
    captured = {}
    monkeypatch.setattr(source_extractor, "resolve_executable", lambda *_: "/mock/sex")

    def run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        Path(command[command.index("-CATALOG_NAME") + 1]).write_bytes(b"ldac")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(source_extractor.subprocess, "run", run)
    result = source_extractor.run_sextractor(
        image,
        catalog,
        executable="sex",
        config_path=config,
        parameters_path=params,
        filter_path=filt,
        threshold_sigma=5.0,
        minimum_area=5,
        timeout_seconds=30,
    )

    assert result == catalog
    assert (
        captured["command"][captured["command"].index("-CATALOG_TYPE") + 1]
        == "FITS_LDAC"
    )
    assert captured["command"][
        captured["command"].index("-PARAMETERS_NAME") + 1
    ] == str(params.resolve())
    assert captured["command"][captured["command"].index("-NTHREADS") + 1] == "1"
    assert captured["kwargs"]["timeout"] == 30


def test_scamp_command_and_header_output(tmp_path, monkeypatch):
    catalog = tmp_path / "one.cat"
    catalog.write_bytes(b"catalog")
    config = tmp_path / "scamp.conf"
    config.write_text("ASTREF_CATALOG GAIA-EDR3\n")
    captured = {}
    monkeypatch.setattr(scamp_wrapper, "resolve_executable", lambda *_: "/mock/scamp")
    monkeypatch.setattr(
        scamp_wrapper,
        "supported_reference_catalogs",
        lambda *_: frozenset({"GAIA-EDR3"}),
    )

    def run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        (tmp_path / "one.head").write_text("SIMPLE  =                    T\nEND\n")
        return subprocess.CompletedProcess(command, 0, "scamp ok", "")

    monkeypatch.setattr(scamp_wrapper.subprocess, "run", run)
    head = scamp_wrapper.run_scamp(
        catalog,
        config,
        executable="scamp",
        timeout_seconds=12,
        parameters={"ASTREF_CATALOG": "GAIA-EDR3"},
    )

    assert head == tmp_path / "one.head"
    assert "-ASTREF_CATALOG" in captured["command"]
    assert (
        captured["command"][captured["command"].index("-ASTREF_CATALOG") + 1]
        == "GAIA-EDR3"
    )
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["kwargs"]["timeout"] == 12


def test_scamp_reference_catalog_choices_are_parsed_from_config_dump():
    scamp_defaults = """
    ASTREF_CATALOG 2MASS # NONE,FILE,USNO-A2,GAIA-DR2,
                          # GAIA-EDR3, or ALLWISE
    ASTREF_BAND DEFAULT # DEFAULT, BLUEST, or REDDEST
    """
    choices = scamp_wrapper.parse_supported_reference_catalogs(scamp_defaults)

    assert {"2MASS", "GAIA-DR2", "GAIA-EDR3", "ALLWISE"} <= choices
    assert "GAIA-DR3" not in choices


def test_unsupported_reference_catalog_has_version_specific_diagnostic(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(scamp_wrapper, "resolve_executable", lambda *_: "/mock/scamp")
    monkeypatch.setattr(
        scamp_wrapper,
        "supported_reference_catalogs",
        lambda *_: frozenset({"GAIA-EDR3", "2MASS"}),
    )
    catalog = tmp_path / "one.cat"
    catalog.write_bytes(b"catalog")
    config = tmp_path / "scamp.conf"
    config.write_text("ASTREF_CATALOG GAIA-EDR3\n")

    with pytest.raises(
        AstrometryError, match="Unsupported astrometry.reference_catalog"
    ) as exc:
        scamp_wrapper.run_scamp(
            catalog,
            config,
            parameters={"ASTREF_CATALOG": "GAIA-DR3"},
        )
    assert "2MASS, GAIA-EDR3" in str(exc.value)


def test_reference_catalog_defaults_match_config_file_and_yaml():
    project_root = Path(__file__).resolve().parents[1]
    config = PipelineConfig.from_yaml(project_root / "configs/default.yaml")
    scamp_config = (project_root / "configs/scamp/default.scamp").read_text()

    assert AstrometryConfig().reference_catalog == "GAIA-EDR3"
    assert config.astrometry.reference_catalog == "GAIA-EDR3"
    assert "ASTREF_CATALOG         GAIA-EDR3" in scamp_config


def test_astrometry_preflight_rejects_catalog_not_supported_by_configured_scamp(
    tmp_path, monkeypatch
):
    config_paths = [
        tmp_path / name for name in ("scamp.conf", "sex.conf", "sex.param", "sex.conv")
    ]
    for path in config_paths:
        path.write_text("# test config\n")
    config = AstrometryConfig(
        scamp_config_path=config_paths[0],
        sextractor_config_path=config_paths[1],
        sextractor_parameters_path=config_paths[2],
        sextractor_filter_path=config_paths[3],
        reference_catalog="GAIA-DR3",
    )
    monkeypatch.setattr(
        scamp_wrapper,
        "resolve_executable",
        lambda executable, name: f"/mock/{name.casefold()}",
    )
    monkeypatch.setattr(
        scamp_wrapper,
        "supported_reference_catalogs",
        lambda *_: frozenset({"GAIA-EDR3", "2MASS"}),
    )

    with pytest.raises(
        AstrometryError, match="Unsupported astrometry.reference_catalog 'GAIA-DR3'"
    ):
        calibrate.check_astrometry_tools(config)


def test_scamp_timeout_and_nonzero_exit_are_explicit(tmp_path, monkeypatch):
    catalog = tmp_path / "one.cat"
    catalog.write_bytes(b"catalog")
    config = tmp_path / "scamp.conf"
    config.write_text("# test\n")
    monkeypatch.setattr(scamp_wrapper, "resolve_executable", lambda *_: "/mock/scamp")
    monkeypatch.setattr(
        scamp_wrapper.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("scamp", 1)
        ),
    )
    with pytest.raises(ScampTimeoutError):
        scamp_wrapper.run_scamp(catalog, config, timeout_seconds=1)

    monkeypatch.setattr(
        scamp_wrapper.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 2, "", "bad catalog"
        ),
    )
    with pytest.raises(AstrometryError, match="bad catalog"):
        scamp_wrapper.run_scamp(catalog, config)


def test_missing_external_executable_is_actionable(monkeypatch):
    monkeypatch.setattr(scamp_wrapper.shutil, "which", lambda _: None)
    with pytest.raises(AstrometryError, match="Install SCAMP"):
        scamp_wrapper.resolve_executable("scamp", "SCAMP")


def test_invalid_or_missing_wcs_fails_astrometry_validation():
    with pytest.raises(AstrometryError, match="no usable celestial WCS"):
        calibrate._valid_wcs(fits.Header(), (32, 32), "unwcs.fits")


def _write_scamp_head(path, cards, end_card="END     "):
    path.write_bytes(("\n".join([*cards, end_card]) + "\n").encode("utf-8"))
    return path


def test_scamp_head_reader_accepts_ascii_cards_and_requires_end(tmp_path):
    path = _write_scamp_head(
        tmp_path / "ascii.head",
        [fits.Card("CRVAL1", 151.25, "World coordinate").image],
    )

    header = calibrate._read_scamp_header(path)

    assert header["CRVAL1"] == pytest.approx(151.25)


def test_scamp_head_reader_normalizes_non_ascii_comment_only(tmp_path, caplog):
    path = _write_scamp_head(
        tmp_path / "unicode-comment.head",
        [
            "COMMENT   Reference catalog from Université".ljust(80),
            fits.Card("CRVAL1", 151.25, "World coordinate").image,
        ],
    )

    header = calibrate._read_scamp_header(path)

    assert "Universit?" in str(header["COMMENT"])
    assert header["CRVAL1"] == pytest.approx(151.25)
    assert "unicode-comment.head" in caplog.text
    assert "COMMENT" in caplog.text


def test_scamp_head_reader_rejects_non_ascii_wcs_values(tmp_path, caplog):
    path = _write_scamp_head(
        tmp_path / "unicode-wcs-value.head",
        ["CTYPE1  = 'RA---TÁN'".ljust(80)],
    )

    with pytest.raises(AstrometryError, match="non-ASCII.*astrometric value"):
        calibrate._read_scamp_header(path)

    assert "unicode-wcs-value.head" in caplog.text
    assert "CTYPE1" in caplog.text


@pytest.mark.parametrize(
    ("end_card", "cards", "message"),
    [
        (None, [fits.Card("CRVAL1", 151.25).image], "missing required END"),
        ("END     trailing data", [], "malformed END card"),
    ],
)
def test_scamp_head_reader_rejects_missing_or_malformed_end(
    tmp_path, end_card, cards, message
):
    path = tmp_path / "bad-end.head"
    if end_card is None:
        path.write_text("\n".join(cards) + "\n", encoding="ascii")
    else:
        _write_scamp_head(path, cards, end_card=end_card)

    with pytest.raises(AstrometryError, match=message):
        calibrate._read_scamp_header(path)


def test_astrometry_calibrates_all_extensions_and_preserves_detector_wcs(
    tmp_path, monkeypatch
):
    source = _make_sky_mef(tmp_path / "sky.fits")
    frame = FrameReference.from_fits(source, expected_extensions=4).load()
    stage = AstrometricCalibration(
        AstrometryConfig(
            scamp_config_path=tmp_path / "scamp.conf",
            sextractor_config_path=tmp_path / "sex.conf",
            sextractor_parameters_path=tmp_path / "sex.param",
            sextractor_filter_path=tmp_path / "sex.conv",
            keep_intermediates=True,
        )
    )
    stage.config.scamp_config_path.write_text("# scamp\n")
    for path in (
        stage.config.sextractor_config_path,
        stage.config.sextractor_parameters_path,
        stage.config.sextractor_filter_path,
    ):
        path.write_text("# sex\n")
    stage.output_dir = tmp_path / "astrometry"
    calls = {"sex": 0, "scamp": 0}
    configured_catalogs = []

    def fake_sex(image, catalog, **kwargs):
        calls["sex"] += 1
        catalog.write_bytes(b"fake fits-ldac catalog")
        return catalog

    def fake_scamp(catalog, config_path, **kwargs):
        calls["scamp"] += 1
        configured_catalogs.append(kwargs["parameters"]["ASTREF_CATALOG"])
        head = catalog.with_suffix(".head")
        calibrated = _wcs_header((16, 20), ra=151.0 + calls["scamp"] * 0.01)
        cards = [
            "COMMENT   SCAMP reference catalog from Université".ljust(80),
            *(card.image for card in calibrated.cards),
            "END".ljust(80),
        ]
        head.write_bytes(("\n".join(cards) + "\n").encode("utf-8"))
        return head

    monkeypatch.setattr(calibrate, "run_sextractor", fake_sex)
    monkeypatch.setattr(calibrate.scamp_wrapper, "run_scamp", fake_scamp)
    result = stage.process(frame)

    assert calls == {"sex": 4, "scamp": 4}
    assert configured_catalogs == ["GAIA-EDR3"] * 4
    assert result.header["ASTROMET"] is True
    assert [result[index].header["DETECTOR"] for index in range(1, 5)] == list(
        DETECTORS
    )
    assert [result[index].header["CRVAL1"] for index in range(1, 5)] == pytest.approx(
        [151.01, 151.02, 151.03, 151.04]
    )
    assert [result[index].header["CRVAL1"] for index in range(1, 5)] != [
        frame[index].header["CRVAL1"] for index in range(1, 5)
    ]
    assert (
        len(
            list(
                (stage.output_dir / "intermediates" / frame.frame_id).glob("detector_*")
            )
        )
        == 4
    )


def test_astrometry_resume_reuses_matching_products_and_invalidates_changed_settings(
    tmp_path, monkeypatch
):
    source = _make_sky_mef(tmp_path / "sky.fits")
    refs = _refs([source])
    config = AstrometryConfig(
        scamp_config_path=tmp_path / "scamp.conf",
        sextractor_config_path=tmp_path / "sex.conf",
        sextractor_parameters_path=tmp_path / "sex.param",
        sextractor_filter_path=tmp_path / "sex.conv",
    )
    config.scamp_config_path.write_text("# scamp\n")
    for path in (
        config.sextractor_config_path,
        config.sextractor_parameters_path,
        config.sextractor_filter_path,
    ):
        path.write_text("# sex\n")
    calls = {"sex": 0, "scamp": 0}

    def fake_sex(image, catalog, **kwargs):
        calls["sex"] += 1
        catalog.write_bytes(b"fake fits-ldac catalog")
        return catalog

    def fake_scamp(catalog, config_path, **kwargs):
        calls["scamp"] += 1
        head = catalog.with_suffix(".head")
        _wcs_header((16, 20), ra=151.0).totextfile(
            str(head), endcard=True, overwrite=True
        )
        return head

    monkeypatch.setattr(calibrate, "run_sextractor", fake_sex)
    monkeypatch.setattr(calibrate.scamp_wrapper, "run_scamp", fake_scamp)
    manifest = tmp_path / "state.json"
    output_dir = tmp_path / "astrometry"

    first = AstrometricCalibration(config)
    first.output_dir = output_dir
    first.run(refs, PipelineState(manifest))
    assert calls == {"sex": 4, "scamp": 4}

    resumed = AstrometricCalibration(config)
    resumed.output_dir = output_dir
    resumed.run(refs, PipelineState(manifest))
    assert calls == {"sex": 4, "scamp": 4}

    changed_config = config.model_copy(update={"detection_threshold_sigma": 6.0})
    changed = AstrometricCalibration(changed_config)
    changed.output_dir = output_dir
    changed.run(refs, PipelineState(manifest))
    assert calls == {"sex": 8, "scamp": 8}


def test_stacking_extracts_per_detector_and_writes_stack_products(
    tmp_path, monkeypatch
):
    paths = [
        _make_sky_mef(tmp_path / f"science-{i}.fits", ra=150 + i * 0.001)
        for i in range(2)
    ]
    stage = ImageStacking(_stack_config(tmp_path, minimum_exposures=2))
    stage.output_dir = tmp_path / "processed" / "stacking"
    calls = []
    _fake_swarp(monkeypatch, calls)

    result = stage.run(_refs(paths), PipelineState(tmp_path / "state.json"))

    assert len(result) == 2
    assert len(calls) == 4
    products = list(stage.output_dir.glob("stack_*.fits"))
    assert len(products) == 4
    with fits.open(products[0], memmap=False) as hdul:
        header = hdul[0].header
        assert hdul[0].data.ndim == 2
        assert header["STACK"] is True
        assert header["NSTACK"] == 2
        assert header["FILTER"] == "HX"
        assert header["OBJECT"] == "Trumpler19"
        assert header["DETECTOR"] in DETECTORS
        assert header["EXTINDEX"] in (1, 2, 3, 4)
        assert header["COMBMETH"] == "MEDIAN"
    manifest = __import__("json").loads(
        (stage.output_dir / "stack_manifest.json").read_text()
    )
    assert manifest["successful_frames"] == 2
    assert len(manifest["groups"]) == 4
    assert all(group["status"] == "done" for group in manifest["groups"])


def test_stacking_resumes_valid_output_and_invalidates_changed_input(
    tmp_path, monkeypatch
):
    paths = [
        _make_sky_mef(tmp_path / f"science-{i}.fits", ra=150 + i * 0.001)
        for i in range(2)
    ]
    state_path = tmp_path / "state.json"
    state = PipelineState(state_path)
    stage = ImageStacking(_stack_config(tmp_path))
    stage.output_dir = tmp_path / "processed" / "stacking"
    calls = []
    _fake_swarp(monkeypatch, calls)

    stage.run(_refs(paths), state)
    assert len(calls) == 4
    stage.run(_refs(paths), PipelineState(state_path))
    assert len(calls) == 4

    paths[0].touch()
    stage.run(_refs(paths), PipelineState(state_path))
    assert len(calls) == 8

    stage.config = stage.config.model_copy(update={"combination_method": "AVERAGE"})
    stage.run(_refs(paths), PipelineState(state_path))
    assert len(calls) == 12


def test_stack_fails_when_there_are_too_few_compatible_inputs(tmp_path, monkeypatch):
    path = _make_sky_mef(tmp_path / "only-one.fits")
    stage = ImageStacking(_stack_config(tmp_path, minimum_exposures=2))
    stage.output_dir = tmp_path / "stacking"
    calls = []
    _fake_swarp(monkeypatch, calls)
    result = stage.run(_refs([path]), PipelineState(tmp_path / "state.json"))

    assert len(result) == 0
    assert not calls
    assert len(PipelineState(tmp_path / "state.json").failures("stacking")) == 1
    manifest = __import__("json").loads(
        (stage.output_dir / "stack_manifest.json").read_text()
    )
    assert all(group["status"] == "failed" for group in manifest["groups"])
    assert all(
        "fewer than minimum_exposures" in group["error"] for group in manifest["groups"]
    )


def test_swarp_nonzero_exit_is_reported(tmp_path, monkeypatch):
    paths = [_make_sky_mef(tmp_path / f"science-{i}.fits") for i in range(2)]
    stage = ImageStacking(_stack_config(tmp_path))
    stage.output_dir = tmp_path / "products"
    groups, errors = stage._groups(_refs(paths))
    assert not errors
    monkeypatch.setattr(stack_module, "resolve_executable", lambda *_: "/mock/swarp")
    monkeypatch.setattr(
        stack_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 3, "", "invalid WCS"
        ),
    )

    with pytest.raises(AstrometryError, match="invalid WCS"):
        stage._stack_group(groups[0])


def test_stacking_limits_concurrent_swarp_groups_to_batch_size(tmp_path, monkeypatch):
    paths = [
        _make_sky_mef(tmp_path / f"science-{i}.fits", ra=150 + i * 0.001)
        for i in range(2)
    ]
    stage = ImageStacking(_stack_config(tmp_path), n_workers=4, batch_size=1)
    stage.output_dir = tmp_path / "products"
    calls = []
    _fake_swarp(monkeypatch, calls)
    limits = {}

    def bounded_parallel(func, items, **kwargs):
        limits.update(kwargs)
        return [func(item) for item in items]

    monkeypatch.setattr(
        "newhope_pipeline.parallel.executor.run_parallel", bounded_parallel
    )
    stage.run(_refs(paths), PipelineState(tmp_path / "state.json"))
    assert limits["n_workers"] == 4
    assert limits["max_in_flight"] == 1
    assert len(calls) == 4


def test_standalone_stack_reads_only_sky_products_and_only_runs_later_stages(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "data" / "P1_2025-05-13"
    sky_dir = run_dir / "processed" / "sky_subtraction"
    _make_sky_mef(sky_dir / "one.fits")
    _make_sky_mef(sky_dir / "two.fits", ra=150.01)
    (run_dir / "raw").mkdir(parents=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "base_dir: data\nreduction: {}\nastrometry: {scamp_config_path: x.scamp}\n"
        "photometry: {vvv_catalog_path: cat.fits}\n"
    )
    monkeypatch.setattr("newhope_pipeline.cli.check_astrometry_tools", lambda *_: None)
    monkeypatch.setattr("newhope_pipeline.cli.check_stacking_tools", lambda *_: None)
    captured = {}

    class FakePipeline:
        def __init__(self, stages, manifest_path):
            captured["stages"] = [stage.name for stage in stages]
            captured["manifest"] = manifest_path

        def run(self, frames):
            captured["paths"] = [frame.path for frame in frames]
            return frames

    monkeypatch.setattr("newhope_pipeline.cli.Pipeline", FakePipeline)
    result = CliRunner().invoke(
        app,
        [
            "stack",
            "--config-path",
            str(config_path),
            "--base-dir",
            str(tmp_path / "data"),
            "--proposal",
            "P1",
            "--night",
            "2025-05-13",
            "--workers",
            "2",
            "--batch-size",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["stages"] == ["astrometric_calibration", "stacking"]
    assert len(captured["paths"]) == 2
    assert all(path.parent == sky_dir for path in captured["paths"])
    assert captured["manifest"] == run_dir / "pipeline_state.json"


def test_stack_help_mentions_input_controls():
    result = CliRunner().invoke(app, ["stack", "--help"])
    assert result.exit_code == 0
    assert "--workers" in result.output
    assert "--batch-size" in result.output
