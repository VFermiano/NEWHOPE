import logging
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

from newhope_pipeline.cli import (
    _LOGGING_HANDLERS,
    _cleanup_logging,
    _setup_logging,
    app,
)
from newhope_pipeline.core.exceptions import PipelineError
from newhope_pipeline.core.frame import Frame, FrameCollection
from newhope_pipeline.core.pipeline import Pipeline
from newhope_pipeline.core.stage import PipelineStage

from .conftest import make_mef


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "base_dir": str(tmp_path / "data"),
                "reduction": {},
                "astrometry": {"scamp_config_path": "scamp.conf"},
                "photometry": {"vvv_catalog_path": "catalog.fits"},
            }
        )
    )
    return path


def test_logging_configuration_info_debug_and_cleanup(tmp_path):
    root = logging.getLogger()
    old_level = root.level
    info_path = tmp_path / "info.log"
    handlers = _setup_logging(log_file=info_path)
    logging.getLogger("test.logging").debug("debug-hidden")
    logging.getLogger("test.logging").info("info-visible")
    _cleanup_logging(handlers)

    info_text = info_path.read_text()
    assert "info-visible" in info_text
    assert "debug-hidden" not in info_text
    assert " INFO " in info_text and "test.logging:" in info_text
    assert all(handler not in root.handlers for handler in handlers)
    assert all(handler._closed for handler in handlers)
    assert root.level == old_level

    debug_path = tmp_path / "debug.log"
    handlers = _setup_logging(verbose=True, log_file=debug_path)
    logging.getLogger("test.logging").debug("debug-visible")
    _cleanup_logging(handlers)
    assert " DEBUG " in debug_path.read_text()
    assert "debug-visible" in debug_path.read_text()


def test_reinitializing_logging_does_not_duplicate_handlers(tmp_path):
    first = _setup_logging(log_file=tmp_path / "first.log")
    second = _setup_logging(log_file=tmp_path / "second.log")
    try:
        root = logging.getLogger()
        assert all(handler not in root.handlers for handler in first)
        assert all(handler in root.handlers for handler in second)
        logging.getLogger("test.logging").info("single-copy")
    finally:
        _cleanup_logging(second)
    assert (tmp_path / "second.log").read_text().count("single-copy") == 1


def test_invalid_log_file_path_is_reported_and_handlers_are_cleaned(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file")
    with pytest.raises(typer.BadParameter, match="cannot create log file"):
        _setup_logging(log_file=blocked / "pipeline.log")
    assert not _LOGGING_HANDLERS
    _cleanup_logging()


def test_frame_collection_reports_actual_loading_progress(tmp_path, caplog):
    make_mef(tmp_path / "a.fits")
    make_mef(tmp_path / "b.fits")
    make_mef(tmp_path / "c.fits")

    with caplog.at_level(logging.INFO):
        frames = FrameCollection.from_directory(tmp_path, expected_extensions=4)

    messages = [record.getMessage() for record in caplog.records]
    assert len(frames) == 3
    assert any("Discovered 3 FITS file(s)" in message for message in messages)
    assert any("1/3 file(s) (33.3%)" in message for message in messages)
    assert any("2/3 file(s) (66.7%)" in message for message in messages)
    assert any("3/3 file(s) (100.0%)" in message for message in messages)
    assert any(
        "Completed FITS metadata discovery: 3/3" in message for message in messages
    )


def test_frame_collection_empty_directory_completes_without_division(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        frames = FrameCollection.from_directory(tmp_path)
    assert len(frames) == 0
    assert any("0/0 file(s) (0.0%)" in record.getMessage() for record in caplog.records)
    assert any(
        "Completed FITS metadata discovery: 0/0" in record.getMessage()
        for record in caplog.records
    )


def test_frame_collection_logs_invalid_fits_and_extension_validation(tmp_path, caplog):
    make_mef(tmp_path / "wrong-count.fits", n_ext=2)
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(PipelineError, match="expected 4"),
    ):
        FrameCollection.from_directory(tmp_path, expected_extensions=4)
    failures = [
        record
        for record in caplog.records
        if "Failed to inspect FITS file" in record.getMessage()
    ]
    assert failures and failures[-1].exc_info is not None
    assert not any("1/1 file(s)" in record.getMessage() for record in caplog.records)

    caplog.clear()
    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "broken.fits").write_text("not a FITS file")
    with caplog.at_level(logging.INFO), pytest.raises(OSError):
        FrameCollection.from_directory(invalid_dir)
    assert any(
        "Failed to inspect FITS file broken.fits" in record.getMessage()
        for record in caplog.records
    )


class _IdentityStage(PipelineStage):
    name = "identity"

    def process(self, frame: Frame) -> Frame:
        return frame


def test_pipeline_logs_stage_start_end_and_worker_count(
    tmp_path, synthetic_frame_file, caplog
):
    frames = FrameCollection([Frame.from_fits(synthetic_frame_file)])
    stage = _IdentityStage(config=None, n_workers=1)
    with caplog.at_level(logging.INFO):
        result = Pipeline([stage], manifest_path=tmp_path / "state.json").run(frames)

    assert len(result) == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Stage identity started: input=1, workers=1" in message for message in messages
    )
    assert any(
        "Stage identity completed: input=1, successful=1, failed=0" in message
        for message in messages
    )
    assert any(
        "Pipeline complete: successful=1/1, failed=0" in message for message in messages
    )


def test_run_cli_default_and_explicit_log_files_and_stage_order(tmp_path, monkeypatch):
    captured = []

    class FakePipeline:
        def __init__(self, stages, manifest_path):
            captured.append([stage.name for stage in stages])

        def run(self, frames):
            return frames

    monkeypatch.setattr("newhope_pipeline.cli.Pipeline", FakePipeline)
    monkeypatch.setattr("newhope_pipeline.cli.check_astrometry_tools", lambda *_: None)
    monkeypatch.setattr("newhope_pipeline.cli.check_stacking_tools", lambda *_: None)
    config_path = _config(tmp_path)

    def run_once(base_dir: Path, extra_args: list[str]):
        raw = base_dir / "P1_2025-05-11" / "raw"
        raw.mkdir(parents=True)
        make_mef(raw / "frame.fits")
        return CliRunner().invoke(
            app,
            [
                "run",
                "--config-path",
                str(config_path),
                "--base-dir",
                str(base_dir),
                "--proposal",
                "P1",
                "--night",
                "2025-05-11",
                *extra_args,
            ],
        )

    default_base = tmp_path / "default"
    default_result = run_once(default_base, [])
    assert default_result.exit_code == 0, default_result.output
    default_log = default_base / "P1_2025-05-11" / "pipeline.log"
    assert default_log.is_file()
    assert "INFO" in default_log.read_text()

    explicit_base = tmp_path / "explicit"
    explicit_log = tmp_path / "custom" / "debug.log"
    explicit_result = run_once(
        explicit_base, ["--verbose", "--log-file", str(explicit_log)]
    )
    assert explicit_result.exit_code == 0, explicit_result.output
    assert explicit_log.is_file()
    assert "DEBUG" in explicit_log.read_text()
    assert not (explicit_base / "P1_2025-05-11" / "pipeline.log").exists()
    expected = [
        "dark_subtraction",
        "flat_fielding",
        "sky_subtraction",
        "astrometric_calibration",
        "stacking",
    ]
    assert captured == [expected] * 2


def test_run_help_includes_verbose_and_log_file_options():
    result = CliRunner().invoke(app, ["run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--verbose" in result.output and "-v" in result.output
    assert "--log-file" in result.output


def test_run_reports_invalid_log_file_path(tmp_path):
    config_path = _config(tmp_path)
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file")
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--config-path",
            str(config_path),
            "--base-dir",
            str(tmp_path / "base"),
            "--proposal",
            "P1",
            "--night",
            "2025-05-11",
            "--log-file",
            str(blocked / "log.txt"),
        ],
    )
    assert result.exit_code != 0
    assert "cannot create log file" in result.output
    assert not _LOGGING_HANDLERS
