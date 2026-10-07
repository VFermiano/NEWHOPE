from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits
from typer.testing import CliRunner

from newhope_pipeline.cli import app
from newhope_pipeline.config import PipelineConfig, make_run_name


def _write_config(tmp_path: Path) -> Path:
    cfg = {
        "base_dir": "data",
        "reduction": {},
        "astrometry": {"scamp_config_path": "x.scamp"},
        "photometry": {"vvv_catalog_path": "cat.fits"},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_make_run_name():
    assert make_run_name("2025A-599150", "2025-05-13") == "2025A-599150_2025-05-13"
    assert "/" not in make_run_name("a/b", "c d")


def test_resolve_paths_uses_cli_base_dir_over_config(tmp_path):
    cfg = PipelineConfig.from_yaml(_write_config(tmp_path))
    paths = cfg.resolve_paths("P1", "2025-05-13", base_dir=tmp_path / "elsewhere")
    assert paths.run_dir == tmp_path / "elsewhere" / "P1_2025-05-13"
    assert paths.raw_dir == paths.run_dir / "raw"
    assert paths.output_dir == paths.run_dir / "processed"
    assert paths.manifest_path == paths.run_dir / "pipeline_state.json"
    assert not paths.run_dir.exists()  # create=False


def test_resolve_paths_create(tmp_path):
    cfg = PipelineConfig.from_yaml(_write_config(tmp_path))
    paths = cfg.resolve_paths("P1", "2025-05-13", base_dir=tmp_path, create=True)
    assert paths.raw_dir.is_dir() and paths.output_dir.is_dir()


def test_reduction_dirs_default_to_run_raw_dir(tmp_path):
    cfg = PipelineConfig.from_yaml(_write_config(tmp_path))
    paths = cfg.resolve_paths("P1", "2025-05-13", base_dir=tmp_path)
    red = cfg.reduction_for(paths)
    assert red.dark_dir == paths.raw_dir and red.flat_dir == paths.raw_dir


def test_download_creates_run_dir_and_passes_it_on(tmp_path, monkeypatch):
    seen = {}

    def fake_download(acq, outdir, dry_run=False):
        seen.update(acq=acq, outdir=outdir)
        from newhope_pipeline.acquisition.noirlab import DownloadResult
        return DownloadResult()

    monkeypatch.setattr("newhope_pipeline.cli.download_night", fake_download)
    res = CliRunner().invoke(app, [
        "download", "--config-path", str(_write_config(tmp_path)),
        "--base-dir", str(tmp_path / "out"), "--proposal", "P1", "--night", "2025-05-13",
    ])
    assert res.exit_code == 0, res.output
    assert seen["outdir"] == tmp_path / "out" / "P1_2025-05-13" / "raw"
    assert seen["outdir"].is_dir()
    assert seen["acq"].proposal == "P1" and seen["acq"].night == "2025-05-13"


def test_run_reads_from_same_run_dir(tmp_path, monkeypatch):
    raw = tmp_path / "out" / "P1_2025-05-13" / "raw"
    raw.mkdir(parents=True)
    h = fits.Header(); h["FILTER"] = "HX"; h["EXPTIME"] = 30.0
    fits.writeto(raw / "a.fits", np.zeros((8, 8), dtype=np.float32), h)

    captured = {}

    class FakePipeline:
        def __init__(self, stages, manifest_path):
            captured["manifest"] = manifest_path
            captured["dark_dir"] = stages[0].config.dark_dir
        def run(self, frames):
            captured["n"] = len(frames)
            return frames

    monkeypatch.setattr("newhope_pipeline.cli.Pipeline", FakePipeline)
    res = CliRunner().invoke(app, [
        "run", "--config-path", str(_write_config(tmp_path)),
        "--base-dir", str(tmp_path / "out"), "--proposal", "P1", "--night", "2025-05-13",
    ])
    assert res.exit_code == 0, res.output
    assert captured["n"] == 1
    assert captured["manifest"] == tmp_path / "out" / "P1_2025-05-13" / "pipeline_state.json"
    assert captured["dark_dir"] == raw


def test_missing_proposal_and_night_errors(tmp_path):
    res = CliRunner().invoke(app, ["run", "--config-path", str(_write_config(tmp_path))])
    assert res.exit_code != 0


def test_force_message_shown_on_every_task(tmp_path, monkeypatch):
    from newhope_pipeline.acquisition.noirlab import DownloadResult
    monkeypatch.setattr("newhope_pipeline.cli.download_night", lambda *a, **k: DownloadResult())
    res = CliRunner().invoke(app, [
        "download", "--config-path", str(_write_config(tmp_path)),
        "--base-dir", str(tmp_path), "--proposal", "P1", "--night", "2025-05-13",
    ])
    assert "May the force be with you" in res.output

    res = CliRunner().invoke(app, ["run", "--config-path", str(_write_config(tmp_path))])
    assert "May the force be with you" in res.output  # even when the run then errors out
