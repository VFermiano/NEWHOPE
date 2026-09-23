"""Command-line entry point.

Usage:
    newhope-pipeline --config-path configs/default.yaml
"""
from __future__ import annotations

import logging
from pathlib import Path

import typer

from .acquisition.noirlab import download_night, save_credentials
from .config import PipelineConfig
from .core.exceptions import AcquisitionError
from .core.frame import FrameCollection
from .core.pipeline import Pipeline
from .reduction.dark import DarkSubtraction
from .reduction.flat import FlatFielding
from .reduction.sky import SkySubtraction

app = typer.Typer()


@app.command()
def download(
    config_path: Path = typer.Option(Path("configs/default.yaml")),
    dry_run: bool = typer.Option(False, help="List matches without downloading"),
    set_credentials: bool = typer.Option(
        False, "--set-credentials", help="Store your NOIRLab email/password locally and exit"
    ),
) -> None:
    """Download raw science + calibration frames from the NOIRLab archive.

    Reads instrument/telescope/proposal/night from the config's
    `acquisition:` section and saves into `raw_dir`.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if set_credentials:
        save_credentials()
        return

    config = PipelineConfig.from_yaml(config_path)
    if config.acquisition is None:
        raise typer.BadParameter(f"No `acquisition:` section found in {config_path}.")

    try:
        result = download_night(config.acquisition, outdir=config.raw_dir, dry_run=dry_run)
    except AcquisitionError as exc:
        typer.echo(f"Download failed: {exc}", err=True)
        raise typer.Exit(code=1)

    if not dry_run:
        typer.echo(
            f"Done. {result.downloaded} downloaded/verified, {result.failed} failed "
            f"(of {result.total_found} matched)."
        )


@app.command()
def run(
    config_path: Path = typer.Option(Path("configs/default.yaml")),
    download_first: bool = typer.Option(
        False, "--download", help="Run the NOIRLab download step before processing"
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = PipelineConfig.from_yaml(config_path)

    if download_first:
        if config.acquisition is None:
            raise typer.BadParameter(f"--download requires an `acquisition:` section in {config_path}.")
        try:
            download_night(config.acquisition, outdir=config.raw_dir)
        except AcquisitionError as exc:
            typer.echo(f"Download failed: {exc}", err=True)
            raise typer.Exit(code=1)

    frames = FrameCollection.from_directory(config.raw_dir)

    stages = [
        DarkSubtraction(config.reduction, n_workers=config.n_workers),
        FlatFielding(config.reduction, n_workers=config.n_workers),
        SkySubtraction(config.reduction, n_workers=config.n_workers),
        # TODO: append AstrometricCalibration(config.astrometry, ...),
        # PSFPhotometryStage(config.photometry, ...),
        # VVVCalibrationStage(config.photometry, ...) once ported.
    ]
    pipeline = Pipeline(stages, manifest_path=config.manifest_path)
    result = pipeline.run(frames)
    typer.echo(f"Done. {len(result)} frames processed.")


if __name__ == "__main__":
    app()
