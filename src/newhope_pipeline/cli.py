"""Command-line entry point.

Usage:
    newhope-pipeline --config-path configs/default.yaml
"""
from __future__ import annotations

import logging
from pathlib import Path

import typer

from .config import PipelineConfig
from .core.frame import FrameCollection
from .core.pipeline import Pipeline
from .reduction.dark import DarkSubtraction
from .reduction.flat import FlatFielding
from .reduction.sky import SkySubtraction

app = typer.Typer()


@app.command()
def run(config_path: Path = typer.Option(Path("configs/default.yaml"))) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = PipelineConfig.from_yaml(config_path)
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
