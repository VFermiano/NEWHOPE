"""Command-line entry point.

Usage:
    newhope-pipeline download --base-dir /path/to/data --proposal 2025A-599150 --night 2025-05-13
    newhope-pipeline run      --base-dir /path/to/data --proposal 2025A-599150 --night 2025-05-13
"""
from __future__ import annotations

import logging
from pathlib import Path

import typer

from .acquisition.noirlab import download_night, save_credentials
from .config import AcquisitionConfig, PipelineConfig
from .core.exceptions import AcquisitionError
from .core.frame import FrameCollection
from .core.pipeline import Pipeline
from .io.decompress import uncompress_directory
from .reduction.dark import DarkSubtraction
from .reduction.flat import FlatFielding
from .reduction.sky import SkySubtraction

app = typer.Typer()


FORCE_MESSAGE = "May the Force be with you"


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(module)s: %(message)s")


def _force_message() -> None:
    """NEWHOPE gimmick: greet every task with a Star Wars blessing."""
    typer.secho(f"\n  ✦  {FORCE_MESSAGE}  ✦\n", fg=typer.colors.YELLOW, bold=True)


def _resolve_target(
    config: PipelineConfig, proposal: str | None, night: str | None, config_path: Path
) -> tuple[str, str]:
    """CLI values win; fall back to the config's `acquisition:` section."""
    acq = config.acquisition
    proposal = proposal or (acq.proposal if acq else None)
    night = night or (acq.night if acq else None)
    if not proposal or not night:
        raise typer.BadParameter(
            f"proposal and night are required (pass --proposal/--night or set them in {config_path})."
        )
    return proposal, night


@app.command()
def download(
    night: str = typer.Option(None, "--night", help="Observing night, YYYY-MM-DD"),
    proposal: str = typer.Option(None, "--proposal", help="Proposal ID, e.g. 2025A-599150"),
    base_dir: Path = typer.Option(
        None, "--base-dir", help="Parent directory; a <proposal>_<night>/ run folder is created inside it"
    ),
    config_path: Path = typer.Option(Path("configs/default.yaml")),
    dry_run: bool = typer.Option(False, help="List matches without downloading"),
    set_credentials: bool = typer.Option(
        False, "--set-credentials", help="Store your NOIRLab email/password locally and exit"
    ),
) -> None:
    """Download raw science + calibration frames from the NOIRLab archive.

    Files are saved to <base-dir>/<proposal>_<night>/raw/, which is created
    if it doesn't exist. Pass the same --base-dir/--proposal/--night to
    `run` and it will keep working inside that same run directory.
    """
    _setup_logging()
    _force_message()

    if set_credentials:
        save_credentials()
        return

    config = PipelineConfig.from_yaml(config_path)
    proposal, night = _resolve_target(config, proposal, night, config_path)

    acquisition = (config.acquisition or AcquisitionConfig()).model_copy(
        update={"proposal": proposal, "night": night}
    )
    paths = config.resolve_paths(proposal, night, base_dir=base_dir, create=not dry_run)
    typer.echo(f"Run directory: {paths.run_dir}")

    try:
        result = download_night(acquisition, outdir=paths.raw_dir, dry_run=dry_run)
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
    night: str = typer.Option(None, "--night", help="Observing night, YYYY-MM-DD"),
    proposal: str = typer.Option(None, "--proposal", help="Proposal ID, e.g. 2025A-599150"),
    base_dir: Path = typer.Option(
        None, "--base-dir", help="Parent directory containing the <proposal>_<night>/ run folder"
    ),
    config_path: Path = typer.Option(Path("configs/default.yaml")),
    download_first: bool = typer.Option(
        False, "--download", help="Run the NOIRLab download step before processing"
    ),
    delete_compressed: bool = typer.Option(
        False,
        "--delete-compressed",
        help="Delete each .fits.fz after it has been uncompressed (default: keep it)",
    ),
) -> None:
    """Run the reduction inside <base-dir>/<proposal>_<night>/.

    Reads from `raw/`, writes to `processed/`, and keeps the resume manifest
    in the same run directory.
    """
    _setup_logging()
    _force_message()

    config = PipelineConfig.from_yaml(config_path)
    proposal, night = _resolve_target(config, proposal, night, config_path)
    paths = config.resolve_paths(proposal, night, base_dir=base_dir, create=True)
    typer.echo(f"Run directory: {paths.run_dir}")

    if download_first:
        acquisition = (config.acquisition or AcquisitionConfig()).model_copy(
            update={"proposal": proposal, "night": night}
        )
        try:
            download_night(acquisition, outdir=paths.raw_dir)
        except AcquisitionError as exc:
            typer.echo(f"Download failed: {exc}", err=True)
            raise typer.Exit(code=1)

    # Step 1: uncompress .fits.fz (no-op if the files are already uncompressed).
    decompressed = uncompress_directory(
        paths.raw_dir, n_workers=config.n_workers, delete_original=delete_compressed
    )
    if decompressed.failed:
        names = ", ".join(p.name for p, _ in decompressed.failed)
        typer.echo(f"Could not uncompress: {names}", err=True)
        raise typer.Exit(code=1)

    frames = FrameCollection.from_directory(paths.raw_dir)
    if len(frames) == 0:
        raise typer.BadParameter(
            f"No FITS files in {paths.raw_dir}. Run `download` first or use --download."
        )

    reduction = config.reduction_for(paths)
    stages = [
        DarkSubtraction(reduction, n_workers=config.n_workers),
        FlatFielding(reduction, n_workers=config.n_workers),
        SkySubtraction(reduction, n_workers=config.n_workers),
        # TODO: append AstrometricCalibration(config.astrometry, ...),
        # PSFPhotometryStage(config.photometry, ...),
        # VVVCalibrationStage(config.photometry, ...) once ported.
    ]
    pipeline = Pipeline(stages, manifest_path=paths.manifest_path)
    result = pipeline.run(frames)
    typer.echo(f"Done. {len(result)} frames processed.")


if __name__ == "__main__":
    app()
