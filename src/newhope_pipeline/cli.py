"""Command-line entry point.

Usage:
    newhope-pipeline download --base-dir /path/to/data --proposal 2025A-599150 --night 2025-05-13
    newhope-pipeline run      --base-dir /path/to/data --proposal 2025A-599150 --night 2025-05-13
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
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
logger = logging.getLogger(__name__)


FORCE_MESSAGE = "May the Force be with you"
_LOGGING_HANDLERS: list[logging.Handler] = []
_PREVIOUS_ROOT_LEVEL: int | None = None


def _setup_logging(verbose: bool = False, log_file: Path | None = None) -> list[logging.Handler]:
    """Set up consistent console/file logging without accumulating handlers."""
    global _PREVIOUS_ROOT_LEVEL
    _cleanup_logging()

    root = logging.getLogger()
    _PREVIOUS_ROOT_LEVEL = root.level
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    root.addHandler(console_handler)

    if log_file is not None:
        try:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        except OSError as exc:
            _cleanup_logging(handlers)
            raise typer.BadParameter(f"cannot create log file {log_file}: {exc}") from exc
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
        root.addHandler(file_handler)

    root.setLevel(level)
    _LOGGING_HANDLERS.extend(handlers)
    return handlers


def _cleanup_logging(handlers: Iterable[logging.Handler] | None = None) -> None:
    """Remove and close handlers installed by this CLI."""
    global _PREVIOUS_ROOT_LEVEL
    root = logging.getLogger()
    to_close = list(handlers) if handlers is not None else list(_LOGGING_HANDLERS)
    for handler in to_close:
        root.removeHandler(handler)
        handler.flush()
        handler.close()
        if handler in _LOGGING_HANDLERS:
            _LOGGING_HANDLERS.remove(handler)
    if not _LOGGING_HANDLERS and _PREVIOUS_ROOT_LEVEL is not None:
        root.setLevel(_PREVIOUS_ROOT_LEVEL)
        _PREVIOUS_ROOT_LEVEL = None


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
    handlers = _setup_logging()
    try:
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
        logger.info(
            "Starting download: proposal=%s, night=%s, output_dir=%s, dry_run=%s",
            proposal,
            night,
            paths.raw_dir,
            dry_run,
        )
        typer.echo(f"Run directory: {paths.run_dir}")

        try:
            result = download_night(acquisition, outdir=paths.raw_dir, dry_run=dry_run)
        except AcquisitionError as exc:
            logger.exception("NOIRLab download failed for proposal=%s night=%s", proposal, night)
            typer.echo(f"Download failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        if not dry_run:
            logger.info(
                "Download complete: downloaded/verified=%d, failed=%d, matched=%d",
                result.downloaded,
                result.failed,
                result.total_found,
            )
            typer.echo(
                f"Done. {result.downloaded} downloaded/verified, {result.failed} failed "
                f"(of {result.total_found} matched)."
            )
    finally:
        _cleanup_logging(handlers)


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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable DEBUG logging (default: INFO)"
    ),
    log_file: Path | None = typer.Option(
        None,
        "--log-file",
        help="Log file path (default: <run directory>/pipeline.log)",
    ),
) -> None:
    """Run the reduction inside <base-dir>/<proposal>_<night>/.

    Reads from `raw/`, writes to `processed/`, and keeps the resume manifest
    in the same run directory. INFO logs report phase and frame progress;
    pass `--verbose`/`-v` for DEBUG details. Logs go to `<run>/pipeline.log`
    unless `--log-file PATH` is supplied.
    """
    _force_message()

    config = PipelineConfig.from_yaml(config_path)
    proposal, night = _resolve_target(config, proposal, night, config_path)
    paths = config.resolve_paths(proposal, night, base_dir=base_dir, create=True)
    selected_log_file = log_file or paths.run_dir / "pipeline.log"
    handlers = _setup_logging(verbose=verbose, log_file=selected_log_file)
    try:
        typer.echo(f"Run directory: {paths.run_dir}")
        logger.info(
            "Starting run: proposal=%s, night=%s, run_dir=%s, input_dir=%s, output_dir=%s",
            proposal,
            night,
            paths.run_dir,
            paths.raw_dir,
            paths.output_dir,
        )
        logger.info("Persistent log file: %s", selected_log_file)

        if download_first:
            acquisition = (config.acquisition or AcquisitionConfig()).model_copy(
                update={"proposal": proposal, "night": night}
            )
            logger.info("Starting NOIRLab download into %s", paths.raw_dir)
            try:
                download_result = download_night(acquisition, outdir=paths.raw_dir)
            except AcquisitionError as exc:
                logger.exception("NOIRLab download failed for proposal=%s night=%s", proposal, night)
                typer.echo(f"Download failed: {exc}", err=True)
                raise typer.Exit(code=1) from exc
            logger.info(
                "NOIRLab download complete: downloaded/verified=%d, failed=%d, matched=%d",
                download_result.downloaded,
                download_result.failed,
                download_result.total_found,
            )

        logger.info(
            "Starting FITS decompression in %s with %d worker(s)", paths.raw_dir, config.n_workers
        )
        decompressed = uncompress_directory(
            paths.raw_dir, n_workers=config.n_workers, delete_original=delete_compressed
        )
        logger.info(
            "FITS decompression complete: found=%d, decompressed=%d, skipped=%d, failed=%d",
            decompressed.found,
            decompressed.decompressed,
            decompressed.skipped,
            len(decompressed.failed),
        )
        if decompressed.failed:
            names = ", ".join(p.name for p, _ in decompressed.failed)
            logger.error("FITS decompression failed for: %s", names)
            typer.echo(f"Could not uncompress: {names}", err=True)
            raise typer.Exit(code=1)

        frames = FrameCollection.from_directory(paths.raw_dir, expected_extensions=config.n_extensions)
        logger.info("FITS loading complete: %d frame(s) loaded from %s", len(frames), paths.raw_dir)
        if len(frames) == 0:
            logger.error("No FITS frames found in %s", paths.raw_dir)
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
        for stage in stages:
            stage.output_dir = paths.output_dir / stage.name
        logger.info("Intermediate and final stage outputs are stored under %s", paths.output_dir)

        pipeline = Pipeline(stages, manifest_path=paths.manifest_path)
        result = pipeline.run(frames)
        n_failed = len(frames) - len(result)
        logger.info(
            "Run summary: successful=%d/%d, failed=%d, manifest=%s, outputs=%s",
            len(result),
            len(frames),
            n_failed,
            paths.manifest_path,
            paths.output_dir,
        )
        typer.echo(f"Done. {len(result)} of {len(frames)} frames processed.")
        if n_failed:
            typer.echo(
                f"{n_failed} frame(s) failed; see {paths.manifest_path} for the per-stage errors.", err=True
            )
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception:
        logger.exception("Run failed for proposal=%s night=%s", proposal, night)
        raise
    finally:
        _cleanup_logging(handlers)


if __name__ == "__main__":
    app()
