"""Group astrometrically calibrated detector images and coadd with SWarp."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

from ..config import StackingConfig
from ..core.exceptions import AstrometryError
from ..core.frame import FrameCollection, FrameLike
from ..core.stage import PipelineStage
from ..core.state import PipelineState
from .calibrate import _STRUCTURAL_KEYS, _WCS_KEY, _file_stamp, _valid_wcs
from .scamp_wrapper import output_tail, resolve_executable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StackInput:
    path: Path
    frame_id: str
    extension_index: int
    detector: str
    shape: tuple[int, int]
    filt: str
    target: str
    acquisition: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class StackGroup:
    group_id: str
    filt: str
    target: str
    detector: str
    extension_index: int
    acquisition: tuple[tuple[str, str], ...]
    inputs: tuple[StackInput, ...]
    output_path: Path


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _clean(value: object) -> str:
    return " ".join(str(value).strip().split())


def _canonical_number(value: object) -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            raise ValueError
        return format(number, ".12g")
    except (TypeError, ValueError):
        raise AstrometryError(f"Invalid numeric acquisition metadata: {value!r}")


def _target(header: fits.Header, path: Path) -> str:
    for key in ("OBJECT", "FIELD", "TARGNAME", "OBJNAME"):
        value = _clean(header.get(key, ""))
        if value and value.upper() not in {"UNKNOWN", "NONE", "N/A"}:
            return value
    raise AstrometryError(
        f"{path.name}: missing science target metadata; expected one of OBJECT, FIELD, TARGNAME, OBJNAME"
    )


def _acquisition_signature(frame: FrameLike) -> tuple[tuple[str, str], ...]:
    values = []
    for key in ("EXPTIME", "EXPCOADD", "COADDS", "FSAMPLE"):
        value = frame.header.get(key)
        if value is None:
            raise AstrometryError(
                f"{frame.path.name}: missing {key}; cannot confirm that the image is compatible with a stack"
            )
        values.append((key, _canonical_number(value)))
    return tuple(values)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._") or "unknown"


def _header_true(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"true", "t", "yes", "y", "1"}


def _group_fingerprint(group: StackGroup, config: StackingConfig) -> str:
    payload = {
        "group": {
            "filter": group.filt,
            "target": group.target,
            "detector": group.detector,
            "extension_index": group.extension_index,
            "acquisition": group.acquisition,
            "shapes": sorted({item.shape for item in group.inputs}),
        },
        "inputs": [
            {
                "path": str(item.path.resolve()),
                "frame_id": item.frame_id,
                "stamp": _file_stamp(item.path),
            }
            for item in group.inputs
        ],
        "config": config.model_dump(mode="json"),
        "swarp_config": _file_stamp(config.swarp_config_path),
        "executable": _file_stamp(
            Path(resolve_executable(config.swarp_executable, "SWarp"))
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _validate_output(path: Path, label: str) -> tuple[tuple[int, int], float]:
    try:
        with fits.open(path, memmap=False) as hdul:
            if not hdul or hdul[0].data is None or hdul[0].data.ndim != 2:
                raise ValueError("expected a 2-D primary image")
            data_shape = tuple(int(size) for size in hdul[0].data.shape)
            _valid_wcs(hdul[0].header, data_shape, label)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"RADECSYS=",
                    category=FITSFixedWarning,
                )
                scales = np.asarray(
                    proj_plane_pixel_scales(WCS(hdul[0].header, relax=True).celestial),
                    dtype=float,
                )
            pixel_scale = float(np.mean(np.abs(scales)) * 3600.0)
            if not np.isfinite(pixel_scale) or pixel_scale <= 0:
                raise ValueError("invalid output pixel scale")
            return data_shape, pixel_scale
    except Exception as exc:
        if isinstance(exc, AstrometryError):
            raise
        raise AstrometryError(
            f"Invalid SWarp output {path} for {label}: {exc}"
        ) from exc


class ImageStacking(PipelineStage):
    """A group stage: one SWarp product per filter, field, detector, and setup."""

    name = "stacking"

    def __init__(
        self, config: StackingConfig, n_workers: int = 1, batch_size: int = 1
    ) -> None:
        super().__init__(
            config,
            n_workers=n_workers,
            timeout=config.timeout_seconds,
            batch_size=batch_size,
        )

    def process(self, frame):
        raise AstrometryError(
            "ImageStacking operates on compatible exposure groups; call run()"
        )

    def _manifest_path(self) -> Path:
        if self.output_dir is None:
            raise AstrometryError(
                "Stacking output_dir must be set before running the stage"
            )
        return Path(self.output_dir) / "stack_manifest.json"

    def _sidecar(self, output_path: Path) -> Path:
        return output_path.with_suffix(output_path.suffix + ".stack.json")

    def _output_name(
        self,
        filt: str,
        target: str,
        detector: str,
        extension_index: int,
        acquisition: tuple[tuple[str, str], ...],
    ) -> tuple[str, str]:
        slug = "_".join(
            (
                "stack",
                _slug(filt),
                _slug(target),
                _slug(detector),
                f"ext{extension_index}",
            )
        )
        short_hash = hashlib.sha1(
            repr(
                (
                    filt.casefold(),
                    target.casefold(),
                    detector.casefold(),
                    extension_index,
                    acquisition,
                )
            ).encode()
        ).hexdigest()[:8]
        filename = f"{slug}_{short_hash}.fits"
        group_id = hashlib.sha1(filename.encode()).hexdigest()[:16]
        return group_id, filename

    def _groups(
        self, frames: FrameCollection
    ) -> tuple[list[StackGroup], dict[str, str]]:
        grouped: dict[tuple, list[StackInput]] = {}
        frame_errors: dict[str, str] = {}
        for frame in frames:
            try:
                if frame.obstype in {"dark", "dflat"}:
                    raise AstrometryError(
                        f"{frame.path.name}: OBSTYPE={frame.obstype} is a calibration frame, not a science input"
                    )
                if not _header_true(frame.header.get("SKYSUB", False)):
                    raise AstrometryError(
                        f"{frame.path.name}: input is not marked SKYSUB=True"
                    )
                if not _header_true(frame.header.get("ASTROMET", False)):
                    raise AstrometryError(
                        f"{frame.path.name}: input is not marked ASTROMET=True; run SCAMP calibration first"
                    )
                filt = _clean(frame.header.get("FILTER", frame.filt))
                if not filt or filt.upper() in {"UNKNOWN", "NONE", "N/A"}:
                    raise AstrometryError(f"{frame.path.name}: missing FILTER metadata")
                target = _target(frame.header, frame.path)
                acquisition = _acquisition_signature(frame)
                for ext in frame.extensions:
                    detector = _clean(ext.header.get("DETECTOR", f"EXT{ext.index}"))
                    if not detector:
                        detector = f"EXT{ext.index}"
                    if len(ext.shape) != 2:
                        raise AstrometryError(
                            f"{frame.path.name}[{ext.index}]: expected a 2-D image, got {ext.shape}"
                        )
                    _valid_wcs(ext.header, ext.shape, f"{frame.path.name}[{ext.index}]")
                    item = StackInput(
                        path=frame.path,
                        frame_id=frame.frame_id,
                        extension_index=ext.index,
                        detector=detector,
                        shape=tuple(int(size) for size in ext.shape),
                        filt=filt,
                        target=target,
                        acquisition=acquisition,
                    )
                    key = (
                        filt.casefold(),
                        target.casefold(),
                        ext.index,
                        detector.casefold(),
                        acquisition,
                    )
                    grouped.setdefault(key, []).append(item)
            except (AstrometryError, OSError, TypeError, ValueError, KeyError) as exc:
                frame_errors[frame.frame_id] = str(exc)

        groups = []
        assert self.output_dir is not None
        for key, all_inputs in grouped.items():
            inputs = [item for item in all_inputs if item.frame_id not in frame_errors]
            if not inputs:
                continue
            shapes = {item.shape for item in inputs}
            filt, target, detector = (
                inputs[0].filt,
                inputs[0].target,
                inputs[0].detector,
            )
            index, acquisition = inputs[0].extension_index, inputs[0].acquisition
            group_id, filename = self._output_name(
                filt, target, detector, index, acquisition
            )
            if len(shapes) != 1:
                message = (
                    f"Incompatible detector dimensions for {filt}/{target}/{detector}: "
                    f"{sorted(shapes)}"
                )
                for item in inputs:
                    frame_errors[item.frame_id] = message
                continue
            groups.append(
                StackGroup(
                    group_id=group_id,
                    filt=filt,
                    target=target,
                    detector=detector,
                    extension_index=index,
                    acquisition=acquisition,
                    inputs=tuple(
                        sorted(
                            inputs,
                            key=lambda item: (item.path.name, item.extension_index),
                        )
                    ),
                    output_path=Path(self.output_dir) / filename,
                )
            )
        # A failure in one detector invalidates the entire exposure. Remove
        # those frames from every other detector's group as well.
        filtered = []
        for group in groups:
            inputs = tuple(
                item for item in group.inputs if item.frame_id not in frame_errors
            )
            if inputs:
                filtered.append(
                    StackGroup(
                        group_id=group.group_id,
                        filt=group.filt,
                        target=group.target,
                        detector=group.detector,
                        extension_index=group.extension_index,
                        acquisition=group.acquisition,
                        inputs=inputs,
                        output_path=group.output_path,
                    )
                )
        return filtered, frame_errors

    def _cached_result(self, group: StackGroup) -> dict | None:
        if not self.config.reuse_intermediates:
            return None
        sidecar = self._sidecar(group.output_path)
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if data.get("fingerprint") != _group_fingerprint(group, self.config):
                return None
            if data.get("output_stamp") != _file_stamp(group.output_path):
                return None
            shape, pixel_scale = _validate_output(group.output_path, group.group_id)
            if tuple(data.get("output_shape", [])) != shape:
                return None
            data["output_pixel_scale_arcsec"] = pixel_scale
            return data
        except (OSError, ValueError, TypeError, AstrometryError):
            return None

    def _extract_detector_image(self, item: StackInput, destination: Path) -> None:
        """Write one MEF extension as a standalone FITS image for SExtractor/SWarp."""
        try:
            with fits.open(item.path, memmap=False, lazy_load_hdus=False) as hdul:
                if item.extension_index >= len(hdul):
                    raise ValueError(f"missing extension {item.extension_index}")
                primary = hdul[0].header.copy()
                extension = hdul[item.extension_index]
                if extension.data is None or extension.data.ndim != 2:
                    raise ValueError(
                        f"extension {item.extension_index} is not a 2-D image"
                    )
                data = np.array(extension.data, dtype=np.float32, copy=True)
                header = primary
                for key in list(header.keys()):
                    if _WCS_KEY.match(key):
                        header.remove(key, remove_all=True, ignore_missing=True)
                for card in extension.header.cards:
                    if card.keyword in _STRUCTURAL_KEYS or card.keyword in {
                        "COMMENT",
                        "HISTORY",
                        "",
                    }:
                        continue
                    header[card.keyword] = (card.value, card.comment)
                for key in ("CHECKSUM", "DATASUM"):
                    header.remove(key, ignore_missing=True)
            fits.PrimaryHDU(data=data, header=header).writeto(
                destination, overwrite=True
            )
        except Exception as exc:
            if isinstance(exc, AstrometryError):
                raise
            raise AstrometryError(
                f"Could not extract detector {item.detector} (extension {item.extension_index}) "
                f"from {item.path}: {exc}"
            ) from exc

    def _stack_group(self, group: StackGroup) -> dict:
        cfg = self.config
        if len(group.inputs) < cfg.minimum_exposures:
            message = (
                f"{group.filt}/{group.target}/{group.detector}: {len(group.inputs)} compatible exposure(s), "
                f"fewer than minimum_exposures={cfg.minimum_exposures}"
            )
            raise AstrometryError(message)
        if not Path(cfg.swarp_config_path).expanduser().is_file():
            raise AstrometryError(
                f"SWarp configuration file does not exist: {cfg.swarp_config_path}"
            )
        executable = resolve_executable(cfg.swarp_executable, "SWarp")
        for item in group.inputs:
            if not item.path.is_file():
                raise AstrometryError(f"Stack input disappeared: {item.path}")
            with fits.open(item.path, memmap=False) as hdul:
                header = hdul[0].header
                _valid_wcs(
                    hdul[item.extension_index].header, item.shape, str(item.path)
                )
                if not _header_true(header.get("SKYSUB", False)) or not _header_true(
                    header.get("ASTROMET", False)
                ):
                    raise AstrometryError(
                        f"{item.path.name}: sky subtraction or SCAMP calibration marker is missing"
                    )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if cfg.keep_intermediates:
            work_path = self.output_dir / "intermediates" / group.group_id
            work_path.mkdir(parents=True, exist_ok=True)
            temp_context = None
            temp_dir = work_path
        else:
            temp_context = tempfile.TemporaryDirectory(
                prefix=f"swarp-{group.group_id}-", dir=self.output_dir
            )
            temp_dir = Path(temp_context.name)
        try:
            temporary_output = temp_dir / "stack.fits"
            temporary_output.unlink(missing_ok=True)
            for stale_resample in temp_dir.glob("*.resamp.fits"):
                stale_resample.unlink(missing_ok=True)
            detector_paths = []
            for index, item in enumerate(group.inputs):
                detector_path = (
                    temp_dir / f"input_{index:04d}_{_slug(item.path.stem)}.fits"
                )
                self._extract_detector_image(item, detector_path)
                detector_paths.append(detector_path)
            command = [
                executable,
                *(str(path) for path in detector_paths),
                "-c",
                str(Path(cfg.swarp_config_path).expanduser().resolve()),
                "-IMAGEOUT_NAME",
                str(temporary_output),
                "-WEIGHT_TYPE",
                "NONE",
                "-SUBTRACT_BACK",
                "N",
                "-PROJECTION_TYPE",
                cfg.projection,
                "-RESAMPLE_TYPE",
                cfg.resampling_method,
                "-COMBINE_TYPE",
                cfg.combination_method,
                "-NTHREADS",
                "1",
                "-WRITE_XML",
                "N",
            ]
            if cfg.output_pixel_scale_arcsec is None:
                command.extend(("-PIXELSCALE_TYPE", "MEDIAN"))
            else:
                command.extend(
                    (
                        "-PIXELSCALE_TYPE",
                        "MANUAL",
                        "-PIXEL_SCALE",
                        str(cfg.output_pixel_scale_arcsec),
                    )
                )
            logger.info(
                "SWarp stacking %d %s %s detector %s image(s)",
                len(group.inputs),
                group.filt,
                group.target,
                group.detector,
            )
            logger.debug("SWarp command: %s", command)
            try:
                result = subprocess.run(
                    command,
                    cwd=temp_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=cfg.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise AstrometryError(
                    f"SWarp exceeded {cfg.timeout_seconds:g}s for {group.filt}/{group.target}/{group.detector}"
                ) from exc
            except OSError as exc:
                raise AstrometryError(
                    f"Could not start SWarp ({executable}): {exc}"
                ) from exc
            summary = output_tail(result.stdout, result.stderr)
            if summary:
                logger.info("SWarp output for %s:\n%s", group.group_id, summary)
            if result.returncode != 0:
                details = output_tail(result.stderr, result.stdout, limit=20)
                raise AstrometryError(
                    f"SWarp failed for {group.filt}/{group.target}/{group.detector} "
                    f"(exit {result.returncode})"
                    + (f": {details[-2000:]}" if details else "")
                )
            if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
                raise AstrometryError(f"SWarp produced no image for {group.group_id}")

            output_shape, output_scale = _validate_output(
                temporary_output, group.group_id
            )
            with fits.open(temporary_output, mode="update", memmap=False) as hdul:
                header = hdul[0].header
                header["STACK"] = (True, "Image coadded from sky-subtracted exposures")
                header["NSTACK"] = (len(group.inputs), "Number of exposures combined")
                header["FILTER"] = (group.filt, "Instrument filter")
                header["OBJECT"] = (group.target, "Science target/field")
                header["DETECTOR"] = (group.detector, "NEWFIRM detector identifier")
                header["EXTINDEX"] = (
                    group.extension_index,
                    "Source MEF image extension index",
                )
                header["COMBMETH"] = (
                    cfg.combination_method,
                    "SWarp combination method",
                )
                header["RESAMPM"] = (cfg.resampling_method, "SWarp resampling kernel")
                header["PIXSCAL"] = (output_scale, "Output pixel scale (arcsec/pixel)")
                header["ASTROMET"] = (True, "All inputs calibrated with SCAMP")
                header["SKYSUB"] = (True, "All inputs sky-subtracted")
                for key, value in group.acquisition:
                    header[key] = (float(value), "Shared acquisition setting")
                hdul.flush(output_verify="exception")
            os.replace(temporary_output, group.output_path)
        finally:
            if temp_context is not None:
                temp_context.cleanup()

        details = {
            "group_id": group.group_id,
            "fingerprint": _group_fingerprint(group, cfg),
            "filter": group.filt,
            "target": group.target,
            "detector": group.detector,
            "extension_index": group.extension_index,
            "acquisition": dict(group.acquisition),
            "input_count": len(group.inputs),
            "inputs": [
                {"path": str(item.path), "frame_id": item.frame_id}
                for item in group.inputs
            ],
            "astrometry": "SCAMP per detector extension; validated WCS",
            "output_path": str(group.output_path),
            "output_stamp": _file_stamp(group.output_path),
            "output_shape": list(output_shape),
            "output_pixel_scale_arcsec": output_scale,
            "combination_method": cfg.combination_method,
            "resampling_method": cfg.resampling_method,
            "failed_inputs": [],
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        _json_write(self._sidecar(group.output_path), details)
        return details

    def run(self, frames: FrameCollection, state: PipelineState) -> FrameCollection:
        """Stack groups while returning successful input references to Pipeline."""
        started = monotonic()
        if self.output_dir is None:
            self.output_dir = state.manifest_path.parent / "processed" / self.name
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        groups, frame_errors = self._groups(frames)
        logger.info(
            "Stacking started: inputs=%d, detector groups=%d, workers=%d, batch_size=%d",
            len(frames),
            len(groups),
            self.n_workers,
            self.batch_size,
        )

        outcomes_by_id: dict[str, dict | Exception] = {}
        pending = []
        for group in groups:
            cached = self._cached_result(group)
            if cached is not None:
                outcomes_by_id[group.group_id] = cached
                logger.info("Reusing validated stack %s", group.output_path.name)
            else:
                pending.append(group)

        from ..parallel.executor import run_parallel

        results = run_parallel(
            self._stack_group,
            pending,
            n_workers=self.n_workers,
            timeout=self.config.timeout_seconds,
            max_in_flight=self.batch_size,
            task_labels=[group.output_path.name for group in pending],
        )
        for group, result in zip(pending, results):
            outcomes_by_id[group.group_id] = result

        successful_groups = []
        failed_group_errors: dict[str, str] = {}
        group_manifest = []
        for group in groups:
            result = outcomes_by_id[group.group_id]
            if isinstance(result, Exception):
                error = str(result)
                failed_group_errors[group.group_id] = error
                for item in group.inputs:
                    frame_errors.setdefault(item.frame_id, error)
                logger.error("Stack group %s failed: %s", group.group_id, error)
                group_manifest.append(
                    {
                        "group_id": group.group_id,
                        "filter": group.filt,
                        "target": group.target,
                        "detector": group.detector,
                        "extension_index": group.extension_index,
                        "input_count": len(group.inputs),
                        "output_path": str(group.output_path),
                        "status": "failed",
                        "error": error,
                        "inputs": [
                            {"path": str(item.path), "frame_id": item.frame_id}
                            for item in group.inputs
                        ],
                        "failed_inputs": [
                            {
                                "path": str(item.path),
                                "frame_id": item.frame_id,
                                "reason": error,
                            }
                            for item in group.inputs
                        ],
                    }
                )
            else:
                successful_groups.append(group)
                group_manifest.append({**result, "status": "done"})

        expected_groups: dict[str, int] = {}
        successful_frame_groups: dict[str, int] = {}
        for group in groups:
            for item in group.inputs:
                expected_groups[item.frame_id] = (
                    expected_groups.get(item.frame_id, 0) + 1
                )
                if group.group_id not in failed_group_errors:
                    successful_frame_groups[item.frame_id] = (
                        successful_frame_groups.get(item.frame_id, 0) + 1
                    )

        successful = FrameCollection()
        for frame in frames:
            if frame.frame_id in frame_errors:
                state.mark(
                    frame.frame_id,
                    self.name,
                    status="failed",
                    error=frame_errors[frame.frame_id],
                )
                continue
            if (
                successful_frame_groups.get(frame.frame_id, 0)
                == expected_groups.get(frame.frame_id, 0)
                and expected_groups.get(frame.frame_id, 0) > 0
            ):
                state.mark(
                    frame.frame_id,
                    self.name,
                    status="done",
                    output_path=str(self._manifest_path()),
                )
                successful.add(frame)
            else:
                state.mark(
                    frame.frame_id,
                    self.name,
                    status="failed",
                    error="One or more detector stack groups failed; see stack_manifest.json",
                )

        manifest = {
            "version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_count": len(frames),
            "successful_frames": len(successful),
            "failed_frames": len(frames) - len(successful),
            "minimum_exposures": self.config.minimum_exposures,
            "combination_method": self.config.combination_method,
            "resampling_method": self.config.resampling_method,
            "output_pixel_scale_arcsec": self.config.output_pixel_scale_arcsec,
            "failed_frame_reasons": frame_errors,
            "groups": group_manifest,
        }
        _json_write(self._manifest_path(), manifest)
        logger.info(
            "Stacking complete: %d/%d frames successful, %d failed, elapsed=%.1fs; products=%s",
            len(successful),
            len(frames),
            len(frames) - len(successful),
            monotonic() - started,
            self.output_dir,
        )
        return successful


def check_stacking_tools(config: StackingConfig) -> None:
    resolve_executable(config.swarp_executable, "SWarp")
    if not Path(config.swarp_config_path).expanduser().is_file():
        raise AstrometryError(
            f"SWarp configuration file does not exist: {config.swarp_config_path}"
        )
