"""Flat fielding and bounded-memory construction of master flats.

The legacy NEWFIRM recipe combines lamp-on and lamp-off flats separately,
scales each exposure by its image mode, median-combines each group, subtracts
the lamp-off master, and normalizes each detector by its own median. The
``FLATNORM`` keyword retains the removed detector scale; flat correction then
restores that scale and applies one median shared by all four detectors.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from astropy.io import fits
from scipy import stats

from ..core.exceptions import ReductionError
from ..core.frame import Extension, Frame, FrameCollection, FrameReference
from ..core.stage import PipelineStage
from ..core.state import PipelineState

logger = logging.getLogger(__name__)

_DETECTORS = (1, 2, 3, 4)
_BAND_ROWS = 64
_FlatKey = tuple[str, str]  # normalized filter, lamp state (on/off)


def _normalize_filter(value: object) -> str:
    """Normalize case and whitespace while preserving instrument filter names."""
    normalized = str(value).strip().upper()
    return {"HX": "HX", "JX": "JX", "KXS": "KXs"}.get(normalized, str(value).strip())


def _header_value(reference: FrameReference, keyword: str):
    value = reference.header.get(keyword)
    if value is not None:
        return value
    for extension in reference.extensions:
        value = extension.header.get(keyword)
        if value is not None:
            return value
    return None


def _active_width(reference: FrameReference, detector: int, image_width: int) -> int:
    """Return the active detector width recorded by DETSEC, if present.

    The legacy TrimImage step cropped NEWFIRM's final 64 overscan columns.
    DETSEC records that active width in raw MEFs. Already-trimmed files (or
    test/reference products without DETSEC) use their complete image width.
    """
    detsec = reference[detector].header.get("DETSEC")
    if detsec is None:
        return image_width
    match = re.match(r"\s*\[\s*(-?\d+)\s*:\s*(-?\d+)\s*,", str(detsec))
    if match is None:
        return image_width
    width = abs(int(match.group(2)) - int(match.group(1))) + 1
    return min(width, image_width) if width > 0 else image_width


class FlatFielding(PipelineStage):
    name = "flat_fielding"

    def __init__(
        self,
        config,
        n_workers: int = 1,
        timeout: float | None = None,
        batch_size: int = 1,
        calibration_frames: FrameCollection | Sequence[FrameReference] | None = None,
    ) -> None:
        super().__init__(config, n_workers=n_workers, timeout=timeout, batch_size=batch_size)
        self.calibration_frames = list(calibration_frames or [])
        self._master_flat_cache: dict[str, Mapping[int, np.ndarray]] = {}
        self._master_flat_errors: dict[str, list[str]] = {}

    def run(self, frames: FrameCollection, state: PipelineState) -> FrameCollection:
        """Construct needed raw-flat masters before science workers start."""
        self._ensure_reference_persistence(frames, state)
        self._prepare_raw_master_flats(frames, state)
        return super().run(frames, state)

    def process(self, frame: Frame) -> Frame:
        logger.debug("Flat-field correction started for %s (filter=%s)", frame.path.name, frame.filt)
        masters = self._load_master_flat(frame.filt)
        missing = [e.index for e in frame if e.index not in masters]
        if missing:
            raise ReductionError(f"no master flat for extension(s) {missing} (filter={frame.filt})")

        def divide(ext: Extension) -> np.ndarray:
            master = masters[ext.index]
            if master.shape != ext.shape:
                raise ReductionError(
                    f"master flat shape {master.shape} does not match frame "
                    f"{frame.frame_id} extension {ext.index} shape {ext.shape}"
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                corrected = ext.data / master
            corrected = np.asarray(corrected, dtype=np.float32)
            corrected[~np.isfinite(corrected)] = 0.0
            return corrected

        result = frame.map_extensions(divide)
        logger.debug("Flat-field correction finished for %s", frame.path.name)
        return result

    def _configured_flat_dir(self) -> Path | None:
        flat_dir = getattr(self.config, "flat_dir", None)
        return None if flat_dir is None else Path(flat_dir)

    def _generated_master_dir(self) -> Path | None:
        return None if self.output_dir is None else Path(self.output_dir) / "master_flats"

    @staticmethod
    def _product_paths(directory: Path, filt: str) -> dict[int, Path]:
        nested = {
            detector: directory / str(detector) / f"nflat_{filt}_{detector}.fits"
            for detector in _DETECTORS
            if (directory / str(detector) / f"nflat_{filt}_{detector}.fits").is_file()
        }
        if len(nested) == len(_DETECTORS):
            return nested
        root = {
            detector: directory / f"nflat_{filt}_{detector}.fits"
            for detector in _DETECTORS
            if (directory / f"nflat_{filt}_{detector}.fits").is_file()
        }
        if len(root) == len(_DETECTORS):
            return root
        # Report a single layout when the set is incomplete. Never combine
        # products from detector subdirectories with products at the root.
        return nested if len(nested) >= len(root) else root

    def _prepare_raw_master_flats(self, frames: FrameCollection, state: PipelineState) -> None:
        """Build only missing filter products, storing them under this stage."""
        self._master_flat_errors.clear()
        if not self.calibration_frames:
            logger.info("No raw OBSTYPE=dflat references were supplied to flat fielding")
            return

        generated_dir = self._generated_master_dir()
        if generated_dir is None:
            logger.debug("No flat-stage output directory; raw masters will not be persisted")
            return

        raw_groups, invalid = self._raw_flat_groups()
        if not raw_groups:
            for detail in invalid:
                logger.error("Ignoring unusable raw flat calibration: %s", detail)
            return

        flat_dir = self._configured_flat_dir()
        pending_filters = set()
        for frame in frames:
            output_path = self._output_path(frame)
            if (
                state.is_done(frame.frame_id, self.name)
                and output_path is not None
                and output_path.is_file()
            ):
                continue
            pending_filters.add(_normalize_filter(frame.filt))
        filters = sorted(pending_filters)
        generated_dir.mkdir(parents=True, exist_ok=True)

        for filt in filters:
            if flat_dir is not None and len(self._product_paths(flat_dir, filt)) == len(_DETECTORS):
                logger.info("Using complete precomputed master-flat set for filter=%s in %s", filt, flat_dir)
                continue

            on_refs = raw_groups.get((filt, "on"), [])
            off_refs = raw_groups.get((filt, "off"), [])
            if not on_refs or not off_refs:
                self._master_flat_errors[filt] = [
                    f"master construction needs both lamp groups (lamp-on={len(on_refs)}, "
                    f"lamp-off={len(off_refs)})"
                ]
                logger.warning(
                    "Cannot construct master flat for filter=%s: lamp-on=%d, lamp-off=%d "
                    "OBSTYPE=dflat frame(s)",
                    filt,
                    len(on_refs),
                    len(off_refs),
                )
                continue

            sources = [*on_refs, *off_refs]
            fingerprint = self._calibration_fingerprint(sources)
            for detector in _DETECTORS:
                destination = self._generated_master_path(filt, detector)
                if self._generated_product_is_current(destination, filt, detector, fingerprint):
                    continue
                try:
                    self._write_master_flat_product(
                        destination, filt, detector, on_refs, off_refs, fingerprint
                    )
                except Exception as exc:
                    self._master_flat_errors.setdefault(filt, []).append(
                        f"detector extension {detector}: {exc}"
                    )
                    logger.exception(
                        "Could not construct master flat for filter=%s detector=%d",
                        filt,
                        detector,
                    )

        for detail in invalid:
            logger.warning("Raw OBSTYPE=dflat calibration was excluded: %s", detail)

    def _load_master_flat(self, filt: str) -> Mapping[int, np.ndarray]:
        """Load one complete four-detector flat set and apply a shared median.

        Per-detector legacy products may live in ``flat_dir/<detector>/`` or
        directly under ``flat_dir``. Generated masters live below the flat
        stage output directory. A complete set is always taken from one root
        so products from different filters or calibration sets cannot mix.
        """
        normalized_filter = _normalize_filter(filt)
        if normalized_filter in self._master_flat_cache:
            logger.debug("Master-flat cache hit for filter=%s", normalized_filter)
            return self._master_flat_cache[normalized_filter]
        self._master_flat_cache.clear()

        flat_dir = self._configured_flat_dir()
        generated_dir = self._generated_master_dir()
        search_dirs = [path for path in (flat_dir, generated_dir) if path is not None]
        if not search_dirs:
            raise ReductionError("flat_dir is not configured and no flat-stage output directory exists")

        logger.info(
            "Loading four master flats for filter=%s from %s",
            normalized_filter,
            ", ".join(str(path) for path in search_dirs),
        )
        product_set = next(
            (
                (directory, self._product_paths(directory, normalized_filter))
                for directory in search_dirs
                if len(self._product_paths(directory, normalized_filter)) == len(_DETECTORS)
            ),
            None,
        )
        if product_set is None:
            expected = []
            for path in search_dirs:
                for detector in _DETECTORS:
                    nested = path / str(detector) / f"nflat_{normalized_filter}_{detector}.fits"
                    root = path / f"nflat_{normalized_filter}_{detector}.fits"
                    expected.append(f"{nested} or {root}")
            raw_groups, invalid = self._raw_flat_groups()
            on_count = len(raw_groups.get((normalized_filter, "on"), []))
            off_count = len(raw_groups.get((normalized_filter, "off"), []))
            detail = (
                f"; raw OBSTYPE=dflat frames for filter={normalized_filter}: "
                f"lamp-on={on_count}, lamp-off={off_count}"
            )
            if invalid:
                detail += "; unusable raw calibration headers: " + "; ".join(invalid)
            if not self.calibration_frames:
                detail += "; no raw OBSTYPE=dflat frames were supplied"
            construction_errors = self._master_flat_errors.get(normalized_filter, [])
            if construction_errors:
                detail += "; master-flat construction failed: " + "; ".join(construction_errors)
            raise ReductionError(
                f"no complete master flat set for filter={normalized_filter}; missing expected "
                + ", ".join(expected)
                + detail
            )

        product_dir, paths = product_set
        raw_flats: dict[int, np.ndarray] = {}
        restored_samples: list[np.ndarray] = []
        for detector in _DETECTORS:
            flat_path = paths[detector]
            logger.debug("Reading detector %d master flat: %s", detector, flat_path)
            try:
                with fits.open(flat_path, memmap=False) as hdul:
                    image_hdu = next(
                        (
                            hdu
                            for hdu in hdul
                            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2
                        ),
                        None,
                    )
                    if image_hdu is None:
                        raise ReductionError(f"master flat {flat_path} contains no 2-D image")
                    flat = np.array(image_hdu.data, dtype=np.float32, copy=True)
                    flatnorm = image_hdu.header.get("FLATNORM", hdul[0].header.get("FLATNORM"))
                    active_width_value = image_hdu.header.get(
                        "ACTWIDTH", hdul[0].header.get("ACTWIDTH", flat.shape[1])
                    )
            except OSError as exc:
                raise ReductionError(f"could not read master flat {flat_path}: {exc}") from exc

            if flatnorm is None:
                raise ReductionError(
                    f"master flat {flat_path} has no FLATNORM value; cannot restore "
                    "the detector sensitivity removed by per-detector normalization"
                )
            try:
                scale = float(flatnorm)
            except (TypeError, ValueError) as exc:
                raise ReductionError(f"invalid FLATNORM value {flatnorm!r} in {flat_path}") from exc
            if not np.isfinite(scale) or scale <= 0:
                raise ReductionError(f"invalid FLATNORM value {flatnorm!r} in {flat_path}")
            try:
                active_width = int(active_width_value)
            except (TypeError, ValueError) as exc:
                raise ReductionError(
                    f"invalid ACTWIDTH value {active_width_value!r} in {flat_path}"
                ) from exc
            if active_width <= 0 or active_width > flat.shape[1]:
                raise ReductionError(
                    f"invalid ACTWIDTH value {active_width_value!r} for image shape "
                    f"{flat.shape} in {flat_path}"
                )

            restored = flat * np.float32(scale)
            active = restored[:, :active_width]
            valid = np.isfinite(active) & (active > 0)
            if not np.any(valid):
                raise ReductionError(f"master flat {flat_path} has no finite positive pixels")
            raw_flats[detector] = restored
            restored_samples.append(active[valid])

        global_median = float(np.median(np.concatenate(restored_samples)))
        if not np.isfinite(global_median) or global_median <= 0:
            raise ReductionError(
                f"invalid global master-flat median for filter={normalized_filter}: {global_median}"
            )

        masters = {
            detector: np.asarray(flat / np.float32(global_median), dtype=np.float32)
            for detector, flat in raw_flats.items()
        }
        self._master_flat_cache[normalized_filter] = masters
        logger.info(
            "Loaded globally normalized master flats for filter=%s from %s using shared median %.6g",
            normalized_filter,
            product_dir,
            global_median,
        )
        return masters

    def _raw_flat_groups(self) -> tuple[dict[_FlatKey, list[FrameReference]], list[str]]:
        groups: dict[_FlatKey, list[FrameReference]] = {}
        invalid: list[str] = []
        for reference in self.calibration_frames:
            if reference.obstype != "dflat":
                invalid.append(f"{reference.path}: expected OBSTYPE=dflat, found {reference.obstype!r}")
                continue
            filt_value = _header_value(reference, "FILTER")
            if filt_value is None or not str(filt_value).strip():
                invalid.append(f"{reference.path}: FILTER is missing")
                continue
            filt = _normalize_filter(filt_value)
            lamp_value = _header_value(reference, "DFLAMPVA")
            if lamp_value is None:
                invalid.append(f"{reference.path}: DFLAMPVA is missing")
                continue
            try:
                lamp = float(lamp_value)
            except (TypeError, ValueError):
                invalid.append(f"{reference.path}: DFLAMPVA has malformed value {lamp_value!r}")
                continue
            if not np.isfinite(lamp):
                invalid.append(f"{reference.path}: DFLAMPVA has non-finite value {lamp_value!r}")
                continue
            if set(reference.shapes) != set(_DETECTORS):
                invalid.append(
                    f"{reference.path}: expected image extensions {_DETECTORS}, "
                    f"found {sorted(reference.shapes)}"
                )
                continue
            key = (filt, "on" if lamp > 0 else "off")
            groups.setdefault(key, []).append(reference)
        for references in groups.values():
            references.sort(key=lambda ref: str(ref.path))
        return groups, invalid

    def _generated_master_path(self, filt: str, detector: int) -> Path:
        directory = self._generated_master_dir()
        if directory is None:
            raise ReductionError("cannot construct raw master flat without a flat-stage output directory")
        return directory / f"nflat_{filt}_{detector}.fits"

    @staticmethod
    def _calibration_fingerprint(references: Sequence[FrameReference]) -> str:
        digest = hashlib.sha256()
        for reference in sorted(references, key=lambda ref: str(ref.path)):
            stat = reference.path.stat()
            digest.update(str(reference.path.resolve()).encode())
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        return digest.hexdigest()

    @staticmethod
    def _generated_product_is_current(
        path: Path, filt: str, detector: int, fingerprint: str
    ) -> bool:
        if not path.is_file():
            return False
        try:
            with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
                header = hdul[0].header
                return (
                    _normalize_filter(header.get("FILTER", "")) == filt
                    and str(header.get("PROCTYPE", "")).strip().lower() == "mastercal"
                    and int(header.get("DETINDEX", -1)) == detector
                    and header.get("CALHASH") == fingerprint
                    and any(int(hdu.header.get("NAXIS", 0)) == 2 for hdu in hdul)
                )
        except (OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _flat_mode(reference: FrameReference, detector: int) -> float:
        try:
            frame = reference.load(extension_indices=[detector])
            data = frame[detector].data
            data = data[:, :_active_width(reference, detector, data.shape[1])]
            valid = data[np.isfinite(data) & (data > 0)]
            if valid.size == 0:
                raise ReductionError("image contains no finite positive pixels")

            integer_values = valid.astype(np.int64)
            if np.array_equal(valid, integer_values):
                lower = int(integer_values.min())
                upper = int(integer_values.max())
                if upper - lower <= 1_000_000:
                    # Raw NEWFIRM flat pixels are integer-valued. Bincount is
                    # exact here and avoids sorting millions of pixels to find
                    # their mode. Ignore zero/negative invalid pixels, which
                    # otherwise dominate some untrimmed detector extensions.
                    mode = float(lower + np.argmax(np.bincount(integer_values - lower)))
                else:
                    mode = float(stats.mode(valid, axis=None, keepdims=False).mode)
            else:
                mode = float(stats.mode(valid, axis=None, keepdims=False).mode)
        except Exception as exc:
            if isinstance(exc, ReductionError):
                raise ReductionError(
                    f"lamp flat {reference.path.name} detector extension {detector}: {exc}"
                ) from exc
            raise ReductionError(
                f"could not calculate lamp-flat mode for {reference.path.name} "
                f"detector extension {detector}: {exc}"
            ) from exc
        if not np.isfinite(mode) or mode <= 0:
            raise ReductionError(
                f"lamp flat {reference.path.name} detector extension {detector} "
                f"has invalid mode {mode}"
            )
        return mode

    @staticmethod
    def _median_band(
        references: Sequence[FrameReference],
        hdulists: Sequence[fits.HDUList],
        detector: int,
        y0: int,
        y1: int,
        scales: Sequence[float],
        width: int,
    ) -> np.ndarray:
        # Keep only one narrow band from each raw image in memory. The stack is
        # bounded by number_of_flats * _BAND_ROWS * detector_width, regardless
        # of the full frame dimensions.
        values = np.empty((len(references), y1 - y0, width), dtype=np.float32)
        for index, (reference, hdul, scale) in enumerate(zip(references, hdulists, scales)):
            try:
                hdu = hdul[detector]
                section = hdu.section[y0:y1, :]
                values[index] = np.asarray(section, dtype=np.float32) * np.float32(scale)
            except Exception as exc:
                raise ReductionError(
                    f"could not read {reference.path.name} detector extension {detector} "
                    f"rows {y0}:{y1}: {exc}"
                ) from exc
        return np.asarray(np.median(values, axis=0), dtype=np.float32)

    def _build_flat_product_data(
        self,
        filt: str,
        detector: int,
        on_refs: Sequence[FrameReference],
        off_refs: Sequence[FrameReference],
    ) -> tuple[np.ndarray, float]:
        if not on_refs or not off_refs:
            raise ReductionError(
                f"filter={filt} requires both lamp-on and lamp-off OBSTYPE=dflat frames "
                f"(lamp-on={len(on_refs)}, lamp-off={len(off_refs)})"
            )
        shape = on_refs[0].shapes[detector]
        for reference in [*on_refs, *off_refs]:
            if reference.shapes[detector] != shape:
                raise ReductionError(
                    f"{reference.path.name}: detector extension {detector} shape "
                    f"{reference.shapes[detector]} differs from {shape} for filter={filt}"
                )

        on_modes = [self._flat_mode(reference, detector) for reference in on_refs]
        off_modes = [self._flat_mode(reference, detector) for reference in off_refs]
        on_scales = [on_modes[0] / mode for mode in on_modes]
        off_scales = [off_modes[0] / mode for mode in off_modes]

        difference = np.empty(shape, dtype=np.float32)
        height, width = shape
        active_width = _active_width(on_refs[0], detector, width)
        try:
            with ExitStack() as stack:
                on_hdulists = [
                    stack.enter_context(fits.open(ref.path, memmap=True, lazy_load_hdus=True))
                    for ref in on_refs
                ]
                off_hdulists = [
                    stack.enter_context(fits.open(ref.path, memmap=True, lazy_load_hdus=True))
                    for ref in off_refs
                ]
                for y0 in range(0, height, _BAND_ROWS):
                    y1 = min(height, y0 + _BAND_ROWS)
                    on_master = self._median_band(
                        on_refs, on_hdulists, detector, y0, y1, on_scales, width
                    )
                    off_master = self._median_band(
                        off_refs, off_hdulists, detector, y0, y1, off_scales, width
                    )
                    difference[y0:y1] = on_master - off_master
        except OSError as exc:
            raise ReductionError(
                f"could not open raw flat inputs for filter={filt}, detector extension {detector}: {exc}"
            ) from exc

        flatnorm = float(np.median(difference[:, :active_width]))
        if not np.isfinite(flatnorm) or flatnorm <= 0:
            raise ReductionError(
                f"lamp-on minus lamp-off master has invalid median {flatnorm} "
                f"for filter={filt}, detector extension {detector}"
            )
        difference /= np.float32(flatnorm)
        return difference, flatnorm

    def _write_master_flat_product(
        self,
        destination: Path,
        filt: str,
        detector: int,
        on_refs: Sequence[FrameReference],
        off_refs: Sequence[FrameReference],
        fingerprint: str,
    ) -> None:
        data, flatnorm = self._build_flat_product_data(filt, detector, on_refs, off_refs)
        first = on_refs[0]
        header = first.header.copy()
        detector_header = first[detector].header
        header.extend(detector_header, update=True, strip=True)
        header["FILTER"] = filt
        header["OBSTYPE"] = "dflat"
        header["PROCTYPE"] = "MasterCal"
        header["PRODTYPE"] = "image"
        header["DETINDEX"] = detector
        header["ACTWIDTH"] = _active_width(first, detector, data.shape[1])
        header["NCOMBINE"] = len(on_refs)
        header["NOFFCOMB"] = len(off_refs)
        header["FLATNORM"] = (flatnorm, "Median before per-detector flat normalization")
        header["CALHASH"] = fingerprint
        for index, reference in enumerate([*on_refs, *off_refs], start=1):
            header[f"IMCMB{index:03d}"] = reference.path.name

        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            fits.PrimaryHDU(data=data, header=header).writeto(temporary_path, overwrite=True)
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        logger.info(
            "Constructed master flat filter=%s detector=%d from %d lamp-on and %d lamp-off "
            "OBSTYPE=dflat frame(s): %s",
            filt,
            detector,
            len(on_refs),
            len(off_refs),
            destination,
        )
