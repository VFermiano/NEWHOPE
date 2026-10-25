"""Dark subtraction and compatibility-aware master-dark calibration."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from astropy.io import fits

from ..core.exceptions import ReductionError
from ..core.frame import Extension, Frame, FrameCollection, FrameLike, FrameReference
from ..core.stage import PipelineStage
from ..core.state import PipelineState

logger = logging.getLogger(__name__)


_MASTER_NAME = re.compile(
    r"^dark_(?P<expcoadd>\d+(?:[p.]\d+)?)s_(?P<coadds>\d+)c_(?P<fsample>\d+)f"
    r"(?:_(?P<detector>[1-4]))?\.fits$",
    re.IGNORECASE,
)
_DarkKey = tuple[float, int, int]  # EXPCOADD, COADDS, FSAMPLE
_DETECTORS = (1, 2, 3, 4)


class DarkSubtraction(PipelineStage):
    name = "dark_subtraction"

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
        self._master_dark_cache: dict[_DarkKey, Mapping[int, np.ndarray]] = {}

    def run(self, frames: FrameCollection, state: PipelineState) -> FrameCollection:
        """Prepare required raw-dark masters before frame workers start."""
        self._ensure_reference_persistence(frames, state)
        self._prepare_raw_master_darks(frames, state)
        return super().run(frames, state)

    def process(self, frame: Frame) -> Frame:
        logger.debug("Dark subtraction started for %s", frame.path.name)
        masters = self._load_master_dark(frame)
        missing = [e.index for e in frame if e.index not in masters]
        if missing:
            raise ReductionError(
                f"no master dark for extension(s) {missing} "
                f"(EXPCOADD/COADDS/FSAMPLE from {frame.path.name})"
            )

        def subtract(ext: Extension) -> np.ndarray:
            master = masters[ext.index]
            if master.shape != ext.shape:
                raise ReductionError(
                    f"master dark shape {master.shape} does not match frame "
                    f"{frame.frame_id} extension {ext.index} shape {ext.shape}"
                )
            return ext.data - master

        result = frame.map_extensions(subtract)
        logger.debug("Dark subtraction finished for %s", frame.path.name)
        return result

    def _prepare_raw_master_darks(
        self, frames: FrameCollection, state: PipelineState
    ) -> None:
        """Build the raw-dark groups needed by pending science frames, once per run.

        Products live below the dark stage output directory. Their source
        fingerprint is recorded in the product header so a changed calibration
        set is recombined on the next run. The stage manifest and science
        outputs remain managed by PipelineStage.
        """
        if not self.calibration_frames or self.output_dir is None:
            return

        raw_groups, invalid_raw = self._raw_dark_groups()
        if not raw_groups:
            for detail in invalid_raw:
                logger.error("Ignoring unusable raw dark calibration: %s", detail)
            return

        dark_dir = self._configured_dark_dir()
        generated_dir = self._generated_master_dir()
        generated_dir.mkdir(parents=True, exist_ok=True)
        external_products = self._discover_master_products([dark_dir] if dark_dir else [])
        generated_products = self._discover_master_products([generated_dir])
        external_keys = self._complete_master_keys(external_products)
        generated_keys = self._complete_master_keys(generated_products)

        prepared: set[_DarkKey] = set()
        for frame in frames:
            output_path = self._output_path(frame)
            if (
                state.is_done(frame.frame_id, self.name)
                and output_path is not None
                and output_path.is_file()
            ):
                continue
            try:
                target = self._frame_dark_key(frame)
            except ReductionError:
                # Let normal stage processing record this as a per-frame failure.
                continue

            available = external_keys | generated_keys | set(raw_groups)
            selected = self._select_dark_key(target, available)
            if selected is None or selected not in raw_groups or selected in external_keys:
                continue
            if selected in prepared:
                continue

            sources = raw_groups[selected]
            destination = self._generated_master_path(selected)
            fingerprint = self._calibration_fingerprint(sources)
            if self._generated_product_is_current(destination, selected, fingerprint):
                prepared.add(selected)
                continue
            try:
                self._write_master_dark_product(destination, selected, sources, fingerprint)
            except Exception:
                # Keep failures frame-scoped: _load_master_dark will retry the
                # combination in its worker and report construction context.
                logger.exception(
                    "Could not prepare raw master dark metadata=%s; frame workers will retry",
                    selected,
                )
            else:
                generated_keys.add(selected)
                prepared.add(selected)

        for detail in invalid_raw:
            logger.warning("Raw OBSTYPE=dark calibration was excluded: %s", detail)

    def _load_master_dark(self, frame: Frame) -> Mapping[int, np.ndarray]:
        """Load or construct a dark with the same COADDS and FSAMPLE setup.

        An exact EXPCOADD match is preferred. If none exists, the nearest
        EXPCOADD is scaled only among masters with the same COADDS and FSAMPLE.
        Raw OBSTYPE=dark exposures are mean-combined per detector, matching the
        legacy SortCombDarks.py recipe. Precomputed legacy master products are
        still supported.
        """
        target = self._frame_dark_key(frame)
        if target in self._master_dark_cache:
            logger.debug("Dark calibration cache hit for %s (%s)", frame.path.name, target)
            return self._master_dark_cache[target]
        # Keep only one scaled calibration key resident per stage instance.
        self._master_dark_cache.clear()

        dark_dir = self._configured_dark_dir()
        generated_dir = self._generated_master_dir()
        search_dirs = [path for path in (dark_dir, generated_dir) if path is not None]
        logger.info(
            "Selecting master dark for %s: EXPCOADD=%s, COADDS=%s, FSAMPLE=%s (searching %s)",
            frame.path.name,
            *target,
            ", ".join(str(path) for path in search_dirs) or "raw OBSTYPE=dark frames",
        )

        per_detector, mef_masters, invalid_products = self._discover_master_products(search_dirs)
        raw_groups, invalid_raw = self._raw_dark_groups()
        available_keys = self._complete_master_keys((per_detector, mef_masters)) | set(raw_groups)
        selected_key = self._select_dark_key(target, available_keys)
        if selected_key is None:
            if not available_keys:
                details = []
                if invalid_products:
                    details.append("invalid precomputed products: " + "; ".join(invalid_products))
                if invalid_raw:
                    details.append("invalid raw dark headers: " + "; ".join(invalid_raw))
                detail_text = "; " + "; ".join(details) if details else ""
                raise ReductionError(
                    f"no master dark calibration available in {', '.join(map(str, search_dirs)) or '<unset>'}: "
                    "no usable precomputed master darks or OBSTYPE=dark frames were found"
                    f"{detail_text}"
                )
            keys = ", ".join(str(key) for key in sorted(available_keys))
            raise ReductionError(
                f"no compatible master dark for EXPCOADD={target[0]}, COADDS={target[1]}, "
                f"FSAMPLE={target[2]}; available keys are [{keys}]. Compatibility requires "
                "matching COADDS and FSAMPLE; only EXPCOADD may be scaled."
            )

        selected_paths: list[Path] = []
        try:
            if selected_key in mef_masters:
                selected_paths = [mef_masters[selected_key]]
                arrays = self._read_mef_master(selected_paths[0], search_dirs)
            elif selected_key in per_detector:
                selected_paths = [per_detector[selected_key][detector] for detector in _DETECTORS]
                arrays = {
                    detector: self._read_single_master(selected_paths[index - 1], search_dirs)
                    for index, detector in enumerate(_DETECTORS, start=1)
                }
            elif selected_key in raw_groups:
                sources = raw_groups[selected_key]
                selected_paths = [reference.path for reference in sources]
                arrays = self._build_master_dark(selected_key, sources)
            else:
                raise ReductionError(
                    f"master dark metadata {selected_key} was indexed but has no usable product or raw inputs"
                )
        except Exception as exc:
            if isinstance(exc, ReductionError) and "failed to construct" in str(exc):
                raise
            if selected_key in raw_groups:
                raise ReductionError(
                    f"failed to construct master dark for metadata={selected_key} from "
                    f"{len(raw_groups[selected_key])} raw dark frame(s): {exc}"
                ) from exc
            raise

        scale = target[0] / selected_key[0]
        if not np.isfinite(scale) or scale <= 0:
            raise ReductionError(
                f"invalid dark scale {scale} for science EXPCOADD={target[0]} "
                f"and master EXPCOADD={selected_key[0]}"
            )

        scaled_masters = {
            detector: np.asarray(data * np.float32(scale), dtype=np.float32)
            for detector, data in arrays.items()
        }
        self._master_dark_cache[target] = scaled_masters
        logger.info(
            "Selected master dark metadata=%s, scale=%.6g, source(s)=%s",
            selected_key,
            scale,
            ", ".join(str(path) for path in selected_paths),
        )
        return scaled_masters

    def _raw_dark_groups(
        self,
    ) -> tuple[dict[_DarkKey, list[FrameReference]], list[str]]:
        groups: dict[_DarkKey, list[FrameReference]] = {}
        invalid: list[str] = []
        for reference in self.calibration_frames:
            if reference.obstype != "dark":
                continue
            try:
                key = self._frame_dark_key(reference)
            except ReductionError as exc:
                invalid.append(str(exc))
                continue
            groups.setdefault(key, []).append(reference)
        for references in groups.values():
            references.sort(key=lambda reference: str(reference.path))
        return groups, invalid

    @staticmethod
    def _frame_dark_key(frame: FrameLike) -> _DarkKey:
        """Read and normalize all required acquisition fields from FITS headers."""
        def header_value(name: str):
            value = frame.header.get(name)
            if value is not None:
                return value
            for extension in frame.extensions:
                value = extension.header.get(name)
                if value is not None:
                    return value
            return None

        def required(name: str):
            value = header_value(name)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ReductionError(
                    f"{frame.path.name}: required dark-matching FITS keyword {name} is missing"
                )
            return value

        expcoadd_value = required("EXPCOADD")
        coadds_value = required("COADDS")
        fsample_value = required("FSAMPLE")
        expcoadd = DarkSubtraction._positive_float(expcoadd_value, "EXPCOADD", frame.path.name)
        coadds = DarkSubtraction._positive_int(coadds_value, "COADDS", frame.path.name)
        fsample = DarkSubtraction._positive_int(fsample_value, "FSAMPLE", frame.path.name)
        return (expcoadd, coadds, fsample)

    @staticmethod
    def _positive_float(value, keyword: str, source: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ReductionError(
                f"{source}: dark-matching FITS keyword {keyword} has non-numeric value {value!r}"
            ) from exc
        if not np.isfinite(parsed) or parsed <= 0:
            raise ReductionError(
                f"{source}: dark-matching FITS keyword {keyword} must be finite and positive; got {value!r}"
            )
        return parsed

    @staticmethod
    def _positive_int(value, keyword: str, source: str) -> int:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ReductionError(
                f"{source}: dark-matching FITS keyword {keyword} has non-numeric value {value!r}"
            ) from exc
        if not np.isfinite(parsed) or parsed <= 0 or not parsed.is_integer():
            raise ReductionError(
                f"{source}: dark-matching FITS keyword {keyword} must be a positive integer; got {value!r}"
            )
        return int(parsed)

    @classmethod
    def _dark_key_from_headers(
        cls,
        primary_header: fits.Header,
        image_header: fits.Header,
        filename_match: re.Match[str] | None,
        source: str = "master dark",
    ) -> _DarkKey | None:
        values = []
        for keyword, filename_field in (
            ("EXPCOADD", "expcoadd"),
            ("COADDS", "coadds"),
            ("FSAMPLE", "fsample"),
        ):
            value = primary_header.get(keyword)
            if value is None:
                value = image_header.get(keyword)
            if value is None and filename_match is not None:
                value = filename_match.group(filename_field)
                if keyword == "EXPCOADD" and isinstance(value, str):
                    value = value.replace("p", ".").replace("P", ".")
            if value is None:
                return None
            try:
                if keyword == "EXPCOADD":
                    values.append(cls._positive_float(value, keyword, source))
                else:
                    values.append(cls._positive_int(value, keyword, source))
            except ReductionError:
                return None
        return (values[0], values[1], values[2])

    @staticmethod
    def _select_dark_key(target: _DarkKey, available: set[_DarkKey]) -> _DarkKey | None:
        if target in available:
            return target
        same_setup = [key for key in available if key[1] == target[1] and key[2] == target[2]]
        if not same_setup:
            return None
        return min(same_setup, key=lambda key: (abs(key[0] - target[0]), key[0]))

    def _configured_dark_dir(self) -> Path | None:
        directory = getattr(self.config, "dark_dir", None)
        return None if directory is None else Path(directory)

    def _generated_master_dir(self) -> Path | None:
        return None if self.output_dir is None else Path(self.output_dir) / "master_darks"

    def _generated_master_path(self, key: _DarkKey) -> Path:
        directory = self._generated_master_dir()
        if directory is None:
            raise ReductionError("dark stage output_dir is required to persist generated master darks")
        exp_token = format(key[0], ".12g").replace(".", "p").replace("-", "m")
        return directory / f"dark_master_{exp_token}s_{key[1]}c_{key[2]}f.fits"

    @staticmethod
    def _complete_master_keys(products) -> set[_DarkKey]:
        per_detector, mef_masters = products[:2]
        return {
            key for key, files in per_detector.items() if all(detector in files for detector in _DETECTORS)
        } | set(mef_masters)

    @classmethod
    def _discover_master_products(
        cls, directories: Sequence[Path]
    ) -> tuple[dict[_DarkKey, dict[int, Path]], dict[_DarkKey, Path], list[str]]:
        per_detector: dict[_DarkKey, dict[int, Path]] = {}
        mef_masters: dict[_DarkKey, Path] = {}
        invalid: list[str] = []
        for directory in directories:
            if not directory.is_dir():
                continue
            candidate_paths = set(directory.glob("dark*.fits"))
            for detector in _DETECTORS:
                candidate_paths.update((directory / str(detector)).glob("dark*.fits"))
            for path in sorted(candidate_paths):
                filename_match = _MASTER_NAME.match(path.name)
                detector_from_name = (
                    int(filename_match.group("detector"))
                    if filename_match and filename_match.group("detector")
                    else None
                )
                detector_from_dir = (
                    int(path.parent.name) if path.parent.name in {"1", "2", "3", "4"} else None
                )
                try:
                    with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
                        if not hdul:
                            continue
                        primary_header = hdul[0].header
                        image_hdus = [
                            (index, hdu)
                            for index, hdu in enumerate(hdul)
                            if int(hdu.header.get("NAXIS", 0)) == 2
                            and (
                                index == 0
                                or str(hdu.header.get("XTENSION", "")).strip().upper() == "IMAGE"
                            )
                        ]
                        if not image_hdus:
                            continue
                        first_image_header = image_hdus[0][1].header
                        is_master_product = filename_match is not None or str(
                            primary_header.get("PROCTYPE", "")
                        ).strip().lower() in {"mastercal", "master"}
                        if not is_master_product:
                            # Raw dark MEFs are also four-image files. Do not treat
                            # them as masters just because their filenames start
                            # with "dark".
                            continue
                        key = cls._dark_key_from_headers(
                            primary_header, first_image_header, filename_match, str(path)
                        )
                        if key is None:
                            invalid.append(f"{path}: missing or malformed EXPCOADD/COADDS/FSAMPLE")
                            continue
                        if int(primary_header.get("NAXIS", 0)) == 0 and len(image_hdus) == 4:
                            mef_masters.setdefault(key, path)
                            continue
                        detector = detector_from_name or detector_from_dir
                        if detector is None or len(image_hdus) != 1:
                            invalid.append(
                                f"{path}: master must be a four-image MEF or detector-specific image"
                            )
                            continue
                        per_detector.setdefault(key, {}).setdefault(detector, path)
                except (OSError, ValueError) as exc:
                    invalid.append(f"{path}: could not inspect master dark: {exc}")
        return per_detector, mef_masters, invalid

    def _build_master_dark(
        self, key: _DarkKey, references: Sequence[FrameReference]
    ) -> Mapping[int, np.ndarray]:
        """Mean-combine one raw dark group with one detector array resident at a time."""
        if not references:
            raise ReductionError(f"no raw dark inputs available for metadata={key}")
        for reference in references:
            if reference.obstype != "dark":
                raise ReductionError(f"{reference.path.name}: raw calibration is not OBSTYPE=dark")
            actual_key = self._frame_dark_key(reference)
            if actual_key != key:
                raise ReductionError(
                    f"{reference.path.name}: raw dark metadata {actual_key} does not match group {key}"
                )
            if set(reference.shapes) != set(_DETECTORS):
                raise ReductionError(
                    f"{reference.path.name}: expected detector extensions {_DETECTORS}, "
                    f"found {sorted(reference.shapes)}"
                )

        masters: dict[int, np.ndarray] = {}
        for detector in _DETECTORS:
            shape = references[0].shapes[detector]
            accumulator = np.zeros(shape, dtype=np.float64)
            for reference in references:
                if reference.shapes[detector] != shape:
                    raise ReductionError(
                        f"{reference.path.name}: detector {detector} shape "
                        f"{reference.shapes[detector]} differs from {shape} in the dark group"
                    )
                try:
                    one_extension = reference.load(extension_indices=[detector])
                    accumulator += one_extension[detector].data
                except Exception as exc:
                    raise ReductionError(
                        f"could not read raw dark {reference.path.name} detector {detector}: {exc}"
                    ) from exc
            masters[detector] = np.asarray(accumulator / len(references), dtype=np.float32)
            del accumulator

        return masters

    @staticmethod
    def _calibration_fingerprint(references: Sequence[FrameReference]) -> str:
        digest = hashlib.sha256()
        for reference in references:
            stat = reference.path.stat()
            digest.update(str(reference.path.resolve()).encode())
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        return digest.hexdigest()

    def _generated_product_is_current(
        self, path: Path, key: _DarkKey, fingerprint: str
    ) -> bool:
        if not path.is_file():
            return False
        try:
            with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
                header = hdul[0].header
                parsed_key = self._dark_key_from_headers(
                    header,
                    next((hdu.header for hdu in hdul[1:] if int(hdu.header.get("NAXIS", 0)) == 2), header),
                    None,
                    str(path),
                )
                return (
                    parsed_key == key
                    and header.get("CALHASH") == fingerprint
                    and str(header.get("PROCTYPE", "")).strip().lower() == "mastercal"
                )
        except (OSError, ValueError):
            return False

    def _write_master_dark_product(
        self,
        destination: Path,
        key: _DarkKey,
        references: Sequence[FrameReference],
        fingerprint: str,
    ) -> None:
        arrays = self._build_master_dark(key, references)
        first = references[0]
        primary_header = first.header.copy()
        primary_header["OBSTYPE"] = "dark"
        primary_header["PROCTYPE"] = "MasterCal"
        primary_header["PRODTYPE"] = "image"
        primary_header["EXPCOADD"] = key[0]
        primary_header["COADDS"] = key[1]
        primary_header["FSAMPLE"] = key[2]
        primary_header["NCOMBINE"] = len(references)
        primary_header["CALHASH"] = fingerprint
        for index, reference in enumerate(references, start=1):
            primary_header[f"IMCMB{index:03d}"] = reference.path.name

        hdus = [fits.PrimaryHDU(header=primary_header)]
        for detector in _DETECTORS:
            extension_header = first[detector].header.copy()
            for scaling_key in ("BSCALE", "BZERO", "BLANK"):
                extension_header.remove(scaling_key, ignore_missing=True)
            extension_header["EXPCOADD"] = key[0]
            extension_header["COADDS"] = key[1]
            extension_header["FSAMPLE"] = key[2]
            extension_header["NCOMBINE"] = len(references)
            hdus.append(
                fits.ImageHDU(
                    data=arrays[detector],
                    header=extension_header,
                    name=first[detector].name or f"DET{detector}",
                )
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            fits.HDUList(hdus).writeto(temporary_path, overwrite=True)
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        logger.info(
            "Constructed master dark metadata=%s from %d OBSTYPE=dark frame(s): %s",
            key,
            len(references),
            destination,
        )

    @staticmethod
    def _read_single_master(path: Path, search_dirs: Sequence[Path]) -> np.ndarray:
        try:
            with fits.open(path, memmap=False) as hdul:
                image_hdu = next(
                    (
                        hdu
                        for hdu in hdul
                        if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2
                    ),
                    None,
                )
                if image_hdu is None:
                    raise ReductionError(f"master dark {path} contains no 2-D image")
                return np.array(image_hdu.data, dtype=np.float32, copy=True)
        except OSError as exc:
            raise ReductionError(f"could not read master dark {path} in {search_dirs}: {exc}") from exc

    @staticmethod
    def _read_mef_master(path: Path, search_dirs: Sequence[Path]) -> Mapping[int, np.ndarray]:
        try:
            with fits.open(path, memmap=False) as hdul:
                images = [
                    hdu.data
                    for hdu in hdul[1:]
                    if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2
                ]
                if len(images) != 4:
                    raise ReductionError(
                        f"MEF master dark {path} must contain four 2-D detector images; found {len(images)}"
                    )
                return {
                    detector: np.array(data, dtype=np.float32, copy=True)
                    for detector, data in enumerate(images, start=1)
                }
        except OSError as exc:
            raise ReductionError(f"could not read master dark {path} in {search_dirs}: {exc}") from exc
