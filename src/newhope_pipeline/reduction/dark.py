"""Dark subtraction using the legacy NEWFIRM master-dark matching rules."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Mapping

import numpy as np
from astropy.io import fits

from ..core.exceptions import ReductionError
from ..core.frame import Extension, Frame
from ..core.stage import PipelineStage

logger = logging.getLogger(__name__)


_MASTER_NAME = re.compile(
    r"^dark_(?P<expcoadd>\d+)s_(?P<coadds>\d+)c_(?P<fsample>\d+)f"
    r"(?:_(?P<detector>[1-4]))?\.fits$",
    re.IGNORECASE,
)
_DarkKey = tuple[float, int, int]  # EXPCOADD, COADDS, FSAMPLE


class DarkSubtraction(PipelineStage):
    name = "dark_subtraction"

    def __init__(self, config, n_workers: int = 1, timeout: float | None = None) -> None:
        super().__init__(config, n_workers=n_workers, timeout=timeout)
        self._master_dark_cache: dict[_DarkKey, Mapping[int, np.ndarray]] = {}

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

    def _load_master_dark(self, frame: Frame) -> Mapping[int, np.ndarray]:
        """Load the closest compatible dark and scale it to the science frame.

        This ports ``find_matching_dark`` from ``DarkFlatCorrect.py``. Exact
        EXPCOADD/COADDS/FSAMPLE masters are preferred. If one is unavailable,
        the nearest EXPCOADD master with matching COADDS and FSAMPLE is used;
        the final fallback matches FSAMPLE only. As in the legacy routine,
        dark signal is scaled by science EXPCOADD / master EXPCOADD.

        Masters can be the legacy per-detector files
        ``dark_<expcoadd>s_<coadds>c_<fsample>f_<detector>.fits`` in
        ``dark_dir/<detector>/`` or directly in ``dark_dir``, or a four-image
        MEF dark master in ``dark_dir``. Per-detector files must provide all
        four detectors for the selected metadata tuple.
        """
        target = self._frame_dark_key(frame)
        if target in self._master_dark_cache:
            logger.debug("Dark calibration cache hit for %s (%s)", frame.path.name, target)
            return self._master_dark_cache[target]

        dark_dir = getattr(self.config, "dark_dir", None)
        if dark_dir is None:
            raise ReductionError("dark_dir is not configured; cannot load master darks")
        dark_dir = Path(dark_dir)
        logger.info(
            "Selecting master dark for %s: EXPCOADD=%s, COADDS=%s, FSAMPLE=%s (searching %s)",
            frame.path.name,
            *target,
            dark_dir,
        )

        per_detector: dict[_DarkKey, dict[int, Path]] = {}
        mef_masters: dict[_DarkKey, Path] = {}
        candidate_paths = set(dark_dir.glob("dark*.fits"))
        for detector in range(1, 5):
            candidate_paths.update((dark_dir / str(detector)).glob("dark*.fits"))

        for path in sorted(candidate_paths):
            filename_match = _MASTER_NAME.match(path.name)
            detector_from_name = (
                int(filename_match.group("detector"))
                if filename_match and filename_match.group("detector")
                else None
            )
            detector_from_dir = int(path.parent.name) if path.parent.name in {"1", "2", "3", "4"} else None

            try:
                with fits.open(path, memmap=False) as hdul:
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

                    primary_header = hdul[0].header
                    first_image_header = image_hdus[0][1].header
                    key = self._dark_key_from_headers(
                        primary_header, first_image_header, filename_match
                    )
                    if key is None:
                        continue

                    if int(primary_header.get("NAXIS", 0)) == 0 and len(image_hdus) == 4:
                        mef_masters.setdefault(key, path)
                        continue

                    detector = detector_from_name or detector_from_dir
                    if detector is None or len(image_hdus) != 1:
                        continue
                    per_detector.setdefault(key, {}).setdefault(detector, path)
            except OSError:
                # Ignore unrelated or unreadable dark-named files while
                # looking for a complete, usable master set.
                continue

        complete_keys = {
            key for key, files in per_detector.items() if all(i in files for i in range(1, 5))
        }
        available_keys = complete_keys | set(mef_masters)
        selected_key = self._select_dark_key(target, available_keys)
        if selected_key is None:
            raise ReductionError(
                f"no compatible master dark in {dark_dir} for "
                f"EXPCOADD={target[0]}, COADDS={target[1]}, FSAMPLE={target[2]}"
            )

        if selected_key == target:
            scale = 1.0
        else:
            scale = target[0] / selected_key[0]
        if not np.isfinite(scale) or scale <= 0:
            raise ReductionError(
                f"invalid dark scale {scale} for science EXPCOADD={target[0]} "
                f"and master EXPCOADD={selected_key[0]}"
            )

        if selected_key in mef_masters:
            selected_paths = [mef_masters[selected_key]]
            arrays = self._read_mef_master(selected_paths[0], dark_dir)
        else:
            selected_paths = [per_detector[selected_key][detector] for detector in range(1, 5)]
            arrays = {
                detector: self._read_single_master(selected_paths[detector - 1], dark_dir)
                for detector in range(1, 5)
            }

        scaled_masters = {
            detector: np.asarray(data * np.float32(scale), dtype=np.float32)
            for detector, data in arrays.items()
        }
        self._master_dark_cache[target] = scaled_masters
        logger.info(
            "Selected master dark metadata=%s, scale=%.6g, file(s)=%s",
            selected_key,
            scale,
            ", ".join(str(path) for path in selected_paths),
        )
        return scaled_masters

    @staticmethod
    def _frame_dark_key(frame: Frame) -> _DarkKey:
        """Read matching fields from the primary header, then detector headers."""
        def header_value(name: str):
            value = frame.header.get(name)
            if value is not None:
                return value
            for extension in frame.extensions:
                value = extension.header.get(name)
                if value is not None:
                    return value
            return None

        try:
            coadds_value = header_value("COADDS")
            fsample_value = header_value("FSAMPLE")
            if coadds_value is None or fsample_value is None:
                raise ValueError("COADDS or FSAMPLE is missing")
            coadds = int(float(coadds_value))
            fsample = int(float(fsample_value))

            expcoadd_value = header_value("EXPCOADD")
            if expcoadd_value is None:
                # Some headers only record total EXPTIME; approximate the
                # per-coadd exposure required by the legacy scale rule.
                expcoadd = float(frame.exptime) / coadds
            else:
                expcoadd = float(expcoadd_value)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ReductionError(
                f"{frame.path.name} must provide numeric EXPCOADD, COADDS, and FSAMPLE"
            ) from exc

        if coadds <= 0 or fsample <= 0 or not np.isfinite(expcoadd) or expcoadd <= 0:
            raise ReductionError(
                f"invalid dark-matching metadata in {frame.path.name}: "
                f"EXPCOADD={expcoadd}, COADDS={coadds}, FSAMPLE={fsample}"
            )
        return (expcoadd, coadds, fsample)

    @staticmethod
    def _dark_key_from_headers(
        primary_header: fits.Header,
        image_header: fits.Header,
        filename_match: re.Match[str] | None,
    ) -> _DarkKey | None:
        def value(name: str, filename_field: str):
            item = primary_header.get(name, image_header.get(name))
            if item is None and filename_match is not None:
                item = filename_match.group(filename_field)
            return item

        try:
            expcoadd = float(value("EXPCOADD", "expcoadd"))
            coadds = int(float(value("COADDS", "coadds")))
            fsample = int(float(value("FSAMPLE", "fsample")))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(expcoadd) or expcoadd <= 0 or coadds <= 0 or fsample <= 0:
            return None
        return (expcoadd, coadds, fsample)

    @staticmethod
    def _select_dark_key(target: _DarkKey, available: set[_DarkKey]) -> _DarkKey | None:
        if target in available:
            return target

        expcoadd, coadds, fsample = target
        same_setup = [key for key in available if key[1] == coadds and key[2] == fsample]
        if not same_setup:
            same_setup = [key for key in available if key[2] == fsample]
        if not same_setup:
            return None

        return min(same_setup, key=lambda key: (abs(key[0] - expcoadd), key[0]))

    @staticmethod
    def _read_single_master(path: Path, dark_dir: Path) -> np.ndarray:
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
            raise ReductionError(f"could not read master dark {path} in {dark_dir}: {exc}") from exc

    @staticmethod
    def _read_mef_master(path: Path, dark_dir: Path) -> Mapping[int, np.ndarray]:
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
            raise ReductionError(f"could not read master dark {path} in {dark_dir}: {exc}") from exc
