"""Flat fielding with one normalization shared by all four detectors.

The legacy flat builder writes one normalized master flat per detector as
``nflat_<filter>_<detector>.fits``.  Its ``FLATNORM`` header keyword records
the median removed when that detector's flat was normalized.  Restoring that
scale before computing a single four-detector median keeps relative detector
sensitivities in the correction and avoids detector-dependent zeropoints.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np
from astropy.io import fits

from ..core.exceptions import ReductionError
from ..core.frame import Extension, Frame
from ..core.stage import PipelineStage

logger = logging.getLogger(__name__)


class FlatFielding(PipelineStage):
    name = "flat_fielding"

    def __init__(self, config, n_workers: int = 1, timeout: float | None = None) -> None:
        super().__init__(config, n_workers=n_workers, timeout=timeout)
        self._master_flat_cache: dict[str, Mapping[int, np.ndarray]] = {}

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
            # Match DarkFlatCorrect.py: divide by the master and replace
            # undefined pixels (for example, zero-valued flat pixels) with 0.
            with np.errstate(divide="ignore", invalid="ignore"):
                corrected = ext.data / master
            corrected = np.asarray(corrected, dtype=np.float32)
            corrected[~np.isfinite(corrected)] = 0.0
            return corrected

        result = frame.map_extensions(divide)
        logger.debug("Flat-field correction finished for %s", frame.path.name)
        return result

    def _load_master_flat(self, filt: str) -> Mapping[int, np.ndarray]:
        """Load and globally normalize the four per-detector master flats.

        Files follow the legacy names ``nflat_<filter>_<detector>.fits`` and
        may either be in ``flat_dir/<detector>/`` (the old calibration tree)
        or directly in ``flat_dir``.  The old builder normalized each detector
        independently, so ``FLATNORM`` is required to restore its removed
        scale. A single median across all restored detector pixels is then
        applied to every detector; per-detector normalization would erase the
        relative chip response and reintroduce zeropoint offsets.
        """
        if filt in self._master_flat_cache:
            logger.debug("Master-flat cache hit for filter=%s", filt)
            return self._master_flat_cache[filt]

        flat_dir = getattr(self.config, "flat_dir", None)
        if flat_dir is None:
            raise ReductionError("flat_dir is not configured; cannot load master flats")
        flat_dir = Path(flat_dir)
        logger.info("Loading four master flats for filter=%s from %s", filt, flat_dir)

        raw_flats: dict[int, np.ndarray] = {}
        restored_samples: list[np.ndarray] = []
        missing_paths: list[str] = []

        for index in range(1, 5):
            filename = f"nflat_{filt}_{index}.fits"
            candidates = (flat_dir / str(index) / filename, flat_dir / filename)
            flat_path = next((path for path in candidates if path.is_file()), None)
            if flat_path is None:
                missing_paths.append(" or ".join(str(path) for path in candidates))
                continue
            logger.debug("Reading detector %d master flat: %s", index, flat_path)

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
                    flatnorm = image_hdu.header.get(
                        "FLATNORM", hdul[0].header.get("FLATNORM")
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

            restored = flat * np.float32(scale)
            valid = np.isfinite(restored) & (restored > 0)
            if not np.any(valid):
                raise ReductionError(f"master flat {flat_path} has no finite positive pixels")

            raw_flats[index] = restored
            restored_samples.append(restored[valid])

        if missing_paths:
            raise ReductionError(
                f"missing master flat(s) for filter={filt}: " + "; ".join(missing_paths)
            )

        # This is intentionally one median over all four chips, not four
        # detector medians. It gives every normalized master one shared scale.
        global_median = float(np.median(np.concatenate(restored_samples)))
        if not np.isfinite(global_median) or global_median <= 0:
            raise ReductionError(f"invalid global master-flat median for filter={filt}: {global_median}")

        masters = {
            index: np.asarray(flat / np.float32(global_median), dtype=np.float32)
            for index, flat in raw_flats.items()
        }
        self._master_flat_cache[filt] = masters
        logger.info(
            "Loaded and globally normalized four master flats for filter=%s using shared median %.6g",
            filt,
            global_median,
        )
        return masters
