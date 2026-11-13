"""Calibrate PSF source catalogs against a local VVV reference catalog."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
from astropy.table import Table

from ..config import PhotometryConfig
from ..core.exceptions import PhotometryError, PipelineError
from ..core.frame import Frame, FrameLike
from ..core.stage import PipelineStage
from .common import (
    atomic_write_json,
    atomic_write_table,
    catalog_filename,
    file_stamp,
    numeric_column,
    product_signature,
)

logger = logging.getLogger(__name__)

_MAG_COLUMNS = {
    "JX": ("mag_1", "Jmag", "Jmag3", "J"),
    "HX": ("mag_2", "Hmag", "Hmag3", "H"),
    "KXS": ("mag_3", "Ksmag", "Kmag", "Ksmag3", "Ks"),
}
_RA_COLUMNS = ("ra", "RAJ2000", "RA_ICRS", "RA")
_DEC_COLUMNS = ("dec", "DEJ2000", "DEC_ICRS", "DEC")


def _empty_matches() -> Table:
    return Table(
        names=(
            "instrument_index",
            "instrumental_mag",
            "vvv_mag",
            "separation_arcsec",
        ),
        dtype=("i8", "f8", "f8", "f8"),
    )


def _column_name(table: Table, choices: tuple[str, ...]) -> str | None:
    by_upper = {name.upper(): name for name in table.colnames}
    for candidate in choices:
        if candidate.upper() in by_upper:
            return by_upper[candidate.upper()]
    return None


@lru_cache(maxsize=4)
def _read_reference_catalog(path_string: str, size: int, mtime_ns: int) -> Table:
    # Including the stamp in the cache key prevents a modified local catalog
    # from being silently reused by long-running workers.
    del size, mtime_ns
    try:
        table = Table.read(path_string)
    except Exception as exc:
        raise PhotometryError(
            f"Could not read VVV reference catalog {path_string}: {exc}"
        ) from exc
    if len(table) == 0:
        raise PhotometryError(f"VVV reference catalog is empty: {path_string}")
    if _column_name(table, _RA_COLUMNS) is None or _column_name(table, _DEC_COLUMNS) is None:
        raise PhotometryError(
            f"VVV catalog {path_string} must contain RA and Dec columns; "
            f"found {', '.join(table.colnames)}"
        )
    return table


class VVVCalibrationStage(PipelineStage):
    """Crossmatch each detector catalog and apply its robust VVV zero point."""

    name = "vvv_calibration"

    def __init__(
        self,
        config: PhotometryConfig,
        input_catalog_dir: Path | None = None,
        n_workers: int = 1,
        batch_size: int = 1,
    ) -> None:
        super().__init__(config, n_workers=n_workers, batch_size=batch_size)
        self.input_catalog_dir = (
            Path(input_catalog_dir) if input_catalog_dir is not None else None
        )

    def _input_catalog_dir(self, frame: FrameLike) -> Path:
        if self.input_catalog_dir is not None:
            return self.input_catalog_dir
        if self.output_dir is not None:
            return Path(self.output_dir).parent / "psf_photometry" / "catalogs"
        return frame.path.parent / "photometry" / "catalogs"

    def _output_catalog_dir(self, frame: FrameLike) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir) / "catalogs"
        return frame.path.parent / "vvv_calibration" / "catalogs"

    def _output_path(self, frame: FrameLike) -> Path | None:
        # Calibration is a catalog sidecar and does not modify the MEF image.
        return None

    def _process_item(self, item: FrameLike) -> Frame | FrameReference:
        if isinstance(item, FrameReference):
            self.process(item.load())
            return item
        return self.process(item)

    def _catalog_path(
        self, directory: Path, frame: FrameLike, extension
    ) -> Path:
        detector = extension.header.get("DETECTOR", extension.name)
        return directory / catalog_filename(frame.frame_id, extension.index, detector)

    def _signature(self, frame: FrameLike, extension) -> str:
        cfg: PhotometryConfig = self.config
        input_catalog = self._catalog_path(self._input_catalog_dir(frame), frame, extension)
        reference_path = Path(cfg.vvv_catalog_path).expanduser().resolve()
        try:
            instrumental_stamp = file_stamp(input_catalog)
        except OSError:
            instrumental_stamp = {"missing": 1}
        dependencies = {
            "instrumental_catalog": instrumental_stamp,
            "vvv_catalog_path": str(reference_path),
            "vvv_catalog": file_stamp(reference_path),
        }
        return product_signature(
            frame.path, extension.index, cfg, dependencies=dependencies
        )

    def _cached_catalog(self, frame: FrameLike, extension) -> bool:
        path = self._catalog_path(self._output_catalog_dir(frame), frame, extension)
        meta_path = path.with_suffix(".json")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            return (
                path.is_file()
                and metadata.get("signature") == self._signature(frame, extension)
                and metadata.get("catalog") == file_stamp(path)
            )
        except (OSError, ValueError, TypeError, PipelineError):
            return False

    def _restore_cached(self, frame: FrameLike, state) -> FrameLike | None:
        restored = super()._restore_cached(frame, state)
        if restored is None:
            return None
        if not all(self._cached_catalog(frame, extension) for extension in frame.extensions):
            return None
        return restored

    def _load_instrumental_catalog(self, frame: FrameLike, extension) -> Table:
        path = self._catalog_path(self._input_catalog_dir(frame), frame, extension)
        if not path.is_file():
            raise PhotometryError(
                f"PSF catalog not found for {frame.path.name}[{extension.index}]: {path}. "
                "Run PSFPhotometryStage before VVVCalibrationStage."
            )
        try:
            return Table.read(path, format="ascii.csv")
        except Exception as exc:
            raise PhotometryError(f"Could not read instrumental catalog {path}: {exc}") from exc

    def _crossmatch_vvv(
        self, instrumental_catalog: Table, filter_name: str, reference: Table
    ) -> tuple[Table, str | None]:
        filter_key = filter_name.strip().upper()
        mag_choices = _MAG_COLUMNS.get(filter_key)
        if mag_choices is None:
            return (
                _empty_matches(),
                f"unsupported filter {filter_name!r}; expected JX, HX, or KXs",
            )
        ra_col = _column_name(reference, _RA_COLUMNS)
        dec_col = _column_name(reference, _DEC_COLUMNS)
        mag_col = _column_name(reference, mag_choices)
        if mag_col is None:
            return (
                _empty_matches(),
                f"VVV catalog has no magnitude column for {filter_name}; "
                f"tried {', '.join(mag_choices)}",
            )

        required = ("ra_deg", "dec_deg", "mag", "mag_err")
        missing = [name for name in required if name not in instrumental_catalog.colnames]
        if missing:
            raise PhotometryError(
                f"Instrumental catalog is missing columns: {', '.join(missing)}"
            )

        inst_ra = numeric_column(instrumental_catalog, "ra_deg")
        inst_dec = numeric_column(instrumental_catalog, "dec_deg")
        inst_mag = numeric_column(instrumental_catalog, "mag")
        inst_err = numeric_column(instrumental_catalog, "mag_err")
        inst_valid = (
            np.isfinite(inst_ra)
            & np.isfinite(inst_dec)
            & np.isfinite(inst_mag)
            & np.isfinite(inst_err)
        )
        vvv_ra = numeric_column(reference, ra_col)
        vvv_dec = numeric_column(reference, dec_col)
        vvv_mag = numeric_column(reference, mag_col)
        vvv_valid = (
            np.isfinite(vvv_ra)
            & np.isfinite(vvv_dec)
            & np.isfinite(vvv_mag)
            & (vvv_ra >= 0)
            & (vvv_ra <= 360)
            & (vvv_dec >= -90)
            & (vvv_dec <= 90)
        )
        if not np.any(inst_valid) or not np.any(vvv_valid):
            return _empty_matches(), None

        inst_indices = np.flatnonzero(inst_valid)
        vvv_indices = np.flatnonzero(vvv_valid)
        instrumental_coords = SkyCoord(
            inst_ra[inst_valid] * u.deg, inst_dec[inst_valid] * u.deg
        )
        vvv_coords = SkyCoord(vvv_ra[vvv_valid] * u.deg, vvv_dec[vvv_valid] * u.deg)
        nearest, separation, _ = instrumental_coords.match_to_catalog_sky(vvv_coords)
        matched = separation <= self.config.match_radius_arcsec * u.arcsec
        matched_instrument_indices = inst_indices[matched]
        matched_vvv_indices = vvv_indices[nearest[matched]]

        return (
            Table(
                {
                    "instrument_index": matched_instrument_indices,
                    "instrumental_mag": inst_mag[matched_instrument_indices],
                    "vvv_mag": vvv_mag[matched_vvv_indices],
                    "separation_arcsec": separation[matched].arcsec,
                }
            ),
            None,
        )

    def _derive_zero_point(
        self, matches: Table, unavailable_detail: str | None = None
    ) -> dict:
        cfg: PhotometryConfig = self.config
        count = len(matches)
        if unavailable_detail is not None:
            return {
                "status": "reference_band_unavailable",
                "detail": unavailable_detail,
                "zero_point": None,
                "zero_point_error": None,
                "n_crossmatches": count,
                "n_zero_point_stars": 0,
            }
        if count < cfg.min_crossmatches:
            return {
                "status": "insufficient_crossmatches",
                "detail": f"only {count} VVV crossmatches; need {cfg.min_crossmatches}",
                "zero_point": None,
                "zero_point_error": None,
                "n_crossmatches": count,
                "n_zero_point_stars": 0,
            }

        vvv_mag = numeric_column(matches, "vvv_mag")
        instrumental_mag = numeric_column(matches, "instrumental_mag")
        delta = vvv_mag - instrumental_mag
        valid = (
            np.isfinite(delta)
            & np.isfinite(vvv_mag)
            & (vvv_mag > cfg.min_vvv_magnitude)
        )
        delta = delta[valid]
        if len(delta) < cfg.min_zero_point_stars:
            return {
                "status": "insufficient_zero_point_stars",
                "detail": (
                    f"only {len(delta)} VVV matches are fainter than "
                    f"{cfg.min_vvv_magnitude:g}; need {cfg.min_zero_point_stars}"
                ),
                "zero_point": None,
                "zero_point_error": None,
                "n_crossmatches": count,
                "n_zero_point_stars": len(delta),
            }

        clipped = np.ma.asarray(
            sigma_clip(
                delta,
                sigma=cfg.sigma_clip_sigma,
                maxiters=cfg.sigma_clip_maxiters,
                masked=True,
            )
        )
        used = np.ma.compressed(clipped)
        if len(used) < cfg.min_zero_point_stars:
            return {
                "status": "insufficient_zero_point_stars_after_clipping",
                "detail": (
                    f"only {len(used)} VVV matches remain after sigma clipping; "
                    f"need {cfg.min_zero_point_stars}"
                ),
                "zero_point": None,
                "zero_point_error": None,
                "n_crossmatches": count,
                "n_zero_point_stars": len(used),
            }
        zp = float(np.ma.median(clipped))
        scatter = float(np.ma.std(clipped))
        return {
            "status": "success",
            "detail": "",
            "zero_point": zp,
            "zero_point_error": scatter,
            "n_crossmatches": count,
            "n_zero_point_stars": len(used),
        }

    def _write_calibrated_catalog(
        self,
        instrumental_catalog: Table,
        matches: Table,
        estimate: dict,
        path: Path,
    ) -> None:
        output = instrumental_catalog.copy(copy_data=True)
        nrows = len(output)
        matched_mag = np.full(nrows, np.nan, dtype=np.float64)
        matched_separation = np.full(nrows, np.nan, dtype=np.float64)
        if len(matches):
            indices = np.asarray(matches["instrument_index"], dtype=np.int64)
            matched_mag[indices] = numeric_column(matches, "vvv_mag")
            matched_separation[indices] = numeric_column(
                matches, "separation_arcsec"
            )

        zp = estimate["zero_point"]
        if "mag" in output.colnames and zp is not None:
            output["mag_cal"] = numeric_column(output, "mag") + float(zp)
        else:
            output["mag_cal"] = np.full(nrows, np.nan, dtype=np.float64)
        output["vvv_mag"] = matched_mag
        output["vvv_match_sep_arcsec"] = matched_separation
        output["zp"] = np.full(
            nrows, np.nan if zp is None else float(zp), dtype=np.float64
        )
        zp_err = estimate["zero_point_error"]
        output["zp_err"] = np.full(
            nrows, np.nan if zp_err is None else float(zp_err), dtype=np.float64
        )
        output["n_zp_stars"] = np.full(
            nrows, int(estimate["n_zero_point_stars"]), dtype=np.int32
        )
        output["calibration_status"] = np.full(
            nrows, estimate["status"], dtype="U64"
        )
        atomic_write_table(output, path)

    def _process_extension(self, frame: Frame, extension) -> None:
        cfg: PhotometryConfig = self.config
        output_path = self._catalog_path(
            self._output_catalog_dir(frame), frame, extension
        )
        signature = self._signature(frame, extension)
        if cfg.reuse_intermediates and self._cached_catalog(frame, extension):
            logger.info("%s: reusing calibrated catalog %s", self.name, output_path.name)
            return

        instrumental = self._load_instrumental_catalog(frame, extension)
        filter_name = (
            str(instrumental["filter"][0]).strip()
            if len(instrumental) and "filter" in instrumental.colnames
            else str(extension.header.get("FILTER", frame.header.get("FILTER", ""))).strip()
        )
        reference_path = Path(cfg.vvv_catalog_path).expanduser().resolve()
        stat = file_stamp(reference_path)
        reference = _read_reference_catalog(
            str(reference_path), stat["size"], stat["mtime_ns"]
        )
        matches, unavailable_detail = self._crossmatch_vvv(
            instrumental, filter_name, reference
        )
        estimate = self._derive_zero_point(matches, unavailable_detail)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_calibrated_catalog(instrumental, matches, estimate, output_path)
        metadata = {
            "signature": signature,
            "catalog": file_stamp(output_path),
            **estimate,
        }
        atomic_write_json(output_path.with_suffix(".json"), metadata)
        if estimate["status"] == "success":
            logger.info(
                "%s: %s[%d] %s zero point %.4f +/- %.4f from %d star(s) (%d matches)",
                self.name,
                frame.path.name,
                extension.index,
                filter_name,
                estimate["zero_point"],
                estimate["zero_point_error"],
                estimate["n_zero_point_stars"],
                estimate["n_crossmatches"],
            )
        else:
            logger.warning(
                "%s: %s[%d] calibration unavailable: %s",
                self.name,
                frame.path.name,
                extension.index,
                estimate["detail"],
            )

    def process(self, frame: Frame) -> Frame:
        for extension in frame.extensions:
            self._process_extension(frame, extension)
        return frame

