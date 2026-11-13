"""Per-detector empirical-PSF photometry for NEWFIRM MEF exposures."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from astropy.wcs import WCS
from photutils.background import LocalBackground, MMMBackground
from photutils.psf import EPSFBuilder, PSFPhotometry, extract_stars
from scipy.spatial import cKDTree

from ..astrometry.scamp_wrapper import resolve_executable
from ..astrometry.source_extractor import run_sextractor
from ..config import AstrometryConfig, PhotometryConfig
from ..core.exceptions import AstrometryError, PhotometryError, PipelineError
from ..core.frame import Extension, Frame, FrameLike, FrameReference
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

_SOURCE_COLUMNS = (
    "NUMBER",
    "X_IMAGE",
    "Y_IMAGE",
    "FLUX_AUTO",
    "FLUX_MAX",
    "FLAGS",
    "FWHM_IMAGE",
    "ELONGATION",
)


def _empty_catalog() -> Table:
    return Table(
        names=(
            "id",
            "x",
            "y",
            "ra_deg",
            "dec_deg",
            "mag",
            "mag_err",
            "flux",
            "flux_err",
            "qfit",
            "flags",
            "filter",
            "filter_raw",
            "detector",
            "extension",
        ),
        dtype=(
            "i8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "i8",
            "U16",
            "U32",
            "U32",
            "i4",
        ),
    )


class PSFPhotometryStage(PipelineStage):
    """Detect sources, build an empirical PSF, and write per-extension catalogs.

    Catalogs and PSF models are sidecar products under
    ``<output_dir>/catalogs``. The Frame is passed through unchanged so later
    image stages retain the same astrometrically calibrated pixels.
    """

    name = "psf_photometry"

    def __init__(
        self,
        config: PhotometryConfig,
        astrometry_config: AstrometryConfig,
        n_workers: int = 1,
        batch_size: int = 1,
    ) -> None:
        super().__init__(
            config,
            n_workers=n_workers,
            timeout=config.timeout_seconds,
            batch_size=batch_size,
        )
        self.astrometry_config = astrometry_config

    def _catalog_dir(self, frame: FrameLike) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir) / "catalogs"
        return frame.path.parent / "photometry" / "catalogs"

    def _output_path(self, frame: FrameLike) -> Path | None:
        # This stage writes catalogs, not replacement image files. Returning
        # the source reference avoids duplicating large MEF images on disk.
        return None

    def _process_item(self, item: FrameLike) -> Frame | FrameReference:
        if isinstance(item, FrameReference):
            self.process(item.load())
            return item
        return self.process(item)

    def _catalog_path(self, frame: FrameLike, extension: Extension) -> Path:
        detector = extension.header.get("DETECTOR", extension.name)
        return self._catalog_dir(frame) / catalog_filename(
            frame.frame_id, extension.index, detector
        )

    def _signature(self, frame: FrameLike, extension: Extension) -> str:
        cfg: PhotometryConfig = self.config
        astrometry_cfg = self.astrometry_config
        executable = resolve_executable(
            astrometry_cfg.sextractor_executable, "SExtractor"
        )
        dependencies = {
            "sextractor_executable": file_stamp(Path(executable)),
            "sextractor_config": file_stamp(
                Path(astrometry_cfg.sextractor_config_path).expanduser().resolve()
            ),
            "sextractor_filter": file_stamp(
                Path(astrometry_cfg.sextractor_filter_path).expanduser().resolve()
            ),
            "sextractor_parameters": file_stamp(
                Path(cfg.sextractor_parameters_path).expanduser().resolve()
            ),
        }
        return product_signature(
            frame.path, extension.index, cfg, dependencies=dependencies
        )

    def _cached_catalog(self, frame: FrameLike, extension: Extension) -> bool:
        path = self._catalog_path(frame, extension)
        meta_path = path.with_suffix(".json")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("signature") != self._signature(frame, extension):
                return False
            if not path.is_file() or metadata.get("catalog") != file_stamp(path):
                return False
            if metadata.get("status") == "success":
                psf_path = path.with_suffix(".psf.fits")
                return psf_path.is_file() and metadata.get("psf") == file_stamp(psf_path)
            return True
        except (OSError, ValueError, TypeError, PipelineError):
            return False

    def _restore_cached(
        self, frame: FrameLike, state
    ) -> FrameLike | None:
        restored = super()._restore_cached(frame, state)
        if restored is None:
            return None
        if not all(self._cached_catalog(frame, extension) for extension in frame.extensions):
            return None
        return restored

    def _read_source_catalog(self, path: Path) -> Table:
        try:
            with fits.open(path, memmap=False) as hdul:
                for hdu in hdul:
                    if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None:
                        names = set(hdu.columns.names or ())
                        if "X_IMAGE" in names and "FLUX_AUTO" in names:
                            return Table(hdu.data)
        except (OSError, ValueError) as exc:
            raise PhotometryError(f"Could not read SExtractor catalog {path}: {exc}") from exc
        raise PhotometryError(
            f"SExtractor catalog {path} has no object table with X_IMAGE and FLUX_AUTO"
        )

    def _select_psf_stars(
        self, table: Table, data_shape: tuple[int, ...], saturation_level: float
    ) -> Table | None:
        missing = [name for name in _SOURCE_COLUMNS if name not in table.colnames]
        if missing:
            raise PhotometryError(
                f"SExtractor catalog is missing PSF-selection columns: {', '.join(missing)}"
            )
        cfg: PhotometryConfig = self.config
        ny, nx = data_shape
        x = numeric_column(table, "X_IMAGE")
        y = numeric_column(table, "Y_IMAGE")
        flux = numeric_column(table, "FLUX_AUTO")
        peak = numeric_column(table, "FLUX_MAX")
        flags = numeric_column(table, "FLAGS")
        elongation = numeric_column(table, "ELONGATION")

        edge = max(cfg.psf_stamp_size // 2 + 1, int(cfg.edge_margin_fraction * min(nx, ny)))
        good = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(flux)
            & (flux > 0)
            & np.isfinite(peak)
            & np.isfinite(flags)
            & (flags <= cfg.max_sextractor_flags)
            & np.isfinite(elongation)
            & (elongation < cfg.max_elongation)
            & (x > edge)
            & (x < nx - edge)
            & (y > edge)
            & (y < ny - edge)
            & (peak < cfg.reject_near_saturation_fraction * saturation_level)
        )
        candidates = table[good]
        if len(candidates) < cfg.minimum_psf_stars:
            return None

        xy = np.column_stack(
            (numeric_column(candidates, "X_IMAGE"), numeric_column(candidates, "Y_IMAGE"))
        )
        distances, _ = cKDTree(xy).query(xy, k=2)
        candidates = candidates[distances[:, 1] >= cfg.min_isolation_pixels]
        if len(candidates) < cfg.minimum_psf_stars:
            return None

        flux = numeric_column(candidates, "FLUX_AUTO")
        order = np.argsort(flux)[::-1]
        count = len(candidates)
        lo = int(cfg.donor_bright_percentile / 100.0 * count)
        hi = max(
            int(cfg.donor_faint_percentile / 100.0 * count),
            lo + cfg.minimum_psf_stars,
        )
        selected = candidates[order[lo:hi]]
        if len(selected) > cfg.max_psf_stars:
            selected = selected[: cfg.max_psf_stars]
        if len(selected) < cfg.minimum_psf_stars:
            return None
        return selected

    def _build_psf_model(
        self, data: np.ndarray, psf_stars: Table
    ) -> tuple[object, int] | None:
        cfg: PhotometryConfig = self.config
        positions = Table()
        # SExtractor positions are one-based; Photutils uses zero-based pixels.
        positions["x"] = numeric_column(psf_stars, "X_IMAGE") - 1.0
        positions["y"] = numeric_column(psf_stars, "Y_IMAGE") - 1.0
        stars = extract_stars(
            NDData(data=np.asarray(data, dtype=np.float32)),
            positions,
            size=cfg.psf_stamp_size,
        )
        if len(stars) < cfg.minimum_psf_stars:
            return None
        builder = EPSFBuilder(
            oversampling=cfg.psf_oversampling,
            maxiters=cfg.psf_maxiters,
            progress_bar=False,
        )
        epsf, fitted_stars = builder(stars)
        n_used = int(getattr(fitted_stars, "n_good_stars", len(stars)))
        if n_used < cfg.minimum_psf_stars:
            return None
        return epsf, n_used

    def _fit_sources(
        self,
        data: np.ndarray,
        sex_table: Table,
        epsf,
        fwhm: float,
    ) -> Table:
        cfg: PhotometryConfig = self.config
        if len(sex_table) == 0:
            return _empty_catalog()

        init_params = Table()
        init_params["x_init"] = numeric_column(sex_table, "X_IMAGE") - 1.0
        init_params["y_init"] = numeric_column(sex_table, "Y_IMAGE") - 1.0
        init_params["flux_init"] = numeric_column(sex_table, "FLUX_AUTO")
        annulus = cfg.background_annulus_fwhm_mult * fwhm
        local_background = LocalBackground(
            inner_radius=annulus,
            outer_radius=annulus + cfg.background_width_fwhm_mult * fwhm,
            bkg_estimator=MMMBackground(),
        )
        fitter = PSFPhotometry(
            psf_model=epsf,
            fit_shape=(cfg.fit_shape, cfg.fit_shape),
            aperture_radius=cfg.aperture_radius_fwhm_mult * fwhm,
            localbkg_estimator=local_background,
            progress_bar=False,
        )
        return fitter(np.asarray(data, dtype=np.float32), init_params=init_params)

    def _output_catalog(
        self,
        frame: Frame,
        extension: Extension,
        result: Table,
        zero_point: float,
        filter_name: str,
        filter_raw: str,
    ) -> Table:
        if len(result) == 0:
            output = _empty_catalog()
            output.meta["STATUS"] = "no_sources"
            return output

        x = numeric_column(result, "x_fit")
        y = numeric_column(result, "y_fit")
        flux = numeric_column(result, "flux_fit")
        flux_err = numeric_column(result, "flux_err")
        qfit = numeric_column(result, "qfit") if "qfit" in result.colnames else np.full(len(result), np.nan)
        flags = numeric_column(result, "flags") if "flags" in result.colnames else np.zeros(len(result))
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = zero_point - 2.5 * np.log10(flux)
            mag_err = (2.5 / np.log(10.0)) * (flux_err / flux)

        try:
            wcs = WCS(extension.header, relax=True).celestial
            if wcs.has_celestial:
                ra, dec = wcs.all_pix2world(x, y, 0)
            else:
                ra, dec = np.full(len(result), np.nan), np.full(len(result), np.nan)
        except Exception:
            ra, dec = np.full(len(result), np.nan), np.full(len(result), np.nan)

        output = Table()
        output["id"] = (
            numeric_column(result, "id").astype(np.int64)
            if "id" in result.colnames
            else np.arange(1, len(result) + 1, dtype=np.int64)
        )
        output["x"] = x
        output["y"] = y
        output["ra_deg"] = ra
        output["dec_deg"] = dec
        output["mag"] = mag
        output["mag_err"] = mag_err
        output["flux"] = flux
        output["flux_err"] = flux_err
        output["qfit"] = qfit
        output["flags"] = flags.astype(np.int64)
        output["filter"] = np.full(len(result), filter_name, dtype="U16")
        output["filter_raw"] = np.full(len(result), filter_raw, dtype="U32")
        output["detector"] = np.full(
            len(result), str(extension.header.get("DETECTOR", extension.name)), dtype="U32"
        )
        output["extension"] = np.full(len(result), extension.index, dtype=np.int32)
        valid = (
            np.isfinite(output["mag"])
            & np.isfinite(output["mag_err"])
            & np.isfinite(output["flux"])
            & (output["flux"] > 0)
        )
        output = output[valid]
        output.meta["STATUS"] = "success"
        return output

    def _write_catalog(
        self,
        frame: Frame,
        extension: Extension,
        catalog: Table,
        psf_model,
        n_psf_stars: int,
        signature: str,
    ) -> None:
        catalog_path = self._catalog_path(frame, extension)
        atomic_write_table(catalog, catalog_path)
        metadata = {
            "signature": signature,
            "status": catalog.meta.get("STATUS", "success"),
            "catalog": file_stamp(catalog_path),
            "psf_stars": n_psf_stars,
        }
        if psf_model is not None:
            psf_path = catalog_path.with_suffix(".psf.fits")
            header = fits.Header()
            header["NSTARS"] = n_psf_stars
            header["OVERSAMP"] = self.config.psf_oversampling
            fits.PrimaryHDU(
                data=np.asarray(psf_model.data, dtype=np.float32), header=header
            ).writeto(psf_path, overwrite=True)
            metadata["psf"] = file_stamp(psf_path)
        else:
            catalog_path.with_suffix(".psf.fits").unlink(missing_ok=True)
        atomic_write_json(catalog_path.with_suffix(".json"), metadata)

    def _process_extension(self, frame: Frame, extension: Extension) -> None:
        cfg: PhotometryConfig = self.config
        ast_cfg = self.astrometry_config
        catalog_path = self._catalog_path(frame, extension)
        signature = self._signature(frame, extension)
        if cfg.reuse_intermediates and self._cached_catalog(frame, extension):
            logger.info("%s: reusing photometry catalog %s", self.name, catalog_path.name)
            return

        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"sex-{frame.frame_id}-{extension.index}-", dir=catalog_path.parent
        ) as directory:
            work_dir = Path(directory)
            image_path = work_dir / "detector.fits"
            source_path = work_dir / "sources.cat"
            image_header = frame.header.copy()
            for card in extension.header.cards:
                keyword = card.keyword
                if (
                    keyword in {"SIMPLE", "XTENSION", "BITPIX", "PCOUNT", "GCOUNT", "EXTEND", "END", "EXTNAME", "EXTVER", "CHECKSUM", "DATASUM"}
                    or keyword.startswith("NAXIS")
                    or keyword in {"COMMENT", "HISTORY", ""}
                ):
                    continue
                image_header[keyword] = (card.value, card.comment)
            fits.PrimaryHDU(
                data=np.asarray(extension.data, dtype=np.float32),
                header=image_header,
            ).writeto(image_path)
            run_sextractor(
                image_path,
                source_path,
                executable=ast_cfg.sextractor_executable,
                config_path=ast_cfg.sextractor_config_path,
                parameters_path=cfg.sextractor_parameters_path,
                filter_path=ast_cfg.sextractor_filter_path,
                threshold_sigma=cfg.detection_threshold_sigma,
                minimum_area=cfg.detection_minimum_area,
                timeout_seconds=cfg.timeout_seconds,
                gain=cfg.gain,
                saturation_level=float(
                    extension.header.get("SATURATE", cfg.saturation_level)
                ),
            )
            source_table = self._read_source_catalog(source_path)

        filter_raw = str(
            frame.header.get(
                "FILTERS",
                extension.header.get("FILTER", frame.header.get("FILTER", "")),
            )
        ).strip()
        filter_name = str(
            extension.header.get("FILTER", frame.header.get("FILTER", ""))
        ).strip()
        fwhm = float(
            extension.header.get(
                "FWHM", frame.header.get("FWHM", cfg.fwhm_pixels)
            )
        )
        zero_point = float(
            extension.header.get(
                cfg.zero_point_keyword,
                frame.header.get(cfg.zero_point_keyword, cfg.zero_point_default),
            )
        )
        saturation = float(
            extension.header.get("SATURATE", cfg.saturation_level)
        )

        psf_stars = self._select_psf_stars(source_table, extension.shape, saturation)
        if psf_stars is None:
            logger.warning(
                "%s: %s[%d] has too few suitable PSF donor stars; writing an empty catalog",
                self.name,
                frame.path.name,
                extension.index,
            )
            catalog = _empty_catalog()
            catalog.meta["STATUS"] = "skipped_too_few_psf_stars"
            self._write_catalog(frame, extension, catalog, None, 0, signature)
            return

        try:
            built = self._build_psf_model(extension.data, psf_stars)
            if built is None:
                catalog = _empty_catalog()
                catalog.meta["STATUS"] = "skipped_psf_model"
                self._write_catalog(frame, extension, catalog, None, 0, signature)
                return
            psf_model, n_psf_stars = built
            result = self._fit_sources(extension.data, source_table, psf_model, fwhm)
            catalog = self._output_catalog(
                frame,
                extension,
                result,
                zero_point,
                filter_name,
                filter_raw,
            )
            self._write_catalog(
                frame, extension, catalog, psf_model, n_psf_stars, signature
            )
        except (ValueError, RuntimeError, IndexError) as exc:
            raise PhotometryError(
                f"PSF photometry failed for {frame.path.name}[{extension.index}]: {exc}"
            ) from exc

        logger.info(
            "%s: %s[%d] wrote %d source(s), %d PSF stars, filter=%s, zero point=%.3f",
            self.name,
            frame.path.name,
            extension.index,
            len(catalog),
            n_psf_stars,
            filter_name or "unknown",
            zero_point,
        )

    def process(self, frame: Frame) -> Frame:
        for extension in frame.extensions:
            self._process_extension(frame, extension)
        return frame


def check_photometry_tools(
    config: PhotometryConfig, astrometry_config: AstrometryConfig
) -> None:
    """Fail before processing if the photometry inputs or tools are missing."""
    for label, path in (
        ("VVV reference catalog", config.vvv_catalog_path),
        ("photometry SExtractor parameter list", config.sextractor_parameters_path),
        ("SExtractor configuration", astrometry_config.sextractor_config_path),
        ("SExtractor filter", astrometry_config.sextractor_filter_path),
    ):
        if not Path(path).expanduser().is_file():
            raise PhotometryError(f"{label} does not exist: {path}")
    try:
        resolve_executable(astrometry_config.sextractor_executable, "SExtractor")
    except AstrometryError as exc:
        raise PhotometryError(str(exc)) from exc
    reference_path = Path(config.vvv_catalog_path).expanduser()
    try:
        reference = Table.read(reference_path)
    except Exception as exc:
        raise PhotometryError(
            f"Could not read VVV reference catalog {reference_path}: {exc}"
        ) from exc
    if len(reference) == 0:
        raise PhotometryError(f"VVV reference catalog is empty: {config.vvv_catalog_path}")
