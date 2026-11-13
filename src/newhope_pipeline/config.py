"""Typed pipeline configuration, loaded from YAML.

pydantic validates at load time (e.g. a bad filter name fails here,
not three stages into a run).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Filter = Literal["JX", "HX", "KXs"]


class SkyConfig(BaseModel):
    """Parameters of the sky-subtraction stage (see reduction/sky.py)."""

    # "scale": sky_i = median(frame_j / level_j) * level_i  (multiplicative; default)
    # "offset": sky_i = median(frame_j - level_j) + level_i  (additive)
    mode: Literal["scale", "offset"] = "scale"
    n_neighbors: int = Field(10, ge=3)  # nearest-in-time exposures used per sky
    min_neighbors: int = Field(4, ge=2)  # fewer than this -> the frame fails
    max_time_gap_s: float = Field(1800.0, gt=0)  # ignore exposures further away in time
    # Header keywords (besides the filter) that must match for two exposures to
    # share a sky, i.e. to belong to the same dither sequence.
    group_keys: list[str] = Field(default_factory=lambda: ["OBJECT"])
    iterations: int = Field(
        2, ge=1
    )  # object-masked passes after the initial unmasked median
    detect_nsigma: float = Field(
        3.0, gt=0
    )  # object-mask threshold on the smoothed image
    # Smallest object peak (as a fraction of the sky level) worth masking; fainter residuals are
    # below flat-field accuracy anyway and masking them only removes good sky pixels.
    detect_floor_frac: float = Field(0.002, ge=0)
    detect_smooth_pix: float = Field(
        2.0, gt=0
    )  # gaussian sigma (pixels) before thresholding
    mask_grow_pix: int = Field(3, ge=0)  # dilation of the object mask
    background_box: int = Field(64, ge=8)  # block size of the coarse background
    min_valid: int = Field(3, ge=1)  # min unmasked samples per pixel; else interpolated

    @model_validator(mode="after")
    def _check_neighbors(self) -> "SkyConfig":
        if self.min_neighbors > self.n_neighbors:
            raise ValueError("min_neighbors cannot exceed n_neighbors")
        return self


class ReductionConfig(BaseModel):
    # None -> use the run's raw directory (darks/flats are downloaded there
    # together with the science frames). Set explicitly to override.
    dark_dir: Path | None = None
    flat_dir: Path | None = None
    filters: list[Filter] = Field(default_factory=lambda: ["JX", "HX", "KXs"])
    sky: SkyConfig = Field(default_factory=SkyConfig)


class AstrometryConfig(BaseModel):
    scamp_config_path: Path = Path("configs/scamp/default.scamp")
    sextractor_executable: str = "sex"
    sextractor_config_path: Path = Path("configs/sextractor/default.sex")
    sextractor_parameters_path: Path = Path("configs/sextractor/default.param")
    sextractor_filter_path: Path = Path("configs/sextractor/default.conv")
    scamp_executable: str = "scamp"
    reference_catalog: str = "GAIA-EDR3"
    reference_catalog_path: Path | None = None
    detection_threshold_sigma: float = Field(5.0, gt=0)
    detection_minimum_area: int = Field(5, ge=1)
    scamp_parameters: dict[str, str | int | float] = Field(default_factory=dict)
    keep_intermediates: bool = True
    reuse_intermediates: bool = True
    timeout_seconds: float = Field(120.0, gt=0)


class StackingConfig(BaseModel):
    """SWarp settings and persistence controls for detector-specific stacks."""

    swarp_executable: str = "swarp"
    swarp_config_path: Path = Path("configs/swarp/default.swarp")
    output_pixel_scale_arcsec: float | None = Field(default=None, gt=0)
    projection: str = "TAN"
    resampling_method: Literal[
        "NEAREST", "BILINEAR", "LANCZOS2", "LANCZOS3", "LANCZOS4"
    ] = "LANCZOS3"
    combination_method: Literal[
        "MEDIAN", "AVERAGE", "WEIGHTED", "SUM", "MIN", "MAX"
    ] = "MEDIAN"
    minimum_exposures: int = Field(2, ge=1)
    keep_intermediates: bool = False
    reuse_intermediates: bool = True
    timeout_seconds: float = Field(600.0, gt=0)

    @model_validator(mode="after")
    def _check_projection(self) -> "StackingConfig":
        if not self.projection.strip():
            raise ValueError("stacking.projection cannot be empty")
        return self


class PhotometryConfig(BaseModel):
    enabled: bool = False
    vvv_catalog_path: Path = Path("data/calib/vvv_catalog.fits")
    sextractor_parameters_path: Path = Path(
        "configs/sextractor/photometry.param"
    )
    detection_threshold_sigma: float = Field(1.5, gt=0)
    detection_minimum_area: int = Field(5, ge=1)
    timeout_seconds: float = Field(900.0, gt=0)
    fwhm_pixels: float = Field(4.0, gt=0)
    zero_point_keyword: str = "MAGZPT"
    zero_point_default: float = 25.0
    saturation_level: float = Field(45000.0, gt=0)
    gain: float = Field(1.0, ge=0)

    psf_stamp_size: int = Field(25, ge=9)
    psf_oversampling: int = Field(1, ge=1)
    psf_maxiters: int = Field(10, ge=1)
    minimum_psf_stars: int = Field(5, ge=3)
    min_isolation_pixels: float = Field(30.0, ge=0)
    edge_margin_fraction: float = Field(0.03, ge=0, lt=0.5)
    donor_bright_percentile: float = Field(5.0, ge=0, le=100)
    donor_faint_percentile: float = Field(40.0, ge=0, le=100)
    max_elongation: float = Field(1.3, gt=0)
    max_sextractor_flags: int = Field(0, ge=0)
    reject_near_saturation_fraction: float = Field(0.9, gt=0, le=1)
    max_psf_stars: int = Field(25, ge=3)
    fit_shape: int = Field(11, ge=3)
    aperture_radius_fwhm_mult: float = Field(1.5, gt=0)
    background_annulus_fwhm_mult: float = Field(5.0, gt=0)
    background_width_fwhm_mult: float = Field(3.0, gt=0)

    min_crossmatches: int = Field(20, ge=1)
    min_zero_point_stars: int = Field(10, ge=1)
    min_vvv_magnitude: float = 12.0
    match_radius_arcsec: float = Field(0.7, gt=0)
    sigma_clip_sigma: float = Field(3.0, gt=0)
    sigma_clip_maxiters: int = Field(5, ge=1)
    reuse_intermediates: bool = True

    @model_validator(mode="after")
    def _check_photometry_shapes(self) -> "PhotometryConfig":
        if self.psf_stamp_size % 2 == 0:
            raise ValueError("photometry.psf_stamp_size must be odd")
        if self.fit_shape % 2 == 0:
            raise ValueError("photometry.fit_shape must be odd")
        if self.donor_faint_percentile < self.donor_bright_percentile:
            raise ValueError(
                "photometry.donor_faint_percentile must be greater than or equal "
                "to donor_bright_percentile"
            )
        return self


class AcquisitionConfig(BaseModel):
    """Config for pulling raw frames from the NOIRLab Astro Data Archive.

    Optional: only needed if you want `newhope-pipeline download` (or
    `run --download`) to fetch data before reduction starts, instead of
    populating `raw_dir` yourself.
    """

    instrument: str = "newfirm"
    telescope: str = "ct4m"
    proposal: str | None = None
    night: str | None = None  # observing night, YYYY-MM-DD (caldat)
    proctype: str = "raw"  # pass "" to not filter on processing type
    limit: int = 5000
    skip_calibrations: bool = False
    object_name: str | None = None  # exact match against the NEWFIRM OBJECT header
    calibrations_only: bool = False  # fetch dark/dflat without proposal science frames

    @model_validator(mode="after")
    def _check_download_selection(self) -> "AcquisitionConfig":
        if self.object_name is not None:
            self.object_name = self.object_name.strip() or None
        if self.calibrations_only and self.object_name:
            raise ValueError("acquisition.object_name cannot be combined with calibrations_only")
        if self.calibrations_only and self.skip_calibrations:
            raise ValueError("acquisition.calibrations_only conflicts with skip_calibrations")
        return self


@dataclass(frozen=True)
class RunPaths:
    """All on-disk locations for ONE run (one proposal + night).

    Layout, given base_dir=data and run_name=2025A-599150_2025-05-13:

        data/2025A-599150_2025-05-13/
            raw/                  <- download target, reduction input
            processed/            <- reduction output
            pipeline_state.json   <- checkpoint/resume manifest
    """

    run_dir: Path
    raw_dir: Path
    output_dir: Path
    manifest_path: Path

    def create(self) -> "RunPaths":
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self


def make_run_name(proposal: str, night: str) -> str:
    """Directory-safe run name, e.g. '2025A-599150_2025-05-13'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{proposal}_{night}")


class PipelineConfig(BaseModel):
    # Parent directory under which each run gets its own subdirectory.
    # Normally supplied on the command line (--base-dir); this is the default.
    base_dir: Path = Path("data")
    n_workers: int = Field(4, ge=1)
    batch_size: int = Field(1, ge=1)
    n_extensions: int = 4
    reduction: ReductionConfig
    astrometry: AstrometryConfig
    stacking: StackingConfig = Field(default_factory=StackingConfig)
    photometry: PhotometryConfig
    acquisition: AcquisitionConfig | None = None

    def resolve_paths(
        self,
        proposal: str,
        night: str,
        base_dir: Path | None = None,
        create: bool = False,
    ) -> RunPaths:
        """Build the run directory layout from base_dir + proposal + night.

        `base_dir` (e.g. from the CLI) overrides the one in the config.
        """
        base = Path(base_dir) if base_dir is not None else self.base_dir
        run_dir = base / make_run_name(proposal, night)
        paths = RunPaths(
            run_dir=run_dir,
            raw_dir=run_dir / "raw",
            output_dir=run_dir / "processed",
            manifest_path=run_dir / "pipeline_state.json",
        )
        return paths.create() if create else paths

    def reduction_for(self, paths: RunPaths) -> ReductionConfig:
        """Reduction config with dark/flat dirs defaulted to the run's raw dir."""
        return self.reduction.model_copy(
            update={
                "dark_dir": self.reduction.dark_dir or paths.raw_dir,
                "flat_dir": self.reduction.flat_dir or paths.raw_dir,
            }
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
