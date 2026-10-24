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
    iterations: int = Field(2, ge=1)  # object-masked passes after the initial unmasked median
    detect_nsigma: float = Field(3.0, gt=0)  # object-mask threshold on the smoothed image
    # Smallest object peak (as a fraction of the sky level) worth masking; fainter residuals are
    # below flat-field accuracy anyway and masking them only removes good sky pixels.
    detect_floor_frac: float = Field(0.002, ge=0)
    detect_smooth_pix: float = Field(2.0, gt=0)  # gaussian sigma (pixels) before thresholding
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
    scamp_config_path: Path
    timeout_seconds: float = 120.0


class PhotometryConfig(BaseModel):
    vvv_catalog_path: Path
    fwhm_pixels: float = 4.0


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
    n_workers: int = 4
    n_extensions: int = 4
    reduction: ReductionConfig
    astrometry: AstrometryConfig
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
