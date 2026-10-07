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
from pydantic import BaseModel, Field

Filter = Literal["JX", "HX", "KXs"]


class ReductionConfig(BaseModel):
    # None -> use the run's raw directory (darks/flats are downloaded there
    # together with the science frames). Set explicitly to override.
    dark_dir: Path | None = None
    flat_dir: Path | None = None
    filters: list[Filter] = Field(default_factory=lambda: ["JX", "HX", "KXs"])


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
