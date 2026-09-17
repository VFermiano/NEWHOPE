"""Typed pipeline configuration, loaded from YAML.

pydantic validates at load time (e.g. a bad filter name fails here,
not three stages into a run).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Filter = Literal["JX", "HX", "KXs"]


class ReductionConfig(BaseModel):
    dark_dir: Path
    flat_dir: Path
    filters: list[Filter] = Field(default_factory=lambda: ["JX", "HX", "KXs"])


class AstrometryConfig(BaseModel):
    scamp_config_path: Path
    timeout_seconds: float = 120.0


class PhotometryConfig(BaseModel):
    vvv_catalog_path: Path
    fwhm_pixels: float = 4.0


class PipelineConfig(BaseModel):
    raw_dir: Path
    output_dir: Path
    manifest_path: Path
    n_workers: int = 4
    reduction: ReductionConfig
    astrometry: AstrometryConfig
    photometry: PhotometryConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
