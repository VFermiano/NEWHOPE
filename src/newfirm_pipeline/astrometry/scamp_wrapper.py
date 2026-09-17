"""Subprocess wrapper around scamp.

TODO: port the existing scamp wrapper here -- the subprocess invocation
itself. Timeout handling and per-frame CSV/status reporting now live in
PipelineStage.run() / PipelineState generically, so this module should
shrink to just "build the scamp command and run it," raising
ScampTimeoutError / AstrometryError on failure. The generic timeout in
PipelineStage.run() (via run_parallel's `timeout`) may make a
scamp-specific subprocess timeout redundant -- decide whether to keep
both or consolidate.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..core.exceptions import AstrometryError


def run_scamp(catalog_path: Path, scamp_config_path: Path) -> None:
    # TODO: port the existing subprocess.run(...) call, argument list, and
    # working-directory handling from the existing scamp wrapper script.
    raise NotImplementedError("port scamp invocation from the existing wrapper")
