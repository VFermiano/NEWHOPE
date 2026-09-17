"""Checkpoint/resume manifest for the pipeline.

Replaces ad hoc checkpoint files: a single JSON manifest, keyed by
(frame_id, stage_name), that every stage reads and writes through.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class PipelineState:
    """Tracks which (frame_id, stage_name) pairs have completed.

    Backed by a JSON file on disk so a run can be killed and resumed
    without redoing finished work. If the manifest grows large or you
    need concurrent writes from multiple worker processes, swap the
    backing store for SQLite -- the public interface (is_done/mark)
    shouldn't need to change.
    """

    manifest_path: Path
    _done: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.manifest_path = Path(self.manifest_path)
        if self.manifest_path.exists():
            self._done = json.loads(self.manifest_path.read_text())

    def is_done(self, frame_id: str, stage_name: str) -> bool:
        return self._done.get(frame_id, {}).get(stage_name, {}).get("status") == "done"

    def mark(
        self,
        frame_id: str,
        stage_name: str,
        status: str,
        output_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self._done.setdefault(frame_id, {})[stage_name] = {
            "status": status,
            "output_path": output_path,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def failures(self, stage_name: Optional[str] = None) -> dict:
        """Return {frame_id: entry} for everything marked 'failed', optionally filtered by stage."""
        out = {}
        for frame_id, stages in self._done.items():
            for name, entry in stages.items():
                if entry.get("status") == "failed" and (stage_name is None or name == stage_name):
                    out[frame_id] = entry
        return out

    def _save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self._done, indent=2))
