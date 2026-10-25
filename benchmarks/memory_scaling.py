"""Compare eager full-dataset loading with the reference-based pipeline.

Run from the repository root after installing the project, for example:

    python benchmarks/memory_scaling.py --frames 8 --size 1024

The script builds a temporary synthetic four-detector FITS dataset and runs
each measurement in a fresh process so ``ru_maxrss`` is a process peak.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits


def _peak_rss_mib() -> float:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak / (1024**2) if sys.platform == "darwin" else peak / 1024)


def _write_dataset(directory: Path, frames: int, size: int) -> None:
    yy, xx = np.mgrid[:size, :size]
    for frame_index in range(frames):
        primary = fits.PrimaryHDU()
        primary.header["FILTER"] = "HX"
        primary.header["EXPTIME"] = 30.0
        hdus = [primary]
        for detector in range(1, 5):
            data = np.asarray(1000.0 * detector + frame_index + xx / size + yy / size, dtype=np.float32)
            hdus.append(fits.ImageHDU(data=data, name=f"im{detector}"))
        fits.HDUList(hdus).writeto(directory / f"frame_{frame_index:04d}.fits")


def _measure(mode: str, directory: Path) -> float:
    from newhope_pipeline.core.frame import Frame, FrameCollection

    if mode == "eager":
        frames = [Frame.from_fits(path, expected_extensions=4) for path in sorted(directory.glob("*.fits"))]
        assert frames
        return _peak_rss_mib()

    from newhope_pipeline.core.pipeline import Pipeline
    from newhope_pipeline.core.stage import PipelineStage

    class AddOne(PipelineStage):
        name = "benchmark_add_one"

        def process(self, frame):
            return frame.map_extensions(lambda ext: ext.data + 1.0)

    refs = FrameCollection.from_directory(directory, expected_extensions=4)
    Pipeline([AddOne(config=None, n_workers=1, batch_size=1)], directory / "state.json").run(refs)
    return _peak_rss_mib()


def _child(mode: str, directory: Path) -> None:
    print(json.dumps({"mode": mode, "peak_rss_mib": _measure(mode, directory)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=1024, help="width and height of each detector")
    parser.add_argument("--mode", choices=("eager", "streaming"), help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frames < 1 or args.size < 8:
        parser.error("--frames must be positive and --size must be at least 8")
    if args.mode:
        if args.directory is None:
            parser.error("internal --mode requires --directory")
        _child(args.mode, args.directory)
        return

    with tempfile.TemporaryDirectory(prefix="newhope-memory-") as temp_dir:
        directory = Path(temp_dir) / "raw"
        directory.mkdir()
        _write_dataset(directory, args.frames, args.size)
        measurements = {}
        for mode in ("eager", "streaming"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                mode,
                "--directory",
                str(directory),
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            measurements[mode] = json.loads(result.stdout.strip().splitlines()[-1])["peak_rss_mib"]
    eager = measurements["eager"]
    streaming = measurements["streaming"]
    reduction = 100.0 * (eager - streaming) / eager if eager else 0.0
    print(f"Synthetic inputs: {args.frames} frames x 4 detectors x {args.size}x{args.size} float32")
    print(f"Eager peak RSS:     {eager:.1f} MiB")
    print(f"Streaming peak RSS: {streaming:.1f} MiB")
    print(f"Peak RSS change:    {reduction:+.1f}%")


if __name__ == "__main__":
    main()
