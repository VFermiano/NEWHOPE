"""Data acquisition: pulling raw frames into the run's `raw/` directory before reduction starts.

Currently just the NOIRLab Astro Data Archive downloader
(`noirlab.download_night`). This package is intentionally separate from
`core/` -- acquisition happens *before* there's a FrameCollection to
build, so it doesn't implement PipelineStage.
"""
