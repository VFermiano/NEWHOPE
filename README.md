# NEWHOPE

Object-oriented reduction, astrometry, and photometry pipeline for NEWFIRM
near-infrared imaging data.

## Status

Scaffold stage. `core/` (Frame, FrameCollection, PipelineStage, Pipeline,
PipelineState) is implemented and tested. `reduction/`, `astrometry/`, and
`photometry/` are stub classes with `TODO` markers where the existing
script-based logic needs to be ported in.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run tests

```bash
pytest -q
```

## Downloading raw data

If your raw frames aren't on disk yet, `newhope-pipeline` can pull them
from the NOIRLab Astro Data Archive first. Pass a base directory and the
pipeline creates one run folder per proposal+night inside it, and uses that
same folder for the whole reduction:

```
<base-dir>/<proposal>_<night>/
    raw/                    <- download target, reduction input
    processed/              <- reduction output
    pipeline_state.json     <- resume manifest
```

Instrument/telescope/etc. come from the `acquisition:` section of your
config (see `configs/default.yaml`); `--proposal`/`--night` on the command
line override it. `base_dir` in the config is only the default for
`--base-dir`. Then either:

```bash
# one-time: store your archive login (or set NOIRLAB_EMAIL / NOIRLAB_PASSWORD instead)
newhope-pipeline download --set-credentials

# download only
newhope-pipeline download --base-dir /data/newfirm --proposal 2025A-599150 --night 2025-05-13
newhope-pipeline download --base-dir /data/newfirm --proposal 2025A-599150 --night 2025-05-13 --dry-run  # list matches first

# or download, then run the pipeline in one go
newhope-pipeline run --base-dir /data/newfirm --proposal 2025A-599150 --night 2025-05-13 --download
```

This fetches every science image for that proposal/night plus same-night
darks/flats for that instrument+telescope (calibrations are archive-wide,
not tied to a single proposal), verifying checksums on re-runs so it's
safe to interrupt and resume. See `src/newhope_pipeline/acquisition/noirlab.py`
for the credential lookup order and auth details; `acquisition:` is
optional -- omit it and put your frames in `<base-dir>/<proposal>_<night>/raw/` yourself.

## Run the pipeline

```bash
# same --base-dir/--proposal/--night as the download -> same run folder
newhope-pipeline run --base-dir /data/newfirm --proposal 2025A-599150 --night 2025-05-13
```

(Will fail until at least one reduction stage's `process()` is filled in --
see `src/newhope_pipeline/reduction/dark.py`.)

## Migration order

See the project notes / conversation history for the recommended order:

1. `Frame` / `FrameCollection` -- done.
2. `PipelineStage` + trivial `Pipeline` -- done.
3. NOIRLab data acquisition (`acquisition/noirlab.py`) -- done.
4. Port `DarkSubtraction.process()` from the existing dark-subtraction
   script; verify output matches the old pipeline on real data.
5. Wire up persistent `PipelineState` resume behavior end to end.
6. Port `FlatFielding`, `SkySubtraction`.
7. Port the scamp wrapper into `AstrometricCalibration`
   (`astrometry/scamp_wrapper.py`, `astrometry/calibrate.py`).
8. Port PSF photometry and VVV calibration
   (`photometry/psf_photometry.py`, `photometry/vvv_calibration.py`).
9. Add tests for each stage as it's ported, not after.
