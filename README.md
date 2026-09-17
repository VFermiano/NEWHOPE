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

## Run the pipeline

```bash
newhope-pipeline --config-path configs/default.yaml
```

(Will fail until at least one reduction stage's `process()` is filled in --
see `src/newhope_pipeline/reduction/dark.py`.)

## Migration order

See the project notes / conversation history for the recommended order:

1. `Frame` / `FrameCollection` -- done.
2. `PipelineStage` + trivial `Pipeline` -- done.
3. Port `DarkSubtraction.process()` from the existing dark-subtraction
   script; verify output matches the old pipeline on real data.
4. Wire up persistent `PipelineState` resume behavior end to end.
5. Port `FlatFielding`, `SkySubtraction`.
6. Port the scamp wrapper into `AstrometricCalibration`
   (`astrometry/scamp_wrapper.py`, `astrometry/calibrate.py`).
7. Port PSF photometry and VVV calibration
   (`photometry/psf_photometry.py`, `photometry/vvv_calibration.py`).
8. Add tests for each stage as it's ported, not after.
