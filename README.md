# NEWHOPE

Object-oriented reduction, astrometry, and photometry pipeline for NEWFIRM
near-infrared imaging data.

## Status

Scaffold stage. `core/` (Frame, FrameCollection, PipelineStage, Pipeline,
PipelineState) is implemented and tested. `reduction/`, `astrometry/`, and
`photometry/` are stub classes with `TODO` markers where the existing
script-based logic needs to be ported in.


## Multi-extension frames

A NEWFIRM raw file is a MEF: empty primary HDU + 4 image extensions (one per
detector). `FrameReference` stores the path, headers, detector shapes, and exposure
metadata without reading pixels. `FrameReference.load()` loads an exposure on demand. A loaded
`Frame` holds the whole exposure -- `frame.header` is the primary header and
`frame.extensions` is a list of `Extension(index, name, data, header)`, indexed 1..4
like the HDUs (`frame[3]` is detector 3). Stages never split the file: use
`frame.map_extensions(lambda ext: ...)` for per-detector operations,
`frame.with_extension_data([...])` / `with_extension_headers([...])` to build
the output, and key master calibrations by `ext.index`. `n_extensions` in the
config (default 4) makes a file with a wrong number of detectors fail at load.

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
newhope-pipeline run \
    --proposal 2025A-599150 \
    --night 2025-05-11 \
    --base-dir /Volumes/monoceros/renew/data/ \
    --batch-size 4 \
    --workers 1 \
    --verbose
```

Each stage writes its result to `<run>/processed/<stage name>/` (dark_subtraction,
flat_fielding, sky_subtraction) and records it in `pipeline_state.json`. Re-running
the same command resumes from structurally valid products without loading completed
raw images. Delete a stage's folder (or the manifest) to force it to be redone. The
command exits with code 1 if any frame failed in any stage.

`FrameCollection.from_directory()` discovers header-only `FrameReference` objects;
pixel arrays are loaded only when a stage needs them. Independent stages load a frame
inside the worker, atomically write the MEF output, and return a metadata reference to
the next stage. `--batch-size` limits submitted frame tasks (default `1`); `--workers`
overrides the configured worker count. The configuration keys are `batch_size` and
`n_workers`. At most `min(batch_size, workers)` independent-frame tasks run at once.
Each active task can hold its input and output arrays together, so ordinary stages
use at most about `2 * min(batch_size, workers)` full science-frame equivalents, plus
the required calibration masters and FITS/library buffers. With the default batch
size of one, independent stages process one frame at a time even if more workers are
configured.

Sky subtraction keeps its existing filter/object grouping, nearest-time neighbor
selection, masks, scaling, and iterative median. Each active target loads its required
neighbor-graph halo and reads neighbor detector extensions one detector at a time.
If `H` is the number of distinct references in a target's dependency halo, that task
holds one target MEF, up to `H - 1` single-detector neighbor arrays, and one output MEF
plus masks and the bounded median workspace. `H` is the closure through
`iterations + 1` neighbor-graph hops and can approach the group size for some settings.
Sky targets use at most `min(batch_size, workers)` concurrent tasks; completed target
products are written before replacement work is submitted. Raising either limit
increases peak RAM.
Dark and flat correction use the existing prebuilt master products; dark and flat
calibration caches are each capped to one matching key/filter per stage instance, so
calibration arrays do not grow with exposure count. FITS data are read with memory
mapping disabled and copied to float32 arrays, which also works for compressed
extensions and avoids retaining arrays tied to open FITS handles.

To start with one worker and a conservative working set:

```bash
newhope-pipeline run \
    --proposal 2025A-599150 \
    --night 2025-05-11 \
    --base-dir /Volumes/monoceros/renew/data/ \
    --batch-size 1 \
    --workers 1
```

The run command writes a timestamped log to `<run>/pipeline.log` by default. Use
`--log-file PATH` to choose another location. INFO output reports phase and frame
progress; `--verbose` (or `-v`) adds DEBUG details such as individual FITS files and
extensions.

### Memory benchmark

The following reproducible benchmark creates synthetic four-detector FITS images and
measures peak process RSS in fresh processes:

```bash
python benchmarks/memory_scaling.py --frames 16 --size 1024
```

On the implementation environment, this measured **322.1 MiB** for eager loading all
16 frames and **147.3 MiB** for the reference-based pipeline, a **54.3% lower peak**.
This synthetic run validates the scaling direction; it does not set a RAM guarantee
for the full observing dataset or the sky-neighbor halo.

```bash
newhope-pipeline run \
    --proposal 2025A-599150 \
    --night 2025-05-11 \
    --base-dir /Volumes/monoceros/renew/data/ \
    --verbose \
    --log-file /tmp/newhope-debug.log
```

## Sky subtraction

`reduction/sky.py`. Per detector: for every frame, the `n_neighbors` exposures of the
same filter + `group_keys` (default `OBJECT`) closest in time (within `max_time_gap_s`,
read from `MJD-OBS`/`DATE-OBS`) are normalized by their own sky level, objects and bad
pixels (non-finite or 0) are masked, and the pixel-wise median is scaled to the frame
and subtracted. A first unmasked median captures the sky gradient, objects are detected
on the residual, and the sky is rebuilt `iterations` times. Bad pixels keep their value.
Output headers get `SKYSUB`, `SKYMODE`, `SKYNFRM`, `SKYITER`, `SKYLEVEL`. All options
live under `reduction.sky` in the config (`SkyConfig`). A frame with fewer than
`min_neighbors` usable exposures fails (it is not subtracted with a bad sky).

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
