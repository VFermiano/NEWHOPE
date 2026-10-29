# NEWHOPE

Object-oriented reduction, astrometry, and photometry pipeline for NEWFIRM
near-infrared imaging data.

## Status

The pipeline performs dark subtraction, flat correction, and sky subtraction,
then detects sources, calibrates each detector extension with SCAMP, and writes
detector-specific SWarp stacks. PSF photometry and VVV photometric calibration
remain future stages.


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

Reduction and astrometry stages write MEFs to `<run>/processed/<stage name>/`
(`dark_subtraction`, `flat_fielding`, `sky_subtraction`, and
`astrometric_calibration`). Stacking writes one detector image per compatible
science group under `processed/stacking/` and records frame status in
`pipeline_state.json`. Re-running resumes from valid products. Astrometry and
stack caches include input file metadata and relevant configuration; changed
inputs or settings trigger recalibration or restacking. The command exits with
code 1 if any frame fails in any stage.

`FrameCollection.from_directory()` discovers header-only `FrameReference` objects;
pixel arrays are loaded only when a stage needs them. Independent stages load a frame
inside the worker, atomically write the MEF output, and return a metadata reference to
the next stage. `--batch-size` limits submitted frame tasks (default `1`); `--workers`
overrides the configured worker count. The configuration keys are `batch_size` and
`n_workers`. At most `min(batch_size, workers)` independent-frame tasks run at once.
Each active task can hold its input and output arrays together, so ordinary stages
use at most about `2 * min(batch_size, workers)` full science-frame equivalents, plus
calibration masters and FITS/library buffers. Astrometry handles one MEF per task.
Stacking extracts one detector extension at a time to temporary single-image FITS
files, then starts no more than `min(batch_size, workers)` SWarp jobs, each configured
for one internal thread. With batch size one, these stages use one task at a time.

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
Dark and flat masters are built from the run's raw `OBSTYPE=dark` and `OBSTYPE=dflat`
frames when needed; calibration caches are capped so arrays do not grow with exposure
count. FITS data are read with memory
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

## Astrometry and stacking

The full `run` command continues from sky subtraction through astrometry and
stacking. The standalone command runs only those final steps on products already
written under `processed/sky_subtraction/`:

```bash
newhope-pipeline stack \
    --proposal 2025B-584787 \
    --night 2026-05-29 \
    --base-dir /path/to/data \
    --workers 2 \
    --batch-size 1
```

`stack` reads only `processed/sky_subtraction/`; it does not run dark, flat, or sky
subtraction. Inputs must have `SKYSUB=True` and compatible filter, target, detector,
image dimensions, `EXPTIME`, `EXPCOADD`, `COADDS`, and `FSAMPLE` headers. Targets are read from
`OBJECT`, falling back to `FIELD`, `TARGNAME`, and `OBJNAME`. Detector identity comes
from each image extension's `DETECTOR` keyword; the 1-based FITS extension index is
also retained so the four NEWFIRM channels are not combined together. The supplied
2026-05-29 products identify extensions as `SCA022`, `SCA011`, `SCA013`, and `SCA019`
at indices 1 through 4.

For astrometry, each MEF detector extension is written to a temporary standalone
image. SExtractor creates a FITS_LDAC catalog (5-sigma threshold and at least five
connected pixels by default), and SCAMP uses that extension's existing celestial
WCS as the initial solution and matches against GAIA-EDR3 by default. The SCAMP header is
validated and merged into the extension WCS. An image without a usable initial WCS
or valid SCAMP result fails explicitly; the pipeline does not invent a sky position
or continue with an arbitrary WCS. Catalogs and `.head` results are kept under
`processed/astrometric_calibration/intermediates/<frame id>/` by default.

SWarp operates on temporary single-image detector FITS files, not on the empty
primary HDU or the whole MEF. It writes one final image per filter, target, detector,
and acquisition setup. Defaults are median combination, LANCZOS3 resampling, TAN
projection, and no second background subtraction (sky has already been removed).
A group needs at least two compatible exposures. The default output pixel scale is
SWarp's median input scale; set `stacking.output_pixel_scale_arcsec` to a measured
value if a fixed scale is desired. No NEWFIRM pixel scale is hardcoded.

```text
<run>/processed/
    sky_subtraction/                 # sky-subtracted four-extension MEFs
    astrometric_calibration/          # MEFs with per-extension SCAMP WCS
        intermediates/<frame id>/     # SExtractor catalogs and SCAMP .head files
    stacking/
        stack_<filter>_<target>_<detector>_ext<index>_<id>.fits
        *.fits.stack.json             # per-stack input mapping and settings
        stack_manifest.json           # all groups and failures
```

Defaults live in `configs/sextractor/default.sex`, `configs/scamp/default.scamp`,
and `configs/swarp/default.swarp`. Matching configuration blocks are `astrometry:`
and `stacking:` in `configs/default.yaml`. Adjust the SExtractor threshold and SCAMP
distortion degree after inspecting catalogs and astrometric residuals for the
observing data. If reference-catalog network access is unavailable, set
`astrometry.reference_catalog: FILE` and provide a SCAMP-compatible local catalog
with `astrometry.reference_catalog_path`.

SExtractor, SCAMP, and SWarp are external programs and are not installed by the
Python package. Check availability with `sex -v`, `scamp -v`, and `swarp -v`, then
configure executable paths if needed. The default `GAIA-EDR3` reference catalog is
supported by SCAMP 2.10.0. At startup, the pipeline reads the available catalog
names from the configured SCAMP executable, so `astrometry.reference_catalog` can
select any catalog that executable supports. Both `run` and `stack` check programs
and configuration files before processing and report missing dependencies. Source
catalogs, SCAMP headers, and stack products are reused only when their inputs and
settings match and the output WCS validates. Set `astrometry.keep_intermediates` or
`stacking.keep_intermediates` to false to remove their intermediates; set the
corresponding `reuse_intermediates` to false to recompute them. SWarp uses the
configured `MEM_MAX` and one thread per job; SExtractor and SCAMP are also launched
with one thread per task. Concurrent tasks are limited by the pipeline's worker and
batch settings.

## Further processing

PSF photometry and VVV photometric calibration are not part of the current
`run` or `stack` commands. They can be added after detector stacks are produced
and the SCAMP/SWarp setup has been validated against the observing data.
