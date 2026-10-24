"""Sky subtraction for dithered NEWFIRM exposures.

Method (per detector, independently -- the four chips never mix):

1. Frames are grouped into dither sequences: same filter plus the header
   keywords in ``sky.group_keys`` (default ``OBJECT``).
2. For every frame the ``n_neighbors`` exposures of its group that are
   closest in time (within ``max_time_gap_s``) are its sky sample.
3. Bad pixels (non-finite or exactly 0, which is what the flat-field stage
   writes for undefined pixels) are always excluded. Objects are masked too:
   a first sky is the plain median of the neighbours (dithering already
   removes most objects and the median keeps the sky gradient); objects are
   then detected on the residual image (smoothed, above a threshold).
4. Each sample is normalized by its own sky level (median of unmasked pixels)
   -- ``mode="scale"``: divided, ``mode="offset"``: subtracted -- and the samples
   are median-combined pixel by pixel, ignoring masked pixels. The result is
   scaled (or shifted) to the frame being corrected by a median fit of the
   frame against it on unmasked pixels, and subtracted.
5. Steps 3-4 are repeated ``iterations`` times (default 2), each time with
   objects re-detected on the latest sky-subtracted frames.

Bad pixels keep their input value, so a 0 from the flat-field stage does not
turn into ``-sky``.

Unlike dark and flat, the sky of one frame depends on *other* frames, so this
stage overrides ``run`` and works on the whole collection at once.
"""
from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from time import monotonic
from typing import Optional, Sequence

import numpy as np
from astropy.time import Time
from scipy import ndimage

from ..config import SkyConfig
from ..core.exceptions import ReductionError
from ..core.frame import Frame, FrameCollection
from ..core.stage import PipelineStage
from ..core.state import PipelineState

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pure array helpers (no pipeline types: easy to test and to reuse)
# --------------------------------------------------------------------------
def bad_pixel_mask(data: np.ndarray) -> np.ndarray:
    """True where a pixel carries no information (NaN/inf, or the 0 left by flat-fielding)."""
    return ~np.isfinite(data) | (data == 0.0)


def coarse_background(data: np.ndarray, bad: Optional[np.ndarray] = None, box: int = 64) -> np.ndarray:
    """Smooth background: block medians (ignoring `bad`) interpolated bilinearly to full size."""
    h, w = data.shape
    ny, nx = max(1, h // box), max(1, w // box)
    ys = np.linspace(0, h, ny + 1).astype(int)
    xs = np.linspace(0, w, nx + 1).astype(int)
    ok = np.isfinite(data) if bad is None else (~bad & np.isfinite(data))

    coarse = np.full((ny, nx), np.nan, dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            sel = ok[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if sel.mean() >= 0.1:  # need a minimum of usable pixels in the block
                coarse[i, j] = np.median(data[ys[i]:ys[i + 1], xs[j]:xs[j + 1]][sel])
    finite = np.isfinite(coarse)
    coarse[~finite] = np.median(coarse[finite]) if finite.any() else 0.0

    yc = 0.5 * (ys[:-1] + ys[1:] - 1)
    xc = 0.5 * (xs[:-1] + xs[1:] - 1)
    rows = np.empty((ny, w), dtype=np.float32)
    x_full = np.arange(w)
    for i in range(ny):
        rows[i] = np.interp(x_full, xc, coarse[i])
    if ny == 1:
        return np.broadcast_to(rows[0], (h, w)).copy()
    t = np.interp(np.arange(h), yc, np.arange(ny))
    i0 = np.minimum(np.floor(t).astype(int), ny - 2)
    frac = (t - i0).astype(np.float32)
    return (1.0 - frac)[:, None] * rows[i0] + frac[:, None] * rows[i0 + 1]


def detect_objects(
    data: np.ndarray,
    bad: np.ndarray,
    nsigma: float = 3.0,
    smooth: float = 2.0,
    grow: int = 3,
    box: int = 64,
    floor: float = 0.0,
) -> np.ndarray:
    """Boolean mask (True = exclude from the sky estimate): objects, their wings, and bad pixels.

    The threshold is ``max(nsigma * noise, floor)``. `floor` (data units) keeps small
    systematic residuals of the sky model -- which are coherent from frame to frame and
    would otherwise be taken for objects once the pixel noise is lower than them -- out of the mask.
    """
    valid = ~bad
    if valid.sum() < 100:
        return bad.copy()
    resid = np.where(bad, 0.0, data - coarse_background(data, bad, box)).astype(np.float32)
    smoothed = ndimage.gaussian_filter(resid, smooth)
    ref = smoothed[valid]
    sigma = 1.4826 * np.median(np.abs(ref - np.median(ref)))
    if not np.isfinite(sigma) or sigma < 0:
        return bad.copy()
    mask = smoothed > max(nsigma * sigma, floor)
    if grow > 0:
        mask = ndimage.binary_dilation(mask, iterations=grow)
    return mask | bad


def sky_level(data: np.ndarray, mask: np.ndarray) -> float:
    """Median of the unmasked pixels."""
    good = data[~mask]
    if good.size < 0.05 * data.size:
        raise ReductionError(
            f"object/bad-pixel mask covers {100 * (1 - good.size / data.size):.0f}% of the detector; "
            "cannot measure a sky level (crowded field? try a larger detect_nsigma)"
        )
    return float(np.median(good))


def fit_level(data: np.ndarray, mask: np.ndarray, norm_sky: np.ndarray, mode: str) -> float:
    """Amount of the normalized sky present in `data`, fitted on the unmasked pixels.

    Comparing data and sky model pixel by pixel (instead of using each image's
    own median) cancels the bias that comes from the two images having
    different object masks over a sky gradient.
    """
    ok = ~mask & np.isfinite(norm_sky)
    if mode == "scale":
        ok &= norm_sky > 0
    if ok.sum() < 0.05 * data.size:
        raise ReductionError("too few unmasked pixels to fit the sky level")
    if mode == "scale":
        return float(np.median(data[ok] / norm_sky[ok]))
    return float(np.median(data[ok] - norm_sky[ok]))


def build_sky(
    samples: Sequence[tuple[np.ndarray, np.ndarray, float]],
    mode: str,
    min_valid: int,
    box: int,
) -> np.ndarray:
    """Median-combine masked samples (data, mask, level) into a *normalized* sky image.

    mode "scale": result is dimensionless (~1) and must be multiplied by the target level.
    mode "offset": result is ~0 and the target level must be added.
    """
    h, w = samples[0][0].shape
    n = len(samples)
    if mode == "scale" and any(level <= 0 for _, _, level in samples):
        raise ReductionError("non-positive sky level in a sample frame; use sky.mode='offset'")

    sky = np.empty((h, w), dtype=np.float32)
    count = np.empty((h, w), dtype=np.int16)
    # Stack in bands of rows so that memory stays at ~n * 8M bytes/n regardless of detector size.
    band = max(1, int(8_000_000 // (n * w)))
    for r0 in range(0, h, band):
        r1 = min(h, r0 + band)
        cube = np.empty((n, r1 - r0, w), dtype=np.float32)
        for j, (data, mask, level) in enumerate(samples):
            part = data[r0:r1]
            cube[j] = part / np.float32(level) if mode == "scale" else part - np.float32(level)
            cube[j][mask[r0:r1]] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices are handled below
            sky[r0:r1] = np.nanmedian(cube, axis=0)
        count[r0:r1] = np.isfinite(cube).sum(axis=0)
        del cube

    low = (count < min_valid) | ~np.isfinite(sky)
    if low.mean() > 0.5:
        raise ReductionError(
            f"sky is undetermined for {100 * low.mean():.0f}% of the pixels (fewer than "
            f"{min_valid} unmasked samples); dither offsets too small or too few neighbours"
        )
    if low.any():
        fill = coarse_background(sky, low, box)
        sky[low] = fill[low]
    return sky.astype(np.float32)


def select_neighbors(index: int, times: Sequence[float], n: int, max_gap: float, allowed: Sequence[int]) -> list[int]:
    """Indices of the `n` allowed frames nearest in time to frame `index` (excluding itself)."""
    cand = [(abs(times[j] - times[index]), j) for j in allowed if j != index]
    cand = [c for c in cand if c[0] <= max_gap]
    cand.sort()
    return [j for _, j in cand[:n]]


# --------------------------------------------------------------------------
# Header helpers
# --------------------------------------------------------------------------
def _header_value(frame: Frame, key: str):
    value = frame.header.get(key)
    if value is None:
        for ext in frame.extensions:
            value = ext.header.get(key)
            if value is not None:
                break
    return value


def obs_time_seconds(frame: Frame) -> float:
    """Observation time in seconds (MJD-OBS, else DATE-OBS[/TIME-OBS])."""
    mjd = _header_value(frame, "MJD-OBS")
    try:
        return float(mjd) * 86400.0
    except (TypeError, ValueError):
        pass
    date = _header_value(frame, "DATE-OBS")
    if date is not None:
        text = str(date).strip()
        if "T" not in text and _header_value(frame, "TIME-OBS") is not None:
            text = f"{text}T{_header_value(frame, 'TIME-OBS')}"
        try:
            return float(Time(text, format="isot", scale="utc").mjd) * 86400.0
        except ValueError:
            pass
    raise ReductionError(f"{frame.path.name}: no usable MJD-OBS or DATE-OBS; cannot order exposures in time")


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------
class SkySubtraction(PipelineStage):
    name = "sky_subtraction"

    def __init__(self, config, n_workers: int = 1, timeout: float | None = None) -> None:
        super().__init__(config, n_workers=n_workers, timeout=timeout)
        self.sky: SkyConfig = getattr(config, "sky", None) or SkyConfig()

    def process(self, frame: Frame) -> Frame:
        raise NotImplementedError(
            "SkySubtraction needs the other exposures of the dither sequence; "
            "call run(frames, state) with the whole collection"
        )

    # ------------------------------------------------------------------ run
    def run(self, frames: FrameCollection, state: PipelineState) -> FrameCollection:
        started = monotonic()
        pool = list(frames)
        restored: dict[str, Frame] = {}
        pending: list[Frame] = []
        for f in pool:
            cached = self._restore_cached(f, state)
            if cached is None:
                pending.append(f)
            else:
                restored[f.frame_id] = cached
        if restored:
            logger.info("%s: skipping %d already-completed frames", self.name, len(restored))
        logger.info(
            "%s: processing started; input=%d, pending=%d, cached=%d, workers=%d, output_dir=%s",
            self.name,
            len(pool),
            len(pending),
            len(restored),
            self.n_workers,
            self.output_dir or "<in-memory>",
        )

        pending_ids = {f.frame_id for f in pending}
        groups: dict[tuple, list[Frame]] = defaultdict(list)
        for f in pool:  # every frame (even a finished one) is a candidate sky sample
            groups[self._group_key(f)].append(f)

        results: dict[str, Frame | Exception] = {}
        for key, members in groups.items():
            targets = [f for f in members if f.frame_id in pending_ids]
            if targets:
                logger.info(
                    "%s: selecting sky frames for group %s: %d candidate(s), %d target(s)",
                    self.name,
                    key,
                    len(members),
                    len(targets),
                )
                results.update(self._reduce_group(members, targets))

        output = FrameCollection()
        progress_interval = max(1, (len(pool) + 99) // 100)
        for index, f in enumerate(pool, start=1):
            if f.frame_id in restored:
                output.add(restored[f.frame_id])
            elif f.frame_id in pending_ids:
                outcome = results.get(f.frame_id) or ReductionError("frame was not processed")
                self._record(f.frame_id, outcome, state, output)
            if index % progress_interval == 0 or index == len(pool):
                logger.info(
                    "%s: result handling progress %d/%d (%.1f%%)",
                    self.name,
                    index,
                    len(pool),
                    100.0 * index / len(pool),
                )
        logger.info(
            "%s: processing complete; input=%d, successful=%d, failed=%d, resumed=%d, elapsed=%.1fs",
            self.name,
            len(pool),
            len(output),
            len(pool) - len(output),
            len(restored),
            monotonic() - started,
        )
        return output

    def _group_key(self, frame: Frame) -> tuple:
        return (frame.filt,) + tuple(str(_header_value(frame, k)) for k in self.sky.group_keys)

    # ---------------------------------------------------------------- group
    def _reduce_group(self, members: list[Frame], targets: list[Frame]) -> dict[str, Frame | Exception]:
        group_started = monotonic()
        cfg = self.sky
        results: dict[str, Frame | Exception] = {}

        times: dict[str, float] = {}
        for f in members:
            try:
                times[f.frame_id] = obs_time_seconds(f)
            except ReductionError as exc:
                if f in targets:
                    results[f.frame_id] = exc
        usable = [f for f in members if f.frame_id in times]
        t_arr = [times[f.frame_id] for f in usable]

        plan: dict[str, list[Frame]] = {}
        for target in targets:
            if target.frame_id not in times:
                continue
            ti = usable.index(target)
            allowed = [j for j, f in enumerate(usable) if f.shapes == target.shapes]
            nbrs = select_neighbors(ti, t_arr, cfg.n_neighbors, cfg.max_time_gap_s, allowed)
            logger.debug(
                "%s: selected %d sky neighbor(s) for %s from %d shape-compatible frame(s)",
                self.name,
                len(nbrs),
                target.path.name,
                len(allowed) - 1,
            )
            if len(nbrs) < cfg.min_neighbors:
                results[target.frame_id] = ReductionError(
                    f"{target.path.name}: only {len(nbrs)} other exposure(s) of the same filter/sequence within "
                    f"{cfg.max_time_gap_s:.0f} s (need {cfg.min_neighbors}); cannot build a sky"
                )
            else:
                plan[target.frame_id] = [usable[j] for j in nbrs]
        if not plan:
            return results

        by_id = {f.frame_id: f for f in usable}
        ext_indices = [e.index for e in by_id[next(iter(plan))].extensions]
        n_threads = max(1, min(self.n_workers, len(ext_indices)))
        logger.info(
            "%s: building skies for %d target frame(s) across %d detector(s) using %d thread(s)",
            self.name,
            len(plan),
            len(ext_indices),
            n_threads,
        )

        def run_ext(idx: int):
            ext_started = monotonic()
            logger.info("%s: detector extension %d processing started", self.name, idx)
            try:
                result = self._reduce_extension(idx, by_id, plan)
                logger.info(
                    "%s: detector extension %d processing complete in %.1fs",
                    self.name,
                    idx,
                    monotonic() - ext_started,
                )
                return result
            except Exception as exc:
                logger.exception("%s: detector extension %d failed", self.name, idx)
                return {tid: exc for tid in plan}, {}

        if n_threads > 1:
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                per_ext = dict(zip(ext_indices, pool.map(run_ext, ext_indices)))
        else:
            per_ext = {idx: run_ext(idx) for idx in ext_indices}

        for tid, nbrs in plan.items():
            errors = [per_ext[i][0][tid] for i in ext_indices if isinstance(per_ext[i][0].get(tid), Exception)]
            if errors:
                results[tid] = errors[0]
                continue
            results[tid] = self._assemble(
                by_id[tid],
                [per_ext[i][0][tid] for i in ext_indices],
                [per_ext[i][1][tid] for i in ext_indices],
                len(nbrs),
            )
        logger.info(
            "%s: group processing complete; targets=%d, elapsed=%.1fs",
            self.name,
            len(targets),
            monotonic() - group_started,
        )
        return results

    # ------------------------------------------------------------ extension
    def _reduce_extension(self, idx: int, by_id: dict[str, Frame], plan: dict[str, list[Frame]]):
        """Sky-subtract extension `idx` of every target. Returns ({id: array | Exception}, {id: level})."""
        cfg = self.sky
        ids = set(plan) | {n.frame_id for nbrs in plan.values() for n in nbrs}
        data = {fid: by_id[fid][idx].data for fid in ids}
        bad = {fid: bad_pixel_mask(d) for fid, d in data.items()}

        def detect(arr, fid, level=None):
            if level is None:  # level of the sky in `arr`, for the threshold floor
                good = arr[~bad[fid]]
                level = float(np.median(good)) if good.size else 0.0
            return detect_objects(
                arr, bad[fid], cfg.detect_nsigma, cfg.detect_smooth_pix, cfg.mask_grow_pix, cfg.background_box,
                floor=cfg.detect_floor_frac * abs(level),
            )

        # Bootstrap: first sky = plain median of the neighbours with only bad pixels excluded.
        # Dithering moves the objects, so the median already removes them, and -- unlike a
        # coarse background -- it captures the (static) sky gradient. Objects are then
        # detected on the *residual* (data - sky), which is flat.
        # Frames that are not being processed (finished earlier) have no residual: they get
        # a one-off detection on their own data.
        masks = {fid: (bad[fid].copy() if fid in plan else detect(data[fid], fid)) for fid in ids}
        outputs: dict[str, np.ndarray | Exception] = {}
        levels: dict[str, float] = {}

        for it in range(cfg.iterations + 1):  # pass 0 = bootstrap, then `iterations` masked passes
            outputs = {tid: err for tid, err in outputs.items() if isinstance(err, Exception)}
            try:
                lv = {fid: sky_level(data[fid], masks[fid]) for fid in ids}
            except ReductionError as exc:
                return {tid: exc for tid in plan}, {}
            for tid, nbrs in plan.items():
                if isinstance(outputs.get(tid), Exception):
                    continue
                try:
                    samples = [(data[n.frame_id], masks[n.frame_id], lv[n.frame_id]) for n in nbrs]
                    norm_sky = build_sky(samples, cfg.mode, cfg.min_valid, cfg.background_box)
                    level = fit_level(data[tid], masks[tid], norm_sky, cfg.mode)
                    sky_img = norm_sky * np.float32(level) if cfg.mode == "scale" else norm_sky + np.float32(level)
                    out = (data[tid] - sky_img).astype(np.float32)
                    out[bad[tid]] = data[tid][bad[tid]]  # keep 0 / NaN pixels as they were
                    outputs[tid] = out
                    levels[tid] = level
                except ReductionError as exc:
                    outputs[tid] = exc
            if it < cfg.iterations:
                for tid, out in outputs.items():
                    if not isinstance(out, Exception):
                        masks[tid] = detect(out, tid, levels[tid])
        return outputs, levels

    # ------------------------------------------------------------- assemble
    def _assemble(self, frame: Frame, arrays: list[np.ndarray], levels: list[float], n_nbrs: int) -> Frame:
        cfg = self.sky
        headers = []
        for ext, level in zip(frame.extensions, levels):
            h = ext.header.copy()
            h["SKYSUB"] = (True, "sky subtracted by NEWHOPE")
            h["SKYMODE"] = (cfg.mode, "sky scaling mode")
            h["SKYNFRM"] = (n_nbrs, "number of exposures in the sky median")
            h["SKYITER"] = (cfg.iterations, "object-masked sky passes")
            h["SKYLEVEL"] = (round(float(level), 4), "sky level of this frame (data units)")
            headers.append(h)
        primary = frame.header.copy()
        primary["SKYSUB"] = (True, "sky subtracted by NEWHOPE")
        new = frame.with_extension_data(arrays).with_extension_headers(headers)
        return new._replace(new.extensions, header=primary)
