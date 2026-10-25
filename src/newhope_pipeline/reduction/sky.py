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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from time import monotonic
from typing import Optional, Sequence

import numpy as np
from astropy.time import Time
from scipy import ndimage

from ..config import SkyConfig
from ..core.exceptions import ReductionError, detach_traceback
from ..core.frame import Frame, FrameCollection, FrameLike, FrameReference
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

    def __init__(
        self,
        config,
        n_workers: int = 1,
        timeout: float | None = None,
        batch_size: int = 1,
    ) -> None:
        super().__init__(config, n_workers=n_workers, timeout=timeout, batch_size=batch_size)
        self.sky: SkyConfig = getattr(config, "sky", None) or SkyConfig()

    def process(self, frame: Frame) -> Frame:
        raise NotImplementedError(
            "SkySubtraction needs the other exposures of the dither sequence; "
            "call run(frames, state) with the whole collection"
        )

    # ------------------------------------------------------------------ run
    def run(self, frames: FrameCollection, state: PipelineState) -> FrameCollection:
        pool = list(frames)
        if all(isinstance(frame, FrameReference) for frame in pool):
            return self._run_references(pool, state)
        return self._run_loaded(pool, state)

    def _run_loaded(self, pool: list[FrameLike], state: PipelineState) -> FrameCollection:
        started = monotonic()
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
                self._record(f, outcome, state, output)
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

    def _run_references(
        self, pool: list[FrameReference], state: PipelineState
    ) -> FrameCollection:
        """Reduce reference inputs one target at a time with a temporal halo in memory."""
        started = monotonic()
        if self.output_dir is None:
            self.output_dir = state.manifest_path.parent / "processed" / self.name
            logger.info("%s: persisting lazy sky products under %s", self.name, self.output_dir)
        restored: dict[str, FrameReference] = {}
        pending: list[FrameReference] = []
        for frame in pool:
            cached = self._restore_cached(frame, state)
            if cached is None:
                pending.append(frame)
            else:
                assert isinstance(cached, FrameReference)
                restored[frame.frame_id] = cached

        pending_ids = {frame.frame_id for frame in pending}
        groups: dict[tuple, list[FrameReference]] = defaultdict(list)
        for frame in pool:
            groups[self._group_key(frame)].append(frame)

        plans: dict[str, list[FrameReference]] = {}
        failures: dict[str, Exception] = {}
        for key, members in groups.items():
            times: dict[str, float] = {}
            for frame in members:
                try:
                    times[frame.frame_id] = obs_time_seconds(frame)
                except ReductionError as exc:
                    if frame.frame_id in pending_ids:
                        failures[frame.frame_id] = exc
            usable = [frame for frame in members if frame.frame_id in times]
            time_values = [times[frame.frame_id] for frame in usable]
            index_by_id = {frame.frame_id: index for index, frame in enumerate(usable)}
            for target in usable:
                if target.frame_id not in pending_ids:
                    continue
                target_index = index_by_id[target.frame_id]
                allowed = [
                    index for index, candidate in enumerate(usable)
                    if candidate.shapes == target.shapes
                ]
                neighbor_indices = select_neighbors(
                    target_index,
                    time_values,
                    self.sky.n_neighbors,
                    self.sky.max_time_gap_s,
                    allowed,
                )
                neighbors = [usable[index] for index in neighbor_indices]
                logger.debug(
                    "%s: selected %d neighbor(s) for %s from %d shape-compatible candidate(s)",
                    self.name,
                    len(neighbors),
                    target.path.name,
                    len(allowed) - 1,
                )
                if len(neighbors) < self.sky.min_neighbors:
                    failures[target.frame_id] = ReductionError(
                        f"{target.path.name}: only {len(neighbors)} other exposure(s) of the same "
                        f"filter/sequence within {self.sky.max_time_gap_s:.0f} s "
                        f"(need {self.sky.min_neighbors}); cannot build a sky"
                    )
                else:
                    plans[target.frame_id] = neighbors
            logger.info(
                "%s: group %s has %d frame(s), %d pending target(s)",
                self.name,
                key,
                len(members),
                sum(frame.frame_id in pending_ids for frame in members),
            )

        if restored:
            logger.info("%s: skipping %d already-completed frame(s)", self.name, len(restored))
        logger.info(
            "%s: processing started; input=%d, pending=%d, cached=%d, batch_size=%d, workers=%d; "
            "sky targets use one target plus its iteration neighbor halo; at most %d task(s) in flight",
            self.name,
            len(pool),
            len(pending),
            len(restored),
            self.batch_size,
            self.n_workers,
            min(self.batch_size, self.n_workers, len(pending)) if pending else 0,
        )

        output = FrameCollection()
        progress_interval = max(1, (len(pool) + 99) // 100)
        next_progress = progress_interval
        failed = 0
        effective_workers = min(self.n_workers, self.batch_size, len(pending)) if pending else 1
        target_executor = (
            ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="sky-target")
            if effective_workers > 1
            else None
        )
        try:
            for start in range(0, len(pool), self.batch_size):
                batch = pool[start : start + self.batch_size]
                targets = [
                    frame for frame in batch
                    if frame.frame_id not in restored
                    and frame.frame_id not in failures
                    and frame.frame_id in plans
                ]
                batch_ids = {frame.frame_id for frame in batch}
                batch_output: dict[str, FrameReference] = {
                    frame_id: restored[frame_id]
                    for frame_id in batch_ids.intersection(restored)
                }

                def record_one(frame: FrameReference, outcome: Frame | Exception) -> None:
                    nonlocal failed
                    recorded = FrameCollection()
                    self._record(frame, outcome, state, recorded)
                    if recorded:
                        result = next(iter(recorded))
                        assert isinstance(result, FrameReference)
                        batch_output[frame.frame_id] = result
                    else:
                        failed += 1

                if target_executor is None:
                    for frame in targets:
                        try:
                            outcome = self._reduce_reference_target(frame, plans, pending_ids)
                        except Exception as exc:  # noqa: BLE001 -- per-frame failure recovery
                            logger.exception("%s failed while reducing target %s", self.name, frame.path.name)
                            outcome = detach_traceback(exc)
                        record_one(frame, outcome)
                        del outcome
                else:
                    next_target = iter(targets)
                    future_targets = {}
                    while True:
                        while len(future_targets) < effective_workers:
                            try:
                                frame = next(next_target)
                            except StopIteration:
                                break
                            future_targets[
                                target_executor.submit(
                                    self._reduce_reference_target, frame, plans, pending_ids
                                )
                            ] = frame
                        if not future_targets:
                            break
                        done, _ = wait(future_targets, return_when=FIRST_COMPLETED)
                        for future in done:
                            frame = future_targets.pop(future)
                            try:
                                outcome = future.result()
                            except Exception as exc:  # noqa: BLE001 -- per-frame failure recovery
                                logger.exception("%s failed while reducing target %s", self.name, frame.path.name)
                                outcome = detach_traceback(exc)
                            record_one(frame, outcome)
                            del outcome, future
                        done.clear()

                for offset, frame in enumerate(batch, start=1):
                    index = start + offset
                    if frame.frame_id not in restored:
                        logger.info(
                            "%s: target %d/%d %s (batch_size=%d, workers=%d)",
                            self.name,
                            index,
                            len(pool),
                            frame.path.name,
                            self.batch_size,
                            effective_workers,
                        )
                        if frame.frame_id in failures:
                            outcome: Frame | Exception = failures[frame.frame_id]
                        elif frame.frame_id not in plans:
                            outcome = ReductionError(f"{frame.path.name}: no usable sky plan was created")
                        else:
                            # Successful targets were persisted as soon as the
                            # worker completed; failures are recorded here.
                            outcome = None
                        if outcome is not None:
                            record_one(frame, outcome)
                        del outcome
                    if frame.frame_id in batch_output:
                        output.add(batch_output[frame.frame_id])
                completed = start + len(batch)
                if completed >= next_progress or completed == len(pool):
                    logger.info(
                        "%s: completed=%d/%d, skipped=%d, failed=%d",
                        self.name,
                        completed,
                        len(pool),
                        len(restored),
                        failed,
                    )
                    next_progress = completed + progress_interval
                batch_output.clear()
        finally:
            if target_executor is not None:
                target_executor.shutdown(wait=True)
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

    def _reduce_reference_target(
        self,
        target: FrameReference,
        plans: dict[str, list[FrameReference]],
        pending_ids: set[str],
    ) -> Frame:
        """Load one target's neighbor/iteration halo and apply the existing sky math."""
        iterations = self.sky.iterations
        target_id = target.frame_id
        needed_masks: list[set[str]] = [set() for _ in range(iterations + 1)]
        needed_masks[iterations] = {target_id} | {neighbor.frame_id for neighbor in plans[target_id]}
        update_outputs: list[set[str]] = [set() for _ in range(iterations)]
        plan_ids = set(plans)
        for iteration in range(iterations, 0, -1):
            update_outputs[iteration - 1] = needed_masks[iteration] & plan_ids
            prior_masks = set(update_outputs[iteration - 1])
            for frame_id in update_outputs[iteration - 1]:
                prior_masks.update(neighbor.frame_id for neighbor in plans[frame_id])
            needed_masks[iteration - 1] = prior_masks

        data_ids = {target_id}
        data_ids.update(neighbor.frame_id for neighbor in plans[target_id])
        for output_ids in update_outputs:
            for frame_id in output_ids:
                data_ids.add(frame_id)
                data_ids.update(neighbor.frame_id for neighbor in plans[frame_id])
        references = {frame.frame_id: frame for neighbors in plans.values() for frame in neighbors}
        references.update({target_id: target})
        # Plan targets can occur only as values in the neighbor lists; add all
        # target references passed through the group plan.
        for frame_id in data_ids:
            if frame_id not in references:
                raise ReductionError(f"missing reference for sky dependency {frame_id}")

        logger.debug(
            "%s: %s requires %d halo frame(s) over %d detector extension(s)",
            self.name,
            target.path.name,
            len(data_ids),
            target.n_ext,
        )
        loaded_target = target.load()
        output_arrays: list[np.ndarray] = []
        levels: list[float] = []
        for extension in target.extensions:
            idx = extension.index
            data: dict[str, np.ndarray] = {target_id: loaded_target[idx].data}
            for frame_id in data_ids - {target_id}:
                reference = references[frame_id]
                data[frame_id] = reference.load(extension_indices=[idx])[idx].data
            bad_masks = {frame_id: bad_pixel_mask(array) for frame_id, array in data.items()}

            def detect(
                arr: np.ndarray,
                bad_mask: np.ndarray,
                level: float | None = None,
            ):
                if level is None:
                    good = arr[~bad_mask]
                    level = float(np.median(good)) if good.size else 0.0
                return detect_objects(
                    arr,
                    bad_mask,
                    self.sky.detect_nsigma,
                    self.sky.detect_smooth_pix,
                    self.sky.mask_grow_pix,
                    self.sky.background_box,
                    floor=self.sky.detect_floor_frac * abs(level),
                )

            masks = {
                frame_id: (
                    bad_masks[frame_id].copy()
                    if frame_id in plan_ids
                    else detect(data[frame_id], bad_masks[frame_id])
                )
                for frame_id in data
            }
            final_output = None
            final_level = None
            failed_outputs: dict[str, ReductionError] = {}
            for iteration in range(iterations + 1):
                active = set(update_outputs[iteration]) if iteration < iterations else set()
                if iteration == iterations:
                    active.add(target_id)
                next_masks: dict[str, np.ndarray] = {}
                for frame_id in active:
                    if frame_id in failed_outputs:
                        continue
                    neighbors = plans.get(frame_id)
                    if not neighbors:
                        continue
                    try:
                        samples = [
                            (data[neighbor.frame_id], masks[neighbor.frame_id],
                             sky_level(data[neighbor.frame_id], masks[neighbor.frame_id]))
                            for neighbor in neighbors
                        ]
                        normalized_sky = build_sky(
                            samples, self.sky.mode, self.sky.min_valid, self.sky.background_box
                        )
                        level = fit_level(data[frame_id], masks[frame_id], normalized_sky, self.sky.mode)
                        sky_image = (
                            normalized_sky * np.float32(level)
                            if self.sky.mode == "scale"
                            else normalized_sky + np.float32(level)
                        )
                        corrected = (data[frame_id] - sky_image).astype(np.float32)
                        corrected[bad_masks[frame_id]] = data[frame_id][bad_masks[frame_id]]
                        if frame_id == target_id and iteration == iterations:
                            final_output, final_level = corrected, level
                        elif iteration < iterations:
                            # Compute masks against this pass's old masks, then
                            # install them together as the next iteration does.
                            next_masks[frame_id] = detect(corrected, bad_masks[frame_id], level)
                    except ReductionError as exc:
                        failed_outputs[frame_id] = detach_traceback(exc)
                        # A failed halo frame keeps its previous mask, matching
                        # the current group reducer's per-target continuation.
                        if frame_id == target_id and iteration == iterations:
                            raise
                masks.update(next_masks)
            if target_id in failed_outputs:
                raise failed_outputs[target_id]
            if final_output is None or final_level is None:
                raise ReductionError(f"{target.path.name}: failed to construct sky for extension {idx}")
            output_arrays.append(final_output)
            levels.append(final_level)
            del data, bad_masks, masks

        return self._assemble(loaded_target, output_arrays, levels, len(plans[target_id]))

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
