import numpy as np
import pytest
from astropy.io import fits

from newhope_pipeline.config import SkyConfig
from newhope_pipeline.core.frame import Frame, FrameCollection
from newhope_pipeline.core.pipeline import Pipeline
from newhope_pipeline.core.stage import PipelineStage
from newhope_pipeline.core.state import PipelineState
from newhope_pipeline.reduction.sky import (
    SkySubtraction,
    bad_pixel_mask,
    build_sky,
    coarse_background,
    detect_objects,
)

N = 128
MJD0 = 60000.0
STAR_AMP = 400.0


class Cfg:
    def __init__(self, **kw):
        self.sky = SkyConfig(**kw)


def _star(shape, y, x, amp=STAR_AMP, sig=2.0):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sig**2))


def make_sequence(tmp_path, n=12, obj="FIELD1", filt="HX", start=0, gap_s=40.0, noise=1.0, seed=0, name="s"):
    """n dithered exposures: smooth sky gradient (different per chip) whose level drifts
    with time, plus one star at a different position in every frame."""
    rng = np.random.default_rng(seed)
    paths, truth = [], []
    for i in range(n):
        t = start + i
        primary = fits.PrimaryHDU()
        primary.header.update({"FILTER": filt, "EXPTIME": 30.0, "OBJECT": obj})
        primary.header["MJD-OBS"] = MJD0 + t * gap_s / 86400.0
        hdus = [primary]
        yy, xx = np.mgrid[:N, :N]
        star_pos = (20 + 7 * i % 90, 15 + 11 * i % 100)
        per_chip = {}
        for k in range(1, 5):
            level = 1000.0 * k * (1 + 0.004 * t)  # slow drift
            sky = level * (1 + 0.05 * (xx / N) - 0.03 * (yy / N))
            star = _star((N, N), *star_pos)
            data = sky + star + rng.normal(0, noise, (N, N))
            hdus.append(fits.ImageHDU(data.astype(np.float32), name=f"im{k}"))
            per_chip[k] = (sky, star)
        p = tmp_path / f"{name}_{obj}_{t:03d}.fits"
        fits.HDUList(hdus).writeto(p)
        paths.append(p)
        truth.append((star_pos, per_chip))
    return paths, truth


def load(paths):
    return FrameCollection([Frame.from_fits(p) for p in paths])


def run_sky(frames, tmp_path, **kw):
    state = PipelineState(manifest_path=tmp_path / "state.json")
    stage = SkySubtraction(Cfg(**kw))
    return stage, stage.run(frames, state), state


def test_sky_removed_stars_kept_per_chip(tmp_path):
    paths, truth = make_sequence(tmp_path)
    stage, out, state = run_sky(load(paths), tmp_path)
    assert len(out) == len(paths) and not state.failures()
    for frame, (pos, per_chip) in zip(out, truth):
        for k in range(1, 5):
            data = frame[k].data
            star_mask = _star((N, N), *pos, amp=1.0) > 1e-3
            background = data[~ndilate(star_mask)]
            assert abs(np.median(background)) < 2.0  # sky gone (was ~1000*k)
            assert np.std(background) < 8.0  # gradient removed too (was ~50*k)
            peak = data[int(pos[0]), int(pos[1])]
            assert peak == pytest.approx(STAR_AMP, rel=0.05)  # star not eaten by the sky model


def test_reference_sky_matches_loaded_group_results(tmp_path):
    paths, _ = make_sequence(tmp_path, n=10)
    config = Cfg(n_neighbors=6, min_neighbors=4)

    eager = SkySubtraction(config).run(load(paths), PipelineState(tmp_path / "eager.json"))
    lazy_stage = SkySubtraction(config, n_workers=2, batch_size=2)
    lazy_stage.output_dir = tmp_path / "sky-products"
    lazy = lazy_stage.run(
        FrameCollection.from_directory(tmp_path, pattern="s_*.fits", expected_extensions=4),
        PipelineState(tmp_path / "lazy.json"),
    )

    assert len(eager) == len(lazy) == len(paths)
    for eager_frame, lazy_frame in zip(eager, lazy):
        lazy_frame = lazy_frame.load()
        assert eager_frame.frame_id == lazy_frame.frame_id
        assert eager_frame.path.name == lazy_frame.path.name
        assert eager_frame.header == lazy_frame.header
        assert [ext.name for ext in eager_frame] == [ext.name for ext in lazy_frame]
        for index in range(1, 5):
            assert eager_frame[index].header == lazy_frame[index].header
            np.testing.assert_allclose(eager_frame[index].data, lazy_frame[index].data, rtol=0, atol=0)


def ndilate(mask):
    from scipy import ndimage

    return ndimage.binary_dilation(mask, iterations=3)


def test_headers_record_the_subtraction(tmp_path):
    paths, _ = make_sequence(tmp_path, n=8)
    _, out, _ = run_sky(load(paths), tmp_path, n_neighbors=5)
    f = list(out)[0]
    assert f.header["SKYSUB"] is True
    assert f[2].header["SKYNFRM"] == 5 and f[2].header["SKYMODE"] == "scale"
    assert f[2].header["SKYLEVEL"] == pytest.approx(2000.0, rel=0.05)


def test_too_few_neighbours_fails_cleanly(tmp_path):
    paths, _ = make_sequence(tmp_path, n=3)
    _, out, state = run_sky(load(paths), tmp_path)  # min_neighbors=4 > 2 available
    assert len(out) == 0
    assert len(state.failures("sky_subtraction")) == 3


def test_sequences_are_separated_by_object_and_filter(tmp_path):
    a, _ = make_sequence(tmp_path, n=8, obj="A", seed=1)
    b, _ = make_sequence(tmp_path, n=8, obj="B", seed=2, start=100)  # B is far brighter
    # make B's sky 3x brighter so that mixing sequences would be obvious
    for p in b:
        with fits.open(p, mode="update") as h:
            for hdu in h[1:]:
                hdu.data *= 3.0
    c, _ = make_sequence(tmp_path, n=3, obj="A", filt="JX", seed=3, name="j")  # other filter, too few
    _, out, state = run_sky(load(a + b + c), tmp_path)
    assert len(out) == 16  # the 3 JX frames lack neighbours and fail
    assert len(state.failures()) == 3
    for f in out:
        assert abs(np.median(f[1].data)) < 3.0 + 1e-4 * f[1].header['SKYLEVEL']  # neither A nor B was corrected with the other's sky


def test_far_away_exposures_are_not_used(tmp_path):
    paths, _ = make_sequence(tmp_path, n=14, gap_s=600.0)  # window 1800 s -> at most 3 neighbours per side
    _, out, state = run_sky(load(paths), tmp_path, min_neighbors=3, n_neighbors=10, min_valid=2)
    assert len(out) == 14 and not state.failures()
    counts = [f[1].header["SKYNFRM"] for f in out]
    assert max(counts) == 6 and min(counts) == 3  # middle frames: 3+3, first/last: 3 (never the distant ones)


def test_bad_pixels_keep_their_value(tmp_path):
    paths, _ = make_sequence(tmp_path, n=8)
    with fits.open(paths[3], mode="update") as h:
        h[1].data[10, 10] = 0.0
        h[1].data[11, 11] = np.nan
    _, out, _ = run_sky(load(paths), tmp_path)
    f = list(out)[3]
    assert f[1].data[10, 10] == 0.0 and np.isnan(f[1].data[11, 11])  # not turned into -sky


def test_offset_mode(tmp_path):
    paths, _ = make_sequence(tmp_path, n=8)
    _, out, state = run_sky(load(paths), tmp_path, mode="offset")
    assert not state.failures()
    assert abs(np.median(list(out)[2][3].data)) < 10.0


def test_input_frames_are_not_mutated(tmp_path):
    paths, _ = make_sequence(tmp_path, n=6)
    frames = load(paths)
    before = [f[1].data.copy() for f in frames]
    run_sky(frames, tmp_path)
    assert all(np.array_equal(b, f[1].data) for b, f in zip(before, frames))


def test_helpers_background_and_detection():
    rng = np.random.default_rng(1)
    img = 500.0 + rng.normal(0, 2, (200, 200)).astype(np.float32)
    img += _star(img.shape, 100, 100).astype(np.float32)
    bad = bad_pixel_mask(img)
    assert np.allclose(coarse_background(img, bad, 64).mean(), 500.0, atol=2.0)
    mask = detect_objects(img, bad)
    assert mask[100, 100] and not mask[10, 10]
    assert mask.mean() < 0.05


def test_build_sky_ignores_masked_pixels():
    samples = []
    for i in range(5):
        d = np.full((64, 64), 100.0, dtype=np.float32)
        m = np.zeros_like(d, dtype=bool)
        if i < 3:  # an object sits at the same place in 3 samples and is masked there
            d[20:24, 20:24] = 9999.0
            m[20:24, 20:24] = True
        if i < 4:  # a second spot is masked in 4 of 5 samples -> too few valid samples
            d[40:44, 40:44] = 7777.0
            m[40:44, 40:44] = True
        samples.append((d, m, 100.0))
    sky = build_sky(samples, "scale", min_valid=2, box=16)
    assert sky[22, 22] == pytest.approx(1.0)  # 2 clean samples remain; contaminated ones were ignored
    assert sky[42, 42] == pytest.approx(1.0, abs=0.01)  # 1 clean sample < min_valid -> interpolated, not 77.77
    assert sky[5, 5] == pytest.approx(1.0)


# ---- stage persistence / resume (the bug where cached frames re-entered the pipeline raw) ----------


class AddStage(PipelineStage):
    def __init__(self, name, amount):
        super().__init__(config=None)
        self.name = name
        self.amount = amount

    def process(self, frame):
        return frame.map_extensions(lambda ext: ext.data + self.amount)


def test_resume_reloads_processed_output_not_raw(tmp_path):
    paths, _ = make_sequence(tmp_path, n=1)
    manifest = tmp_path / "state.json"

    def stages():
        a, b = AddStage("add10", 10.0), AddStage("add1", 1.0)
        a.output_dir, b.output_dir = tmp_path / "out" / "add10", tmp_path / "out" / "add1"
        return a, b

    a, _ = stages()
    first = list(Pipeline([a], manifest_path=manifest).run(load(paths)))[0]
    raw_value = Frame.from_fits(paths[0])[1].data[0, 0]
    assert first[1].data[0, 0] == pytest.approx(raw_value + 10.0)

    # "new session": add10 is already done, add1 is new
    a, b = stages()
    final = list(Pipeline([a, b], manifest_path=manifest).run(load(paths)))[0]
    assert final[1].data[0, 0] == pytest.approx(raw_value + 11.0)  # +10 was NOT lost
    assert (tmp_path / "out" / "add1" / paths[0].name).is_file()
    assert final.frame_id == first.frame_id
