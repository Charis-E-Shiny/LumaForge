import numpy as np
import pytest

from lumaforge.degrade import CONDITIONS, add_sensor_noise, apply_condition, colour_cast
from lumaforge.isp import classical as c
from lumaforge.isp.analysis import analyze, estimate_noise_profile, plan
from lumaforge.isp.pipeline import ISPConfig, run_classical
from lumaforge.metrics_lite import delta_e_76, psnr


def smooth_scene(size=128):
    """A gently varying colour image in linear camera space."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    return np.stack((0.2 + 0.3 * x, 0.25 + 0.25 * y, 0.15 + 0.2 * (x + y) / 2), axis=-1)


def mosaic(rgb):
    out = np.empty(rgb.shape[:2], dtype=np.float32)
    out[0::2, 0::2] = rgb[0::2, 0::2, 0]
    out[0::2, 1::2] = rgb[0::2, 1::2, 1]
    out[1::2, 0::2] = rgb[1::2, 0::2, 1]
    out[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return out


def test_pack_unpack_round_trip():
    bayer = np.random.default_rng(0).random((16, 24)).astype(np.float32)
    assert np.array_equal(c.unpack(c.pack(bayer)), bayer)


@pytest.mark.parametrize("cfa", ["rggb", "bggr", "grbg", "gbrg"])
def test_to_rggb_puts_red_at_origin(cfa):
    # Mark red samples with 1.0, using the layout that this pattern implies.
    bayer = np.zeros((8, 8), dtype=np.float32)
    index = cfa.index("r")
    bayer[index // 2 :: 2, index % 2 :: 2] = 1.0
    out = c.to_rggb(bayer, cfa)
    assert out[0, 0] == 1.0 and out[0, 1] == 0.0 and out[1, 0] == 0.0
    assert out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0


def test_to_rggb_rejects_unknown_pattern():
    with pytest.raises(ValueError):
        c.to_rggb(np.zeros((4, 4)), "rgbw")


def test_demosaic_recovers_smooth_scene():
    rgb = smooth_scene()
    out = c.demosaic_malvar(mosaic(rgb))
    assert psnr(out[4:-4, 4:-4], rgb[4:-4, 4:-4]) > 45


def test_demosaic_constant_image_stays_constant():
    # The Malvar filters each have unit DC gain, a direct check on their coefficients.
    out = c.demosaic_malvar(np.full((32, 32), 0.3, dtype=np.float32))
    assert np.allclose(out, 0.3, atol=1e-5)


def test_box_filter_matches_brute_force():
    image = np.random.default_rng(1).random((20, 20)).astype(np.float32)
    out = c.box_filter(image, 2)
    assert out.shape == image.shape
    assert out[10, 10] == pytest.approx(image[8:13, 8:13].mean(), abs=1e-5)


def test_guided_filter_keeps_edges_better_than_box():
    step = np.zeros((64, 64), dtype=np.float32)
    step[:, 32:] = 1.0
    noisy = step + np.random.default_rng(2).normal(0, 0.02, step.shape).astype(np.float32)
    guided = c.guided_filter(noisy, noisy, 3, 1e-3)
    assert np.abs(guided - step).mean() < np.abs(c.box_filter(noisy, 3) - step).mean()


def test_srgb_gamma_round_trip():
    values = np.linspace(0, 1, 101, dtype=np.float32)
    assert np.allclose(c.srgb_degamma(c.srgb_gamma(values)), values, atol=1e-5)


def test_ycbcr_round_trip():
    rgb = smooth_scene(16)
    assert np.allclose(c.ycbcr_to_rgb(c.rgb_to_ycbcr(rgb)), rgb, atol=1e-5)


def test_tone_map_preserves_chromaticity():
    rgb = np.array([[[0.2, 0.4, 0.1]]], dtype=np.float32)
    out = c.tone_map(rgb, 1.5, 2.0, 0.9)
    assert np.allclose(out / out.sum(), rgb / rgb.sum(), atol=1e-5)


def test_white_balance_removes_grey_world_cast():
    grey = np.full((64, 64, 4), 0.3, dtype=np.float32)
    grey += np.random.default_rng(3).normal(0, 0.01, grey.shape).astype(np.float32)
    grey[..., 0] *= 0.5
    grey[..., 3] *= 0.7
    gains = c.estimate_white_balance(grey)
    assert gains[0] == pytest.approx(2.0, rel=0.1)
    assert gains[2] == pytest.approx(1 / 0.7, rel=0.1)


def test_white_balance_prior_full_weight_returns_prior():
    packed = np.random.default_rng(4).random((32, 32, 4)).astype(np.float32) * 0.5
    prior = np.array([2.0, 1.0, 1.7])
    assert np.allclose(c.estimate_white_balance(packed, prior=prior, prior_weight=1.0), prior, atol=1e-4)


def test_noise_profile_recovers_known_parameters():
    rng = np.random.default_rng(5)
    rgb = smooth_scene(512)
    bayer = add_sensor_noise(mosaic(rgb), 2e-3, 2e-5, rng)
    shot, read = estimate_noise_profile(c.pack(bayer))
    expected = 2e-3 * 0.3 + 2e-5
    assert shot * 0.3 + read == pytest.approx(expected, rel=0.35)


def test_measured_noise_rises_with_added_noise():
    rng = np.random.default_rng(6)
    clean = mosaic(smooth_scene(256))
    low = analyze(c.pack(add_sensor_noise(clean, 1e-4, 1e-6, rng))).sigma
    high = analyze(c.pack(add_sensor_noise(clean, 5e-3, 3e-5, rng))).sigma
    assert high > low * 3


def test_plan_routes_noisy_capture_to_quality_model():
    rng = np.random.default_rng(7)
    clean = mosaic(smooth_scene(256))
    calm = plan(analyze(c.pack(add_sensor_noise(clean, 1e-5, 1e-7, rng))))
    noisy = plan(analyze(c.pack(add_sensor_noise(clean, 1e-5, 1e-7, rng))), compensation=16.0)
    assert calm.route == "fast"
    assert noisy.exposure == 16.0
    assert calm.sharpen >= noisy.sharpen


def test_pipeline_output_is_valid_and_each_switch_runs():
    bayer = add_sensor_noise(mosaic(smooth_scene(64)), 1e-3, 1e-5, np.random.default_rng(8))
    for config in (ISPConfig(), ISPConfig(denoise=False), ISPConfig(white_balance=False), ISPConfig(ccm=False),
                   ISPConfig(tone=False), ISPConfig(chroma=False), ISPConfig(adaptive=False)):
        out = run_classical(bayer, "IP", config)
        assert out.rgb.shape == (64, 64, 3)
        assert out.rgb.min() >= 0 and out.rgb.max() <= 1


def test_conditions_are_reproducible_and_change_input():
    bayer = mosaic(smooth_scene(32))
    for name, condition in CONDITIONS.items():
        first = apply_condition(bayer, condition, np.random.default_rng(1))
        second = apply_condition(bayer, condition, np.random.default_rng(1))
        assert np.array_equal(first, second)
        assert np.array_equal(first, bayer) == (name == "clean")


def test_colour_cast_touches_only_red_and_blue_sites():
    bayer = np.ones((4, 4), dtype=np.float32)
    out = colour_cast(bayer, 2.0, 0.5)
    assert out[0, 0] == 2.0 and out[1, 1] == 0.5 and out[0, 1] == 1.0 and out[1, 0] == 1.0


def test_delta_e_is_zero_for_identical_images():
    rgb = np.clip(smooth_scene(16), 0, 1)
    assert delta_e_76(rgb, rgb) == 0
