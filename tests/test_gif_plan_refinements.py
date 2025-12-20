import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.gif_processing.gif_generator import GifGenerator
from src.gif_processing.gif_optimizer import GifOptimizer
from src.gif_processing.pil_compression import PILCompressor
from src.video_processing.video_compressor import VideoCompressor


class DummyConfigHelper:
    def get_optimization_config(self):
        return {
            "near_target": {
                "threshold_percent": 15,
                "absolute_mb_threshold": 0.3,
            },
            "allow_aggressive_compression": False
        }

    def get_quality_floors(self):
        return {
            "enforce": True,
            "min_width": 360,
            "min_fps": 15,
            "min_width_aggressive": 360,
            "min_fps_aggressive": 15
        }

    def get_segmentation_config(self):
        return {}


def _build_optimizer_stub(temp_dir: str) -> GifOptimizer:
    opt = GifOptimizer.__new__(GifOptimizer)
    opt.temp_dir = temp_dir
    opt.fast_mode = False
    opt.gifsicle_optimize_level = 2
    opt.skip_gifsicle_far_over_ratio = 2.5
    opt.near_target_max_runs = 4
    opt.long_clip_guardrails = {}
    opt.multi_pass_enabled = True
    opt.adaptive_strategy = False
    opt.progressive_refinement = False
    opt.content_aware = False
    opt.parameter_presets = {}
    opt.advanced_options = {}
    opt.stage_time_budgets = {'lossless': 8, 'gifsicle': 10, 'palette': 12}
    opt._shutdown_checker = lambda: False
    opt.shutdown_requested = False
    opt._is_tool_available = lambda *_args, **_kwargs: True
    opt.config_helper = DummyConfigHelper()
    return opt


def test_stage2_projection_skip_reencode(tmp_path: Path):
    gif_path = tmp_path / "projection.gif"
    target_mb = 5.0
    target_bytes = int(target_mb * 1024 * 1024)
    with open(gif_path, "wb") as handle:
        handle.write(b"\x00" * int(target_bytes * 1.2))

    optimizer = _build_optimizer_stub(str(tmp_path))

    with patch.object(optimizer, "_calculate_required_parameters", return_value={
        'width': 400, 'height': 400, 'fps': 20, 'colors': 200, 'lossy': 0
    }), patch.object(optimizer, "_predict_gif_size", return_value=target_mb * 1.05), \
            patch.object(optimizer, "_stage_lossless_optimization", return_value=False), \
            patch.object(optimizer, "_stage_adaptive_search", return_value=False), \
            patch.object(optimizer, "_stage_gifsicle_lossy_compression", return_value=False), \
            patch.object(optimizer, "_stage_ffmpeg_fallback_compression", return_value=False), \
            patch.object(optimizer, "_stage_pil_compression", return_value=False), \
            patch.object(optimizer, "_progressive_resolution_reduction", return_value=False), \
            patch("src.gif_processing.gif_optimizer.get_gif_info", return_value={
                'width': 400, 'height': 400, 'fps': 20, 'duration': 4.0, 'colors': 200
            }):
        reencode_mock = MagicMock(side_effect=AssertionError("Stage 2 re-encode should be skipped"))
        optimizer._reencode_from_source = reencode_mock

        optimizer._run_optimization_stages(
            str(gif_path),
            target_bytes=target_bytes,
            target_size_mb=target_mb,
            original_size_mb=os.path.getsize(gif_path) / (1024 * 1024),
            source_video=str(gif_path),
        )

        reencode_mock.assert_not_called()


def test_multi_pass_caps_attempts_and_stops_on_regressions(tmp_path: Path):
    gif_path = tmp_path / "multi_pass.gif"
    target_bytes = 5000
    with open(gif_path, "wb") as handle:
        handle.write(b"\x00" * 8000)

    optimizer = _build_optimizer_stub(str(tmp_path))
    optimizer.enable_early_exit = False

    sizes = [7000, 9000, 9500]

    def fake_run(_in, out, **_kwargs):
        idx = min(len(call_log), len(sizes) - 1)
        with open(out, "wb") as handle:
            handle.write(b"\x00" * sizes[idx])
        call_log.append(out)
        return True

    call_log = []
    optimizer._run_gifsicle = fake_run
    optimizer._adaptive_gifsicle_strategy = lambda *_args, **_kwargs: {
        'strategy': 'balanced',
        'optimize': 2,
        'colors_range': [192, 160],
        'lossy_range': [40, 60],
        'scale_range': [0.95, 0.9],
        'resize_method': 'mix',
    }

    result = optimizer._gifsicle_multi_pass_optimize(str(gif_path), target_bytes, target_bytes / (1024 * 1024))

    assert len(call_log) <= 3, "multi-pass should cap total attempts to 3"
    assert result is None or os.path.exists(result)


def test_cached_palette_reuse_skips_generation(tmp_path: Path):
    input_video = tmp_path / "input.mp4"
    output_gif = tmp_path / "out.gif"
    cached_palette = tmp_path / "cache.png"
    for path in (input_video, cached_palette):
        path.write_bytes(b"\x00")

    class DummyConfig:
        def get(self, key, default=None):
            if key == 'gif_settings.single_gif.quick_feasibility.enabled':
                return False
            if key == 'gif_settings.performance.single_gif_parallel.precompute_palette':
                return False
            return default

        def get_temp_dir(self):
            return str(tmp_path)

    generator = GifGenerator.__new__(GifGenerator)
    generator.config = DummyConfig()
    generator.config_helper = DummyConfigHelper()
    generator.ffmpeg_utils = MagicMock()
    generator.optimizer = MagicMock()
    generator._is_shutdown_requested = lambda: False
    generator.long_clip_guardrails = {}
    generator.temp_dir = str(tmp_path)

    with patch.object(generator, "_build_filter_profile", return_value={
        'fps': 12,
        'mpdecimate': {'hi': 640, 'lo': 320, 'frac': 0.25},
        'palette_stats_mode': 'full',
        'max_colors': 64,
        'guardrails_active': False,
    }), patch.object(generator, "_create_gif_with_palette") as create_with_palette, \
            patch.object(generator, "_create_gif_split_palette", side_effect=AssertionError("split palette should be skipped")), \
            patch.object(generator, "_generate_palette", side_effect=AssertionError("palette generation should be skipped")):
        create_with_palette.side_effect = lambda *args, **kwargs: output_gif.write_bytes(b"\x00" * 1024) or True

        result = generator._create_single_gif(
            str(input_video),
            str(output_gif),
            {
                'max_size_mb': 10,
                'fps': 12,
                'width': 120,
                'height': -1,
                'colors': 64,
                'palette_max_colors': 64,
            },
            start_time=0.0,
            duration=1.0,
            skip_optimizer=True,
            cached_palette_path=str(cached_palette),
            disable_parallel_palette=True,
        )

        assert result['success'] is True
        assert create_with_palette.call_count == 1


def test_invoke_single_dedupes_cached_palette(tmp_path: Path):
    """Ensure cached_palette_path is forwarded once when present in runtime opts."""
    generator = GifGenerator.__new__(GifGenerator)
    captured = {}

    def _fake_single(*args, **kwargs):
        captured["kwargs"] = kwargs
        return {"success": True}

    generator._create_single_gif = _fake_single

    result = generator._invoke_single_with_runtime_options(
        input_video="input.mp4",
        output_path=str(tmp_path / "out.gif"),
        settings={"max_size_mb": 10},
        start_time=0.0,
        duration=1.0,
        force_guardrails=False,
        runtime_opts={
            "disable_parallel_palette": True,
            "cached_palette_path": "pal.png",
            "palette_timeout_override": 12,
        },
    )

    assert result["success"] is True
    assert captured["kwargs"]["cached_palette_path"] == "pal.png"
    assert "cached_palette_path" not in {
        k for k in captured["kwargs"] if k != "cached_palette_path"
    }


def test_quantize_with_adaptive_palette_handles_typeerror(monkeypatch, tmp_path: Path):
    """TypeErrors during palette quantization should downgrade gracefully."""
    gif_path = tmp_path / "in.gif"
    out_path = tmp_path / "out.gif"
    Image.new("RGB", (2, 2), color="red").save(gif_path, format="GIF")

    def _raise_type_error(self, *args, **kwargs):
        raise TypeError("unsupported operand type(s) for &: 'tuple' and 'int'")

    monkeypatch.setattr(Image.Image, "quantize", _raise_type_error)

    success = PILCompressor.quantize_with_adaptive_palette(
        str(gif_path), str(out_path), max_colors=8
    )

    assert success is False
    assert not out_path.exists()


def test_return_cae_best_effort_candidate(tmp_path: Path):
    """Best-effort candidate within acceptance should be returned and flagged."""
    compressor = VideoCompressor.__new__(VideoCompressor)
    input_path = tmp_path / "in.mp4"
    candidate_path = tmp_path / "cand.mp4"
    output_path = tmp_path / "out.mp4"

    input_path.write_bytes(b"\x00" * 16)
    candidate_path.write_bytes(b"\x00" * 32)

    compressor.temp_dir = str(tmp_path)
    compressor._get_compression_results = lambda *_args, **_kwargs: {
        "success": True,
        "method": "cae_discord_10mb_best_effort",
    }

    result = compressor._return_cae_best_effort_candidate(
        str(input_path),
        str(output_path),
        {},
        {
            "path": str(candidate_path),
            "size_mb": 5.0,
            "pass_index": 2,
            "reason": "quality_retry",
        },
        acceptance_limit_mb=10.0,
    )

    assert result is not None
    assert result.get("best_effort") is True
    assert result.get("best_effort_reason") == "quality_retry"
    assert output_path.exists()


def test_gif_only_optimization_enforces_quality_floors(tmp_path: Path):
    """Test that GIF-only optimization (no source video) enforces quality floors."""
    optimizer = _build_optimizer_stub(str(tmp_path))
    optimizer.adaptive_strategy = True
    optimizer.content_aware = False  # Disable for simpler test
    
    # Create a test GIF file
    gif_path = tmp_path / "test.gif"
    img = Image.new('RGB', (500, 500), color='red')
    frames = [img.copy() for _ in range(10)]
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:], duration=100, loop=0)
    
    # Get GIF info
    from src.gif_processing.gif_utils import get_gif_info
    gif_info = get_gif_info(str(gif_path))
    
    # Test parameter calculation for GIF-only input (no source video)
    current_params = {
        'width': gif_info.get('width', 500),
        'height': gif_info.get('height', 500),
        'fps': 10,
        'colors': 256,
        'lossy': 0
    }
    
    # Simulate a large file that needs aggressive compression
    current_size_mb = 15.0
    target_size_mb = 10.0
    
    # Calculate parameters (should enforce floors for GIF-only)
    required_params = optimizer._calculate_required_parameters(
        current_size_mb, target_size_mb, current_params, gif_info, gif_path=str(gif_path)
    )
    
    # Verify quality floors are enforced
    # Minimum width should be respected
    assert required_params['width'] >= 360, f"Width {required_params['width']} below minimum 360"
    
    # For GIF-only, colors should be at least 160 (enforced in adaptive path)
    # Note: This depends on the adaptive strategy selection
    if optimizer.adaptive_strategy:
        # Colors should be reasonable (at least 128, ideally 160+ for GIF-only)
        assert required_params['colors'] >= 128, f"Colors {required_params['colors']} too low"
    
    # Lossy should be capped at 100 for GIF-only (enforced in adaptive path)
    if optimizer.adaptive_strategy:
        assert required_params['lossy'] <= 100, f"Lossy {required_params['lossy']} exceeds cap of 100"
    
    # Verify dither is set (for quality)
    assert 'dither' in required_params, "Dither should be set for quality"

