"""
Unit tests for GifOptimizer-specific helper behaviors.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from src.gif_processing.gif_optimizer import GifOptimizer


class TestGifOptimizerGifsicleCommand(unittest.TestCase):
    """Validate gifsicle command construction details."""

    def setUp(self):
        fd, self.input_path = tempfile.mkstemp(suffix=".gif")
        os.close(fd)
        with open(self.input_path, "wb") as handle:
            handle.write(b"GIF89a")
        self.output_path = self.input_path + ".optimized.gif"

        # Instantiate without running GifOptimizer.__init__ (avoids heavy deps)
        self.optimizer = GifOptimizer.__new__(GifOptimizer)
        self.optimizer.fast_mode = False
        self.optimizer.gifsicle_optimize_level = 3
        self.optimizer._shutdown_checker = lambda: False
        self.optimizer.shutdown_requested = False
        self.optimizer._is_tool_available = lambda *_args, **_kwargs: True

        self.captured_cmd = None

    def tearDown(self):
        for path in (self.input_path, self.output_path):
            if path and os.path.exists(path):
                os.remove(path)

    def test_lossy_argument_uses_equals_sign(self):
        """Ensure --lossy flag is rendered as --lossy=<value>."""

        def fake_run(cmd, **_kwargs):
            self.captured_cmd = cmd
            with open(self.output_path, "wb") as handle:
                handle.write(b"\x00")

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        with patch("src.gif_processing.gif_optimizer.subprocess.run", side_effect=fake_run):
            result = self.optimizer._run_gifsicle(
                self.input_path, self.output_path, colors=128, lossy=190, scale=0.5
            )

        self.assertTrue(result, "gifsicle wrapper should report success when subprocess succeeds")
        self.assertIsNotNone(self.captured_cmd, "subprocess command should have been captured")
        self.assertIn(
            "--lossy=150",
            self.captured_cmd,
            "Lossy flag should stay within gifsicle arg and use equals-sign form",
        )


class TestGifOptimizerNearTargetBehavior(unittest.TestCase):
    """Tests around Stage 3 near-target behavior and FFmpeg fallbacks."""

    def setUp(self):
        # Create a temporary GIF file to operate on
        fd, self.gif_path = tempfile.mkstemp(suffix=".gif")
        os.close(fd)
        # Start with a file clearly over the target (1.3x)
        self.target_bytes = 1_000_000
        with open(self.gif_path, "wb") as handle:
            handle.write(b"\x00" * int(self.target_bytes * 1.3))
        self.original_bytes = os.path.getsize(self.gif_path)

        # Lightweight GifOptimizer with stubbed configuration and helpers
        self.optimizer = GifOptimizer.__new__(GifOptimizer)
        self.optimizer.temp_dir = tempfile.gettempdir()
        self.optimizer.fast_mode = False
        self.optimizer.gifsicle_optimize_level = 2
        self.optimizer.skip_gifsicle_far_over_ratio = 0.35
        self.optimizer.near_target_max_runs = 8
        self.optimizer._shutdown_checker = lambda: False
        self.optimizer.shutdown_requested = False
        self.optimizer._is_tool_available = lambda *_args, **_kwargs: True

        class DummyConfigHelper:
            def get_optimization_config(self_inner):
                return {
                    "near_target": {
                        "threshold_percent": 15,
                        "mode": "both",
                        "fine_tune_threshold_percent": 10,
                        "max_attempts": 8,
                    },
                    "allow_aggressive_compression": False,
                }

            def get_quality_floors(self_inner):
                return {}

        self.optimizer.config_helper = DummyConfigHelper()

    def tearDown(self):
        if self.gif_path and os.path.exists(self.gif_path):
            os.remove(self.gif_path)

    def test_near_target_skips_ffmpeg_and_progressive(self):
        """When gifsicle gets very close to target, skip FFmpeg fallback and progressive reduction."""

        # Simulate Stage 1/2 doing nothing
        with patch.object(self.optimizer, "_stage_lossless_optimization", return_value=False), \
            patch.object(self.optimizer, "_reencode_from_source", return_value=False), \
            patch.object(self.optimizer, "_stage_adaptive_search", return_value=False), \
            patch.object(self.optimizer, "_stage_pil_compression", return_value=False), \
            patch.object(self.optimizer, "_stage_final_polish", return_value=False), \
            patch.object(self.optimizer, "_stage_gifsicle_lossy_compression") as gifsicle_mock, \
            patch.object(self.optimizer, "_stage_ffmpeg_fallback_compression", return_value=False) as ffmpeg_mock, \
            patch.object(self.optimizer, "_progressive_resolution_reduction", return_value=False) as progressive_mock:

            # Stage 3: gifsicle reduces size to just 1% over target
            def _fake_gifsicle(path, target_bytes, target_size_mb):
                with open(path, "wb") as handle:
                    handle.write(b"\x00" * int(target_bytes * 1.01))
                return False  # Not under target yet

            gifsicle_mock.side_effect = _fake_gifsicle

            success = self.optimizer._run_optimization_stages(
                self.gif_path,
                target_bytes=self.target_bytes,
                target_size_mb=1.0,
                original_size_mb=self.original_bytes / 1024 / 1024,
                source_video=None,
            )

            # Optimization cannot claim success because we're still over target
            self.assertFalse(success)
            # FFmpeg-heavy fallbacks must be skipped for near-target cases
            ffmpeg_mock.assert_not_called()
            progressive_mock.assert_not_called()

    def test_far_over_target_allows_ffmpeg(self):
        """When still far over target after gifsicle, FFmpeg fallback is allowed and can meet target."""

        with patch.object(self.optimizer, "_stage_lossless_optimization", return_value=False), \
            patch.object(self.optimizer, "_reencode_from_source", return_value=False), \
            patch.object(self.optimizer, "_stage_adaptive_search", return_value=False), \
            patch.object(self.optimizer, "_stage_pil_compression", return_value=False), \
            patch.object(self.optimizer, "_stage_final_polish", return_value=False), \
            patch.object(self.optimizer, "_stage_gifsicle_lossy_compression") as gifsicle_mock, \
            patch.object(self.optimizer, "_stage_ffmpeg_fallback_compression") as ffmpeg_mock, \
            patch.object(self.optimizer, "_progressive_resolution_reduction") as progressive_mock:

            # Stage 3: gifsicle improves a bit but leaves file still 80% over target
            def _fake_gifsicle(path, target_bytes, target_size_mb):
                with open(path, "wb") as handle:
                    handle.write(b"\x00" * int(target_bytes * 1.8))
                return False

            gifsicle_mock.side_effect = _fake_gifsicle

            # FFmpeg fallback succeeds and brings file exactly to target size
            def _fake_ffmpeg(path, target_bytes, target_size_mb):
                with open(path, "wb") as handle:
                    handle.write(b"\x00" * target_bytes)
                return True

            ffmpeg_mock.side_effect = _fake_ffmpeg
            progressive_mock.return_value = False

            success = self.optimizer._run_optimization_stages(
                self.gif_path,
                target_bytes=self.target_bytes,
                target_size_mb=1.0,
                original_size_mb=self.original_bytes / 1024 / 1024,
                source_video=None,
            )

            self.assertTrue(success)
            ffmpeg_mock.assert_called_once()
            # Because FFmpeg already met the target, progressive reduction should not run
            progressive_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

