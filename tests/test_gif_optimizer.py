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


if __name__ == "__main__":
    unittest.main()

