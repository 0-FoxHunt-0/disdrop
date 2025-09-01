"""Disdrop package root.

Export primary classes and CLI for convenience when installed via pip.
"""

from .cli import main as cli_main  # noqa: F401
from .video_compressor import DynamicVideoCompressor  # noqa: F401
from .gif_generator import GifGenerator  # noqa: F401
from .automated_workflow import AutomatedWorkflow  # noqa: F401