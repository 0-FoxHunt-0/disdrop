import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.gif_operations import GIFOptimizer  # Use the canonical optimizer

logger = logging.getLogger(__name__)


def process_gifs(compression_settings: Optional[Dict[str, Any]] = None) -> List[Path]:
    """Main entry point for GIF optimization."""
    optimizer = GIFOptimizer(compression_settings)
    return asyncio.run(optimizer.process_all())
