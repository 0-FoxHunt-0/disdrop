import threading
from typing import Dict


class QualityManager:
    """Manages quality settings adaptation based on results."""

    def __init__(self):
        self.quality_history = {}
        self._lock = threading.Lock()

    def get_settings(self, file_size: float, target_size: float) -> Dict:
        """Get optimized quality settings based on file size."""
        with self._lock:
            ratio = target_size / file_size
            return {
                'scale_factor': min(1.0, max(0.3, ratio ** 0.5)),
                'colors': 256 if ratio > 0.5 else 192,
                'lossy_value': min(100, int(50 * (1/ratio)))
            }

    def update_settings(self, settings: Dict, result_size: float, target_size: float) -> Dict:
        """Update settings based on optimization results."""
        with self._lock:
            new_settings = settings.copy()
            ratio = result_size / target_size

            # More conservative adjustments
            if ratio > 1.5:
                # Don't adjust all parameters at once
                if new_settings['scale_factor'] > 0.3:
                    new_settings['scale_factor'] *= 0.9
                elif new_settings['colors'] > 128:
                    new_settings['colors'] = max(
                        128, new_settings['colors'] - 32)
                else:
                    new_settings['lossy_value'] = min(
                        90, new_settings['lossy_value'] + 10)
            elif ratio > 1.2:
                # Even more gentle adjustments
                if new_settings['scale_factor'] > 0.4:
                    new_settings['scale_factor'] *= 0.95
                else:
                    new_settings['lossy_value'] = min(
                        80, new_settings['lossy_value'] + 5)

            # Add minimum thresholds
            new_settings['scale_factor'] = max(
                0.25, new_settings['scale_factor'])
            new_settings['colors'] = max(64, new_settings['colors'])
            new_settings['lossy_value'] = min(90, new_settings['lossy_value'])

            return new_settings
