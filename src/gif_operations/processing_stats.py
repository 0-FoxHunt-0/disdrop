from pathlib import Path
import threading


class ProcessingStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.total_processed = 0
            self.successful = 0
            self.failed = 0
            self.total_size_saved = 0
            self.results = []

    def add_result(self, file_path: Path, initial_size: float, final_size: float, success: bool):
        with self._lock:
            self.total_processed += 1
            if success:
                self.successful += 1
                self.total_size_saved += (initial_size - final_size)
            else:
                self.failed += 1
            self.results.append({
                'file': file_path.name,
                'initial_size': initial_size,
                'final_size': final_size,
                'reduction': ((initial_size - final_size) / initial_size) * 100 if initial_size > 0 else 0,
                'success': success
            })

    def get_summary(self) -> str:
        with self._lock:
            if not self.total_processed:
                return "No files processed"

            summary = [
                f"\nProcessing Summary:",
                f"Total files processed: {self.total_processed}",
                f"Successfully optimized: {self.successful}",
                f"Failed to optimize: {self.failed}",
                f"Total size reduction: {self.total_size_saved:.2f}MB",
                f"Success rate: {(self.successful / self.total_processed) * 100:.1f}%"
            ]

            if self.failed > 0:
                failed_files = [r['file']
                                for r in self.results if not r['success']]
                summary.append("\nFailed files:")
                summary.extend([f"  - {f}" for f in failed_files])

            return "\n".join(summary)
