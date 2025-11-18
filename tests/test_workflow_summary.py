import contextlib
import io
import time
import types

from src.automated_workflow import AutomatedWorkflow, AnalysisTracker
from src.error_handler import ErrorHandler


def _build_stub_workflow():
    workflow = AutomatedWorkflow.__new__(AutomatedWorkflow)
    workflow.error_handler = ErrorHandler()
    workflow.analysis_tracker = AnalysisTracker()
    workflow.use_cache = False
    workflow._print_cache_stats = types.MethodType(lambda self, tracker_counts=None: None, workflow)
    return workflow


def test_workflow_summary_reports_errors_and_diagnostics():
    workflow = _build_stub_workflow()

    workflow.error_handler.handle_error(ValueError("Bitrate floor too low"), "input_a", continue_processing=False)
    workflow.error_handler.handle_error(PermissionError("Permission denied writing output"), "input_b", continue_processing=False)

    workflow.analysis_tracker.record_retry('mp4')
    workflow.analysis_tracker.record_validation_failure('mp4', amount=2)
    workflow.analysis_tracker.record_timeout('ffmpeg')

    stats = {'processed': 4, 'successful': 2, 'errors': 2, 'skipped': 0}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        workflow._print_workflow_summary(stats, start_time=time.time() - 1)

    output = buffer.getvalue()
    assert "Retryable vs non-retryable" in output
    assert "Top failure signals" in output
    assert "📈 Diagnostics" in output


def test_workflow_summary_handles_zero_processed_without_errors():
    workflow = _build_stub_workflow()
    stats = {'processed': 0, 'successful': 0, 'errors': 0, 'skipped': 0}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        workflow._print_workflow_summary(stats, start_time=time.time())
    output = buffer.getvalue()
    assert "Files processed: 0" in output
    assert "Error Analysis" not in output

