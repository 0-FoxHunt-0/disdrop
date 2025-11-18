from src.error_handler import ErrorHandler


def test_error_handler_top_failures_and_retryable_summary():
    handler = ErrorHandler()

    handler.handle_error(ValueError("Bitrate floor constraint"), "file_a", continue_processing=False)
    handler.handle_error(PermissionError("Permission denied"), "file_b", continue_processing=False)
    handler.handle_error(PermissionError("Permission denied again"), "file_c", continue_processing=False)

    summary = handler.get_error_summary()
    assert summary['retryable_errors'] == 1
    assert summary['non_retryable_errors'] == 2

    top_failures = handler.get_top_failures(limit=2)
    assert top_failures
    categories = [entry['category'] for entry in top_failures]
    assert 'permission' in categories
    assert 'bitrate_validation' in categories

    sample_messages = [entry['sample_message'] for entry in top_failures]
    assert any("Permission denied again" in msg for msg in sample_messages)

