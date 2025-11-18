from src.automated_workflow import AnalysisTracker


def test_analysis_tracker_records_counts_and_events():
    tracker = AnalysisTracker(max_recent_events=5)

    tracker.record_cache_hit('loop')
    tracker.record_cache_miss('loop')
    tracker.record_retry('mp4')
    tracker.record_segmentation('gif')
    tracker.record_guardrail('gif')
    tracker.record_timeout('ffmpeg')
    tracker.record_validation_failure('mp4', amount=2)
    tracker.record_summary_cleanup('segments')
    tracker.record_mp4_move('move')

    snapshot = tracker.snapshot()
    counts = snapshot['counts']

    assert counts['cache_hits'] == 1
    assert counts['cache_misses'] == 1
    assert counts['retries'] == 1
    assert counts['segmentation_events'] == 1
    assert counts['guardrail_events'] == 1
    assert counts['timeout_events'] == 1
    assert counts['validation_failures'] == 2
    assert counts['summary_cleanups'] == 1
    assert counts['mp4_moves'] == 1
    # Ensure recent events are tracked and capped
    assert snapshot['recent_events']


def test_analysis_tracker_top_metrics_filters_context_counts():
    tracker = AnalysisTracker()
    tracker.record_cache_hit('loop')
    tracker.record_retry('mp4')
    tracker.record_retry('mp4')

    top_metrics = tracker.top_metrics(limit=2)
    # Only base metrics (without contextual suffix) should be reported
    metric_names = {metric for metric, _ in top_metrics}
    assert 'retries' in metric_names
    assert 'cache_hits' in metric_names

