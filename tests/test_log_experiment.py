from datetime import datetime, timezone

from mlops.log_experiment import aggregate_metrics


def test_no_calls_returns_zero_total():
    result = aggregate_metrics([], {}, {}, appointment_count=0, cost_per_second=0.05)
    assert result == {"total_calls": 0}


def test_aggregates_success_cost_and_booking_rate():
    conversation_logs = [
        {"call_id": "c1", "duration": 60, "end_time": datetime.now(timezone.utc), "metadata": {"cost": 3.0}},
        {"call_id": "c2", "duration": 40, "end_time": None, "metadata": None},
    ]
    call_logs_by_id = {
        "c1": {"status": "completed", "cost": 3.0},
        "c2": {"status": "failed", "cost": None},
    }
    summaries_by_id = {
        "c1": {"sentiment": "positive"},
    }

    result = aggregate_metrics(
        conversation_logs, call_logs_by_id, summaries_by_id, appointment_count=1, cost_per_second=0.05
    )

    assert result["total_calls"] == 2
    assert result["success_rate"] == 0.5  # only c1 completed
    assert result["avg_duration"] == 50  # (60 + 40) / 2
    # c1 cost=3.0 from metadata, c2 falls back to duration * cost_per_second = 40 * 0.05 = 2.0
    assert result["total_cost"] == 5.0
    assert result["avg_cost_per_call"] == 2.5
    assert result["booking_conversion_rate"] == 0.5  # 1 appointment / 2 calls
    assert result["positive_sentiment_rate"] == 1.0  # only c1 has a sentiment, and it's positive


def test_falls_back_to_call_log_cost_when_metadata_missing():
    conversation_logs = [
        {"call_id": "c1", "duration": 100, "end_time": None, "metadata": {}},
    ]
    call_logs_by_id = {"c1": {"status": "answered", "cost": 7.5}}

    result = aggregate_metrics(conversation_logs, call_logs_by_id, {}, appointment_count=0, cost_per_second=0.05)

    assert result["total_cost"] == 7.5
    assert result["success_rate"] == 1.0  # status "answered" counts as completed
