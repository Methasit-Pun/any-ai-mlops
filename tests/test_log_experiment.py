from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from mlops import log_experiment
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


def test_log_agent_run_leaves_recent_calls_for_the_next_run(tmp_path, monkeypatch):
    monkeypatch.setattr(log_experiment.checkpoint.Config, "CHECKPOINT_PATH", str(tmp_path / "checkpoint.json"))
    monkeypatch.setattr(log_experiment.Config, "IN_PROGRESS_GRACE_SECONDS", 3600)
    windows = []

    def fake_logs(conn, config_id, since, until):
        windows.append((since, until))
        return [{"call_id": "c1", "duration": 10, "end_time": None, "metadata": None}]

    monkeypatch.setattr(log_experiment.db, "get_conversation_logs", fake_logs)
    monkeypatch.setattr(log_experiment.db, "get_call_logs_by_ids", lambda conn, ids: {})
    monkeypatch.setattr(log_experiment.db, "get_call_summaries_by_ids", lambda conn, ids: {})
    monkeypatch.setattr(log_experiment.db, "get_appointment_count_by_call_ids", lambda conn, ids: 0)
    for name in ("set_experiment", "log_params", "log_text", "log_metrics", "set_tags"):
        monkeypatch.setattr(log_experiment.mlflow, name, MagicMock())

    @contextmanager
    def fake_start_run(run_name=None):
        yield MagicMock()

    monkeypatch.setattr(log_experiment.mlflow, "start_run", fake_start_run)

    agent = {
        "id": "a1", "name": "Reception", "model": "m", "temperature": 0.5, "voice": "v",
        "language": "th", "max_duration": 300, "prompt": "hi", "updated_at": datetime.now(timezone.utc),
    }
    before = datetime.now(timezone.utc)
    log_experiment.log_agent_run(conn=object(), agent=agent)

    _, until = windows[0]
    assert until <= before - timedelta(seconds=3600) + timedelta(seconds=5)
    assert log_experiment.checkpoint.get_last_run("a1") == until
