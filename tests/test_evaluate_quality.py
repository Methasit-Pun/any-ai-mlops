import json
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from mlops import evaluate_quality


def _fake_gemini_response(score: int, rationale: str):
    response = MagicMock()
    response.text = json.dumps({"score": score, "rationale": rationale})
    return response


def test_score_transcript_parses_judge_response():
    client = MagicMock()
    client.models.generate_content.return_value = _fake_gemini_response(4, "Collected all details.")

    result = evaluate_quality.score_transcript(client, "gemini-2.5-flash", "caller: hi... agent: ...")

    assert result == {"score": 4.0, "rationale": "Collected all details."}
    client.models.generate_content.assert_called_once()


@pytest.fixture(autouse=True)
def _isolated_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_quality.checkpoint.Config, "CHECKPOINT_PATH", str(tmp_path / "checkpoint.json"))


def test_score_transcript_rejects_out_of_range_score():
    client = MagicMock()
    client.models.generate_content.return_value = _fake_gemini_response(7, "??")

    with pytest.raises(ValueError, match="out of range"):
        evaluate_quality.score_transcript(client, "gemini-2.5-flash", "...")


def test_score_with_retries_recovers_from_a_bad_response():
    bad = MagicMock()
    bad.text = "not json"
    client = MagicMock()
    client.models.generate_content.side_effect = [bad, _fake_gemini_response(5, "great")]

    result = evaluate_quality.score_with_retries(client, "gemini-2.5-flash", "...", attempts=3)

    assert result == {"score": 5.0, "rationale": "great"}
    assert client.models.generate_content.call_count == 2


def test_score_with_retries_returns_none_when_every_attempt_fails():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("api down")

    assert evaluate_quality.score_with_retries(client, "gemini-2.5-flash", "...", attempts=2) is None
    assert client.models.generate_content.call_count == 2


def test_run_evaluation_raises_for_unknown_agent(monkeypatch):
    monkeypatch.setattr(evaluate_quality.db, "get_agent_config_by_id", lambda conn, agent_id: None)

    with pytest.raises(ValueError, match="no agent_config found"):
        evaluate_quality.run_evaluation(conn=object(), client=MagicMock(), agent_id="missing", limit=5)


def test_run_evaluation_logs_aggregate_and_table(monkeypatch):
    monkeypatch.setattr(
        evaluate_quality.db, "get_agent_config_by_id", lambda conn, agent_id: {"id": agent_id, "name": "Reception"}
    )
    newest = datetime(2026, 9, 1, 12, 0)
    seen_since = []

    def fake_transcriptions(conn, agent_id, limit, since):
        seen_since.append(since)
        return [
            {"call_id": "c1", "start_time": newest, "transcription": "..."},
            {"call_id": "c2", "start_time": datetime(2026, 9, 1, 11, 0), "transcription": "..."},
        ]

    monkeypatch.setattr(evaluate_quality.db, "get_recent_transcriptions", fake_transcriptions)
    monkeypatch.setattr(
        evaluate_quality,
        "score_transcript",
        lambda client, model, transcript: {"score": 3.0, "rationale": "ok"},
    )

    logged_metrics = {}
    logged_tables = {}
    logged_tags = {}

    monkeypatch.setattr(evaluate_quality.mlflow, "set_experiment", lambda name: logged_metrics.setdefault("_experiment", name))
    monkeypatch.setattr(evaluate_quality.mlflow, "log_metric", lambda k, v: logged_metrics.__setitem__(k, v))
    monkeypatch.setattr(evaluate_quality.mlflow, "log_table", lambda data, artifact_file: logged_tables.__setitem__(artifact_file, data))
    monkeypatch.setattr(evaluate_quality.mlflow, "set_tags", lambda tags: logged_tags.update(tags))

    @contextmanager
    def fake_start_run(run_name=None):
        yield MagicMock()

    monkeypatch.setattr(evaluate_quality.mlflow, "start_run", fake_start_run)

    evaluate_quality.run_evaluation(conn=object(), client=MagicMock(), agent_id="agent-1", limit=5)

    assert logged_metrics["avg_quality_score"] == 3.0
    assert logged_metrics["eval_sample_size"] == 2
    assert logged_metrics["eval_failed_count"] == 0
    assert logged_tables["quality_eval.json"] == {
        "call_id": ["c1", "c2"],
        "score": [3.0, 3.0],
        "rationale": ["ok", "ok"],
    }
    assert logged_tags == {
        "agent_id": "agent-1",
        "run_type": "quality_eval",
        "judge_model": evaluate_quality.Config.JUDGE_MODEL,
        "rubric_hash": evaluate_quality.RUBRIC_HASH,
    }
    # Checkpoint advances to the newest scored call, so the next run only sees newer ones.
    assert evaluate_quality.checkpoint.get_last_run("quality_eval:agent-1") == newest.replace(tzinfo=timezone.utc)
    evaluate_quality.run_evaluation(conn=object(), client=MagicMock(), agent_id="agent-1", limit=5)
    assert seen_since[-1] == newest.replace(tzinfo=timezone.utc)


def test_run_evaluation_keeps_checkpoint_when_judge_fails_on_everything(monkeypatch):
    monkeypatch.setattr(
        evaluate_quality.db, "get_agent_config_by_id", lambda conn, agent_id: {"id": agent_id, "name": "Reception"}
    )
    monkeypatch.setattr(
        evaluate_quality.db,
        "get_recent_transcriptions",
        lambda conn, agent_id, limit, since: [
            {"call_id": "c1", "start_time": datetime(2026, 9, 1), "transcription": "..."}
        ],
    )
    monkeypatch.setattr(evaluate_quality, "score_with_retries", lambda client, model, transcript, attempts: None)

    with pytest.raises(RuntimeError, match="judge failed on all"):
        evaluate_quality.run_evaluation(conn=object(), client=MagicMock(), agent_id="agent-1", limit=5)

    assert evaluate_quality.checkpoint.get_last_run("quality_eval:agent-1").year == 1970
