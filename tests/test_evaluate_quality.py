import json
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from mlops import evaluate_quality


def _fake_openai_response(score: int, rationale: str):
    message = MagicMock()
    message.content = json.dumps({"score": score, "rationale": rationale})
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def test_score_transcript_parses_judge_response():
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_openai_response(4, "Collected all details.")

    result = evaluate_quality.score_transcript(client, "gpt-4o-mini", "caller: hi... agent: ...")

    assert result == {"score": 4.0, "rationale": "Collected all details."}
    client.chat.completions.create.assert_called_once()


def test_run_evaluation_raises_for_unknown_agent(monkeypatch):
    monkeypatch.setattr(evaluate_quality.db, "get_agent_config_by_id", lambda conn, agent_id: None)

    with pytest.raises(ValueError, match="no agent_config found"):
        evaluate_quality.run_evaluation(conn=object(), client=MagicMock(), agent_id="missing", limit=5)


def test_run_evaluation_logs_aggregate_and_table(monkeypatch):
    monkeypatch.setattr(
        evaluate_quality.db, "get_agent_config_by_id", lambda conn, agent_id: {"id": agent_id, "name": "Reception"}
    )
    monkeypatch.setattr(
        evaluate_quality.db,
        "get_recent_transcriptions",
        lambda conn, agent_id, limit: [
            {"call_id": "c1", "transcription": "..."},
            {"call_id": "c2", "transcription": "..."},
        ],
    )
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
    assert logged_tables["quality_eval.json"] == [
        {"call_id": "c1", "score": 3.0, "rationale": "ok"},
        {"call_id": "c2", "score": 3.0, "rationale": "ok"},
    ]
    assert logged_tags == {"agent_id": "agent-1", "run_type": "quality_eval"}
