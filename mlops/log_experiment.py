"""Aggregates recorded call outcomes per AgentConfig into MLflow runs.

This is the "experiment tracking + cost/latency monitoring" half of the
plan: every time this runs, each active agent gets one MLflow run capturing
its current config (prompt/model/temperature) as params and its call
outcomes since the last run as metrics, so config versions can be compared
over time in the MLflow UI.

Usage:
    python -m mlops.log_experiment
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

import mlflow

from . import checkpoint, db
from .config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def aggregate_metrics(
    conversation_logs: list[dict[str, Any]],
    call_logs_by_id: dict[str, dict[str, Any]],
    summaries_by_id: dict[str, dict[str, Any]],
    appointment_count: int,
    cost_per_second: float,
) -> dict[str, float]:
    """Pure aggregation over already-fetched rows — no I/O, easy to unit test."""
    total_calls = len(conversation_logs)
    if total_calls == 0:
        return {"total_calls": 0}

    completed = 0
    total_duration = 0
    total_cost = 0.0
    positive_sentiment = 0
    sentiment_count = 0

    for log in conversation_logs:
        call_id = log["call_id"]
        duration = log.get("duration") or 0
        total_duration += duration

        meta = log.get("metadata") or {}
        cost = meta.get("cost") if isinstance(meta, dict) else None
        call_log = call_logs_by_id.get(call_id)
        if cost is None:
            cost = (call_log or {}).get("cost")
        if cost is None:
            cost = duration * cost_per_second
        total_cost += cost

        status = (call_log or {}).get("status")
        is_completed = bool(log.get("end_time")) or status in ("completed", "answered")
        if is_completed:
            completed += 1

        summary = summaries_by_id.get(call_id)
        if summary and summary.get("sentiment"):
            sentiment_count += 1
            if summary["sentiment"] == "positive":
                positive_sentiment += 1

    metrics: dict[str, float] = {
        "total_calls": total_calls,
        "success_rate": completed / total_calls,
        "avg_duration": total_duration / total_calls,
        "total_cost": round(total_cost, 4),
        "avg_cost_per_call": round(total_cost / total_calls, 4),
        "booking_conversion_rate": appointment_count / total_calls,
    }
    if sentiment_count > 0:
        metrics["positive_sentiment_rate"] = positive_sentiment / sentiment_count

    return metrics


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def log_agent_run(conn, agent: dict[str, Any]) -> None:
    agent_id = agent["id"]
    since = checkpoint.get_last_run(agent_id)
    now = datetime.now(timezone.utc)

    conversation_logs = db.get_conversation_logs(conn, agent_id, since)
    if not conversation_logs:
        logger.info("agent %s: no new calls since %s, skipping", agent_id, since)
        return

    call_ids = [log["call_id"] for log in conversation_logs]
    call_logs_by_id = db.get_call_logs_by_ids(conn, call_ids)
    summaries_by_id = db.get_call_summaries_by_ids(conn, call_ids)
    appointment_count = db.get_appointment_count(conn, agent_id, since)

    metrics = aggregate_metrics(
        conversation_logs,
        call_logs_by_id,
        summaries_by_id,
        appointment_count,
        Config.COST_PER_SECOND,
    )

    experiment_name = f"agent-{agent_id}-{agent['name']}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=now.isoformat()):
        mlflow.log_params(
            {
                "model": agent["model"],
                "temperature": agent["temperature"],
                "voice": agent["voice"],
                "language": agent["language"],
                "max_duration": agent["max_duration"],
                "prompt_hash": _prompt_hash(agent["prompt"] or ""),
            }
        )
        mlflow.log_text(agent["prompt"] or "", "prompt.txt")
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "agent_id": agent_id,
                "window_start": since.isoformat(),
                "window_end": now.isoformat(),
                "agent_config_updated_at": agent["updated_at"].isoformat(),
            }
        )

    logger.info("agent %s: logged run with metrics %s", agent_id, metrics)
    checkpoint.set_last_run(agent_id, now)


def main() -> None:
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    with db.get_connection() as conn:
        agents = db.get_active_agent_configs(conn)
        logger.info("found %d active agent(s)", len(agents))
        for agent in agents:
            log_agent_run(conn, agent)


if __name__ == "__main__":
    main()
