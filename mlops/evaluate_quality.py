"""LLM-as-judge quality evaluation for a booking voice agent's transcripts.

Samples an agent's transcripts that haven't been scored yet (tracked per agent
in the checkpoint file), scores each with a Gemini judge against a
booking-flow-specific rubric, and logs the per-example scores plus the
aggregate as an MLflow run — tracking and quality live under the same
experiment (`agent-<id>-<name>`) so they can be viewed together.

Usage:
    python -m mlops.evaluate_quality --agent-id <id> --limit 20
    python -m mlops.evaluate_quality  # no --agent-id: evaluates every active agent
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import mlflow
from google import genai
from google.genai import types

from . import checkpoint, db
from .config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RUBRIC_PROMPT = """You are grading a transcript of an AI voice agent handling a restaurant/clinic \
booking phone call. Score it 1-5 on how well the agent:
- collected the required booking details (date, time, party size / service, contact info) \
without asking the caller to repeat themselves
- gave accurate information (didn't invent availability, prices, or policies)
- ended the call in a clear outcome: a confirmed booking or an explicit handoff to a human

Respond with ONLY a JSON object: {{"score": <1-5 integer>, "rationale": "<one sentence>"}}

Transcript:
{transcript}
"""


# Tagged on every eval/calibration run so a rubric edit shows up as a new version
# in MLflow instead of silently shifting the score trend.
RUBRIC_HASH = hashlib.sha256(RUBRIC_PROMPT.encode("utf-8")).hexdigest()[:12]


def judge_tags() -> dict[str, str]:
    return {"judge_model": Config.JUDGE_MODEL, "rubric_hash": RUBRIC_HASH}


def rows_to_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """mlflow.log_table wants a dict of columns (or a DataFrame), not a list of row-dicts."""
    return {key: [row[key] for row in rows] for key in rows[0]}


def eval_checkpoint_key(agent_id: str) -> str:
    """Separate key from log_experiment's per-agent checkpoint in the same file."""
    return f"quality_eval:{agent_id}"


def score_transcript(client: genai.Client, model: str, transcript: str) -> dict[str, Any]:
    response = client.models.generate_content(
        model=model,
        contents=RUBRIC_PROMPT.format(transcript=transcript),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    parsed = json.loads(response.text)
    score = float(parsed["score"])
    if not 1 <= score <= 5:
        raise ValueError(f"judge score out of range 1-5: {score}")
    return {"score": score, "rationale": parsed.get("rationale", "")}


def score_with_retries(client: genai.Client, model: str, transcript: str, attempts: int) -> dict[str, Any] | None:
    """Returns None instead of raising when every attempt fails (bad JSON, out-of-range
    score, API error), so one bad transcript doesn't abort the whole run."""
    for attempt in range(1, attempts + 1):
        try:
            return score_transcript(client, model, transcript)
        except Exception as exc:  # noqa: BLE001 - any judge failure is retryable here
            logger.warning("judge attempt %d/%d failed: %s", attempt, attempts, exc)
    return None


def _as_utc(ts: datetime) -> datetime:
    # conversation_logs.start_time is a Prisma timestamp without time zone, stored in UTC.
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def run_evaluation(conn, client: genai.Client, agent_id: str, limit: int) -> None:
    agent = db.get_agent_config_by_id(conn, agent_id)
    if agent is None:
        raise ValueError(f"no agent_config found with id={agent_id}")

    since = checkpoint.get_last_run(eval_checkpoint_key(agent_id))
    transcripts = db.get_recent_transcriptions(conn, agent_id, limit, since)
    if not transcripts:
        logger.info("agent %s: no new transcripts since %s to evaluate", agent_id, since)
        return

    rows = []
    failed = 0
    for row in transcripts:
        result = score_with_retries(client, Config.JUDGE_MODEL, row["transcription"], Config.JUDGE_MAX_ATTEMPTS)
        if result is None:
            failed += 1
            logger.warning("agent %s: call %s could not be scored, skipping", agent_id, row["call_id"])
            continue
        rows.append({"call_id": row["call_id"], **result})

    if not rows:
        # Leave the checkpoint where it is so these transcripts are retried next run.
        raise RuntimeError(f"agent {agent_id}: judge failed on all {len(transcripts)} transcripts")

    avg_score = sum(r["score"] for r in rows) / len(rows)

    experiment_name = f"agent-{agent_id}-{agent['name']}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="quality-eval"):
        mlflow.log_metric("avg_quality_score", avg_score)
        mlflow.log_metric("eval_sample_size", len(rows))
        mlflow.log_metric("eval_failed_count", failed)
        mlflow.log_table(data=rows_to_columns(rows), artifact_file="quality_eval.json")
        mlflow.set_tags({"agent_id": agent_id, "run_type": "quality_eval", **judge_tags()})

    newest = max(_as_utc(row["start_time"]) for row in transcripts)
    checkpoint.set_last_run(eval_checkpoint_key(agent_id), newest)
    logger.info("agent %s: avg_quality_score=%.2f over %d transcripts", agent_id, avg_score, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", help="Evaluate a single agent; omit to evaluate every active agent")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    client = genai.Client(api_key=Config.GOOGLE_API_KEY)

    with db.get_connection() as conn:
        if args.agent_id:
            run_evaluation(conn, client, args.agent_id, args.limit)
            return

        agents = db.get_active_agent_configs(conn)
        logger.info("found %d active agent(s)", len(agents))
        failed_agents = []
        for agent in agents:
            try:
                run_evaluation(conn, client, agent["id"], args.limit)
            except Exception:  # noqa: BLE001 - keep evaluating the other agents
                logger.exception("agent %s: evaluation failed", agent["id"])
                failed_agents.append(agent["id"])

    if failed_agents:
        raise SystemExit(f"evaluation failed for agent(s): {', '.join(failed_agents)}")


if __name__ == "__main__":
    main()
