"""LLM-as-judge quality evaluation for a booking voice agent's transcripts.

Samples recent CallLog.transcription rows for one agent, scores each with an
OpenAI judge against a booking-flow-specific rubric, and logs the per-example
scores plus the aggregate as an MLflow run — tracking and quality live under
the same experiment (`agent-<id>-<name>`) so they can be viewed together.

Usage:
    python -m mlops.evaluate_quality --agent-id <id> --limit 20
"""

import argparse
import json
import logging
from typing import Any

import mlflow
from openai import OpenAI

from . import db
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


def score_transcript(client: OpenAI, model: str, transcript: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": RUBRIC_PROMPT.format(transcript=transcript)}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    parsed = json.loads(content)
    return {"score": float(parsed["score"]), "rationale": parsed.get("rationale", "")}


def run_evaluation(conn, client: OpenAI, agent_id: str, limit: int) -> None:
    agent = db.get_agent_config_by_id(conn, agent_id)
    if agent is None:
        raise ValueError(f"no agent_config found with id={agent_id}")

    transcripts = db.get_recent_transcriptions(conn, agent_id, limit)
    if not transcripts:
        logger.info("agent %s: no transcripts available to evaluate", agent_id)
        return

    rows = []
    for row in transcripts:
        result = score_transcript(client, Config.JUDGE_MODEL, row["transcription"])
        rows.append({"call_id": row["call_id"], **result})

    avg_score = sum(r["score"] for r in rows) / len(rows)

    experiment_name = f"agent-{agent_id}-{agent['name']}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="quality-eval"):
        mlflow.log_metric("avg_quality_score", avg_score)
        mlflow.log_metric("eval_sample_size", len(rows))
        mlflow.log_table(data=rows, artifact_file="quality_eval.json")
        mlflow.set_tags({"agent_id": agent_id, "run_type": "quality_eval"})

    logger.info("agent %s: avg_quality_score=%.2f over %d transcripts", agent_id, avg_score, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    with db.get_connection() as conn:
        run_evaluation(conn, client, args.agent_id, args.limit)


if __name__ == "__main__":
    main()
