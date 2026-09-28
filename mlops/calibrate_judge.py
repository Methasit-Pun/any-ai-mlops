"""Calibrates the Gemini judge (see evaluate_quality.score_transcript) against
a human-labeled baseline.

The baseline is data/human_labels.csv: `call_id,human_score` rows (score 1-5,
using the same rubric as evaluate_quality.RUBRIC_PROMPT). This re-scores those same transcripts with the judge and
reports how well it agrees with the human scores (MAE, correlation, % within
1 point), logged to MLflow under the `judge-calibration` experiment.

The run fails (non-zero exit) when the judge misses the pass bar, so a
scheduled job flags a judge that can't be trusted. Defaults: MAE <= 0.75,
within-1-point rate >= 0.8, at least 10 matched labels.

Usage:
    python -m mlops.calibrate_judge --labels data/human_labels.csv
    python -m mlops.calibrate_judge --max-mae 0.5 --min-within-1 0.9
"""

import argparse
import csv
import logging
from typing import Any

import mlflow
from google import genai

from . import db
from .config import Config
from .evaluate_quality import judge_tags, rows_to_columns, score_with_retries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_human_labels(path: str) -> dict[str, float]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["call_id"]: float(row["human_score"]) for row in csv.DictReader(f) if row.get("call_id")}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def check_pass_bar(metrics: dict[str, Any], max_mae: float, min_within_1: float, min_sample_size: int) -> list[str]:
    """Returns the reasons the judge fails the bar; empty means it passes."""
    failures = []
    if metrics["sample_size"] < min_sample_size:
        failures.append(f"sample_size {metrics['sample_size']} < {min_sample_size}")
    if metrics["mae"] > max_mae:
        failures.append(f"mae {metrics['mae']} > {max_mae}")
    if metrics["within_1_point_rate"] < min_within_1:
        failures.append(f"within_1_point_rate {metrics['within_1_point_rate']} < {min_within_1}")
    return failures


def label_coverage_warning(human_labels: dict[str, float]) -> str | None:
    """Agreement numbers mean little if the labels never include a good (4-5) or a
    bad (1-2) call — the judge could miss that whole end of the scale unnoticed."""
    scores = human_labels.values()
    missing = []
    if not any(s >= 4 for s in scores):
        missing.append("good (4-5)")
    if not any(s <= 2 for s in scores):
        missing.append("bad (1-2)")
    if missing:
        return f"labels contain no {' or '.join(missing)} calls; add some for a meaningful calibration"
    return None


def run_calibration(conn, client: genai.Client, model: str, human_labels: dict[str, float]) -> dict[str, Any]:
    """Pure-ish aggregation over already-fetched labels — only score_transcript does I/O."""
    call_logs_by_id = db.get_call_logs_by_ids(conn, list(human_labels))

    rows = []
    for call_id, human_score in human_labels.items():
        call_log = call_logs_by_id.get(call_id)
        if not call_log or not call_log.get("transcription"):
            logger.warning("call %s: no transcription found, skipping", call_id)
            continue
        judge = score_with_retries(client, model, call_log["transcription"], Config.JUDGE_MAX_ATTEMPTS)
        if judge is None:
            logger.warning("call %s: judge could not score it, skipping", call_id)
            continue
        rows.append(
            {
                "call_id": call_id,
                "human_score": human_score,
                "judge_score": judge["score"],
                "abs_diff": abs(judge["score"] - human_score),
                "rationale": judge["rationale"],
            }
        )

    if not rows:
        raise ValueError("no labeled call_ids matched a transcription in call_logs")

    n = len(rows)
    metrics = {
        "mae": round(sum(r["abs_diff"] for r in rows) / n, 4),
        "within_1_point_rate": round(sum(1 for r in rows if r["abs_diff"] <= 1) / n, 4),
        "exact_match_rate": round(sum(1 for r in rows if r["abs_diff"] == 0) / n, 4),
        "sample_size": n,
    }
    correlation = _pearson([r["human_score"] for r in rows], [r["judge_score"] for r in rows])
    if correlation is not None:
        metrics["correlation"] = round(correlation, 4)

    return {"rows": rows, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/human_labels.csv")
    parser.add_argument("--max-mae", type=float, default=0.75)
    parser.add_argument("--min-within-1", type=float, default=0.8)
    parser.add_argument("--min-sample-size", type=int, default=10)
    args = parser.parse_args()

    human_labels = load_human_labels(args.labels)
    if not human_labels:
        raise SystemExit(f"no labeled rows in {args.labels} -- add call_id,human_score rows first")
    coverage_warning = label_coverage_warning(human_labels)
    if coverage_warning:
        logger.warning(coverage_warning)

    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    client = genai.Client(api_key=Config.GOOGLE_API_KEY)

    with db.get_connection() as conn:
        result = run_calibration(conn, client, Config.JUDGE_MODEL, human_labels)

    failures = check_pass_bar(result["metrics"], args.max_mae, args.min_within_1, args.min_sample_size)

    mlflow.set_experiment("judge-calibration")
    with mlflow.start_run(run_name=Config.JUDGE_MODEL):
        mlflow.log_metrics({**result["metrics"], "passed": 0 if failures else 1})
        mlflow.log_table(data=rows_to_columns(result["rows"]), artifact_file="calibration.json")
        mlflow.set_tags({**judge_tags(), "labels_file": args.labels})

    logger.info("calibration: %s", result["metrics"])
    if failures:
        raise SystemExit("judge failed calibration: " + "; ".join(failures))
    logger.info("judge passed calibration")


if __name__ == "__main__":
    main()
