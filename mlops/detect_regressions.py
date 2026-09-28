"""Compares each active agent's latest MLflow runs against its recent history and
flags regressions, so a bad prompt/config change or a judge-score drop gets
noticed instead of sitting unread in MLflow.

For every active agent and each run type (`tracking` from log_experiment.py,
`quality_eval` from evaluate_quality.py), the latest run's watched metrics are
compared with the mean of the previous `--baseline-runs` runs. A metric that
moved the wrong way by more than `--max-relative-change` is a regression.
Quality runs are only compared with runs scored under the same rubric_hash.

It also reports agent config changes (prompt/model/voice/...) between the last
two tracking runs, with the before/after metrics side by side.

Exits non-zero when any regression is found. If ALERT_WEBHOOK_URL is set, the
report is also POSTed there as Slack-compatible JSON ({"text": ...}).

Usage:
    python -m mlops.detect_regressions
    python -m mlops.detect_regressions --baseline-runs 14 --max-relative-change 0.2
"""

import argparse
import json
import logging
import os
import time
import urllib.request
from typing import Any

import mlflow

from . import db
from .config import Config
from .log_experiment import experiment_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HIGHER_IS_BETTER = "higher"
LOWER_IS_BETTER = "lower"

WATCHED_METRICS: dict[str, dict[str, str]] = {
    "tracking": {
        "success_rate": HIGHER_IS_BETTER,
        "booking_conversion_rate": HIGHER_IS_BETTER,
        "positive_sentiment_rate": HIGHER_IS_BETTER,
        "avg_cost_per_call": LOWER_IS_BETTER,
    },
    "quality_eval": {
        "avg_quality_score": HIGHER_IS_BETTER,
    },
}

# A run built from a handful of calls is too noisy to call a regression on.
MIN_SAMPLE: dict[str, tuple[str, int]] = {
    "tracking": ("total_calls", 20),
    "quality_eval": ("eval_sample_size", 5),
}

CONFIG_PARAMS = ("prompt_hash", "model", "temperature", "voice", "language", "max_duration")

MIN_BASELINE_RUNS = 3


def find_regressions(
    latest: dict[str, float],
    baseline: list[dict[str, float]],
    watched: dict[str, str],
    max_relative_change: float,
) -> list[dict[str, Any]]:
    """Pure comparison of one run's metrics against earlier runs' metrics."""
    regressions = []
    for metric, direction in watched.items():
        if metric not in latest:
            continue
        history = [run[metric] for run in baseline if metric in run]
        if len(history) < MIN_BASELINE_RUNS:
            continue
        base = sum(history) / len(history)
        if base == 0:
            continue
        change = (latest[metric] - base) / abs(base)
        worse = change < -max_relative_change if direction == HIGHER_IS_BETTER else change > max_relative_change
        if worse:
            regressions.append(
                {"metric": metric, "latest": latest[metric], "baseline": round(base, 4), "change": round(change, 4)}
            )
    return regressions


def describe_config_change(latest_params: dict[str, str], previous_params: dict[str, str]) -> list[str]:
    return [
        f"{param}: {previous_params.get(param)} -> {latest_params.get(param)}"
        for param in CONFIG_PARAMS
        if latest_params.get(param) != previous_params.get(param)
    ]


def fetch_runs(experiment: str, run_type: str, max_results: int) -> list[dict[str, Any]]:
    """Newest first. Returns [] when the experiment doesn't exist yet."""
    if mlflow.get_experiment_by_name(experiment) is None:
        return []
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"tags.run_type = '{run_type}'",
        order_by=["attributes.start_time DESC"],
        max_results=max_results,
        output_format="list",
    )
    return [
        {
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
            "start_time_ms": run.info.start_time,
        }
        for run in runs
    ]


def check_agent(agent: dict[str, Any], baseline_runs: int, max_relative_change: float, max_age_hours: float) -> dict[str, list]:
    experiment = experiment_name(agent["id"], agent["name"])
    report: dict[str, list] = {"regressions": [], "config_changes": []}
    now_ms = time.time() * 1000

    for run_type, watched in WATCHED_METRICS.items():
        runs = fetch_runs(experiment, run_type, baseline_runs + 1)
        if not runs:
            continue
        latest, history = runs[0], runs[1:]
        if now_ms - latest["start_time_ms"] > max_age_hours * 3600 * 1000:
            # No fresh run (e.g. no new calls today) — already reported when it was new.
            continue

        sample_metric, min_sample = MIN_SAMPLE[run_type]
        if latest["metrics"].get(sample_metric, 0) < min_sample:
            logger.info("%s/%s: %s below %d, not checking", experiment, run_type, sample_metric, min_sample)
        else:
            if run_type == "quality_eval":
                rubric = latest["tags"].get("rubric_hash")
                history = [run for run in history if run["tags"].get("rubric_hash") == rubric]
            for regression in find_regressions(
                latest["metrics"], [run["metrics"] for run in history], watched, max_relative_change
            ):
                report["regressions"].append({"experiment": experiment, "run_type": run_type, **regression})

        if run_type == "tracking" and history:
            changes = describe_config_change(latest["params"], history[0]["params"])
            if changes:
                report["config_changes"].append(
                    {
                        "experiment": experiment,
                        "changes": changes,
                        "before": {m: history[0]["metrics"].get(m) for m in watched},
                        "after": {m: latest["metrics"].get(m) for m in watched},
                    }
                )
    return report


def format_report(regressions: list[dict[str, Any]], config_changes: list[dict[str, Any]]) -> str:
    lines = []
    if regressions:
        lines.append(f"Regressions ({len(regressions)}):")
        for r in regressions:
            lines.append(
                f"- {r['experiment']} [{r['run_type']}] {r['metric']}: {r['latest']} vs baseline "
                f"{r['baseline']} ({r['change']:+.0%})"
            )
    if config_changes:
        lines.append(f"Config changes ({len(config_changes)}):")
        for c in config_changes:
            lines.append(f"- {c['experiment']}: {', '.join(c['changes'])}")
            lines.append(f"  before: {c['before']}")
            lines.append(f"  after:  {c['after']}")
    return "\n".join(lines) if lines else "No regressions or config changes."


def _send_alert(url: str, text: str) -> None:
    request = urllib.request.Request(
        url, data=json.dumps({"text": text}).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-runs", type=int, default=7)
    parser.add_argument("--max-relative-change", type=float, default=0.15)
    parser.add_argument("--max-age-hours", type=float, default=36)
    args = parser.parse_args()

    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    with db.get_connection() as conn:
        agents = db.get_active_agent_configs(conn)

    regressions, config_changes = [], []
    for agent in agents:
        report = check_agent(agent, args.baseline_runs, args.max_relative_change, args.max_age_hours)
        regressions += report["regressions"]
        config_changes += report["config_changes"]

    text = format_report(regressions, config_changes)
    logger.info("\n%s", text)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"## MLOps regression check\n\n```\n{text}\n```\n")

    if Config.ALERT_WEBHOOK_URL and (regressions or config_changes):
        _send_alert(Config.ALERT_WEBHOOK_URL, text)

    if regressions:
        raise SystemExit(f"{len(regressions)} regression(s) found")


if __name__ == "__main__":
    main()
