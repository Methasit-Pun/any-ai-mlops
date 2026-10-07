"""Weekly per-scenario accuracy run: results -> MLflow -> gate -> report -> email.

Consumes eval/results.json produced by any-chat-backend's `npm run eval:weekly`
(scripts/eval-suite.ts), which drives the real booking FSM with simulated
callers and scores every trial with deterministic checks. This module adds:

  * an optional Gemini judge (quality 1-5 + hallucination, checked against the
    FAQ knowledge base shipped in results.json)
  * one MLflow parent run per ISO week with a nested child run per scenario
  * a regression gate against the last passing run of the same dataset version
  * a markdown/HTML report, emailed when the run completes

Usage:
    python -m mlops.eval_suite --results ../any-chat-backend/eval/results.json
    python -m mlops.eval_suite --results results.json --no-email --no-judge
"""

import argparse
import html
import json
import logging
import math
import os
import smtplib
import ssl
from collections import defaultdict
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Callable

import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "any-ai-weekly-accuracy"
DEFAULT_REPORT_TO = "methasitpun@gmail.com"

# Initial targets from the test plan. Treat them as a starting point and tune
# once a few weeks of real baselines exist.
TARGETS = {
    "task_success": 0.90,      # min, per scenario
    "routing_accuracy": 0.95,  # min, per scenario
    "slot_accuracy": 0.95,     # min, per scenario
    "judge_score": 4.0,        # min, overall (only when the judge ran)
    "hallucination_rate": 0.03,  # max, overall (only when the judge ran)
    "error_rate": 0.10,        # max, overall — infra errors are excluded from accuracy
}
MAX_REGRESSION = 0.05  # allowed week-over-week drop in task_success per scenario


# ── Metrics (pure) ────────────────────────────────────────────────────────────


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return float(ordered[k])


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def compute_metrics(trials: list[dict[str, Any]]) -> dict[str, float]:
    """Metrics for one group of trials. Errored trials (rate limits, network) count
    toward error_rate only — they say nothing about the agent's accuracy."""
    valid = [t for t in trials if not t.get("error")]
    metrics: dict[str, float] = {"n_trials": len(trials), "error_rate": _rate(len(trials) - len(valid), len(trials)) or 0.0}
    if not valid:
        return metrics

    metrics["task_success"] = _rate(sum(1 for t in valid if t["passed"]), len(valid)) or 0.0

    routed = [t for t in valid if t["checks"]["routing"] is not None]
    if routed:
        metrics["routing_accuracy"] = _rate(sum(1 for t in routed if t["checks"]["routing"]), len(routed)) or 0.0

    metrics["outcome_accuracy"] = _rate(sum(1 for t in valid if t["checks"]["outcome"]), len(valid)) or 0.0
    metrics["path_accuracy"] = _rate(sum(1 for t in valid if t["checks"]["path"]), len(valid)) or 0.0

    slot_total = sum(t["slot_total"] for t in valid)
    if slot_total:
        metrics["slot_accuracy"] = _rate(sum(t["slot_correct"] for t in valid), slot_total) or 0.0

    by_case: dict[str, list[bool]] = defaultdict(list)
    for t in valid:
        by_case[t["case_id"]].append(t["passed"])
    # Share of cases that passed on every trial — a flakiness signal the average hides.
    metrics["case_consistency"] = _rate(sum(1 for r in by_case.values() if all(r)), len(by_case)) or 0.0

    metrics["avg_turns"] = round(sum(t["turns"] for t in valid) / len(valid), 2)
    metrics["avg_stuck_turns"] = round(sum(t["unexpected_stuck_turns"] for t in valid) / len(valid), 2)

    latencies = [ms for t in valid for ms in t["fsm_ms"]]
    metrics["fsm_p50_ms"] = _percentile(latencies, 50)
    metrics["fsm_p95_ms"] = _percentile(latencies, 95)

    judged = [t for t in valid if t.get("judge")]
    if judged:
        metrics["judge_score"] = round(sum(t["judge"]["score"] for t in judged) / len(judged), 3)
        metrics["hallucination_rate"] = _rate(sum(1 for t in judged if t["judge"]["hallucination"]), len(judged)) or 0.0
    return metrics


def metrics_by_scenario(trials: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trials:
        groups[t["scenario"]].append(t)
    return {name: compute_metrics(group) for name, group in sorted(groups.items())}


# ── Gate (pure) ───────────────────────────────────────────────────────────────


def evaluate_gate(
    overall: dict[str, float],
    per_scenario: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]] | None,
) -> list[str]:
    """Returns the list of failure reasons; empty means the gate passed."""
    reasons: list[str] = []

    if overall.get("error_rate", 0) > TARGETS["error_rate"]:
        reasons.append(f"error_rate {overall['error_rate']:.0%} > {TARGETS['error_rate']:.0%} — run is unreliable (rate limits / outages?)")

    for scenario, m in per_scenario.items():
        for key in ("task_success", "routing_accuracy", "slot_accuracy"):
            if key in m and m[key] < TARGETS[key]:
                reasons.append(f"{scenario}: {key} {m[key]:.0%} below target {TARGETS[key]:.0%}")
        prev = (baseline or {}).get(scenario, {}).get("task_success")
        if prev is not None and "task_success" in m and prev - m["task_success"] > MAX_REGRESSION:
            reasons.append(f"{scenario}: task_success dropped {prev:.0%} -> {m['task_success']:.0%} vs last passing run")

    if "judge_score" in overall and overall["judge_score"] < TARGETS["judge_score"]:
        reasons.append(f"judge_score {overall['judge_score']:.2f} below target {TARGETS['judge_score']:.1f}")
    if "hallucination_rate" in overall and overall["hallucination_rate"] > TARGETS["hallucination_rate"]:
        reasons.append(f"hallucination_rate {overall['hallucination_rate']:.0%} above limit {TARGETS['hallucination_rate']:.0%}")
    return reasons


# ── Judge (optional) ──────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are grading a chat between a customer and a restaurant booking assistant.
Scenario under test: {scenario}

Knowledge base (the ONLY source of truth for prices, policies, hours, parking, menus):
{knowledge_base}

Score the ASSISTANT 1-5 for: staying on task for this scenario, being polite and concise, never asking for
information the customer already gave, and reaching a clear outcome (confirmed booking, answered question,
or handoff to a human).
Set "hallucination" to true if the assistant stated any price, discount, policy, menu item, availability or
capability that is NOT supported by the knowledge base or the conversation itself. Booking slot availability
shown in the chat counts as supported. Declining or handing off is never a hallucination.

Respond with ONLY JSON: {{"score": <1-5 integer>, "hallucination": <true|false>, "rationale": "<one sentence>"}}

Conversation:
{transcript}
"""


def format_transcript(turns: list[dict[str, Any]]) -> str:
    return "\n".join(f"customer: {t['customer']}\nassistant: {t['bot']}" for t in turns)


def make_gemini_judge(model: str, api_key: str) -> Callable[[str, str, str], dict[str, Any]]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def judge(scenario: str, knowledge_base: str, transcript: str) -> dict[str, Any]:
        response = client.models.generate_content(
            model=model,
            contents=JUDGE_PROMPT.format(scenario=scenario, knowledge_base=knowledge_base, transcript=transcript),
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
        )
        parsed = json.loads(response.text)
        return {
            "score": float(parsed["score"]),
            "hallucination": bool(parsed["hallucination"]),
            "rationale": parsed.get("rationale", ""),
        }

    return judge


def apply_judge(trials: list[dict[str, Any]], knowledge_base: str, judge: Callable[[str, str, str], dict[str, Any]]) -> int:
    """Attaches t['judge'] to every non-errored trial with a transcript. Returns how many were judged."""
    judged = 0
    for t in trials:
        if t.get("error") or not t["transcript"]:
            continue
        try:
            t["judge"] = judge(t["scenario"], knowledge_base, format_transcript(t["transcript"]))
            judged += 1
        except Exception as exc:  # one bad judge call must not sink the whole weekly run
            logger.warning("judge failed for %s #%s: %s", t["case_id"], t["trial"], exc)
    return judged


# ── Report (pure) ─────────────────────────────────────────────────────────────


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v:.0%}"


def _delta(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return "–"
    d = (current - previous) * 100
    return f"{d:+.0f} pts" if abs(d) >= 0.5 else "±0"


def failure_rows(trials: list[dict[str, Any]], limit: int = 25) -> list[dict[str, str]]:
    rows = []
    for t in trials:
        if t["passed"]:
            continue
        if t.get("error"):
            why = f"error: {t['error'][:120]}"
        else:
            parts = []
            if t["checks"]["routing"] is False:
                parts.append(f"routed to {t['state_path'][1] if len(t['state_path']) > 1 else '?'}")
            if not t["checks"]["outcome"]:
                parts.append(f"ended in {t['final_state']}")
            if not t["checks"]["path"]:
                parts.append("bad state path")
            bad = [f"{k}: want {v['expected']!r} got {v['actual']!r}" for k, v in t["slot_detail"].items() if not v["ok"]]
            parts.extend(bad)
            why = "; ".join(parts) or "failed"
        last_bot = t["transcript"][-1]["bot"].replace("\n", " ")[:140] if t["transcript"] else ""
        rows.append({"case": f"{t['case_id']} #{t['trial']}", "scenario": t["scenario"], "why": why, "last_bot": last_bot})
    return rows[:limit]


def build_report(
    meta: dict[str, Any],
    week: str,
    overall: dict[str, float],
    per_scenario: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]] | None,
    reasons: list[str],
    failures: list[dict[str, str]],
    run_url: str | None,
) -> tuple[str, str]:
    """Returns (markdown, html)."""
    passed = not reasons
    status = "PASSED" if passed else "REGRESSION / BELOW TARGET"
    cols = ["Scenario", "Trials", "Success", "vs last", "Routing", "Slots", "Judge", "Halluc.", "p95 ms"]
    rows = []
    for name, m in per_scenario.items():
        prev = (baseline or {}).get(name, {}).get("task_success")
        rows.append([
            name, str(int(m["n_trials"])), _pct(m.get("task_success")), _delta(m.get("task_success"), prev),
            _pct(m.get("routing_accuracy")), _pct(m.get("slot_accuracy")),
            f"{m['judge_score']:.1f}" if "judge_score" in m else "–", _pct(m.get("hallucination_rate")),
            f"{m['fsm_p95_ms']:.0f}" if "fsm_p95_ms" in m else "–",
        ])
    rows.append([
        "**OVERALL**", str(int(overall["n_trials"])), _pct(overall.get("task_success")), "–",
        _pct(overall.get("routing_accuracy")), _pct(overall.get("slot_accuracy")),
        f"{overall['judge_score']:.1f}" if "judge_score" in overall else "–", _pct(overall.get("hallucination_rate")),
        f"{overall['fsm_p95_ms']:.0f}" if "fsm_p95_ms" in overall else "–",
    ])

    summary = (
        f"Overall task success {_pct(overall.get('task_success'))} across {int(overall['n_trials'])} trials "
        f"({meta.get('case_count', '?')} cases x {meta.get('trials_per_case', '?')}); "
        f"infra error rate {_pct(overall.get('error_rate'))}; case consistency {_pct(overall.get('case_consistency'))}."
    )
    config = (
        f"agent model `{meta.get('agent_model')}`, prompt hash `{meta.get('prompt_hash')}`, "
        f"fsm hash `{meta.get('fsm_hash')}`, git `{meta.get('git_sha')}`, dataset `{meta.get('dataset_version')}`"
    )

    md = [f"# Weekly accuracy report — {week}: {status}", "", summary, "", f"Config: {config}", ""]
    if run_url:
        md += [f"MLflow run: {run_url}", ""]
    md += ["## Gate", ""]
    md += [f"- {r}" for r in reasons] if reasons else ["- All scenarios meet targets and have not regressed."]
    md += ["", "## Per scenario", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    if baseline is None:
        md += ["", "_No previous passing run for this dataset version — this run becomes the baseline._"]
    md += ["", "## Failures", ""]
    if failures:
        md += [f"- **{f['case']}** ({f['scenario']}): {f['why']}" + (f' — last bot: "{f["last_bot"]}"' if f["last_bot"] else "") for f in failures]
    else:
        md += ["- None."]
    markdown = "\n".join(md) + "\n"

    color = "#1a7f37" if passed else "#cf222e"
    th = "".join(f"<th style='text-align:left;padding:4px 8px;border-bottom:1px solid #ccc'>{html.escape(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td style='padding:4px 8px;border-bottom:1px solid #eee'>{html.escape(c.replace('**', ''))}</td>" for c in r) + "</tr>"
        for r in rows
    )
    gate_html = "".join(f"<li>{html.escape(r)}</li>" for r in reasons) or "<li>All scenarios meet targets and have not regressed.</li>"
    fail_html = "".join(
        f"<li><b>{html.escape(f['case'])}</b> ({html.escape(f['scenario'])}): {html.escape(f['why'])}</li>" for f in failures
    ) or "<li>None.</li>"
    link = f"<p><a href='{html.escape(run_url)}'>Open MLflow run</a></p>" if run_url else ""
    html_doc = (
        "<div style='font-family:Segoe UI,Arial,sans-serif;font-size:14px;color:#1f2328;max-width:820px'>"
        f"<h2 style='margin-bottom:4px'>Weekly accuracy report — {html.escape(week)}</h2>"
        f"<p style='font-weight:600;color:{color};margin-top:0'>{status}</p>"
        f"<p>{html.escape(summary)}</p><p style='color:#656d76'>{html.escape(config.replace('`', ''))}</p>{link}"
        f"<h3>Gate</h3><ul>{gate_html}</ul>"
        f"<h3>Per scenario</h3><table style='border-collapse:collapse'><tr>{th}</tr>{body}</table>"
        f"<h3>Failures</h3><ul>{fail_html}</ul></div>"
    )
    return markdown, html_doc


# ── Email ─────────────────────────────────────────────────────────────────────


def send_report_email(subject: str, markdown: str, html_body: str, attachments: dict[str, bytes] | None = None) -> None:
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    if not user or not password:
        raise RuntimeError("SMTP_USER / SMTP_PASSWORD are not set — cannot email the report (use --no-email to skip)")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("REPORT_FROM", user)
    msg["To"] = os.environ.get("REPORT_TO", DEFAULT_REPORT_TO)
    msg.set_content(markdown)
    msg.add_alternative(html_body, subtype="html")
    for name, data in (attachments or {}).items():
        msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=name)

    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=60) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)
    logger.info("report emailed to %s", msg["To"])


# ── MLflow ────────────────────────────────────────────────────────────────────


def find_baseline(dataset_version: str) -> dict[str, dict[str, float]] | None:
    """Per-scenario metrics of the most recent run that passed the gate on the same dataset."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return None
    parents = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.gate_passed = 'true' and tags.dataset_version = '{dataset_version}' and tags.run_type = 'weekly_accuracy'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if parents.empty:
        return None
    parent_id = parents.iloc[0]["run_id"]
    children = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{parent_id}'",
    )
    baseline: dict[str, dict[str, float]] = {}
    for _, row in children.iterrows():
        scenario = row.get("tags.scenario")
        if scenario:
            baseline[scenario] = {c.removeprefix("metrics."): row[c] for c in children.columns if c.startswith("metrics.") and row[c] == row[c]}
    return baseline or None


def _clean(metrics: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if v is not None}


def log_to_mlflow(
    week: str,
    meta: dict[str, Any],
    overall: dict[str, float],
    per_scenario: dict[str, dict[str, float]],
    trials: list[dict[str, Any]],
    reasons: list[str],
    markdown: str,
    results_path: str,
) -> str:
    """Logs the weekly run; returns the parent run's id."""
    with mlflow.start_run(run_name=week) as parent:
        mlflow.set_tags({
            "run_type": "weekly_accuracy",
            "gate_passed": str(not reasons).lower(),
            "dataset_version": meta.get("dataset_version", "unknown"),
            "git_sha": meta.get("git_sha", "unknown"),
            "prompt_hash": meta.get("prompt_hash", "unknown"),
            "fsm_hash": meta.get("fsm_hash", "unknown"),
        })
        mlflow.log_params({
            "agent_model": meta.get("agent_model"),
            "trials_per_case": meta.get("trials_per_case"),
            "case_count": meta.get("case_count"),
        })
        mlflow.log_metrics(_clean(overall))
        for scenario, metrics in per_scenario.items():
            with mlflow.start_run(run_name=scenario, nested=True):
                mlflow.set_tags({"scenario": scenario, "run_type": "weekly_accuracy_scenario"})
                mlflow.log_metrics(_clean(metrics))
        mlflow.log_text(markdown, "report.md")
        mlflow.log_artifact(results_path, artifact_path="raw")
        rows = [
            {
                "case_id": t["case_id"], "scenario": t["scenario"], "trial": t["trial"], "passed": t["passed"],
                "final_state": t["final_state"], "turns": t["turns"], "error": t.get("error") or "",
                "judge_score": (t.get("judge") or {}).get("score"), "hallucination": (t.get("judge") or {}).get("hallucination"),
            }
            for t in trials
        ]
        mlflow.log_table(data={k: [r[k] for r in rows] for k in rows[0]}, artifact_file="trials.json")
        return parent.info.run_id


# ── Orchestration ─────────────────────────────────────────────────────────────


def iso_week(when: datetime) -> str:
    year, week, _ = when.isocalendar()
    return f"{year}-W{week:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="eval/results.json")
    parser.add_argument("--report-out", default="reports")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--no-email", action="store_true")
    parser.add_argument("--no-gate", action="store_true", help="log and report only; never exit non-zero on a failed gate")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    meta, trials = data["meta"], data["trials"]
    if not trials:
        raise SystemExit("results file has no trials")

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not args.no_judge and api_key:
        judged = apply_judge(trials, meta.get("knowledge_base", ""), make_gemini_judge(os.environ.get("JUDGE_MODEL", "gemini-2.5-flash"), api_key))
        logger.info("judged %d/%d trials", judged, len(trials))
    else:
        logger.info("judge skipped (%s)", "--no-judge" if args.no_judge else "no GOOGLE_API_KEY")

    overall = compute_metrics(trials)
    per_scenario = metrics_by_scenario(trials)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    baseline = find_baseline(meta.get("dataset_version", "unknown"))
    reasons = evaluate_gate(overall, per_scenario, baseline)

    week = iso_week(datetime.now(timezone.utc))
    failures = failure_rows(trials)
    markdown, html_body = build_report(meta, week, overall, per_scenario, baseline, reasons, failures, run_url=None)
    run_id = log_to_mlflow(week, meta, overall, per_scenario, trials, reasons, markdown, args.results)

    tracking = os.environ.get("MLFLOW_TRACKING_URI", "")
    run_url = f"{tracking.rstrip('/')}/#/experiments/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}/runs/{run_id}" if tracking.startswith("http") else None
    if run_url:
        markdown, html_body = build_report(meta, week, overall, per_scenario, baseline, reasons, failures, run_url)

    os.makedirs(args.report_out, exist_ok=True)
    report_path = os.path.join(args.report_out, f"{week}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    logger.info("report written to %s (gate %s)", report_path, "PASSED" if not reasons else "FAILED")

    if not args.no_email:
        status = "PASSED" if not reasons else "ATTENTION"
        send_report_email(
            f"[{status}] Weekly agent accuracy {week}: {overall.get('task_success', 0):.0%} task success",
            markdown,
            html_body,
            attachments={f"{week}.md": markdown.encode("utf-8")},
        )

    if reasons and not args.no_gate:
        for r in reasons:
            logger.error("gate: %s", r)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
