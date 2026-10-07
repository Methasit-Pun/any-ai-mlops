import pytest

from mlops import eval_suite
from mlops.eval_suite import (
    build_report,
    compute_metrics,
    evaluate_gate,
    failure_rows,
    metrics_by_scenario,
    send_report_email,
)


def make_trial(case_id="c1", scenario="booking_happy", passed=True, error=None, routing=True, slot_total=2, slot_correct=2, **extra):
    trial = {
        "case_id": case_id,
        "scenario": scenario,
        "trial": 1,
        "error": error,
        "passed": passed,
        "checks": {"routing": routing, "outcome": passed, "path": True},
        "slot_total": slot_total,
        "slot_correct": slot_correct,
        "slot_detail": {},
        "final_state": "CONFIRMED" if passed else "ASK_DATE",
        "state_path": ["IDLE", "ASK_DATE"],
        "turns": 6,
        "unexpected_stuck_turns": 0,
        "fsm_ms": [100, 200, 300],
        "transcript": [{"customer": "hi", "bot": "hello", "state": "ASK_DATE", "fsm_ms": 100}],
    }
    trial.update(extra)
    return trial


def test_errored_trials_count_only_toward_error_rate():
    trials = [make_trial(), make_trial(passed=False), make_trial(passed=False, error="429")]
    m = compute_metrics(trials)
    assert m["n_trials"] == 3
    assert m["error_rate"] == pytest.approx(0.3333, abs=1e-3)
    assert m["task_success"] == 0.5  # 1 of the 2 non-errored trials


def test_all_errored_returns_only_error_rate():
    m = compute_metrics([make_trial(passed=False, error="boom")])
    assert m == {"n_trials": 1, "error_rate": 1.0}


def test_slot_accuracy_is_field_level_and_skips_cases_without_slots():
    trials = [make_trial(slot_total=4, slot_correct=3), make_trial(case_id="faq", slot_total=0, slot_correct=0, routing=None)]
    m = compute_metrics(trials)
    assert m["slot_accuracy"] == 0.75
    assert m["routing_accuracy"] == 1.0  # the routing=None trial is excluded


def test_case_consistency_penalises_flaky_cases():
    trials = [make_trial("a", passed=True), make_trial("a", passed=False), make_trial("b", passed=True), make_trial("b", passed=True)]
    assert compute_metrics(trials)["case_consistency"] == 0.5


def test_latency_percentiles():
    t = make_trial(fsm_ms=list(range(1, 101)))
    m = compute_metrics([t])
    assert m["fsm_p50_ms"] == 50
    assert m["fsm_p95_ms"] == 95


def test_judge_metrics_only_when_judged():
    assert "judge_score" not in compute_metrics([make_trial()])
    trials = [
        make_trial(judge={"score": 5.0, "hallucination": False}),
        make_trial(judge={"score": 3.0, "hallucination": True}),
    ]
    m = compute_metrics(trials)
    assert m["judge_score"] == 4.0
    assert m["hallucination_rate"] == 0.5


def test_gate_passes_when_everything_meets_targets():
    per = {"booking_happy": {"task_success": 1.0, "routing_accuracy": 1.0, "slot_accuracy": 1.0}}
    assert evaluate_gate({"error_rate": 0.0}, per, baseline=None) == []


def test_gate_flags_below_target_and_regression():
    per = {"booking_happy": {"task_success": 0.80, "routing_accuracy": 1.0}}
    reasons = evaluate_gate({"error_rate": 0.0}, per, baseline={"booking_happy": {"task_success": 0.95}})
    assert any("below target" in r for r in reasons)
    assert any("dropped" in r for r in reasons)


def test_gate_small_dip_within_tolerance_is_not_a_regression():
    per = {"faq": {"task_success": 0.93}}
    assert evaluate_gate({"error_rate": 0.0}, per, baseline={"faq": {"task_success": 0.96}}) == []


def test_gate_fails_on_high_infra_error_rate_and_hallucination():
    reasons = evaluate_gate({"error_rate": 0.4, "hallucination_rate": 0.2, "judge_score": 3.0}, {}, baseline=None)
    assert len(reasons) == 3


def test_failure_rows_describe_why_a_trial_failed():
    failing = make_trial(
        passed=False,
        routing=False,
        slot_detail={"partySize": {"expected": 2, "actual": 3, "ok": False}, "name": {"expected": "a", "actual": "a", "ok": True}},
    )
    rows = failure_rows([make_trial(), failing])
    assert len(rows) == 1
    assert "partySize: want 2 got 3" in rows[0]["why"]
    assert "ended in ASK_DATE" in rows[0]["why"]


def test_build_report_flags_status_and_first_run_baseline():
    trials = [make_trial(), make_trial(passed=False)]
    overall, per = compute_metrics(trials), metrics_by_scenario(trials)
    md, html_doc = build_report({"case_count": 1, "trials_per_case": 2}, "2026-W41", overall, per, None, ["booking_happy: task_success 50% below target 90%"], failure_rows(trials), None)
    assert "2026-W41" in md and "REGRESSION" in md
    assert "becomes the baseline" in md
    assert "booking_happy" in html_doc


def test_email_requires_smtp_credentials(monkeypatch):
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="SMTP_USER"):
        send_report_email("s", "body", "<p>body</p>")


def test_email_sends_to_default_recipient(monkeypatch):
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    monkeypatch.delenv("REPORT_TO", raising=False)
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, context=None, timeout=None):
            sent["host"] = host

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, user, password):
            sent["login"] = (user, password)

        def send_message(self, msg):
            sent["to"] = msg["To"]
            sent["subject"] = msg["Subject"]

    monkeypatch.setattr(eval_suite.smtplib, "SMTP_SSL", FakeSMTP)
    send_report_email("Weekly", "body", "<p>body</p>", attachments={"r.md": b"x"})
    assert sent["to"] == "methasitpun@gmail.com"
    assert sent["login"] == ("bot@example.com", "pw")
    assert sent["subject"] == "Weekly"
