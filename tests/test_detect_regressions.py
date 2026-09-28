import time

import mlflow

from mlops import detect_regressions as dr

WATCHED = {"success_rate": dr.HIGHER_IS_BETTER, "avg_cost_per_call": dr.LOWER_IS_BETTER}


def test_flags_drop_in_higher_is_better_metric():
    baseline = [{"success_rate": 0.9}] * 3

    result = dr.find_regressions({"success_rate": 0.6}, baseline, WATCHED, max_relative_change=0.15)

    assert [r["metric"] for r in result] == ["success_rate"]
    assert result[0]["baseline"] == 0.9


def test_flags_rise_in_lower_is_better_metric():
    baseline = [{"avg_cost_per_call": 1.0}] * 3

    result = dr.find_regressions({"avg_cost_per_call": 1.5}, baseline, WATCHED, max_relative_change=0.15)

    assert [r["metric"] for r in result] == ["avg_cost_per_call"]


def test_ignores_improvements_and_small_moves():
    baseline = [{"success_rate": 0.8, "avg_cost_per_call": 1.0}] * 3

    assert dr.find_regressions({"success_rate": 0.95, "avg_cost_per_call": 1.1}, baseline, WATCHED, 0.15) == []


def test_needs_enough_baseline_runs():
    assert dr.find_regressions({"success_rate": 0.1}, [{"success_rate": 0.9}] * 2, WATCHED, 0.15) == []


def test_describe_config_change_lists_only_changed_params():
    before = {"prompt_hash": "aaa", "model": "m1", "temperature": "0.5"}
    after = {"prompt_hash": "bbb", "model": "m1", "temperature": "0.5"}

    assert dr.describe_config_change(after, before) == ["prompt_hash: aaa -> bbb"]


def _run(metrics, params=None, tags=None, age_hours=0):
    return {
        "metrics": metrics,
        "params": params or {},
        "tags": tags or {},
        "start_time_ms": (time.time() - age_hours * 3600) * 1000,
    }


def test_check_agent_reports_regression_and_config_change(monkeypatch):
    tracking = [_run({"total_calls": 50, "success_rate": 0.5}, {"prompt_hash": "new"})] + [
        _run({"total_calls": 50, "success_rate": 0.9}, {"prompt_hash": "old"}, age_hours=24 * i) for i in range(1, 4)
    ]
    monkeypatch.setattr(
        dr, "fetch_runs", lambda experiment, run_type, max_results: tracking if run_type == "tracking" else []
    )

    report = dr.check_agent({"id": "a1", "name": "Reception"}, baseline_runs=7, max_relative_change=0.15, max_age_hours=36)

    assert [r["metric"] for r in report["regressions"]] == ["success_rate"]
    assert report["config_changes"][0]["changes"] == ["prompt_hash: old -> new"]
    assert report["config_changes"][0]["before"]["success_rate"] == 0.9
    assert report["config_changes"][0]["after"]["success_rate"] == 0.5


def test_check_agent_skips_small_samples_stale_runs_and_other_rubrics(monkeypatch):
    runs = {
        "tracking": [_run({"total_calls": 3, "success_rate": 0.1})] + [_run({"total_calls": 50, "success_rate": 0.9})] * 3,
        "quality_eval": [_run({"eval_sample_size": 20, "avg_quality_score": 2.0}, tags={"rubric_hash": "v2"})]
        + [_run({"eval_sample_size": 20, "avg_quality_score": 4.5}, tags={"rubric_hash": "v1"})] * 3,
    }
    monkeypatch.setattr(dr, "fetch_runs", lambda experiment, run_type, max_results: runs[run_type])

    report = dr.check_agent({"id": "a1", "name": "Reception"}, baseline_runs=7, max_relative_change=0.15, max_age_hours=36)
    assert report["regressions"] == []

    runs["tracking"][0] = _run({"total_calls": 50, "success_rate": 0.1}, age_hours=48)
    report = dr.check_agent({"id": "a1", "name": "Reception"}, baseline_runs=7, max_relative_change=0.15, max_age_hours=36)
    assert report["regressions"] == []


def test_fetch_runs_against_a_real_file_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}")
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    assert dr.fetch_runs("agent-a1-Reception", "tracking", 5) == []

    mlflow.set_experiment("agent-a1-Reception")
    for rate in (0.9, 0.8):
        with mlflow.start_run():
            mlflow.log_metric("success_rate", rate)
            mlflow.log_param("prompt_hash", "abc")
            mlflow.set_tag("run_type", "tracking")
        time.sleep(0.01)  # distinct start_time so the newest-first order is deterministic
    with mlflow.start_run():
        mlflow.set_tag("run_type", "quality_eval")

    runs = dr.fetch_runs("agent-a1-Reception", "tracking", 5)

    assert [r["metrics"]["success_rate"] for r in runs] == [0.8, 0.9]
    assert runs[0]["params"] == {"prompt_hash": "abc"}


def test_format_report_when_nothing_found():
    assert dr.format_report([], []) == "No regressions or config changes."
