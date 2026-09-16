import csv
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from mlops import calibrate_judge


def _write_labels(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["call_id", "human_score"])
        writer.writerows(rows)


def test_load_human_labels_reads_call_id_and_score(tmp_path):
    path = tmp_path / "labels.csv"
    _write_labels(path, [["c1", "4"], ["c2", "2"]])

    assert calibrate_judge.load_human_labels(str(path)) == {"c1": 4.0, "c2": 2.0}


def test_load_human_labels_empty_file_returns_empty_dict(tmp_path):
    path = tmp_path / "labels.csv"
    _write_labels(path, [])

    assert calibrate_judge.load_human_labels(str(path)) == {}


def test_pearson_perfect_agreement_is_one():
    assert calibrate_judge._pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_pearson_constant_series_returns_none():
    assert calibrate_judge._pearson([2, 2, 2], [1, 2, 3]) is None


def test_run_calibration_raises_when_no_labels_match_a_transcription(monkeypatch):
    monkeypatch.setattr(calibrate_judge.db, "get_call_logs_by_ids", lambda conn, call_ids: {})

    with pytest.raises(ValueError, match="no labeled call_ids"):
        calibrate_judge.run_calibration(conn=object(), client=MagicMock(), model="gemini-2.5-flash", human_labels={"c1": 4})


def test_run_calibration_computes_agreement_metrics(monkeypatch):
    monkeypatch.setattr(
        calibrate_judge.db,
        "get_call_logs_by_ids",
        lambda conn, call_ids: {
            "c1": {"transcription": "..."},
            "c2": {"transcription": "..."},
        },
    )
    scores = iter([{"score": 4.0, "rationale": "ok"}, {"score": 3.0, "rationale": "meh"}])
    monkeypatch.setattr(calibrate_judge, "score_transcript", lambda client, model, transcript: next(scores))

    result = calibrate_judge.run_calibration(
        conn=object(), client=MagicMock(), model="gemini-2.5-flash", human_labels={"c1": 4.0, "c2": 5.0}
    )

    assert result["metrics"]["sample_size"] == 2
    assert result["metrics"]["mae"] == pytest.approx(1.0)  # |4-4| and |3-5| averaged
    assert result["metrics"]["exact_match_rate"] == 0.5
    assert result["metrics"]["within_1_point_rate"] == 0.5


def test_main_exits_when_labels_file_has_no_rows(tmp_path, monkeypatch):
    path = tmp_path / "labels.csv"
    _write_labels(path, [])
    monkeypatch.setattr("sys.argv", ["calibrate_judge", "--labels", str(path)])

    with pytest.raises(SystemExit, match="no labeled rows"):
        calibrate_judge.main()
