# data/

`human_labels.csv` is the baseline for `mlops.calibrate_judge`. It's currently
empty (header only) — no human scoring has happened yet.

To use it: pick some `call_id`s from `call_logs` (ones with a non-empty
`transcription`), have a human score each 1-5 against the same rubric as
`mlops/evaluate_quality.py`'s `RUBRIC_PROMPT`, and add a row per call:

```csv
call_id,human_score
c1a2b3,4
d4e5f6,2
```

Then run `python -m mlops.calibrate_judge` to see how well the Gemini judge
agrees with those human scores.
