# data/

`human_labels.csv` is the baseline for `mlops.calibrate_judge`. It's currently
empty (header only) — no human scoring has happened yet, and nothing in this
repo can fill it in for you: a human score is the whole point of comparison,
so generating it automatically would just be comparing the judge to itself.

To use it:

```bash
python -m mlops.prepare_labeling_sheet
```

This writes every transcript currently in `conversation_logs` (across every
agent, active or not) to `data/transcripts_for_labeling.csv`, with a blank
`human_score` column. Open it, read a transcript, score it 1-5 against the
same rubric as `mlops/evaluate_quality.py`'s `RUBRIC_PROMPT`, then copy the
`call_id,human_score` columns for the rows you scored into `human_labels.csv`:

```csv
call_id,human_score
c1a2b3,4
d4e5f6,2
```

Then run `python -m mlops.calibrate_judge` to see how well the Gemini judge
agrees with those human scores.

Note: as of this writing, every transcript in the database is from dev/test
traffic (`test-*`/`call_*` call_ids, agents named "test"/"(Test)", or configs
that are no longer active) — none of it is a real customer call. That's fine
for calibrating the judge itself (the rubric doesn't care whether the call was
real), but don't read anything into *which* agent or prompt scored well from
this data.
