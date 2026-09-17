"""Dumps every transcript currently in conversation_logs (any agent, active or not)
to a CSV so a human can read them and assign 1-5 scores.

This deliberately does NOT write data/human_labels.csv itself and does not invent
scores — scoring a transcript against the rubric is a human judgment call that
calibrate_judge exists to check the AI judge against, so faking it here would
defeat the entire point. It only removes the friction of finding transcripts.

Usage:
    python -m mlops.prepare_labeling_sheet
    python -m mlops.prepare_labeling_sheet --out data/transcripts_for_labeling.csv

Workflow:
    1. Run this script.
    2. Open the output CSV, read each transcript, fill in human_score (1-5) for
       the rows you choose to label (see evaluate_quality.RUBRIC_PROMPT for the
       scoring criteria — same rubric the AI judge uses).
    3. Copy the call_id,human_score columns for your scored rows into
       data/human_labels.csv.
    4. Run `python -m mlops.calibrate_judge`.
"""

import argparse
import csv

from . import db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/transcripts_for_labeling.csv")
    args = parser.parse_args()

    with db.get_connection() as conn:
        rows = db.get_all_transcripts(conn)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["call_id", "agent_name", "start_time", "human_score", "transcript"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "call_id": row["call_id"],
                    "agent_name": row["agent_name"],
                    "start_time": row["start_time"],
                    "human_score": "",
                    "transcript": row["transcription"],
                }
            )

    print(f"wrote {len(rows)} transcripts to {args.out}")
    print("fill in human_score (1-5) per row you choose to label, then copy")
    print("call_id,human_score into data/human_labels.csv")


if __name__ == "__main__":
    main()
