# any-ai-mlops

MLOps sidecar for any-ai-backend's voice agents: experiment tracking, cost/latency
monitoring, and LLM-as-judge quality evaluation, all logged to MLflow. Reads the
same Postgres database any-ai-backend (Prisma) owns — this service never writes
to it (see [mlops/db.py](mlops/db.py)).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # fill in DATABASE_URL and GOOGLE_API_KEY
```

You also need an MLflow tracking server for `MLFLOW_TRACKING_URI` to point at.
For local development, the simplest option is the file-store server built into
the `mlflow` package already in `requirements.txt` — no Docker required:

```bash
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Then open http://localhost:5000 for the UI. A Postgres-backed, containerized
version of the same stack (plus a runner container for the scripts below) is
defined one level up, in the sibling `docker-compose.mlops.yml` — use that if
you want persistence beyond a single machine or a shared team instance.

## Running the scripts

```bash
python -m mlops.log_experiment                          # logs one MLflow run per active agent
python -m mlops.evaluate_quality --agent-id <id>         # quality-evaluates one agent
python -m mlops.evaluate_quality                         # quality-evaluates every active agent
python -m mlops.calibrate_judge                          # checks the judge against data/human_labels.csv
```

`log_experiment.py` checkpoints the last-processed timestamp per agent in
`CHECKPOINT_PATH` (default `./checkpoint.json`) so repeated runs don't
double-count calls — see [mlops/checkpoint.py](mlops/checkpoint.py).

## Judge calibration

`mlops/calibrate_judge.py` compares the Gemini judge's scores against a human
baseline. There's no baseline yet — `data/human_labels.csv` is an empty
template. See [data/README.md](data/README.md) for how to fill it in.

## Tests

```bash
pytest
```

Tests are unit tests over pure functions with the DB, MLflow, and Gemini
clients mocked/monkeypatched — they don't need a live database or MLflow
server. `tests/conftest.py` sets placeholder env vars so `mlops.config.Config`
can import cleanly under pytest.

## CI / automation

- [.github/workflows/ci.yml](.github/workflows/ci.yml) runs the test suite on
  every push to `main` and on every pull request. No secrets required.
- [.github/workflows/scheduled.yml](.github/workflows/scheduled.yml) runs
  `log_experiment.py` and `evaluate_quality.py` (all active agents) daily at
  03:00 UTC, plus on manual dispatch. This needs repo secrets/variables set
  under **Settings > Secrets and variables > Actions** before it does
  anything useful:
  - `secrets.DATABASE_URL`
  - `secrets.GOOGLE_API_KEY`
  - `vars.MLFLOW_TRACKING_URI` — must be a real, internet-reachable MLflow
    server; `localhost` doesn't exist from a GitHub-hosted runner.

  GitHub-hosted runners are ephemeral, so the workflow caches
  `checkpoint.json` between runs via `actions/cache` rather than relying on
  local disk state — see the comments in that workflow file for how.

There's no calibration job scheduled, since `data/human_labels.csv` has no
real data yet; run `calibrate_judge.py` manually once it does.
