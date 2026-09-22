import os

from dotenv import load_dotenv

load_dotenv()


def _require_database_url() -> str:
    url = os.environ["DATABASE_URL"]
    if not url.startswith(("postgres://", "postgresql://")):
        raise ValueError(
            "DATABASE_URL must start with 'postgres://' or 'postgresql://'. "
            f"Got: {url[:20]!r}... This usually means the secret was saved with "
            "surrounding quotes (e.g. copied from a .env file as "
            'DATABASE_URL="...") — re-save it in GitHub Secrets with just the raw URL.'
        )
    return url


class Config:
    DATABASE_URL = _require_database_url()
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")
    CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "./checkpoint.json")
    COST_PER_SECOND = float(os.environ.get("COST_PER_SECOND", "0.05"))
