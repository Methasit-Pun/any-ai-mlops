import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    DATABASE_URL = os.environ["DATABASE_URL"]
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")
    CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "./checkpoint.json")
    COST_PER_SECOND = float(os.environ.get("COST_PER_SECOND", "0.05"))
