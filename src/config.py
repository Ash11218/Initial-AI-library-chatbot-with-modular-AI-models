from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

BOOKS_CSV = DATA_DIR / "books.csv"
INTENTS_CSV = DATA_DIR / "intents.csv"
MONTHLY_DEMAND_CSV = DATA_DIR / "monthly_demand.csv"
