import pandas as pd
from .config import BOOKS_CSV, INTENTS_CSV, MONTHLY_DEMAND_CSV


def load_books() -> pd.DataFrame:
    return pd.read_csv(BOOKS_CSV)


def load_intents() -> pd.DataFrame:
    return pd.read_csv(INTENTS_CSV)


def load_monthly_demand() -> pd.DataFrame:
    return pd.read_csv(MONTHLY_DEMAND_CSV)
