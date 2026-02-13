import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


class DemandPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model: LinearRegression | None = None

    def train(self, months, borrow_counts):
        X = np.array(months).reshape(-1, 1)
        y = np.array(borrow_counts)
        self.model = LinearRegression()
        self.model.fit(X, y)

    def predict_next(self, next_month: int) -> float:
        if not self.model:
            raise RuntimeError("Demand model not loaded.")
        return float(self.model.predict(np.array([[next_month]]))[0])

    def save(self):
        if not self.model:
            raise RuntimeError("Nothing to save.")
        joblib.dump(self.model, self.model_path)

    def load(self):
        self.model = joblib.load(self.model_path)
