from pathlib import Path  # noqa: F401
from src.config import MODELS_DIR
from src.datasets import load_intents, load_monthly_demand
from src.intent_model import IntentClassifier
from src.demand_model import DemandPredictor


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    # Train Intent Classifier
    intents = load_intents()
    intent_model_path = str(MODELS_DIR / "intent_model.joblib")
    intent_clf = IntentClassifier(intent_model_path)
    intent_clf.train(intents["text"].tolist(), intents["intent"].tolist())
    intent_clf.save()
    print("✅ Saved intent model:", intent_model_path)

    # Train Demand Predictor
    demand = load_monthly_demand()
    demand_model_path = str(MODELS_DIR / "demand_model.joblib")
    demand_pred = DemandPredictor(demand_model_path)
    demand_pred.train(demand["month"].tolist(), demand["borrow_count"].tolist())
    demand_pred.save()
    print("✅ Saved demand model:", demand_model_path)


if __name__ == "__main__":
    main()
