from .intent_model import IntentClassifier
from .recommender import BookRecommender
from .learning_path import generate_learning_path
from .demand_model import DemandPredictor
from .datasets import load_books, load_monthly_demand


class AIDecisionLayer:
    def __init__(self, intent_model_path: str, demand_model_path: str):
        self.intent = IntentClassifier(intent_model_path)
        self.demand = DemandPredictor(demand_model_path)

        self.books_df = load_books()
        self.recommender = BookRecommender(self.books_df)

    def load_models(self):
        self.intent.load()
        self.demand.load()

    def handle(self, user_text: str) -> dict:
        intent_result = self.intent.predict(user_text)

        if intent_result.intent == "RECOMMEND":
            recs = self.recommender.recommend(user_text, top_k=3)
            return {
                "intent": intent_result.intent,
                "confidence": intent_result.confidence,
                "response": recs,
            }

        if intent_result.intent == "LEARNING_PATH":
            path = generate_learning_path(self.books_df, user_text)
            return {
                "intent": intent_result.intent,
                "confidence": intent_result.confidence,
                "response": path,
            }

        if intent_result.intent == "DEMAND_PREDICT":
            demand_df = load_monthly_demand()
            # predict next month based on last month number
            last_month = int(demand_df["month"].max())
            next_month = last_month + 1
            pred = self.demand.predict_next(next_month)
            return {
                "intent": intent_result.intent,
                "confidence": intent_result.confidence,
                "response": {
                    "next_month": next_month,
                    "predicted_borrows": round(pred, 2),
                },
            }

        # SMALLTALK or unknown
        return {
            "intent": intent_result.intent,
            "confidence": intent_result.confidence,
            "response": "Hi! I can recommend books, generate a learning path, or predict demand. Try: 'recommend AI books'.",
        }
