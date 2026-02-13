from dataclasses import dataclass
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class IntentResult:
    intent: str
    confidence: float


class IntentClassifier:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline: Pipeline | None = None

    def train(self, texts, labels):
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        )
        self.pipeline.fit(texts, labels)

    def predict(self, text: str) -> IntentResult:
        if not self.pipeline:
            raise RuntimeError("Intent model not loaded. Train or load it first.")
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        best_idx = proba.argmax()
        return IntentResult(
            intent=str(classes[best_idx]), confidence=float(proba[best_idx])
        )

    def save(self):
        if not self.pipeline:
            raise RuntimeError("Nothing to save.")
        joblib.dump(self.pipeline, self.model_path)

    def load(self):
        self.pipeline = joblib.load(self.model_path)
