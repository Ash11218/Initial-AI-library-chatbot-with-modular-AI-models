import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BookRecommender:
    def __init__(self, books_df: pd.DataFrame):
        self.books = books_df.copy()
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        # combine subject + difficulty + description for better matching
        self.books["features"] = (
            self.books["subject"].astype(str)
            + " "
            + self.books["difficulty"].astype(str)
            + " "
            + self.books["description"].astype(str)
        )
        self.matrix = self.vectorizer.fit_transform(self.books["features"])

    def recommend(self, query: str, top_k: int = 3) -> list[dict]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]

        results = []
        for i in top_idx:
            row = self.books.iloc[i]
            results.append(
                {
                    "title": row["title"],
                    "subject": row["subject"],
                    "difficulty": row["difficulty"],
                    "score": float(sims[i]),
                }
            )
        return results
