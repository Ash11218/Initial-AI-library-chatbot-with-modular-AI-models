import pandas as pd

DIFFICULTY_ORDER = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}


def generate_learning_path(books_df: pd.DataFrame, topic_query: str) -> list[dict]:
    # Filter by topic match (subject or description)
    topic = topic_query.lower().strip()

    df = books_df.copy()
    df["combined"] = (
        df["subject"].astype(str) + " " + df["description"].astype(str)
    ).str.lower()
    filtered = df[df["combined"].str.contains(topic, na=False)]

    # If nothing matches, fallback to all books (still make a path)
    if filtered.empty:
        filtered = df

    filtered["diff_rank"] = filtered["difficulty"].map(DIFFICULTY_ORDER).fillna(99)
    filtered = filtered.sort_values(["diff_rank", "title"])

    path = []
    for step, (_, row) in enumerate(filtered.iterrows(), start=1):
        # simple estimated hours by difficulty
        hours = (
            6
            if row["difficulty"] == "Beginner"
            else 10
            if row["difficulty"] == "Intermediate"
            else 14
        )
        path.append(
            {
                "step": step,
                "book": row["title"],
                "difficulty": row["difficulty"],
                "estimated_hours": hours,
            }
        )
    return path[:6]
