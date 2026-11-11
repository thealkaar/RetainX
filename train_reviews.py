import csv
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "olist"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_CSV = DATA_DIR / "olist_order_reviews_dataset.csv"


def train_olist_reviews():
    print(f"Loading real Olist e-commerce customer reviews from {REVIEW_CSV}...")
    texts = []
    labels = []
    groups = []

    with open(REVIEW_CSV, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            msg = row.get("review_comment_message", "").strip()
            score = row.get("review_score", "").strip()
            if not msg or len(msg) < 3 or not score:
                continue
            try:
                val = int(score)
                if val in (1, 2):
                    texts.append(msg)
                    labels.append(0)  # Negative sentiment
                    groups.append(row.get("order_id", ""))
                elif val in (4, 5):
                    texts.append(msg)
                    labels.append(1)  # Positive sentiment
                    groups.append(row.get("order_id", ""))
            except ValueError:
                continue

    print(f"Loaded {len(texts):,} customer reviews (Positive: {sum(labels):,}, Negative: {len(labels)-sum(labels):,})")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(texts, labels, groups))
    x_train = [texts[i] for i in train_idx]
    x_test = [texts[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, strip_accents="unicode")
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.5)
    clf.fit(x_train_vec, y_train)

    probs = clf.predict_proba(x_test_vec)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    print(f"Olist Review Sentiment Model -> ROC-AUC: {auc:.3f}, F1: {f1:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")

    bundle = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "metrics": {
            "roc_auc": round(auc, 3),
            "f1": round(f1, 3),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
        },
        "evaluation_split": "20% order-grouped holdout; reviews from one order stay in one split.",
        "label_definition": "Negative=1-2 stars; positive=4-5 stars; 3-star and blank comments excluded.",
        "training_examples": len(train_idx),
        "test_examples": len(test_idx),
        "limitations": "Predicts star-rating-derived sentiment for Olist Portuguese comments, not customer satisfaction independently verified by humans. Three-star and blank reviews are excluded. Performance may differ on other stores, time periods, and writing styles.",
    }

    joblib.dump(bundle, MODEL_DIR / "reviews.joblib")
    print(f"Saved review model to {MODEL_DIR / 'reviews.joblib'}")


if __name__ == "__main__":
    train_olist_reviews()
