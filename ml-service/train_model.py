"""Train the LIAR pipeline using the case study's split and preprocessing method."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
LOWER_TRUTH = {"pants-fire", "false", "barely-true"}


def load_partition(name, seen):
    path = DATA_DIR / f"{name}.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Download the LIAR {name}.tsv partition into ml-service/data/.")
    raw = pd.read_csv(path, sep="\t", header=None)
    if raw.shape[1] < 3:
        raise ValueError(f"{path} must contain LIAR columns including label (column 2) and statement (column 3).")
    frame = raw[[1, 2]].copy()
    frame.columns = ["label", "statement"]
    frame = frame.dropna(subset=["label", "statement"])
    frame["statement"] = frame["statement"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    frame = frame[frame.statement.str.len() > 0].copy()
    frame["key"] = frame.statement.str.lower()
    frame = frame.drop_duplicates("key")
    frame = frame[~frame.key.isin(seen)].copy()
    seen.update(frame.key)
    frame["y"] = frame.label.isin(LOWER_TRUTH).astype(int)
    return frame


def make_model(c_value):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            min_df=2, max_df=0.95, sublinear_tf=True)),
        ("clf", LogisticRegression(
            C=c_value, max_iter=2000, solver="liblinear", random_state=42)),
    ])


def main():
    seen = set()
    train, valid, test = [load_partition(name, seen) for name in ("train", "valid", "test")]
    candidates = []
    for c_value in (0.1, 1, 10):
        candidate = make_model(c_value).fit(train.statement, train.y)
        score = f1_score(valid.y, candidate.predict(valid.statement), average="macro")
        print(f"C={c_value}: validation macro F1={score:.4f}")
        candidates.append((score, c_value, candidate))
    _, selected_c, fitted = max(candidates, key=lambda item: item[0])
    test_prediction = fitted.predict(test.statement)
    print(f"Selected C: {selected_c}")
    print(f"Test accuracy: {accuracy_score(test.y, test_prediction):.4f}")
    print(f"Test macro F1: {f1_score(test.y, test_prediction, average='macro'):.4f}")
    print(classification_report(test.y, test_prediction, target_names=["Real", "Fake"], digits=4))
    print("Confusion matrix (actual rows, predicted columns; Real=0, Fake=1):")
    print(confusion_matrix(test.y, test_prediction, labels=[0, 1]))
    output = ROOT / "model.joblib"
    joblib.dump(fitted, output)
    print(f"Saved selected C={selected_c} pipeline to {output}")


if __name__ == "__main__":
    main()
