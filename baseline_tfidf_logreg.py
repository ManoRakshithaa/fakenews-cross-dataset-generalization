"""
Baseline Model 1 of many — TF-IDF + Logistic Regression, WELFake in-domain
=============================================================================
This is the FIRST model script. Scope on purpose: WELFake -> WELFake only,
using title+body combined as the text input. Once this is verified working
and you understand it, the same `run_experiment()` function gets reused for
the other 3 dataset combos and the title-only / body-only variants — no
rewriting, just different arguments.

WHY THIS ORDER (train/vectorize/evaluate) MATTERS
----------------------------------------------------
1. Split train/test FIRST, before vectorizing.
2. Fit the TF-IDF vectorizer ONLY on the training text.
3. Use that already-fitted vectorizer to *transform* (not re-fit) the test
   text.
If you vectorize before splitting, or fit the vectorizer on all the data,
the model gets to "see" test-set vocabulary and statistics during training.
That's a data leakage bug that quietly inflates your accuracy number —
exactly the kind of unsupported result your research rules told me to
avoid producing.

Run: python3 baseline_tfidf_logreg.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

PROCESSED_DIR = Path("./processed")
RESULTS_DIR = Path("./results")

# Reproducibility: same seed every run = same split, same result, so anyone
# (including you, six months from now) can rerun this and get the same number.
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_clean(dataset_name: str) -> pd.DataFrame:
    """Load a cleaned dataset produced by clean_datasets.py."""
    path = PROCESSED_DIR / f"{dataset_name}_clean.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["text"])          # safety net; shouldn't drop anything at this stage
    return df


def build_text_field(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Create the single text column the model will actually read, based on
    which of your 3 planned variants you're running:
      "title_body" -> title + " " + text   (this baseline's default)
      "title_only" -> just the title
      "body_only"  -> just the article body
    """
    df = df.copy()
    title = df["title"].fillna("")
    text = df["text"].fillna("")
    if mode == "title_body":
        df["_model_input"] = title + " " + text
    elif mode == "title_only":
        df["_model_input"] = title
    elif mode == "body_only":
        df["_model_input"] = text
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'title_body', 'title_only', or 'body_only'.")
    return df


def run_experiment(train_dataset: str, test_dataset: str, mode: str = "title_body"):
    """
    train_dataset / test_dataset: "WELFake" or "ISOT". For this first
    baseline they're the SAME (in-domain). Later, passing different names
    here is literally all that changes for the cross-dataset experiments.
    """
    print(f"Loading {train_dataset}_clean.csv and {test_dataset}_clean.csv ...")
    train_source = build_text_field(load_clean(train_dataset), mode)
    test_source = build_text_field(load_clean(test_dataset), mode)

    if train_dataset == test_dataset:
        # In-domain: split ONE dataset into train and test portions.
        train_df, test_df = train_test_split(
            train_source,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=train_source["label"],     # LINE A: keep class ratio equal in both halves
        )
    else:
        # Cross-dataset: train on ALL of one dataset, test on ALL of the other.
        train_df = train_source
        test_df = test_source

    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    # LINE B: fit the vectorizer on TRAIN TEXT ONLY.
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    X_train = vectorizer.fit_transform(train_df["_model_input"])
    X_test = vectorizer.transform(test_df["_model_input"])   # LINE C: transform only, never fit again

    y_train = train_df["label"]
    y_test = test_df["label"]

    # class_weight="balanced" corrects for the ~55/45 label skew we measured
    # after cleaning, instead of silently letting the majority class win ties.
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 (label=1): {f1:.4f}")
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)

    RESULTS_DIR.mkdir(exist_ok=True)
    experiment_name = f"{train_dataset}_to_{test_dataset}_{mode}"
    result = {
        "experiment": experiment_name,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "text_mode": mode,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "accuracy": acc,
        "f1": f1,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "random_state": RANDOM_STATE,
        "test_size_if_in_domain": TEST_SIZE if train_dataset == test_dataset else None,
        "model": "TF-IDF (1,2-gram, max_features=20000) + LogisticRegression(class_weight=balanced)",
    }
    out_path = RESULTS_DIR / f"{experiment_name}_tfidf_logreg.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_path}")

    return result


if __name__ == "__main__":
    # Full grid: all 4 dataset combos x all 3 text-field modes = 12 runs.
    # Each is the same run_experiment() call -- only the arguments change.
    combos = [
        ("WELFake", "WELFake"),   # in-domain
        ("WELFake", "ISOT"),      # cross-dataset
        ("ISOT", "ISOT"),         # in-domain
        ("ISOT", "WELFake"),      # cross-dataset
    ]
    modes = ["title_body", "title_only", "body_only"]

    all_results = []
    for mode in modes:
        for train_name, test_name in combos:
            print(f"\n{'#' * 70}\n# EXPERIMENT: train={train_name}  test={test_name}  mode={mode}\n{'#' * 70}")
            result = run_experiment(train_dataset=train_name, test_dataset=test_name, mode=mode)
            all_results.append(result)

    # Side-by-side summary across all 12 runs, grouped by mode, since the
    # per-experiment JSON files are easy to lose track of at this scale.
    print(f"\n{'=' * 70}\nSUMMARY (all modes)\n{'=' * 70}")
    for mode in modes:
        print(f"\n--- mode: {mode} ---")
        for r in all_results:
            if r["text_mode"] == mode:
                print(f"{r['train_dataset']:8s} -> {r['test_dataset']:8s}  "
                      f"accuracy={r['accuracy']:.4f}  f1={r['f1']:.4f}")