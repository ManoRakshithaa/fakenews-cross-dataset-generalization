"""
Baseline Model — TF-IDF + Logistic Regression — FINAL DATA RERUN
=============================================================================
Apples-to-apples rerun of the existing baseline against processed_final/
(the Reuters-regex-fixed cleaned datasets), instead of processed/ (v1).

NOTHING about the modeling logic changed from baseline_tfidf_logreg.py:
  - same TF-IDF settings (1,2-grams, max_features=20000, stop_words=english)
  - same LogisticRegression settings (class_weight=balanced, random_state=42)
  - same split strategy (80/20 stratified for in-domain, full-dataset for
    cross-dataset)
  - same random_state=42 throughout
The ONLY changes are: (1) which folder/files are loaded, and (2) where
results are saved -- results_final/, kept completely separate from the
original results/ folder so old and new numbers are never mixed.

ROC-AUC: NOT computed here. It wasn't part of the original baseline's
evaluation methodology, and the instruction for this rerun was to report
ROC-AUC "if the existing script already calculates it" -- it doesn't, so
per that same instruction it's correctly left out. Easy to add later as a
separate, explicitly-requested addition if wanted (predict_proba is already
available from the fitted model -- would need no retraining).

Scope: all 4 dataset combos, title+body mode only (matches "keep the
existing setup" -- title+body combined is listed as a fixed setting, not a
variable to sweep, for this particular rerun).

Run: python3 baseline_tfidf_logreg_final.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

PROCESSED_DIR = Path("./processed_final")
RESULTS_DIR = Path("./results_final")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Expected final row counts, from final_clean_and_audit.py's actual output.
# Used only as a sanity check -- printed loudly if the loaded files don't
# match, since "do not call any result final until you've confirmed
# processed_final/ was actually used" was an explicit requirement.
EXPECTED_SIZES = {"WELFake": 23903, "ISOT": 38478}


def load_clean(dataset_name: str) -> pd.DataFrame:
    """Load a FINAL cleaned dataset produced by final_clean_and_audit.py."""
    path = PROCESSED_DIR / f"{dataset_name}_final_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run final_clean_and_audit.py first to produce "
            f"processed_final/{dataset_name}_final_clean.csv."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["text"])
    n = len(df)
    expected = EXPECTED_SIZES.get(dataset_name)
    if expected is not None and n != expected:
        print(f"[WARNING] Loaded {path} has {n} rows, expected {expected} from the "
              f"final_clean_and_audit.py run. This may not be the data you think it is -- "
              f"double check before trusting these results.")
    else:
        print(f"Confirmed: loaded {path} ({n} rows, matches expected final size).")
    return df


def build_text_field(df: pd.DataFrame, mode: str) -> pd.DataFrame:
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
    print(f"Loading {train_dataset}_final_clean.csv and {test_dataset}_final_clean.csv ...")
    train_source = build_text_field(load_clean(train_dataset), mode)
    test_source = build_text_field(load_clean(test_dataset), mode)

    if train_dataset == test_dataset:
        train_df, test_df = train_test_split(
            train_source,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=train_source["label"],
        )
    else:
        train_df = train_source
        test_df = test_source

    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    X_train = vectorizer.fit_transform(train_df["_model_input"])
    X_test = vectorizer.transform(test_df["_model_input"])

    y_train = train_df["label"]
    y_test = test_df["label"]

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
        "data_version": "processed_final (Reuters-regex-fixed)",
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
    combos = [
        ("WELFake", "WELFake"),   # in-domain
        ("WELFake", "ISOT"),      # cross-dataset
        ("ISOT", "ISOT"),         # in-domain
        ("ISOT", "WELFake"),      # cross-dataset
    ]

    all_results = []
    for train_name, test_name in combos:
        print(f"\n{'#' * 70}\n# EXPERIMENT (FINAL DATA): train={train_name}  test={test_name}  mode=title_body\n{'#' * 70}")
        result = run_experiment(train_dataset=train_name, test_dataset=test_name, mode="title_body")
        all_results.append(result)

    print(f"\n{'=' * 70}\nSUMMARY (FINAL DATA, title+body)\n{'=' * 70}")
    for r in all_results:
        print(f"{r['train_dataset']:8s} -> {r['test_dataset']:8s}  "
              f"accuracy={r['accuracy']:.4f}  f1={r['f1']:.4f}")

    print(f"\nAll 4 results saved under {RESULTS_DIR.resolve()}")
    print("These are FINAL-data results. Compare against your earlier results/ "
          "(v1 data) folder separately -- nothing here overwrites or merges with those.")