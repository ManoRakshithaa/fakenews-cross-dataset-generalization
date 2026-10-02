"""
Baseline Diagnostics — same protocol as baseline_tfidf_logreg.py, extended
=============================================================================
Does NOT change dataset cleaning, the model, hyperparameters, or the
train/test protocol used in baseline_tfidf_logreg.py. It imports the exact
same loading/splitting/vectorizing/training pieces from that script (not
retyped), so there is zero risk of the two scripts' protocols drifting
apart. What this script ADDS is diagnostic output on top:
  - confusion matrix (raw numbers + a saved PNG plot)
  - accuracy, precision, recall, F1 (binary, positive class = FAKE)
  - macro F1, weighted F1
  - per-class precision/recall/F1
  - predicted-class counts vs actual-class counts (for spotting whether
    a model is just predicting one class most of the time)
  - raw predictions + true labels saved to CSV, for later manual error
    inspection

LABEL CONVENTION (confirmed, not assumed, from clean_datasets.py):
  label = 0 -> REAL   (ISOT: True.csv rows; WELFake: label=0 in raw file)
  label = 1 -> FAKE   (ISOT: Fake.csv rows; WELFake: label=1 in raw file)
This is the same convention already used throughout the project (see
clean_datasets.py's forced_label=0/1 assignment for ISOT, and the sample
titles printed during the original audit run).

Scope: title+body mode only, matching the original 4-experiment baseline
you already have results for. Text-field mode is a parameter here too,
in case you want to run the same diagnostics on title_only/body_only later
— nothing else needs to change to do that.

Run: python3 diagnose_baseline.py
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, classification_report, ConfusionMatrixDisplay,
)

# Reuse the EXACT same loading helpers + split constants as the baseline
# script, imported rather than retyped, so the protocol cannot drift.
from baseline_tfidf_logreg import load_clean, build_text_field, RANDOM_STATE, TEST_SIZE

DIAGNOSTICS_DIR = Path("./diagnostics")
PLOTS_DIR = DIAGNOSTICS_DIR / "confusion_matrix_plots"
PREDICTIONS_DIR = DIAGNOSTICS_DIR / "predictions"

LABEL_NAMES = {0: "REAL", 1: "FAKE"}

# A predicted-vs-actual class share gap bigger than this is flagged as a
# possible prediction imbalance. Stated explicitly here (not hidden in
# code) so you can see/change the threshold used for the final flag.
IMBALANCE_FLAG_THRESHOLD_PP = 0.15


def run_diagnostic(train_dataset: str, test_dataset: str, mode: str = "title_body") -> dict:
    print(f"\n{'=' * 70}\nDIAGNOSTIC: train={train_dataset}  test={test_dataset}  mode={mode}\n{'=' * 70}")

    train_source = build_text_field(load_clean(train_dataset), mode)
    test_source = build_text_field(load_clean(test_dataset), mode)

    if train_dataset == test_dataset:
        # SAME protocol as baseline_tfidf_logreg.py: stratified 80/20 split.
        train_df, test_df = train_test_split(
            train_source, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=train_source["label"],
        )
        split_desc = f"20% stratified held-out split of {train_dataset} (random_state={RANDOM_STATE})"
    else:
        # SAME protocol: cross-dataset uses the full external dataset, no split.
        train_df = train_source
        test_df = test_source
        split_desc = f"full {test_dataset} dataset (never seen during training)"

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    X_train = vectorizer.fit_transform(train_df["_model_input"])
    X_test = vectorizer.transform(test_df["_model_input"])
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ---------------- metrics ----------------
    acc = accuracy_score(y_test, y_pred)
    precision_fake = precision_score(y_test, y_pred, pos_label=1, zero_division=0)   # FAKE = positive class
    recall_fake = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_fake = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report_dict = classification_report(
        y_test, y_pred, target_names=["REAL(0)", "FAKE(1)"], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # rows/cols ordered REAL, FAKE

    # ---------------- class counts: predicted vs actual ----------------
    actual_counts_raw = pd.Series(y_test).value_counts().to_dict()
    predicted_counts_raw = pd.Series(y_pred).value_counts().to_dict()
    actual_counts = {LABEL_NAMES[k]: int(actual_counts_raw.get(k, 0)) for k in [0, 1]}
    predicted_counts = {LABEL_NAMES[k]: int(predicted_counts_raw.get(k, 0)) for k in [0, 1]}

    print(f"Evaluation set: {split_desc}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (FAKE): {precision_fake:.4f}  Recall (FAKE): {recall_fake:.4f}  F1 (FAKE): {f1_fake:.4f}")
    print(f"Macro F1: {macro_f1:.4f}   Weighted F1: {weighted_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=["REAL(0)", "FAKE(1)"], zero_division=0))
    print(f"Actual class counts:    {actual_counts}")
    print(f"Predicted class counts: {predicted_counts}")
    print(f"Confusion matrix (rows=actual [REAL,FAKE], cols=predicted [REAL,FAKE]):\n{cm}")

    # ---------------- save predictions + true labels ----------------
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    exp_name = f"{train_dataset}_to_{test_dataset}_{mode}"
    preds_df = pd.DataFrame({
        "true_label": y_test,
        "true_label_name": [LABEL_NAMES[v] for v in y_test],
        "predicted_label": y_pred,
        "predicted_label_name": [LABEL_NAMES[v] for v in y_pred],
        "correct": (y_test == y_pred),
        "title": test_df["title"].values,
        "text": test_df["text"].values,
    })
    preds_path = PREDICTIONS_DIR / f"{exp_name}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    # ---------------- confusion matrix plot ----------------
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL (0)", "FAKE (1)"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{train_dataset} -> {test_dataset} ({mode})\nAccuracy={acc:.3f}   Macro F1={macro_f1:.3f}")
    fig.tight_layout()
    plot_path = PLOTS_DIR / f"{exp_name}_confusion_matrix.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # ---------------- prediction-imbalance flag ----------------
    total = len(y_pred)
    pred_fake_share = predicted_counts["FAKE"] / total
    actual_fake_share = actual_counts["FAKE"] / total
    imbalance_flag = abs(pred_fake_share - actual_fake_share) > IMBALANCE_FLAG_THRESHOLD_PP

    result = {
        "experiment": exp_name,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "mode": mode,
        "eval_set_description": split_desc,
        "n_test": int(total),
        "accuracy": float(acc),
        "precision_fake": float(precision_fake),
        "recall_fake": float(recall_fake),
        "f1_fake": float(f1_fake),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_label_order": ["REAL(0)", "FAKE(1)"],
        "actual_class_counts": actual_counts,
        "predicted_class_counts": predicted_counts,
        "actual_fake_share": float(actual_fake_share),
        "predicted_fake_share": float(pred_fake_share),
        "prediction_imbalance_flag": bool(imbalance_flag),
        "imbalance_flag_threshold_pp": IMBALANCE_FLAG_THRESHOLD_PP,
        "predictions_csv": str(preds_path),
        "confusion_matrix_plot": str(plot_path),
    }

    DIAGNOSTICS_DIR.mkdir(exist_ok=True)
    result_path = DIAGNOSTICS_DIR / f"{exp_name}_diagnostics.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {preds_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {result_path}")

    return result


def main():
    combos = [
        ("WELFake", "WELFake"),   # in-domain
        ("WELFake", "ISOT"),      # cross-dataset
        ("ISOT", "ISOT"),         # in-domain
        ("ISOT", "WELFake"),      # cross-dataset
    ]
    all_results = []
    for train_name, test_name in combos:
        r = run_diagnostic(train_name, test_name, mode="title_body")
        all_results.append(r)

    print(f"\n{'=' * 70}\nPREDICTION-IMBALANCE CHECK ACROSS ALL 4 EXPERIMENTS\n{'=' * 70}")
    for r in all_results:
        flag = "*** POSSIBLE PREDICTION IMBALANCE ***" if r["prediction_imbalance_flag"] else "no strong imbalance flagged"
        print(f"{r['train_dataset']:8s} -> {r['test_dataset']:8s}  "
              f"actual FAKE share={r['actual_fake_share']:.3f}  "
              f"predicted FAKE share={r['predicted_fake_share']:.3f}   -> {flag}")


if __name__ == "__main__":
    main()