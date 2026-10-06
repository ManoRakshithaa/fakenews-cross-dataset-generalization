"""
DistilBERT Fine-Tuning Pipeline — Cross-Dataset Fake-News Generalization
=============================================================================
Implements the same four-direction evaluation design as the TF-IDF +
Logistic Regression baseline, using distilbert-base-uncased fine-tuned for
binary classification. Reads ONLY from processed_final/ (the corrected,
final cleaned datasets) -- never from processed/ (v1) or raw_data/.

THIS SCRIPT IS RUN IN TWO MODES, ON PURPOSE:

  --mode estimate   Loads the REAL training data for the requested
                     direction, runs a small number of real training steps
                     (default 20) on your actual hardware, measures wall
                     time per step, and extrapolates a total-time estimate
                     for the full configured run. Does NOT save a model,
                     does NOT evaluate, does NOT touch the test/target
                     data at all. Use this FIRST to decide whether your
                     configured batch size / epochs / max_length is
                     practical on your hardware before committing to a
                     real run.

  --mode run         The real experiment: full fine-tuning on the source
                     training data, then a single evaluation pass on the
                     held-out or external test data, then save everything
                     (metrics, predictions, confusion matrix, model
                     checkpoint, configs, package versions).

LEAKAGE PREVENTION (how this script guarantees it, not just claims it):
  1. The tokenizer and the pretrained distilbert-base-uncased weights are
     both FIXED, downloaded artifacts -- neither is fit, tuned, or adapted
     to any dataset before fine-tuning starts. A tokenizer has a fixed
     vocabulary; it is not "trained" on your text here.
  2. Training (`trainer.train()`) is called with ONLY the source-dataset
     training split as `train_dataset`. The target/test dataframe is never
     passed to `.train()`, never touched until the SINGLE final
     `trainer.predict()` call after training has fully finished.
  3. `eval_strategy="no"` during training -- there is no periodic
     evaluation against the test set during fine-tuning, so nothing about
     training (early stopping, checkpoint selection, logging-driven
     decisions) can be influenced by target-dataset performance. This
     applies identically to in-domain and cross-dataset runs, so the
     methodology is the same across all four directions.
  4. For in-domain runs, `assert_no_row_overlap()` checks train/test
     indices are disjoint after the stratified split. For cross-dataset
     runs, the two dataframes are loaded from different files entirely, so
     overlap is structurally impossible (and was already independently
     verified to be zero exact-text-hash overlap by final_clean_and_audit.py).
  5. No hyperparameter search of any kind is performed. One fixed
     configuration is used, decided BEFORE looking at any result (see
     --mode estimate above), so no choice here is informed by test-set
     performance.

Does NOT modify or overwrite results_final/ or diagnostics_final/ (the
TF-IDF baseline's outputs). Writes to distilbert_results/ and
distilbert_diagnostics/ instead.

Run (first):  python3 train_distilbert.py --direction WELFake_to_WELFake --mode estimate
Run (second): python3 train_distilbert.py --direction WELFake_to_WELFake --mode run
"""

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("./processed_final")
RESULTS_DIR = Path("./distilbert_results")
DIAGNOSTICS_DIR = Path("./distilbert_diagnostics")
PLOTS_DIR = DIAGNOSTICS_DIR / "confusion_matrix_plots"
PREDICTIONS_DIR = DIAGNOSTICS_DIR / "predictions"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

MODEL_NAME = "distilbert-base-uncased"
RANDOM_STATE = 42
TEST_SIZE = 0.2
LABEL_NAMES = {0: "REAL", 1: "FAKE"}

# Expected final row counts, same sanity check used by the TF-IDF rerun --
# printed/warned, never silently trusted.
EXPECTED_SIZES = {"WELFake": 23903, "ISOT": 38478}

# ============================================================
# DEFAULT TRAINING CONFIG -- a single fixed, documented choice.
# Not tuned against any result. Override via CLI flags if the
# --mode estimate run shows these are impractical on your hardware.
# ============================================================
DEFAULT_CONFIG = {
    "max_length": 256,          # see module docstring section on truncation below
    "batch_size": 8,            # conservative default for CPU/limited-memory hardware
    "eval_batch_size": 16,
    "gradient_accumulation_steps": 2,   # effective batch size = batch_size * this = 16
    "num_train_epochs": 2,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "seed": RANDOM_STATE,
}

# WHY max_length=256: article bodies in this dataset are long (WELFake REAL
# mean ~891 words, well beyond DistilBERT's 512-token hard limit even before
# subword splitting is considered). 256 is a practical middle ground that
# keeps per-step compute bounded on modest hardware. This script computes
# and reports the ACTUAL percentage of examples this truncates (see
# compute_truncation_stats) rather than assuming 256 is fine -- if that
# percentage turns out high, that is a reportable limitation, not something
# to silently accept.


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_training_arguments(**desired_kwargs):
    """
    Build a transformers.TrainingArguments object defensively: different
    transformers versions have renamed/removed some constructor parameters
    over time (e.g. warmup_ratio, eval_strategy have both had naming
    changes across versions). Rather than guessing at the exact current
    name, this inspects the INSTALLED version's actual accepted parameters
    and only passes what it accepts.

    Anything requested but not accepted by this version is NOT silently
    dropped -- it's printed loudly and returned alongside the object, so
    it can be recorded in the run's saved config as an explicit, visible
    deviation from the intended settings (reproducibility requires this to
    be documented, not hidden).
    """
    import inspect
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    valid_params = set(sig.parameters.keys())

    accepted = {k: v for k, v in desired_kwargs.items() if k in valid_params}
    dropped = {k: v for k, v in desired_kwargs.items() if k not in valid_params}

    # Known, confirmed rename (not a guess): transformers v5 removed
    # warmup_ratio in favor of warmup_steps, which v5 ALSO accepts as a
    # float < 1.0 meaning "fraction of total steps" -- identical semantics
    # to warmup_ratio. If warmup_ratio was dropped and warmup_steps is
    # available and not already set, translate it instead of losing it.
    if "warmup_ratio" in dropped and "warmup_steps" in valid_params and "warmup_steps" not in accepted:
        accepted["warmup_steps"] = dropped.pop("warmup_ratio")
        print(f"[INFO] This transformers version removed warmup_ratio; translated to "
              f"warmup_steps={accepted['warmup_steps']} (v5's warmup_steps accepts a float < 1.0 "
              f"as a fraction of total steps, equivalent to the old warmup_ratio semantics).")

    if dropped:
        print(f"[WARNING] This installed transformers version's TrainingArguments does not "
              f"accept these parameter(s): {list(dropped.keys())}. They were NOT applied "
              f"(values that would have been used: {dropped}). This is recorded in the saved "
              f"run config as 'training_arguments_dropped' -- check if an equivalent parameter "
              f"exists under a different name in your installed version before treating a run "
              f"as final.")

    return TrainingArguments(**accepted), dropped


def detect_device() -> str:
    """CUDA > Apple Silicon MPS > CPU, in that order. Returns a string used
    both for reporting and for actually placing the model."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "unknown (torch not installed)"


# ============================================================
# DATA LOADING -- mirrors baseline_tfidf_logreg_final.py's conventions,
# reads from processed_final/ only.
# ============================================================

def load_clean(dataset_name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{dataset_name}_final_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run final_clean_and_audit.py first."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    n = len(df)
    expected = EXPECTED_SIZES.get(dataset_name)
    if expected is not None and n != expected:
        print(f"[WARNING] Loaded {path} has {n} rows, expected {expected}. "
              f"Double check this is the data you think it is.")
    else:
        print(f"Confirmed: loaded {path} ({n} rows, matches expected final size).")
    return df


def build_text_field(df: pd.DataFrame) -> pd.DataFrame:
    """title + body only, per this experiment's scope (no title-only /
    body-only variants yet)."""
    df = df.copy()
    title = df["title"].fillna("")
    text = df["text"].fillna("")
    df["_model_input"] = title + " " + text
    return df


def split_for_direction(train_dataset: str, test_dataset: str):
    """
    Returns (train_df, test_df, split_description).
    In-domain (train_dataset == test_dataset): 80/20 stratified split of
    that one dataset, random_state=42 -- IDENTICAL call signature to the
    TF-IDF baseline's split, for apples-to-apples comparability.
    Cross-dataset: full source dataset for training, full target dataset
    for evaluation, no split.
    """
    from sklearn.model_selection import train_test_split

    if train_dataset == test_dataset:
        full = build_text_field(load_clean(train_dataset))
        train_df, test_df = train_test_split(
            full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=full["label"],
        )
        desc = f"20% stratified held-out split of {train_dataset} (random_state={RANDOM_STATE})"
        assert_no_row_overlap(train_df, test_df)
    else:
        train_df = build_text_field(load_clean(train_dataset))
        test_df = build_text_field(load_clean(test_dataset))
        desc = f"trained on 100% of {train_dataset}, evaluated on 100% of {test_dataset} (external, never seen during training)"

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), desc


def assert_no_row_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Leakage check for in-domain splits: train/test row indices (from the
    original loaded dataframe, before reset_index) must be completely
    disjoint. Raises if not -- this is a hard stop, not a warning, since a
    leakage bug here would invalidate the whole experiment."""
    overlap = set(train_df.index) & set(test_df.index)
    if overlap:
        raise RuntimeError(
            f"LEAKAGE DETECTED: {len(overlap)} row(s) appear in both train and "
            f"test splits. Refusing to proceed. This should be structurally "
            f"impossible with sklearn's train_test_split -- investigate immediately."
        )
    print(f"Leakage check passed: train ({len(train_df)} rows) and test ({len(test_df)} rows) "
          f"splits are fully disjoint.")


# ============================================================
# TOKENIZATION + TRUNCATION REPORTING
# ============================================================

def compute_truncation_stats(tokenizer, texts: pd.Series, max_length: int) -> dict:
    """Tokenizes WITHOUT truncation to find each example's true token
    count, then reports what fraction would be cut off at max_length.
    This is a real, computed number -- not assumed to be negligible."""
    lengths = []
    for t in texts:
        ids = tokenizer(t, truncation=False, add_special_tokens=True)["input_ids"]
        lengths.append(len(ids))
    lengths = np.array(lengths)
    n_truncated = int((lengths > max_length).sum())
    return {
        "n_examples": len(lengths),
        "max_length_used": max_length,
        "n_truncated": n_truncated,
        "pct_truncated": round(100 * n_truncated / len(lengths), 2) if len(lengths) else 0.0,
        "mean_token_length": float(lengths.mean()),
        "median_token_length": float(np.median(lengths)),
        "p90_token_length": float(np.percentile(lengths, 90)),
        "max_token_length": int(lengths.max()),
    }


# ============================================================
# METRICS -- pure numpy/sklearn, no torch dependency. Used both by the
# Trainer's compute_metrics callback AND testable standalone.
# ============================================================

def compute_metrics_from_arrays(y_true, y_pred, y_prob_fake=None) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score,
    )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision_fake = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_fake = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_fake = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=["REAL(0)", "FAKE(1)"], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    roc_auc = None
    if y_prob_fake is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob_fake))
        except ValueError:
            roc_auc = None  # e.g. only one class present

    actual_counts = pd.Series(y_true).value_counts().to_dict()
    predicted_counts = pd.Series(y_pred).value_counts().to_dict()

    return {
        "accuracy": float(acc),
        "precision_fake": float(precision_fake),
        "recall_fake": float(recall_fake),
        "f1_fake": float(f1_fake),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc": roc_auc,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_label_order": ["REAL(0)", "FAKE(1)"],
        "actual_class_counts": {LABEL_NAMES[k]: int(actual_counts.get(k, 0)) for k in [0, 1]},
        "predicted_class_counts": {LABEL_NAMES[k]: int(predicted_counts.get(k, 0)) for k in [0, 1]},
        "n_examples": int(len(y_true)),
    }


def plot_confusion_matrix(cm, title: str, out_path: Path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=["REAL (0)", "FAKE (1)"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# ENVIRONMENT / CONFIG SNAPSHOT -- for reproducibility
# ============================================================

def collect_environment_info(device: str) -> dict:
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "device_used": device,
    }
    for pkg in ["torch", "transformers", "sklearn", "numpy", "pandas"]:
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{pkg}_version"] = "not installed"
    return info


# ============================================================
# TORCH DATASET WRAPPER
# ============================================================

def build_torch_dataset(encodings, labels):
    import torch

    class FakeNewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(int(self.labels[idx]))
            return item

    return FakeNewsDataset(encodings, labels)


# ============================================================
# MODE: estimate
# ============================================================

def run_time_estimate(direction: str, config: dict, n_probe_steps: int = 20):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

    train_name, test_name = direction.split("_to_")
    device = detect_device()
    print(f"Device detected: {device}")

    set_all_seeds(config["seed"])

    train_df, test_df, split_desc = split_for_direction(train_name, test_name)
    print(f"[estimate mode] Split: {split_desc}")
    print(f"[estimate mode] Full training set size: {len(train_df)} (used for realistic per-step timing)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device if device in ("cuda", "mps") else "cpu")

    encodings = tokenizer(
        train_df["_model_input"].tolist(),
        truncation=True, padding=True, max_length=config["max_length"],
    )
    train_torch_ds = build_torch_dataset(encodings, train_df["label"].tolist())

    args, dropped_args = build_training_arguments(
        output_dir=str(RESULTS_DIR / "estimate_tmp"),
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        max_steps=n_probe_steps,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        seed=config["seed"],
        fp16=(device == "cuda"),
        logging_steps=max(1, n_probe_steps // 4),
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
    )
    # No eval_dataset is passed to Trainer here (or anywhere in this script)
    # -- this is a structural leakage guard independent of whatever
    # eval_strategy parameter name this transformers version uses: with no
    # eval_dataset configured, periodic evaluation cannot happen regardless.
    trainer = Trainer(model=model, args=args, train_dataset=train_torch_ds)

    print(f"\nRunning {n_probe_steps} real training steps to measure timing...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    time_per_step = elapsed / n_probe_steps

    steps_per_epoch = len(train_torch_ds) / (config["batch_size"] * config["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * config["num_train_epochs"]
    estimated_total_seconds = total_steps * time_per_step

    result = {
        "direction": direction,
        "device": device,
        "probe_steps_run": n_probe_steps,
        "measured_time_per_step_seconds": round(time_per_step, 3),
        "train_rows": len(train_df),
        "steps_per_epoch": round(steps_per_epoch, 1),
        "configured_epochs": config["num_train_epochs"],
        "estimated_total_steps": round(total_steps, 1),
        "estimated_total_training_time_seconds": round(estimated_total_seconds, 1),
        "estimated_total_training_time_minutes": round(estimated_total_seconds / 60, 1),
        "estimated_total_training_time_hours": round(estimated_total_seconds / 3600, 2),
        "config_used": config,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{direction}_time_estimate.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 70}\nTIME ESTIMATE\n{'=' * 70}")
    print(f"Device: {device}")
    print(f"Measured: {time_per_step:.3f} sec/step over {n_probe_steps} real steps")
    print(f"Full run: ~{steps_per_epoch:.0f} steps/epoch x {config['num_train_epochs']} epochs "
          f"= ~{total_steps:.0f} steps")
    print(f"ESTIMATED total training time: {estimated_total_seconds/60:.1f} minutes "
          f"({estimated_total_seconds/3600:.2f} hours)")
    print(f"(This does not include tokenization time, the final evaluation pass, or model saving, "
          f"which add a smaller, roughly constant amount on top.)")
    print(f"\nSaved: {out_path}")
    print("\nNo model or results were saved from this probe run -- this was timing-only. "
          "Nothing here touched the test/target dataset.")


# ============================================================
# MODE: run
# ============================================================

def run_experiment(direction: str, config: dict):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

    train_name, test_name = direction.split("_to_")
    device = detect_device()
    print(f"Device detected: {device}")

    set_all_seeds(config["seed"])

    train_df, test_df, split_desc = split_for_direction(train_name, test_name)
    print(f"Split: {split_desc}")
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nComputing truncation statistics (train set)...")
    train_trunc = compute_truncation_stats(tokenizer, train_df["_model_input"], config["max_length"])
    print("Computing truncation statistics (test set)...")
    test_trunc = compute_truncation_stats(tokenizer, test_df["_model_input"], config["max_length"])
    print(f"Train: {train_trunc['pct_truncated']}% truncated at max_length={config['max_length']} "
          f"(mean token length {train_trunc['mean_token_length']:.1f})")
    print(f"Test:  {test_trunc['pct_truncated']}% truncated at max_length={config['max_length']} "
          f"(mean token length {test_trunc['mean_token_length']:.1f})")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device if device in ("cuda", "mps") else "cpu")

    train_encodings = tokenizer(
        train_df["_model_input"].tolist(), truncation=True, padding=True, max_length=config["max_length"],
    )
    test_encodings = tokenizer(
        test_df["_model_input"].tolist(), truncation=True, padding=True, max_length=config["max_length"],
    )
    train_torch_ds = build_torch_dataset(train_encodings, train_df["label"].tolist())
    test_torch_ds = build_torch_dataset(test_encodings, test_df["label"].tolist())

    RESULTS_DIR.mkdir(exist_ok=True)
    checkpoint_dir = CHECKPOINTS_DIR / direction
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    args, dropped_args = build_training_arguments(
        output_dir=str(RESULTS_DIR / "train_tmp" / direction),
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        seed=config["seed"],
        fp16=(device == "cuda"),
        logging_steps=50,
        save_strategy="no",     # we save the FINAL model explicitly below, not per-epoch checkpoints
        eval_strategy="no",     # NO periodic eval against test data during training -- see module docstring
        report_to=[],
    )
    # No eval_dataset passed to Trainer -- structural leakage guard, see note above.
    trainer = Trainer(model=model, args=args, train_dataset=train_torch_ds)

    print(f"\n{'=' * 70}\nTRAINING\n{'=' * 70}")
    start = time.time()
    trainer.train()
    train_elapsed = time.time() - start
    print(f"Training finished in {train_elapsed/60:.1f} minutes.")

    print(f"\n{'=' * 70}\nEVALUATION (single pass, test data never touched until now)\n{'=' * 70}")
    pred_output = trainer.predict(test_torch_ds)
    logits = pred_output.predictions
    y_true = test_df["label"].values
    y_pred = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_prob_fake = probs[:, 1]

    metrics = compute_metrics_from_arrays(y_true, y_pred, y_prob_fake)
    metrics["direction"] = direction
    metrics["train_dataset"] = train_name
    metrics["test_dataset"] = test_name
    metrics["split_description"] = split_desc
    metrics["train_rows"] = len(train_df)
    metrics["test_rows"] = len(test_df)
    metrics["training_time_minutes"] = round(train_elapsed / 60, 2)
    metrics["model"] = MODEL_NAME

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (FAKE): {metrics['precision_fake']:.4f}  Recall (FAKE): {metrics['recall_fake']:.4f}  "
          f"F1 (FAKE): {metrics['f1_fake']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}   Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print(f"Confusion matrix (rows=actual [REAL,FAKE], cols=predicted [REAL,FAKE]):")
    print(np.array(metrics["confusion_matrix"]))
    print(f"Actual class counts:    {metrics['actual_class_counts']}")
    print(f"Predicted class counts: {metrics['predicted_class_counts']}")

    # ---------------- save everything ----------------
    DIAGNOSTICS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_DIR / f"{direction}_distilbert_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    preds_df = pd.DataFrame({
        "true_label": y_true,
        "true_label_name": [LABEL_NAMES[v] for v in y_true],
        "predicted_label": y_pred,
        "predicted_label_name": [LABEL_NAMES[v] for v in y_pred],
        "predicted_prob_fake": y_prob_fake,
        "correct": (y_true == y_pred),
        "title": test_df["title"].values,
        "text": test_df["text"].values,
    })
    preds_path = PREDICTIONS_DIR / f"{direction}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    plot_path = PLOTS_DIR / f"{direction}_confusion_matrix.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        f"{train_name} -> {test_name} (DistilBERT)\nAccuracy={metrics['accuracy']:.3f}  Macro F1={metrics['macro_f1']:.3f}",
        plot_path,
    )

    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    run_config = {
        "direction": direction,
        "model_name": MODEL_NAME,
        "config": config,
        "split_description": split_desc,
        "train_truncation_stats": train_trunc,
        "test_truncation_stats": test_trunc,
        "environment": collect_environment_info(device),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "training_arguments_dropped": dropped_args,  # non-empty means some intended
                                                       # settings (e.g. warmup_ratio) could
                                                       # not be applied on this transformers
                                                       # version -- see console warning
    }
    config_path = RESULTS_DIR / f"{direction}_run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nSaved metrics:     {metrics_path}")
    print(f"Saved predictions: {preds_path}")
    print(f"Saved plot:        {plot_path}")
    print(f"Saved run config:  {config_path}")
    print(f"Saved checkpoint:  {checkpoint_dir} (NOTE: this is a full model checkpoint, "
          f"typically 200-300MB+ -- do NOT commit this to git; add distilbert_results/checkpoints/ "
          f"to .gitignore)")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="DistilBERT fine-tuning for cross-dataset fake-news generalization")
    parser.add_argument("--direction", required=True,
                         choices=["WELFake_to_WELFake", "WELFake_to_ISOT", "ISOT_to_ISOT", "ISOT_to_WELFake"])
    parser.add_argument("--mode", required=True, choices=["estimate", "run"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--eval_batch_size", type=int, default=DEFAULT_CONFIG["eval_batch_size"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_train_epochs"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--probe_steps", type=int, default=20, help="only used in --mode estimate")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update({
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
    })

    if args.mode == "estimate":
        run_time_estimate(args.direction, config, n_probe_steps=args.probe_steps)
    else:
        run_experiment(args.direction, config)


if __name__ == "__main__":
    main()