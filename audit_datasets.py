"""
Dataset Audit Pipeline — WELFake & ISOT (Fake News Detection)
================================================================

PURPOSE
-------
This script performs a *read-only* audit of the WELFake and ISOT datasets
before any modeling work begins. It never modifies or overwrites the raw
files — it only reads them and writes NEW files into an `audit_output/`
folder.

WHY THIS EXISTS (research rationale)
-------------------------------------
Before comparing in-domain vs. cross-dataset generalization
(WELFake -> WELFake, WELFake -> ISOT, ISOT -> ISOT, ISOT -> WELFake), you
need to know, with evidence (not assumption):
  - What each dataset actually contains (columns, label meaning, size)
  - Whether there is missing/duplicated/degenerate data that would bias
    a model's apparent accuracy
  - Whether WELFake and ISOT *overlap* with each other. This matters a lot
    for cross-dataset generalization claims: ISOT is a known component of
    several "combined" fake-news datasets circulating online, and if
    WELFake secretly contains ISOT articles (or vice versa), then a
    "cross-dataset" WELFake -> ISOT experiment would partly be testing on
    training data in disguise, and any generalization number would be
    inflated/meaningless.

HOW TO USE THIS SCRIPT
-----------------------
1. Edit the CONFIG section below: point RAW_DATA_DIR at the folder that
   contains your WELFake and ISOT files.
2. Run: `python audit_datasets.py --discover`
   This ONLY lists files it finds and their columns/shape. It does not
   assume anything about which file is which dataset. Use this output to
   fill in the FILE PATHS in the CONFIG section.
3. Once paths are filled in, run: `python audit_datasets.py`
   This runs the full audit and writes results to `audit_output/`.

IMPORTANT — COLUMN MAPPING IS EXPLICIT, NOT GUESSED
-----------------------------------------------------
Per your research rules, this script does NOT silently assume which
column is the text, title, or label. You must confirm/set the
`COLUMN_MAP` for each dataset after running `--discover` and inspecting
the actual column names printed. Sensible *hints* (not assumptions) for
the commonly-distributed versions of WELFake and ISOT are included as
comments, but the script will refuse to run the full audit on a dataset
until you've explicitly confirmed its column map (see CONFIG section).

ISOT-SPECIFIC NOTE (flagged, not assumed)
-------------------------------------------
The commonly-distributed ISOT dataset ships as TWO files: True.csv and
Fake.csv, with NO label column — the label is implied entirely by which
file a row came from. If that's what you have, this script's discovery
step will show you two separate files with identical columns and no
label column, and you'll set `ISOT_TRUE_FILE` / `ISOT_FAKE_FILE`
separately in CONFIG rather than a single `ISOT_FILE`. Do not assume
this is your layout until `--discover` confirms it.

WELFake label-convention note: your own project notes mention a past
label-convention mismatch (0/1 meaning flipped) that required a retrain.
This script explicitly prints a sample of rows per label value so you
can visually confirm which integer means "fake" vs "real" for THIS file,
rather than assuming it matches any prior project.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# CONFIG — EDIT THIS SECTION
# ============================================================

# Folder containing your raw dataset files. Change this to your local path.
RAW_DATA_DIR = "./raw_data"

# Where audit outputs get written. Created if it doesn't exist.
AUDIT_OUTPUT_DIR = "./audit_output"

# --- WELFake ---
# Fill this in after running --discover. Example: "./raw_data/WELFake_Dataset.csv"
WELFAKE_FILE = "./raw_data/raw_data/WELFake_Dataset.csv"

# Column map for WELFake. Set these to the ACTUAL column names in your file
# after inspecting --discover output. Do not trust the placeholders below —
# they are only common-convention hints, not verified facts about your file.
WELFAKE_COLUMN_MAP = {
    "title": "title",
    "text": "text",
    "label": "label",
}

# --- ISOT ---
# The commonly distributed ISOT dataset is TWO files (True.csv, Fake.csv)
# with no label column. If your copy is different (single file with a
# label column), set ISOT_FILE instead and fill ISOT_COLUMN_MAP.
ISOT_TRUE_FILE = "./raw_data/raw_data/True.csv"
ISOT_FAKE_FILE = "./raw_data/raw_data/Fake.csv"
ISOT_FILE = None

ISOT_COLUMN_MAP = {
    "title": "title",
    "text": "text",
    "label": None,
}

# Threshold below which an article is flagged "very short" (word count).
# 10 words is a common floor for "this is probably not a real article
# body" but it is a judgment call — flagged here explicitly, adjust as
# you see fit for your paper's methodology section.
SHORT_ARTICLE_WORD_THRESHOLD = 10

# Near-duplicate cosine similarity threshold (see method explanation in
# the module docstring / README output). 0.9 is a common conservative
# threshold in dedup literature; not a magic number, and is reported in
# the output so it's easy to change and re-run.
NEAR_DUP_COSINE_THRESHOLD = 0.9

# Safety cap: near-duplicate detection is O(n^2) within blocks. If a
# block is larger than this, we skip full pairwise comparison for it and
# report it as "too large to compare exhaustively" rather than silently
# taking hours. See methodology section for the blocking strategy.
MAX_BLOCK_SIZE_FOR_PAIRWISE = 3000


# ============================================================
# SECTION 1 — DATASET DISCOVERY
# ============================================================
# Goal: never assume a filename or schema. Just look at what's on disk
# and report it, so the user (you) can confirm the CONFIG section above.

def discover_files(raw_dir: str):
    """
    Scan raw_dir for candidate dataset files (.csv, .json, .jsonl, .tsv)
    and print their shape + column names + first row preview, WITHOUT
    assuming which file is WELFake vs ISOT vs True vs Fake.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        print(f"[ERROR] RAW_DATA_DIR does not exist: {raw_dir.resolve()}")
        print("Edit RAW_DATA_DIR at the top of this script.")
        return

    candidates = sorted(
        [p for p in raw_dir.rglob("*") if p.suffix.lower() in (".csv", ".json", ".jsonl", ".tsv")]
    )

    if not candidates:
        print(f"[WARNING] No .csv/.json/.jsonl/.tsv files found under {raw_dir.resolve()}")
        return

    print(f"Found {len(candidates)} candidate file(s) under {raw_dir.resolve()}:\n")
    for path in candidates:
        print("=" * 70)
        print(f"FILE: {path}")
        print(f"SIZE: {path.stat().st_size / 1e6:.2f} MB")
        try:
            df = _load_any(path, nrows=5)
            print(f"COLUMNS: {list(df.columns)}")
            print("FIRST 2 ROWS (truncated to 120 chars per cell):")
            with pd.option_context("display.max_colwidth", 120):
                print(df.head(2))
        except Exception as e:
            print(f"[Could not preview this file: {e}]")
        print()

    print("=" * 70)
    print("NEXT STEP: open this script and fill in the CONFIG section")
    print("(WELFAKE_FILE / ISOT_TRUE_FILE / ISOT_FAKE_FILE / column maps)")
    print("using the column names actually printed above. Do not guess.")


def _load_any(path: Path, nrows=None) -> pd.DataFrame:
    """Load a csv/tsv/json/jsonl file. Format is inferred from extension,
    not assumed to be csv by default."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    elif suffix == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows)
    elif suffix == ".jsonl":
        return pd.read_json(path, lines=True, nrows=nrows) if nrows else pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
        return df.head(nrows) if nrows else df
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")


# ============================================================
# SECTION 2 — LOADING WITH EXPLICIT COLUMN MAPS
# ============================================================

def load_and_standardize(path: str, column_map: dict, dataset_name: str,
                          forced_label: int = None) -> pd.DataFrame:
    """
    Load a raw file and rename its columns to standardized names
    (title, text, label) using the user-confirmed column_map.

    forced_label: if given (e.g. 0 or 1), a 'label' column is created
    with this constant value for every row. Used for ISOT's True.csv /
    Fake.csv, where the label comes from the filename, not a column.

    Raises a clear error (does not silently guess) if the map is
    incomplete for a required field.
    """
    path = Path(path)
    df = _load_any(path)
    df["_source_file"] = str(path.name)
    df["_source_dataset"] = dataset_name

    missing_map_entries = [
        k for k, v in column_map.items()
        if v is None and not (k == "label" and forced_label is not None)
    ]
    if missing_map_entries:
        raise ValueError(
            f"[{dataset_name}] Column map is incomplete for: {missing_map_entries}. "
            f"Run --discover, inspect actual columns, and fill in CONFIG before "
            f"running the full audit. Refusing to guess."
        )

    rename_dict = {}
    for standard_name, actual_col in column_map.items():
        if actual_col is not None:
            if actual_col not in df.columns:
                raise ValueError(
                    f"[{dataset_name}] Column '{actual_col}' (mapped to '{standard_name}') "
                    f"not found in {path.name}. Actual columns: {list(df.columns)}"
                )
            rename_dict[actual_col] = standard_name

    df = df.rename(columns=rename_dict)

    if forced_label is not None:
        df["label"] = forced_label

    keep_cols = [c for c in ["title", "text", "label", "_source_file", "_source_dataset"] if c in df.columns]
    return df[keep_cols]


# ============================================================
# SECTION 3 — PER-DATASET AUDIT
# ============================================================

def normalize_text(s) -> str:
    """Lowercase + collapse whitespace. Used ONLY for hashing/dedup
    comparisons, never to overwrite the original text column."""
    if pd.isna(s):
        return ""
    return " ".join(str(s).lower().split())


def hash_text(s) -> str:
    """SHA-256 hash of normalized text. Deterministic, reproducible,
    used for exact-duplicate detection (within and across datasets)."""
    return hashlib.sha256(normalize_text(s).encode("utf-8")).hexdigest()


def audit_dataset(df: pd.DataFrame, name: str, output_dir: Path) -> dict:
    """
    Run the full battery of checks the user asked for on a single
    standardized dataframe (columns: title, text, label, ...).
    Returns a dict of results and writes supporting CSVs to output_dir.
    """
    report = {"dataset": name}
    n_rows = len(df)
    report["n_rows"] = n_rows
    report["columns"] = list(df.columns)

    # --- Missing values ---
    missing = df.isna().sum().to_dict()
    missing_pct = (df.isna().mean() * 100).round(2).to_dict()
    report["missing_values"] = missing
    report["missing_pct"] = missing_pct

    # --- Exact duplicate rows (all columns identical) ---
    report["n_duplicate_rows_all_cols"] = int(df.duplicated().sum())

    # --- Duplicate article texts (based on normalized text hash) ---
    if "text" in df.columns:
        df = df.copy()
        df["_text_hash"] = df["text"].apply(hash_text)
        dup_text_count = int(df["_text_hash"].duplicated().sum())
        report["n_duplicate_texts"] = dup_text_count
        dup_text_rows = df[df.duplicated("_text_hash", keep=False)]
        dup_text_rows.to_csv(output_dir / f"{name}_duplicate_texts.csv", index=False)
    else:
        report["n_duplicate_texts"] = None

    # --- Class distribution ---
    if "label" in df.columns:
        vc = df["label"].value_counts(dropna=False)
        report["class_distribution"] = vc.to_dict()
        report["unique_labels"] = sorted(df["label"].dropna().unique().tolist())

        # Print a small sample per label so the user can visually confirm
        # what each label value actually means for THIS file — no
        # assumption carried over from any other project.
        print(f"\n[{name}] Sample rows per label value (confirm meaning yourself):")
        for lbl in report["unique_labels"]:
            sample = df[df["label"] == lbl].head(1)
            if "title" in sample.columns and len(sample) > 0:
                print(f"  label={lbl!r} sample title: {str(sample['title'].values[0])[:100]!r}")
    else:
        report["class_distribution"] = None
        report["unique_labels"] = None

    # --- Empty / very short articles ---
    if "text" in df.columns:
        word_counts = df["text"].apply(lambda s: len(normalize_text(s).split()))
        df["_word_count"] = word_counts
        report["n_empty_articles"] = int((word_counts == 0).sum())
        report["n_short_articles"] = int(
            (word_counts < SHORT_ARTICLE_WORD_THRESHOLD).sum()
        )
        report["text_length_stats_words"] = {
            "mean": float(word_counts.mean()),
            "median": float(word_counts.median()),
            "std": float(word_counts.std()),
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
            "p1": float(word_counts.quantile(0.01)),
            "p99": float(word_counts.quantile(0.99)),
        }
        short_rows = df[word_counts < SHORT_ARTICLE_WORD_THRESHOLD]
        short_rows.drop(columns=["_text_hash"], errors="ignore").to_csv(
            output_dir / f"{name}_short_or_empty_articles.csv", index=False
        )
    else:
        report["n_empty_articles"] = None
        report["n_short_articles"] = None
        report["text_length_stats_words"] = None

    # --- Title length stats ---
    if "title" in df.columns:
        title_word_counts = df["title"].apply(lambda s: len(normalize_text(s).split()))
        report["title_length_stats_words"] = {
            "mean": float(title_word_counts.mean()),
            "median": float(title_word_counts.median()),
            "std": float(title_word_counts.std()),
            "min": int(title_word_counts.min()),
            "max": int(title_word_counts.max()),
        }
        report["n_empty_titles"] = int((title_word_counts == 0).sum())
    else:
        report["title_length_stats_words"] = None
        report["n_empty_titles"] = None

    # --- Obvious formatting problems (flagged, not auto-fixed) ---
    formatting_flags = {}
    if "text" in df.columns:
        # Rows where text looks like it's just a URL / boilerplate wire-service tag
        # (e.g. "WASHINGTON (Reuters) - ") is a KNOWN ISOT quirk that can leak
        # source identity into the model. We flag prevalence, we do not strip it.
        reuters_tag_count = int(df["text"].astype(str).str.contains(
            r"\(Reuters\)", regex=True, na=False
        ).sum())
        formatting_flags["rows_containing_(Reuters)_tag"] = reuters_tag_count

        all_caps_title_count = None
        if "title" in df.columns:
            all_caps_title_count = int(df["title"].fillna("").astype(str).apply(
                lambda s: s.isupper() and len(s) > 3
            ).sum())
        formatting_flags["all_caps_titles"] = all_caps_title_count

        html_entity_count = int(df["text"].astype(str).str.contains(
            r"&amp;|&lt;|&gt;|<[a-zA-Z]+>", regex=True, na=False
        ).sum())
        formatting_flags["rows_with_leftover_html_entities_or_tags"] = html_entity_count
    report["formatting_flags"] = formatting_flags

    return report


# ============================================================
# SECTION 4 — CROSS-DATASET OVERLAP (WELFake <-> ISOT)
# ============================================================

def cross_dataset_overlap(df_a: pd.DataFrame, name_a: str,
                           df_b: pd.DataFrame, name_b: str,
                           output_dir: Path) -> dict:
    """
    Exact-overlap check between two datasets using normalized-text
    and normalized-title SHA-256 hashes. This is the most important
    leakage check for the planned cross-dataset experiment: if the
    same article appears in both WELFake and ISOT, then a
    "cross-dataset" train/test split isn't actually testing
    generalization for those rows.
    """
    result = {"dataset_a": name_a, "dataset_b": name_b}

    a = df_a.copy()
    b = df_b.copy()

    if "text" in a.columns and "text" in b.columns:
        a["_text_hash"] = a["text"].apply(hash_text)
        b["_text_hash"] = b["text"].apply(hash_text)
        overlap_hashes = set(a["_text_hash"]) & set(b["_text_hash"])
        result["n_exact_text_overlaps"] = len(overlap_hashes)

        overlap_rows_a = a[a["_text_hash"].isin(overlap_hashes)]
        overlap_rows_b = b[b["_text_hash"].isin(overlap_hashes)]
        pd.concat([overlap_rows_a, overlap_rows_b]).to_csv(
            output_dir / f"exact_text_overlap_{name_a}_vs_{name_b}.csv", index=False
        )
    else:
        result["n_exact_text_overlaps"] = None

    if "title" in a.columns and "title" in b.columns:
        a["_title_hash"] = a["title"].apply(hash_text)
        b["_title_hash"] = b["title"].apply(hash_text)
        title_overlap_hashes = set(a["_title_hash"]) & set(b["_title_hash"])
        result["n_exact_title_overlaps"] = len(title_overlap_hashes)
    else:
        result["n_exact_title_overlaps"] = None

    return result


def near_duplicate_check(df_a: pd.DataFrame, name_a: str,
                          df_b: pd.DataFrame, name_b: str,
                          output_dir: Path,
                          threshold: float = NEAR_DUP_COSINE_THRESHOLD) -> dict:
    """
    Near-duplicate detection between two datasets using TF-IDF (word
    n-grams) + cosine similarity, computed WITHIN BLOCKS to keep it
    tractable.

    METHOD (explained per your request, before running):
      1. Block candidate pairs by the first 8 normalized words of the
         text ("prefix blocking"). Two articles can only be compared if
         they share the same prefix block. This is a simple, fully
         transparent heuristic — it will MISS near-duplicates that open
         differently (e.g. one has an extra sentence prepended), so it is
         a conservative/partial check, not exhaustive. This is stated
         explicitly rather than implied to be complete.
      2. Within each block (skipped if the block exceeds
         MAX_BLOCK_SIZE_FOR_PAIRWISE, reported separately), compute
         TF-IDF vectors (word 1-2 grams) and pairwise cosine similarity.
      3. Pairs with cosine similarity >= threshold are reported as
         near-duplicates, along with the exact score, so you can inspect
         and adjust the threshold yourself.

    This is intentionally simple and explainable over something like
    embedding-based semantic similarity, because for a reproducibility-
    focused audit, TF-IDF cosine similarity is fully deterministic and
    does not depend on any pretrained model or its version.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    a = df_a[["title", "text"]].copy() if "text" in df_a.columns else None
    b = df_b[["title", "text"]].copy() if "text" in df_b.columns else None
    if a is None or b is None:
        return {"error": "text column missing in one or both datasets; skipped"}

    a["_norm"] = a["text"].apply(normalize_text)
    b["_norm"] = b["text"].apply(normalize_text)
    a["_prefix"] = a["_norm"].apply(lambda s: " ".join(s.split()[:8]))
    b["_prefix"] = b["_norm"].apply(lambda s: " ".join(s.split()[:8]))

    a["_orig_idx"] = a.index
    b["_orig_idx"] = b.index

    shared_prefixes = set(a["_prefix"]) & set(b["_prefix"])
    shared_prefixes.discard("")  # empty text should never be treated as a "match block"

    near_dup_pairs = []
    skipped_blocks = []

    for prefix in shared_prefixes:
        block_a = a[a["_prefix"] == prefix]
        block_b = b[b["_prefix"] == prefix]
        if len(block_a) * len(block_b) == 0:
            continue
        if max(len(block_a), len(block_b)) > MAX_BLOCK_SIZE_FOR_PAIRWISE:
            skipped_blocks.append({"prefix": prefix, "size_a": len(block_a), "size_b": len(block_b)})
            continue

        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        try:
            combined = pd.concat([block_a["_norm"], block_b["_norm"]])
            tfidf = vectorizer.fit_transform(combined)
        except ValueError:
            continue  # e.g. block is all-empty strings after stripping

        tfidf_a = tfidf[: len(block_a)]
        tfidf_b = tfidf[len(block_a):]
        sims = cosine_similarity(tfidf_a, tfidf_b)

        idx_a_list = block_a["_orig_idx"].tolist()
        idx_b_list = block_b["_orig_idx"].tolist()
        for i, row_i in enumerate(idx_a_list):
            for j, row_j in enumerate(idx_b_list):
                score = sims[i, j]
                if score >= threshold:
                    near_dup_pairs.append({
                        f"{name_a}_index": row_i,
                        f"{name_b}_index": row_j,
                        "cosine_similarity": float(score),
                        f"{name_a}_title": df_a.loc[row_i, "title"] if "title" in df_a.columns else None,
                        f"{name_b}_title": df_b.loc[row_j, "title"] if "title" in df_b.columns else None,
                    })

    pairs_df = pd.DataFrame(near_dup_pairs)
    pairs_df.to_csv(output_dir / f"near_duplicate_candidates_{name_a}_vs_{name_b}.csv", index=False)

    if skipped_blocks:
        pd.DataFrame(skipped_blocks).to_csv(
            output_dir / f"near_dup_skipped_blocks_{name_a}_vs_{name_b}.csv", index=False
        )

    return {
        "method": "prefix-blocked TF-IDF (1,2-gram) cosine similarity",
        "threshold": threshold,
        "n_near_duplicate_pairs_found": len(near_dup_pairs),
        "n_blocks_skipped_too_large": len(skipped_blocks),
        "note": (
            "This is a PARTIAL check limited to articles sharing an identical "
            "8-word normalized prefix. It will under-count near-duplicates that "
            "differ near the start of the article. Treat n_near_duplicate_pairs_found "
            "as a lower bound, not an exhaustive count."
        ),
    }


# ============================================================
# SECTION 5 — MAIN ORCHESTRATION
# ============================================================

def _json_safe(obj):
    """Recursively convert numpy/pandas types to plain Python for json.dump."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def print_report_table(report: dict):
    print(f"\n{'=' * 70}\nDATASET AUDIT: {report['dataset']}\n{'=' * 70}")
    print(f"Rows: {report['n_rows']}")
    print(f"Columns: {report['columns']}")
    print(f"Missing values (count): {report['missing_values']}")
    print(f"Duplicate rows (all cols identical): {report['n_duplicate_rows_all_cols']}")
    print(f"Duplicate article texts: {report['n_duplicate_texts']}")
    print(f"Class distribution: {report['class_distribution']}")
    print(f"Unique labels: {report['unique_labels']}")
    print(f"Empty articles: {report['n_empty_articles']}")
    print(f"Short articles (< {SHORT_ARTICLE_WORD_THRESHOLD} words): {report['n_short_articles']}")
    print(f"Text length stats (words): {report['text_length_stats_words']}")
    print(f"Title length stats (words): {report['title_length_stats_words']}")
    print(f"Empty titles: {report['n_empty_titles']}")
    print(f"Formatting flags: {report['formatting_flags']}")


def main():
    parser = argparse.ArgumentParser(description="WELFake / ISOT dataset audit pipeline")
    parser.add_argument("--discover", action="store_true",
                         help="Only scan RAW_DATA_DIR and print file/column info. Does not run the full audit.")
    args = parser.parse_args()

    output_dir = Path(AUDIT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.discover:
        discover_files(RAW_DATA_DIR)
        return

    all_reports = {}
    loaded = {}

    # --- Load WELFake ---
    if WELFAKE_FILE:
        try:
            welfake_df = load_and_standardize(WELFAKE_FILE, WELFAKE_COLUMN_MAP, "WELFake")
            loaded["WELFake"] = welfake_df
            report = audit_dataset(welfake_df, "WELFake", output_dir)
            all_reports["WELFake"] = report
            print_report_table(report)
        except Exception as e:
            print(f"[ERROR loading WELFake] {e}")
    else:
        print("[SKIP] WELFAKE_FILE not set in CONFIG. Run --discover first.")

    # --- Load ISOT ---
    if ISOT_TRUE_FILE and ISOT_FAKE_FILE:
        try:
            # NOTE: label convention for ISOT is a DECISION you must record,
            # not an assumption baked in silently. Here: True.csv -> label 0
            # (real), Fake.csv -> label 1 (fake). This mirrors a common
            # convention but is stated explicitly so it can be checked
            # against your paper's chosen convention and WELFake's convention.
            true_df = load_and_standardize(ISOT_TRUE_FILE, ISOT_COLUMN_MAP, "ISOT_true", forced_label=0)
            fake_df = load_and_standardize(ISOT_FAKE_FILE, ISOT_COLUMN_MAP, "ISOT_fake", forced_label=1)
            isot_df = pd.concat([true_df, fake_df], ignore_index=True)
            loaded["ISOT"] = isot_df
            report = audit_dataset(isot_df, "ISOT", output_dir)
            all_reports["ISOT"] = report
            print_report_table(report)
        except Exception as e:
            print(f"[ERROR loading ISOT] {e}")
    elif ISOT_FILE:
        try:
            isot_df = load_and_standardize(ISOT_FILE, ISOT_COLUMN_MAP, "ISOT")
            loaded["ISOT"] = isot_df
            report = audit_dataset(isot_df, "ISOT", output_dir)
            all_reports["ISOT"] = report
            print_report_table(report)
        except Exception as e:
            print(f"[ERROR loading ISOT] {e}")
    else:
        print("[SKIP] ISOT files not set in CONFIG. Run --discover first.")

    # --- Cross-dataset overlap ---
    if "WELFake" in loaded and "ISOT" in loaded:
        print(f"\n{'=' * 70}\nCROSS-DATASET OVERLAP: WELFake vs ISOT\n{'=' * 70}")
        overlap_report = cross_dataset_overlap(loaded["WELFake"], "WELFake", loaded["ISOT"], "ISOT", output_dir)
        print(overlap_report)
        all_reports["cross_dataset_exact_overlap"] = overlap_report

        print("\nRunning near-duplicate check (see method notes above / in script docstring)...")
        near_dup_report = near_duplicate_check(loaded["WELFake"], "WELFake", loaded["ISOT"], "ISOT", output_dir)
        print(near_dup_report)
        all_reports["cross_dataset_near_duplicates"] = near_dup_report
    else:
        print("\n[SKIP] Cross-dataset overlap check requires both WELFake and ISOT to be loaded.")

    # --- Save full report as JSON ---
    all_reports["_generated_at"] = datetime.now().isoformat()
    report_path = output_dir / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(_json_safe(all_reports), f, indent=2)
    print(f"\nFull machine-readable report saved to: {report_path.resolve()}")
    print(f"Supporting CSVs (duplicates, overlaps, short articles) saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
