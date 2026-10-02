"""
FINAL Dataset Cleaning + Audit Pass — WELFake / ISOT
=============================================================================
This is the final cleaning pass before any dataset sizes or model results
are treated as paper-ready. It does NOT reuse or trust any previously
printed/cached numbers from earlier runs -- every count in this script's
output is recomputed from the raw data, in this run, using the SAME
methodology as clean_datasets.py (step order, dedup logic, class-label
convention) with ONE change: the fixed Reuters-tag regex (see
clean_datasets.py's REUTERS_TAG_PATTERN v2 docstring for what changed and
why).

METHODOLOGY (unchanged from clean_datasets.py, recomputed not assumed):
  Step 1: drop exact-duplicate article texts WITHIN each raw dataset
          (SHA-256 hash of normalized text).
  Step 2: remove WELFake rows whose text hash matches an ISOT article
          (computed on the step-1-deduped data, same as before). ISOT is
          left untouched at this stage.
  Step 3: strip leading "(Reuters)" dateline tags -- FIXED regex.
  Step 4: drop articles under SHORT_ARTICLE_WORD_THRESHOLD words, measured
          AFTER step 3.

ADDITIONALLY, this script:
  - reports title-only overlap (computed at the same stage as step 2,
    informational -- does NOT remove additional rows; row removal still
    happens via text-hash match only, unchanged from the established
    methodology)
  - re-verifies zero text-hash overlap remains between the FINAL WELFake
    and FINAL ISOT (does not assume step 2 alone guarantees this --
    checks it explicitly after step 3's text modification, since
    stripping text could in principle change a hash)
  - reports residual "(Reuters)" markers still present in the FINAL
    cleaned data (both datasets), with counts, so nothing is silently
    assumed to be fully clean
  - saves the final datasets under NEW, explicitly-named paths (not
    overwriting the previous processed/ files) so there is no ambiguity
    about which files are "final" for modeling / the paper

Does NOT train, tune, or run any model. Cleaning + audit only.

Run: python3 final_clean_and_audit.py
"""

import json
from pathlib import Path

import pandas as pd

from audit_datasets import (
    load_and_standardize, hash_text, normalize_text,
    WELFAKE_FILE, WELFAKE_COLUMN_MAP,
    ISOT_TRUE_FILE, ISOT_FAKE_FILE, ISOT_COLUMN_MAP,
)
from clean_datasets import (
    step1_drop_internal_duplicates,
    step2_remove_cross_dataset_overlap,
    step3_strip_source_tags,
    step4_drop_short_articles,
    REUTERS_TAG_PATTERN,
)
from analyze_distributions import REUTERS_ANYWHERE_PATTERN

FINAL_DIR = Path("./processed_final")
REPORT_PATH = FINAL_DIR / "final_cleaning_report.json"

# The numbers previously reported to you, BEFORE this final pass. Used only
# to print an explicit side-by-side comparison and flag any mismatch in the
# parts of the pipeline that should NOT have changed (steps 1-2, 4).
PRELIMINARY_NUMBERS = {
    "raw_welfake": 72134,
    "raw_isot": 44898,
    "welfake_internal_duplicates": 9432,
    "isot_internal_duplicates": 6260,
    "cross_dataset_text_overlap": 38638,
    "welfake_final_size_v1": 23903,
    "isot_final_size_v1": 38478,
}


def title_hash_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame) -> int:
    hashes_a = set(df_a["title"].apply(hash_text))
    hashes_b = set(df_b["title"].apply(hash_text))
    return len(hashes_a & hashes_b)


def text_hash_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame) -> int:
    hashes_a = set(df_a["text"].apply(hash_text))
    hashes_b = set(df_b["text"].apply(hash_text))
    return len(hashes_a & hashes_b)


def residual_reuters_count(df: pd.DataFrame) -> dict:
    text = df["text"].fillna("").astype(str)
    leading_still_present = int(text.apply(lambda t: bool(REUTERS_TAG_PATTERN.match(t))).sum())
    anywhere = int(text.str.contains(REUTERS_ANYWHERE_PATTERN, regex=True).sum())
    return {
        "leading_tag_still_unstripped": leading_still_present,
        "anywhere_in_text": anywhere,
        "n_rows": len(df),
        "anywhere_pct": round(100 * anywhere / len(df), 2) if len(df) else 0.0,
    }


def main():
    FINAL_DIR.mkdir(exist_ok=True)
    report = {}

    # ---------------- Load raw ----------------
    print(f"{'=' * 70}\nLOADING RAW DATA\n{'=' * 70}")
    welfake_raw = load_and_standardize(WELFAKE_FILE, WELFAKE_COLUMN_MAP, "WELFake")
    true_df = load_and_standardize(ISOT_TRUE_FILE, ISOT_COLUMN_MAP, "ISOT_true", forced_label=0)
    fake_df = load_and_standardize(ISOT_FAKE_FILE, ISOT_COLUMN_MAP, "ISOT_fake", forced_label=1)
    isot_raw = pd.concat([true_df, fake_df], ignore_index=True)

    report["raw_welfake_size"] = len(welfake_raw)
    report["raw_isot_size"] = len(isot_raw)
    print(f"Raw WELFake: {len(welfake_raw)} rows")
    print(f"Raw ISOT: {len(isot_raw)} rows")

    # ---------------- Step 1: internal duplicates ----------------
    print(f"\n{'=' * 70}\nSTEP 1 — INTERNAL DUPLICATES (recomputed)\n{'=' * 70}")
    welfake_s1 = step1_drop_internal_duplicates(welfake_raw, "WELFake")
    isot_s1 = step1_drop_internal_duplicates(isot_raw, "ISOT")
    report["welfake_internal_duplicates_removed"] = len(welfake_raw) - len(welfake_s1)
    report["isot_internal_duplicates_removed"] = len(isot_raw) - len(isot_s1)
    report["welfake_size_after_step1"] = len(welfake_s1)
    report["isot_size_after_step1"] = len(isot_s1)

    # ---------------- Title-only overlap (informational, same stage as step 2) ----------------
    print(f"\n{'=' * 70}\nTITLE-ONLY CROSS-DATASET OVERLAP (informational, not used for removal)\n{'=' * 70}")
    title_overlap_n = title_hash_overlap(welfake_s1, isot_s1)
    report["title_only_overlap_count"] = title_overlap_n
    print(f"Title-hash overlap between step-1-deduped WELFake and ISOT: {title_overlap_n}")

    # ---------------- Step 2: cross-dataset text overlap ----------------
    print(f"\n{'=' * 70}\nSTEP 2 — CROSS-DATASET EXACT TEXT OVERLAP (recomputed)\n{'=' * 70}")
    welfake_s2 = step2_remove_cross_dataset_overlap(welfake_s1, isot_s1)
    isot_s2 = isot_s1  # untouched at this stage, same as clean_datasets.py
    report["cross_dataset_text_overlap_removed"] = len(welfake_s1) - len(welfake_s2)
    report["welfake_size_after_step2"] = len(welfake_s2)

    # ---------------- Step 3: Reuters tag stripping (FIXED regex) ----------------
    print(f"\n{'=' * 70}\nSTEP 3 — STRIP (REUTERS) TAGS (FIXED regex)\n{'=' * 70}")
    welfake_s3 = step3_strip_source_tags(welfake_s2, "WELFake")
    isot_s3 = step3_strip_source_tags(isot_s2, "ISOT")

    # ---------------- Step 4: short/empty articles ----------------
    print(f"\n{'=' * 70}\nSTEP 4 — DROP SHORT/EMPTY ARTICLES\n{'=' * 70}")
    welfake_final = step4_drop_short_articles(welfake_s3, "WELFake")
    isot_final = step4_drop_short_articles(isot_s3, "ISOT")
    report["welfake_short_removed"] = len(welfake_s3) - len(welfake_final)
    report["isot_short_removed"] = len(isot_s3) - len(isot_final)
    report["welfake_final_size"] = len(welfake_final)
    report["isot_final_size"] = len(isot_final)

    # ---------------- Final class distribution ----------------
    report["welfake_final_class_distribution"] = welfake_final["label"].value_counts().to_dict()
    report["isot_final_class_distribution"] = isot_final["label"].value_counts().to_dict()

    # ---------------- Verify: no duplicate leakage remains ----------------
    print(f"\n{'=' * 70}\nVERIFICATION — RE-CHECKING FOR LEAKAGE AFTER FINAL CLEANING\n{'=' * 70}")
    remaining_overlap = text_hash_overlap(welfake_final, isot_final)
    report["remaining_text_overlap_after_cleaning"] = remaining_overlap
    if remaining_overlap == 0:
        print("CONFIRMED: zero exact-text-hash overlap between final WELFake and final ISOT.")
    else:
        print(f"[WARNING] {remaining_overlap} overlapping rows STILL remain after cleaning -- "
              f"this should not happen if step 2 worked as intended. Investigate before using "
              f"these datasets for modeling.")

    # ---------------- Residual Reuters markers in FINAL data ----------------
    print(f"\n{'=' * 70}\nRESIDUAL REUTERS MARKERS IN FINAL CLEANED DATA\n{'=' * 70}")
    welfake_residual = residual_reuters_count(welfake_final)
    isot_residual = residual_reuters_count(isot_final)
    report["welfake_final_residual_reuters"] = welfake_residual
    report["isot_final_residual_reuters"] = isot_residual
    print(f"WELFake final: {welfake_residual}")
    print(f"ISOT final: {isot_residual}")
    print("'leading_tag_still_unstripped' should be 0 for both (the fixed regex should catch "
          "every leading dateline it was designed to). 'anywhere_in_text' counts ANY remaining "
          "mention, including genuine mid-article references this cleaning step was never meant "
          "to remove -- a non-zero number here is not necessarily a bug.")

    # ---------------- Save final datasets ----------------
    keep_cols = ["title", "text", "label"]
    welfake_path = FINAL_DIR / "WELFake_final_clean.csv"
    isot_path = FINAL_DIR / "ISOT_final_clean.csv"
    welfake_final[keep_cols].to_csv(welfake_path, index=False)
    isot_final[keep_cols].to_csv(isot_path, index=False)
    report["welfake_final_path"] = str(welfake_path.resolve())
    report["isot_final_path"] = str(isot_path.resolve())

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ---------------- FINAL REPORT: clearly separated from preliminary numbers ----------------
    print(f"\n{'=' * 70}\nFINAL REPORT\n{'=' * 70}")

    print("\n--- PRELIMINARY numbers (reported before this final pass -- DO NOT use in the paper) ---")
    for k, v in PRELIMINARY_NUMBERS.items():
        print(f"  {k}: {v}")

    print("\n--- FINAL numbers (recomputed in this run -- use these) ---")
    print(f"  raw_welfake_size: {report['raw_welfake_size']}")
    print(f"  raw_isot_size: {report['raw_isot_size']}")
    print(f"  welfake_internal_duplicates_removed: {report['welfake_internal_duplicates_removed']}")
    print(f"  isot_internal_duplicates_removed: {report['isot_internal_duplicates_removed']}")
    print(f"  title_only_overlap_count: {report['title_only_overlap_count']}")
    print(f"  cross_dataset_text_overlap_removed: {report['cross_dataset_text_overlap_removed']}")
    print(f"  welfake_short_removed (step 4): {report['welfake_short_removed']}")
    print(f"  isot_short_removed (step 4): {report['isot_short_removed']}")
    print(f"  welfake_final_size: {report['welfake_final_size']}")
    print(f"  isot_final_size: {report['isot_final_size']}")
    print(f"  welfake_final_class_distribution: {report['welfake_final_class_distribution']}")
    print(f"  isot_final_class_distribution: {report['isot_final_class_distribution']}")
    print(f"  remaining_text_overlap_after_cleaning: {report['remaining_text_overlap_after_cleaning']}")
    print(f"  welfake_final_residual_reuters: {report['welfake_final_residual_reuters']}")
    print(f"  isot_final_residual_reuters: {report['isot_final_residual_reuters']}")

    print("\n--- CONSISTENCY CHECK: stages NOT touched by the regex fix should match the preliminary numbers ---")
    checks = [
        ("raw_welfake_size", "raw_welfake"),
        ("raw_isot_size", "raw_isot"),
        ("welfake_internal_duplicates_removed", "welfake_internal_duplicates"),
        ("isot_internal_duplicates_removed", "isot_internal_duplicates"),
        ("cross_dataset_text_overlap_removed", "cross_dataset_text_overlap"),
    ]
    for report_key, prelim_key in checks:
        match = report[report_key] == PRELIMINARY_NUMBERS[prelim_key]
        status = "MATCHES (expected)" if match else "*** MISMATCH -- INVESTIGATE ***"
        print(f"  {report_key}: {report[report_key]} vs preliminary {PRELIMINARY_NUMBERS[prelim_key]} -- {status}")

    print(f"\n--- FINAL DATASET FILES (use these paths for modeling / the paper) ---")
    print(f"  WELFake: {report['welfake_final_path']}")
    print(f"  ISOT: {report['isot_final_path']}")
    print(f"  Full JSON report: {REPORT_PATH.resolve()}")
    print(f"\n  NOTE: previous processed/WELFake_clean.csv and processed/ISOT_clean.csv "
          f"(from the v1 Reuters regex) are UNCHANGED and left in place for reference, "
          f"but should NOT be used going forward -- use the processed_final/ files above.")

    print("\nNo model training was run. This script performed cleaning and audit only.")


if __name__ == "__main__":
    main()