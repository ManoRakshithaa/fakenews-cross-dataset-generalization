"""
Cleaning Pipeline — Step 2 of the WELFake / ISOT Research Project
=====================================================================
Turns the RAW datasets into CLEAN, train-ready versions. Never touches
the raw files — writes new files into `./processed/`.

Run this AFTER audit_datasets.py (it reuses that script's CONFIG and
functions, so make sure yours is already filled in correctly).

STEPS (in order — this order matters, see notes inline):
  1. Drop exact duplicate article texts WITHIN each dataset.
  2. Remove WELFake rows whose text exactly matches an ISOT article,
     so ISOT stays a genuinely unseen cross-dataset test set.
  3. Strip the leading "(Reuters)" source-agency tag from article text,
     so a model can't just learn to detect that tag instead of learning
     anything about veracity.
  4. Drop empty / very short articles (measured AFTER tag stripping).

Every step PRINTS how many rows it removed and why — nothing is
dropped silently.
"""

import re
from pathlib import Path

import pandas as pd

# Reuse the already-confirmed loading logic + CONFIG from audit_datasets.py
# instead of retyping paths/column maps here (avoids the two ever drifting
# out of sync).
from audit_datasets import (
    load_and_standardize,
    hash_text,
    normalize_text,
    WELFAKE_FILE, WELFAKE_COLUMN_MAP,
    ISOT_TRUE_FILE, ISOT_FAKE_FILE, ISOT_COLUMN_MAP,
    SHORT_ARTICLE_WORD_THRESHOLD,
)

PROCESSED_DIR = Path("./processed")


# ------------------------------------------------------------------
# STEP 1 — internal duplicates
# ------------------------------------------------------------------
def step1_drop_internal_duplicates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Within ONE dataset, keep only the FIRST occurrence of each unique
    article text (by normalized-text hash). Drop the rest."""
    before = len(df)
    df = df.copy()
    df["_text_hash"] = df["text"].apply(hash_text)                    # LINE A: one hash per row
    df = df.drop_duplicates(subset="_text_hash", keep="first")        # LINE B: keep 1st, drop rest
    after = len(df)
    print(f"[{name}] Step 1 - internal duplicates: {before} -> {after} rows "
          f"({before - after} removed)")
    return df


# ------------------------------------------------------------------
# STEP 2 — cross-dataset overlap (WELFake vs ISOT)
# ------------------------------------------------------------------
def step2_remove_cross_dataset_overlap(welfake_df: pd.DataFrame,
                                        isot_df: pd.DataFrame) -> pd.DataFrame:
    """Remove any WELFake row whose text hash also appears in ISOT.
    ISOT is left untouched — it's the clean external test set. WELFake
    is trimmed so it no longer secretly contains ISOT's articles."""
    before = len(welfake_df)
    isot_hashes = set(isot_df["text"].apply(hash_text))               # LINE A: set of ISOT hashes
    welfake_df = welfake_df.copy()
    welfake_df["_text_hash"] = welfake_df["text"].apply(hash_text)    # LINE B: hash WELFake rows
    welfake_df = welfake_df[~welfake_df["_text_hash"].isin(isot_hashes)]  # LINE C: keep non-overlap
    after = len(welfake_df)
    print(f"[WELFake] Step 2 - cross-dataset overlap with ISOT: {before} -> {after} rows "
          f"({before - after} removed)")
    return welfake_df


# ------------------------------------------------------------------
# STEP 3 — strip leading "(Reuters)" source tag
# ------------------------------------------------------------------
# VERSION 2 of this pattern. A manual audit of 50 residual "(Reuters)"
# mentions remaining after v1 cleaning found 50/50 were leading datelines
# v1 failed to catch, not genuine mid-article citations. Three failure
# modes were identified and are each handled by one optional piece below:
#
#   B1 - no city before the tag at all:            "(Reuters) - U.S. ..."
#   B2 - multi-city byline (v1's character class
#        didn't allow "/" or "and" between cities): "NEW YORK/WASHINGTON
#                                                     (Reuters) - ..."
#   B3 - a leading wire-service correction note
#        and/or "By <author>" byline pushes the
#        dateline past position 0:                  "(This Oct 15 story
#                                                     corrects ...) By Julia
#                                                     Harte ASHEBORO, N.C.
#                                                     (Reuters) - ..."
#
# Still fully anchored with ^ -- every piece is optional, but the whole
# match must run contiguously from the very start of the text through to
# "(Reuters)". This is what keeps it safe: a genuine mid-article citation
# is preceded by ordinary lowercase prose and sentence punctuation, which
# none of these pieces can match, so the pattern simply fails to match at
# all for those cases (verified directly -- see note below).
#
# Safety: tested against 510 synthetic non-dateline sentences, including
# ones with a genuine mid-sentence "(Reuters)" mention -- zero false
# positives. Also tested against the real failing patterns from the manual
# audit (21 real B1/B2 strings, 4 reconstructed B3 strings) -- all stripped
# correctly, leaving only the genuine article body.
REUTERS_TAG_PATTERN = re.compile(
    r"^(?:\([^()]{0,200}\)\s*)?"                                             # B3a: optional leading correction note
    r"(?:[Bb]y\s+[A-Z][\w.\-']*(?:\s+[A-Z][\w.\-']*){0,3}"
    r"(?:\s+and\s+[A-Z][\w.\-']*(?:\s+[A-Z][\w.\-']*){0,3})?\s+)?"           # B3b: optional "By <author(s)>" byline
    r"(?:[A-Z][A-Za-z.]{0,20}(?:(?:[,/]\s*|\s+(?:and\s+)?)[A-Z][A-Za-z.]{0,20}){0,5}\s*)?"  # B1/B2: optional dateline location(s)
    r"\(Reuters\)\s*[-\u2013\u2014]\s*"                                      # the tag itself + dash (hyphen/en-dash/em-dash)
)


def step3_strip_source_tags(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Remove the leading (Reuters) dateline tag from article text."""
    df = df.copy()
    n_matched = df["text"].fillna("").astype(str).str.contains(REUTERS_TAG_PATTERN, regex=True).sum()
    df["text"] = df["text"].fillna("").astype(str).apply(
        lambda t: REUTERS_TAG_PATTERN.sub("", t)                      # LINE A: strip if present
    )
    print(f"[{name}] Step 3 - stripped leading (Reuters) tag from {n_matched} rows")
    return df


# ------------------------------------------------------------------
# STEP 4 — drop short/empty articles (measured AFTER stripping)
# ------------------------------------------------------------------
def step4_drop_short_articles(df: pd.DataFrame, name: str,
                               min_words: int = SHORT_ARTICLE_WORD_THRESHOLD) -> pd.DataFrame:
    """Drop rows whose article text has fewer than min_words words.
    Must run AFTER step 3, so we're measuring real article content,
    not counting the (Reuters) tag as part of the word count."""
    before = len(df)
    df = df.copy()
    word_counts = df["text"].apply(lambda s: len(normalize_text(s).split()))  # LINE A
    df = df[word_counts >= min_words]                                        # LINE B
    after = len(df)
    print(f"[{name}] Step 4 - short/empty articles (< {min_words} words): "
          f"{before} -> {after} rows ({before - after} removed)")
    return df


# ------------------------------------------------------------------
# ORCHESTRATION
# ------------------------------------------------------------------
def clean_pipeline():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw datasets...")
    welfake = load_and_standardize(WELFAKE_FILE, WELFAKE_COLUMN_MAP, "WELFake")
    true_df = load_and_standardize(ISOT_TRUE_FILE, ISOT_COLUMN_MAP, "ISOT_true", forced_label=0)
    fake_df = load_and_standardize(ISOT_FAKE_FILE, ISOT_COLUMN_MAP, "ISOT_fake", forced_label=1)
    isot = pd.concat([true_df, fake_df], ignore_index=True)

    print("\n--- STEP 1: internal duplicates ---")
    welfake = step1_drop_internal_duplicates(welfake, "WELFake")
    isot = step1_drop_internal_duplicates(isot, "ISOT")

    print("\n--- STEP 2: cross-dataset overlap ---")
    welfake = step2_remove_cross_dataset_overlap(welfake, isot)

    print("\n--- STEP 3: strip (Reuters) source tags ---")
    welfake = step3_strip_source_tags(welfake, "WELFake")
    isot = step3_strip_source_tags(isot, "ISOT")

    print("\n--- STEP 4: drop short/empty articles ---")
    welfake = step4_drop_short_articles(welfake, "WELFake")
    isot = step4_drop_short_articles(isot, "ISOT")

    keep_cols = ["title", "text", "label"]
    welfake_out = welfake[keep_cols]
    isot_out = isot[keep_cols]

    welfake_path = PROCESSED_DIR / "WELFake_clean.csv"
    isot_path = PROCESSED_DIR / "ISOT_clean.csv"
    welfake_out.to_csv(welfake_path, index=False)
    isot_out.to_csv(isot_path, index=False)

    print(f"\nSaved: {welfake_path}  ({len(welfake_out)} rows)")
    print(f"Saved: {isot_path}  ({len(isot_out)} rows)")
    print("\nRaw files were NOT modified. Cleaned files are in ./processed/")


if __name__ == "__main__":
    clean_pipeline()