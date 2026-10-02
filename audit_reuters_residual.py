"""
Residual Reuters-Mention Audit — ISOT REAL (cleaned data)
=============================================================================
PURPOSE: the analysis script found that 12.84% of cleaned ISOT REAL articles
still contain "(Reuters)" somewhere in the text after cleaning (leading tags
were stripped, but other mentions weren't touched). This script samples 50
of those articles reproducibly, extracts the context around every mention,
and applies a HEURISTIC first-pass classification -- clearly labeled as a
suggestion, not a verified judgment.

WHY A HEURISTIC AND NOT A REAL CLASSIFICATION:
Classifying "is this a genuine mid-article citation or a leading tag our
regex missed" requires actually reading the sentence -- that's a human (or
an AI reading the real output) judgment call, not something this script can
responsibly claim to do with certainty. What this script DOES do reliably:
  - sample the right 50 articles, reproducibly (fixed random_state)
  - find every "(Reuters)" occurrence and extract clean surrounding context
  - apply one clearly-documented, deterministic heuristic rule as a
    starting-point suggestion (column name ends in "_suggested")
  - save everything needed for a real read-through to a CSV

THE HEURISTIC RULE (stated plainly, not hidden in code):
  If a match starts within the first LEADING_WINDOW_CHARS characters of the
  article, AND the text before it looks dateline-like (short, mostly
  uppercase letters, no sentence-ending ". " yet), it's suggested as
  CATEGORY B (a leading marker the anchored strip-regex likely missed --
  e.g. a multi-city byline like "NEW YORK/WASHINGTON (Reuters) -" containing
  a "/" character that the original regex's character class doesn't allow,
  or a leading quote mark before the location name).
  If a match is clearly deep in the article body (past that window), it's
  suggested as CATEGORY A (genuine mid-article reference).
  Anything that doesn't clearly fit either pattern is suggested as
  CATEGORY C (ambiguous).
These are SUGGESTIONS ONLY. The "context" column is included specifically
so a human (or Claude, reading the real printed output) can confirm or
override every single one.

Does NOT modify processed/ISOT_clean.csv or any other existing file -- only
reads it and writes a new CSV under reuters_residual_audit/.

Run: python3 audit_reuters_residual.py
"""

import re
from pathlib import Path

import pandas as pd

from baseline_tfidf_logreg import load_clean
from analyze_distributions import REUTERS_ANYWHERE_PATTERN

OUT_DIR = Path("./reuters_residual_audit")

RANDOM_STATE = 42          # fixed seed -- same 50 articles every run, as long
                            # as processed/ISOT_clean.csv itself doesn't change
SAMPLE_SIZE = 50
CONTEXT_WINDOW_CHARS = 100  # characters of context shown before/after each match
LEADING_WINDOW_CHARS = 120  # heuristic cutoff for "near the start of the article"


def find_all_matches(text: str):
    """Return every '(Reuters)' match position in this article's text."""
    return list(REUTERS_ANYWHERE_PATTERN.finditer(text))


def extract_context(text: str, match_start: int, match_end: int, window: int = CONTEXT_WINDOW_CHARS) -> str:
    start = max(0, match_start - window)
    end = min(len(text), match_end + window)
    snippet = text[start:end].replace("\n", " ").replace("\r", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def suggest_classification(text: str, match_start: int) -> str:
    """
    Deterministic heuristic, documented in the module docstring above.
    Returns one of: 'B_suggested', 'A_suggested', 'C_suggested'.
    """
    if match_start <= LEADING_WINDOW_CHARS:
        prefix = text[:match_start]
        letters = [c for c in prefix if c.isalpha()]
        if not letters:
            return "C_suggested"   # essentially nothing before the match -- ambiguous
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        looks_dateline_like = upper_ratio > 0.6 and ". " not in prefix.strip(" .")
        if looks_dateline_like:
            return "B_suggested"
        return "C_suggested"
    return "A_suggested"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading cleaned ISOT data (identical to what the model trained/tested on)...")
    isot = load_clean("ISOT")
    isot_real = isot[isot["label"] == 0].copy()
    isot_real["text"] = isot_real["text"].fillna("").astype(str)

    has_marker = isot_real["text"].str.contains(REUTERS_ANYWHERE_PATTERN, regex=True)
    candidates = isot_real[has_marker]
    print(f"ISOT REAL rows with a residual '(Reuters)' mention: {len(candidates)}")

    if len(candidates) < SAMPLE_SIZE:
        print(f"[WARNING] Only {len(candidates)} candidates available, fewer than "
              f"the requested sample of {SAMPLE_SIZE}. Sampling all of them.")
        sample = candidates
    else:
        sample = candidates.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    print(f"Sampled {len(sample)} articles (random_state={RANDOM_STATE}, reproducible).")

    rows = []
    for sample_id, (orig_idx, row) in enumerate(sample.iterrows(), start=1):
        text = row["text"]
        title = row["title"]
        matches = find_all_matches(text)
        for match_num, m in enumerate(matches, start=1):
            context = extract_context(text, m.start(), m.end())
            suggestion = suggest_classification(text, m.start())
            rows.append({
                "sample_id": sample_id,
                "original_index": orig_idx,
                "title": title,
                "match_number_in_article": match_num,
                "total_matches_in_article": len(matches),
                "match_char_position": m.start(),
                "context": context,
                "suggested_classification": suggestion,
                "human_classification": "",   # left blank -- fill in A / B / C after reading context
                "notes": "",
            })

    audit_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "sampled_reuters_mentions.csv"
    audit_df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(audit_df)} mention-rows (across {len(sample)} articles) to {csv_path}")

    # ---------------- printed listing (for reading / pasting back) ----------------
    print(f"\n{'=' * 70}\nSAMPLED ARTICLES AND CONTEXT\n{'=' * 70}")
    for sample_id, (orig_idx, row) in enumerate(sample.iterrows(), start=1):
        text = row["text"]
        title = row["title"]
        matches = find_all_matches(text)
        print(f"\n--- Sample #{sample_id} (original_index={orig_idx}) ---")
        print(f"Title: {title}")
        for match_num, m in enumerate(matches, start=1):
            context = extract_context(text, m.start(), m.end())
            suggestion = suggest_classification(text, m.start())
            print(f"  Match {match_num}/{len(matches)} (suggested: {suggestion}): {context}")

    # ---------------- heuristic summary (explicitly caveated) ----------------
    print(f"\n{'=' * 70}\nHEURISTIC SUGGESTED-CLASSIFICATION SUMMARY (NOT a verified count)\n{'=' * 70}")
    print("This counts SUGGESTIONS from the position-based heuristic rule above, at the "
          "mention level (one article may have multiple mentions). Verify against the "
          "context column before treating these as real findings.")
    counts = audit_df["suggested_classification"].value_counts()
    total = len(audit_df)
    for label in ["A_suggested", "B_suggested", "C_suggested"]:
        n = int(counts.get(label, 0))
        pct = round(100 * n / total, 1) if total else 0.0
        print(f"  {label}: {n} ({pct}%)")

    print(f"\nAll {len(sample)} sampled articles' full context printed above and saved to:\n{csv_path.resolve()}")
    print("\nNEXT STEP: paste the printed listing above back to Claude (or review it yourself) "
          "to assign the real A/B/C classification per example -- the suggested labels are a "
          "starting point only.")


if __name__ == "__main__":
    main()