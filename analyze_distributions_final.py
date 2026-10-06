"""
Distribution Analysis — FINAL DATA RERUN (against processed_final/)
=============================================================================
PURPOSE: investigate WHY the ISOT -> WELFake baseline shows severe
prediction imbalance (79.2% predicted FAKE vs 43.8% actual FAKE), WITHOUT
training any new model. Pure descriptive statistics and exploratory text
analysis.

This script does NOT:
  - train a classifier of any kind
  - change clean_datasets.py's output files
  - change the baseline experiment protocol
It DOES reuse the raw loading helpers from audit_datasets.py and the
cleaned-data loading helper from baseline_tfidf_logreg.py, so the data
being analyzed here is IDENTICAL to what the model actually saw.

IMPORTANT DISTINCTION USED THROUGHOUT:
  RAW data    = straight from WELFake_Dataset.csv / True.csv / Fake.csv,
                before clean_datasets.py touched it. Used ONLY to see what
                the Reuters tag situation looked like before stripping.
  CLEANED data = processed/WELFake_clean.csv, processed/ISOT_clean.csv —
                this is what the baseline model actually trained/tested on.
                All length/vocabulary analysis uses THIS, not raw.

Everything this script prints or plots is a computed number from your
actual data. The "OBSERVATIONS" section states only what the numbers show.
The "HYPOTHESES SUPPORTED" / "HYPOTHESES NOT ESTABLISHED" sections are a
scaffold for you (and Claude, once you send back the output) to fill in
against the real printed numbers — they are not pre-written conclusions.

Uses baseline_tfidf_logreg_final.py (processed_final/ data). Output goes
to distribution_analysis_final/, kept fully separate from the original
distribution_analysis/ folder (v1 data). REUTERS_TAG_PATTERN is imported
from clean_datasets.py, which already has the fixed v2 regex, so the
leading-tag detection in this rerun automatically reflects the fix.

Run: python3 analyze_distributions_final.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Raw-data loading (pre-cleaning) — reused from audit_datasets.py so the
# Reuters-tag "before" picture is built from the exact same raw files.
from audit_datasets import (
    load_and_standardize, normalize_text,
    WELFAKE_FILE, WELFAKE_COLUMN_MAP,
    ISOT_TRUE_FILE, ISOT_FAKE_FILE, ISOT_COLUMN_MAP,
)

# Cleaned-data loading — reused from baseline_tfidf_logreg.py so this is
# IDENTICAL to what the model trained/tested on.
from baseline_tfidf_logreg_final import load_clean, build_text_field

# Same anchored "leading Reuters tag" pattern used in clean_datasets.py's
# step 3, imported so the "beginning" detection here matches exactly what
# was stripped during cleaning (not a slightly different regex).
from clean_datasets import REUTERS_TAG_PATTERN

OUT_DIR = Path("./distribution_analysis_final")
PLOTS_DIR = OUT_DIR / "plots"

LABEL_NAMES = {0: "REAL", 1: "FAKE"}
GROUP_ORDER = [("WELFake", 0), ("WELFake", 1), ("ISOT", 0), ("ISOT", 1)]
GROUP_LABELS = [f"{ds} {LABEL_NAMES[lbl]}" for ds, lbl in GROUP_ORDER]

# A simple "anywhere in text" marker check — broader than the anchored
# leading-tag pattern, catches "(Reuters)" appearing anywhere in the body.
REUTERS_ANYWHERE_PATTERN = re.compile(r"\(Reuters\)")


# ============================================================
# LOADING
# ============================================================

def load_raw_labeled() -> dict:
    """Load the RAW (pre-cleaning) datasets, same as clean_datasets.py does
    before any of its 4 cleaning steps run."""
    welfake = load_and_standardize(WELFAKE_FILE, WELFAKE_COLUMN_MAP, "WELFake")
    true_df = load_and_standardize(ISOT_TRUE_FILE, ISOT_COLUMN_MAP, "ISOT_true", forced_label=0)
    fake_df = load_and_standardize(ISOT_FAKE_FILE, ISOT_COLUMN_MAP, "ISOT_fake", forced_label=1)
    isot = pd.concat([true_df, fake_df], ignore_index=True)
    return {"WELFake": welfake, "ISOT": isot}


def load_cleaned_labeled() -> dict:
    """Load the CLEANED datasets — identical to what the baseline model
    trained/tested on."""
    welfake = build_text_field(load_clean("WELFake"), mode="title_body")
    isot = build_text_field(load_clean("ISOT"), mode="title_body")
    return {"WELFake": welfake, "ISOT": isot}


# ============================================================
# ANALYSIS 1 — REUTERS / SOURCE MARKERS
# ============================================================

def reuters_marker_stats(raw_data: dict, cleaned_data: dict) -> pd.DataFrame:
    rows = []
    for ds, lbl in GROUP_ORDER:
        raw_df = raw_data[ds]
        raw_sub = raw_df[raw_df["label"] == lbl]
        raw_text = raw_sub["text"].fillna("").astype(str)
        n_raw = len(raw_sub)
        raw_begin_n = int(raw_text.apply(lambda t: bool(REUTERS_TAG_PATTERN.match(t))).sum())
        raw_anywhere_n = int(raw_text.str.contains(REUTERS_ANYWHERE_PATTERN, regex=True).sum()) if n_raw else 0

        cleaned_df = cleaned_data[ds]
        cleaned_sub = cleaned_df[cleaned_df["label"] == lbl]
        cleaned_text = cleaned_sub["text"].fillna("").astype(str)
        n_cleaned = len(cleaned_sub)
        cleaned_begin_n = int(cleaned_text.apply(lambda t: bool(REUTERS_TAG_PATTERN.match(t))).sum()) if n_cleaned else 0
        cleaned_anywhere_n = int(cleaned_text.str.contains(REUTERS_ANYWHERE_PATTERN, regex=True).sum()) if n_cleaned else 0

        rows.append({
            "dataset": ds, "class": LABEL_NAMES[lbl],
            "n_raw": n_raw,
            "raw_beginning_n": raw_begin_n,
            "raw_beginning_pct": round(100 * raw_begin_n / n_raw, 2) if n_raw else None,
            "raw_anywhere_n": raw_anywhere_n,
            "raw_anywhere_pct": round(100 * raw_anywhere_n / n_raw, 2) if n_raw else None,
            "n_cleaned": n_cleaned,
            "cleaned_beginning_n": cleaned_begin_n,
            "cleaned_beginning_pct": round(100 * cleaned_begin_n / n_cleaned, 2) if n_cleaned else None,
            "cleaned_anywhere_n": cleaned_anywhere_n,
            "cleaned_anywhere_pct": round(100 * cleaned_anywhere_n / n_cleaned, 2) if n_cleaned else None,
        })
    return pd.DataFrame(rows)


# ============================================================
# ANALYSIS 2 & 3 — LENGTH STATS (article + title)
# ============================================================

def length_stats(values: pd.Series) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
    }


def compute_length_tables(cleaned_data: dict):
    word_rows, char_rows, title_rows = [], [], []
    word_count_cache = {}   # (ds,lbl) -> pd.Series, reused for plots
    title_count_cache = {}

    for ds, lbl in GROUP_ORDER:
        sub = cleaned_data[ds]
        sub = sub[sub["label"] == lbl]

        word_counts = sub["text"].apply(lambda t: len(normalize_text(t).split()))
        char_counts = sub["text"].fillna("").astype(str).apply(len)
        title_word_counts = sub["title"].apply(lambda t: len(normalize_text(t).split()))

        word_count_cache[(ds, lbl)] = word_counts
        title_count_cache[(ds, lbl)] = title_word_counts

        word_rows.append({"dataset": ds, "class": LABEL_NAMES[lbl], **length_stats(word_counts)})
        char_rows.append({"dataset": ds, "class": LABEL_NAMES[lbl], **length_stats(char_counts)})
        title_rows.append({"dataset": ds, "class": LABEL_NAMES[lbl], **length_stats(title_word_counts)})

    return (pd.DataFrame(word_rows), pd.DataFrame(char_rows), pd.DataFrame(title_rows),
            word_count_cache, title_count_cache)


# ============================================================
# ANALYSIS 4 — VOCAB / STYLE
# ============================================================

def most_frequent_words(text_series: pd.Series, top_n: int = 20, max_features: int = 2000):
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 1), stop_words="english")
    counts = vectorizer.fit_transform(text_series.fillna("").astype(str))
    terms = vectorizer.get_feature_names_out()
    freqs = np.asarray(counts.sum(axis=0)).ravel()
    order = np.argsort(freqs)[::-1][:top_n]
    return [(terms[i], int(freqs[i])) for i in order]


def top_discriminative_terms(text_a: pd.Series, text_b: pd.Series, label_a: str, label_b: str,
                              top_n: int = 20, max_features: int = 5000):
    """
    Mean-TF-IDF difference between two groups' text. EXPLORATORY / CORRELATIONAL
    ONLY -- a term appearing here means it has different average TF-IDF weight
    between the two corpora, not that it CAUSES any classifier's behavior.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1), stop_words="english")
    combined = pd.concat([text_a.fillna("").astype(str), text_b.fillna("").astype(str)])
    tfidf = vectorizer.fit_transform(combined)
    terms = vectorizer.get_feature_names_out()

    n_a = len(text_a)
    mean_a = np.asarray(tfidf[:n_a].mean(axis=0)).ravel()
    mean_b = np.asarray(tfidf[n_a:].mean(axis=0)).ravel()
    diff = mean_a - mean_b

    order = np.argsort(diff)
    top_b = [(terms[i], float(diff[i])) for i in order[:top_n]]
    top_a = [(terms[i], float(diff[i])) for i in order[::-1][:top_n]]
    return {f"top_terms_more_{label_a}": top_a, f"top_terms_more_{label_b}": top_b}


# ============================================================
# PLOTS
# ============================================================

def plot_length_boxplot(word_count_cache: dict, out_path: Path, title: str, ylabel: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = [word_count_cache[(ds, lbl)] for ds, lbl in GROUP_ORDER]
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        ax.boxplot(data, tick_labels=GROUP_LABELS, showfliers=False)
    except TypeError:
        # Older matplotlib (<3.9) doesn't have tick_labels yet.
        ax.boxplot(data, labels=GROUP_LABELS, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reuters_marker_bars(reuters_df: pd.DataFrame, out_path: Path):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(GROUP_LABELS))
    width = 0.35
    raw_pcts = [reuters_df.loc[(reuters_df["dataset"] == ds) & (reuters_df["class"] == LABEL_NAMES[lbl]),
                                "raw_anywhere_pct"].values[0] for ds, lbl in GROUP_ORDER]
    cleaned_pcts = [reuters_df.loc[(reuters_df["dataset"] == ds) & (reuters_df["class"] == LABEL_NAMES[lbl]),
                                    "cleaned_anywhere_pct"].values[0] for ds, lbl in GROUP_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, raw_pcts, width, label="Raw (before cleaning)")
    ax.bar(x + width / 2, cleaned_pcts, width, label="Cleaned (after leading-tag strip)")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, rotation=20)
    ax.set_ylabel("% of articles containing \"(Reuters)\" anywhere")
    ax.set_title("Reuters marker prevalence: raw vs cleaned")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_real_vs_real_overlay(word_count_cache: dict, out_path: Path):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    welfake_real = word_count_cache[("WELFake", 0)]
    isot_real = word_count_cache[("ISOT", 0)]
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, max(welfake_real.quantile(0.99), isot_real.quantile(0.99)), 40)
    ax.hist(welfake_real, bins=bins, alpha=0.5, label="WELFake REAL", density=True)
    ax.hist(isot_real, bins=bins, alpha=0.5, label="ISOT REAL", density=True)
    ax.set_xlabel("Article word count")
    ax.set_ylabel("Density")
    ax.set_title("Analysis 5: WELFake REAL vs ISOT REAL — article length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    print("Loading raw and cleaned datasets...")
    raw_data = load_raw_labeled()
    cleaned_data = load_cleaned_labeled()

    # Safety check: if any dataset x class group is empty after cleaning,
    # length/vocab stats and plots for that group are meaningless (or will
    # crash outright). Flag it loudly rather than silently producing NaNs.
    for ds, lbl in GROUP_ORDER:
        n = len(cleaned_data[ds][cleaned_data[ds]["label"] == lbl])
        if n == 0:
            print(f"[WARNING] {ds} {LABEL_NAMES[lbl]} has ZERO rows in the cleaned data. "
                  f"All stats/plots for this group will be skipped or show as empty.")

    # ---------------- Analysis 1 ----------------
    print(f"\n{'=' * 70}\nANALYSIS 1 — REUTERS / SOURCE MARKERS\n{'=' * 70}")
    reuters_df = reuters_marker_stats(raw_data, cleaned_data)
    print(reuters_df.to_string(index=False))
    reuters_df.to_csv(OUT_DIR / "analysis1_reuters_markers.csv", index=False)

    # ---------------- Analysis 2 & 3 ----------------
    print(f"\n{'=' * 70}\nANALYSIS 2 — ARTICLE LENGTH (word count, cleaned data)\n{'=' * 70}")
    word_df, char_df, title_df, word_cache, title_cache = compute_length_tables(cleaned_data)
    print(word_df.to_string(index=False))
    word_df.to_csv(OUT_DIR / "analysis2_word_length.csv", index=False)

    print(f"\n{'=' * 70}\nANALYSIS 2b — ARTICLE LENGTH (character count, cleaned data)\n{'=' * 70}")
    print(char_df.to_string(index=False))
    char_df.to_csv(OUT_DIR / "analysis2b_char_length.csv", index=False)

    print(f"\n{'=' * 70}\nANALYSIS 3 — TITLE LENGTH (word count, cleaned data)\n{'=' * 70}")
    print(title_df.to_string(index=False))
    title_df.to_csv(OUT_DIR / "analysis3_title_length.csv", index=False)

    # ---------------- Analysis 4 ----------------
    print(f"\n{'=' * 70}\nANALYSIS 4 — VOCAB / STYLE (most frequent words per group)\n{'=' * 70}")
    freq_summary = {}
    for ds, lbl in GROUP_ORDER:
        sub = cleaned_data[ds]
        sub = sub[sub["label"] == lbl]
        top_words = most_frequent_words(sub["_model_input"])
        freq_summary[f"{ds}_{LABEL_NAMES[lbl]}"] = top_words
        print(f"\nTop 20 words — {ds} {LABEL_NAMES[lbl]}:")
        print(", ".join(f"{w}({c})" for w, c in top_words))

    print(f"\n{'=' * 70}\nANALYSIS 4b — WELFake vs ISOT discriminative terms (TF-IDF, all classes combined)\n{'=' * 70}")
    print("(Correlational only -- does NOT imply these terms cause any classifier's behavior)")
    disc_all = top_discriminative_terms(
        cleaned_data["WELFake"]["_model_input"], cleaned_data["ISOT"]["_model_input"],
        "WELFake", "ISOT",
    )
    for key, terms in disc_all.items():
        print(f"\n{key}:")
        print(", ".join(f"{w}({d:+.4f})" for w, d in terms))

    # ---------------- Analysis 5 ----------------
    print(f"\n{'=' * 70}\nANALYSIS 5 — WELFake REAL vs ISOT REAL\n{'=' * 70}")
    welfake_real = cleaned_data["WELFake"][cleaned_data["WELFake"]["label"] == 0]
    isot_real = cleaned_data["ISOT"][cleaned_data["ISOT"]["label"] == 0]

    print("Length comparison (from Analysis 2/3 tables above, REAL rows only):")
    print(word_df[word_df["class"] == "REAL"].to_string(index=False))
    print(title_df[title_df["class"] == "REAL"].to_string(index=False))

    print("\nReuters marker comparison (from Analysis 1 table above, REAL rows only):")
    print(reuters_df[reuters_df["class"] == "REAL"].to_string(index=False))

    print("\nDiscriminative terms, WELFake REAL vs ISOT REAL (TF-IDF, correlational only):")
    disc_real = top_discriminative_terms(
        welfake_real["_model_input"], isot_real["_model_input"],
        "WELFake_REAL", "ISOT_REAL",
    )
    for key, terms in disc_real.items():
        print(f"\n{key}:")
        print(", ".join(f"{w}({d:+.4f})" for w, d in terms))

    # ---------------- Plots ----------------
    print(f"\n{'=' * 70}\nSAVING PLOTS\n{'=' * 70}")
    plot_length_boxplot(word_cache, PLOTS_DIR / "article_length_boxplot.png",
                         "Article length by dataset x class", "Word count")
    plot_length_boxplot(title_cache, PLOTS_DIR / "title_length_boxplot.png",
                         "Title length by dataset x class", "Title word count")
    plot_reuters_marker_bars(reuters_df, PLOTS_DIR / "reuters_marker_bars.png")
    plot_real_vs_real_overlay(word_cache, PLOTS_DIR / "real_vs_real_length_overlay.png")
    print(f"Saved 4 plots to {PLOTS_DIR.resolve()}")

    # ---------------- Concise summary table ----------------
    summary = reuters_df.merge(word_df, on=["dataset", "class"], suffixes=("", "_wordlen"))
    summary = summary.merge(title_df, on=["dataset", "class"], suffixes=("", "_titlelen"))
    summary_path = OUT_DIR / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n{'=' * 70}\nCONCISE SUMMARY TABLE (saved to {summary_path})\n{'=' * 70}")
    print(summary[["dataset", "class", "n_raw", "raw_anywhere_pct", "cleaned_anywhere_pct",
                    "mean", "median", "mean_titlelen", "median_titlelen"]].to_string(index=False))

    # ---------------- Data-driven observation bullets ----------------
    print(f"\n{'=' * 70}\nOBSERVATIONS (computed directly from the numbers above)\n{'=' * 70}")
    def get(ds, lbl, col, df):
        return df.loc[(df["dataset"] == ds) & (df["class"] == LABEL_NAMES[lbl]), col].values[0]

    wf_real_raw_pct = get("WELFake", 0, "raw_anywhere_pct", reuters_df)
    isot_real_raw_pct = get("ISOT", 0, "raw_anywhere_pct", reuters_df)
    wf_real_clean_pct = get("WELFake", 0, "cleaned_anywhere_pct", reuters_df)
    isot_real_clean_pct = get("ISOT", 0, "cleaned_anywhere_pct", reuters_df)
    wf_real_mean_len = get("WELFake", 0, "mean", word_df)
    isot_real_mean_len = get("ISOT", 0, "mean", word_df)

    print(f"1. Raw-data Reuters marker prevalence (anywhere in text): "
          f"WELFake REAL = {wf_real_raw_pct}%, ISOT REAL = {isot_real_raw_pct}% "
          f"(gap = {abs(wf_real_raw_pct - isot_real_raw_pct):.2f} percentage points).")
    print(f"2. Cleaned-data residual Reuters marker prevalence (after leading-tag stripping): "
          f"WELFake REAL = {wf_real_clean_pct}%, ISOT REAL = {isot_real_clean_pct}%.")
    print(f"3. Mean article word count: WELFake REAL = {wf_real_mean_len:.1f}, "
          f"ISOT REAL = {isot_real_mean_len:.1f} "
          f"(gap = {abs(wf_real_mean_len - isot_real_mean_len):.1f} words).")
    print("4. See analysis4b/5 discriminative-terms output above for the specific vocabulary "
          "differences driving these two REAL classes apart (or not) in TF-IDF space.")
    print("5. Full per-group stats, including std/p25/p75 and FAKE-class comparisons, are in "
          f"the CSVs saved under {OUT_DIR.resolve()} — this section only calls out the REAL-vs-REAL "
          "numbers most relevant to the current hypothesis.")

    # ---------------- Hypothesis scaffold (NOT pre-decided) ----------------
    print(f"\n{'=' * 70}\nHYPOTHESIS SCAFFOLD — fill in against the numbers above\n{'=' * 70}")
    print("""
This section is a checklist, not a conclusion. Compare it against the printed
numbers (and send both to Claude/your research chat to finalize):

Would be SUPPORTED if the numbers show a large, consistent gap:
  - "ISOT REAL articles are more strongly associated with wire-service/Reuters
    style than WELFake REAL articles" -- IF raw_anywhere_pct is much higher for
    ISOT REAL than WELFake REAL (see Observation 1).
  - "WELFake REAL and ISOT REAL differ in typical article length" -- IF the
    mean/median word counts in Observation 3 differ substantially and the
    boxplot/overlay histogram show clearly separated distributions rather than
    heavy overlap.
  - "WELFake REAL and ISOT REAL use measurably different vocabulary" -- IF the
    Analysis 5 discriminative-terms list contains clearly topic/style-specific
    words (not just noise) with non-trivial TF-IDF differences.

NOT established by this analysis, regardless of the numbers above:
  - That any observed distributional difference CAUSES the model's ISOT->WELFake
    prediction-imbalance behavior. This analysis is descriptive/correlational
    only -- confirming a causal mechanism would require a controlled follow-up
    (e.g. retraining ISOT with Reuters tags fully removed and re-testing on
    WELFake, which has NOT been done here).
  - That these findings generalize beyond WELFake and ISOT specifically to fake
    news detection or dataset generalization in general.
  - That vocabulary/length differences are the ONLY or PRIMARY explanation --
    other unmeasured factors (topic distribution, time period, political
    slant, editorial style beyond word choice) have not been ruled out.
""")

    print(f"\nAll CSVs and plots saved under: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()