# WELFake / ISOT Dataset Audit — Setup & Usage

## 1. Locate your files first
Put `audit_datasets.py` anywhere in your project, then run:

```bash
python audit_datasets.py --discover
```

This scans `RAW_DATA_DIR` (edit this path at the top of the script — default
`./raw_data`) and prints every `.csv`/`.json`/`.jsonl`/`.tsv` file it finds,
with actual column names and a 2-row preview. It does **not** guess which
file is WELFake vs ISOT — you confirm that yourself from the printed output.

## 2. Fill in the CONFIG section
At the top of `audit_datasets.py`:
- `WELFAKE_FILE` → path to your WELFake CSV
- `WELFAKE_COLUMN_MAP` → set `title`/`text`/`label` to the **actual** column
  names `--discover` showed you (don't guess — the script will refuse to
  run if any entry is still `None`)
- `ISOT_TRUE_FILE` / `ISOT_FAKE_FILE` → if your ISOT copy is the common
  two-file layout (`True.csv`, `Fake.csv`, no label column). If instead you
  have a single labeled ISOT file, use `ISOT_FILE` + `ISOT_COLUMN_MAP`
  (including `label`) instead.

**Label convention flag:** the script assigns `True.csv → label 0`,
`Fake.csv → label 1` by default (see comment in `main()`). This is a
recorded *decision*, not a verified fact about your file — check it matches
whatever convention you plan to use in the paper (and cross-check against
WELFake's own label meaning, which the script prints a sample of for you to
eyeball).

## 3. Run the full audit
```bash
python audit_datasets.py
```

Outputs land in `AUDIT_OUTPUT_DIR` (default `./audit_output/`):
- `audit_report.json` — every statistic, machine-readable
- `<name>_duplicate_texts.csv` — rows involved in within-dataset duplicates
- `<name>_short_or_empty_articles.csv` — flagged short/empty articles
- `exact_text_overlap_WELFake_vs_ISOT.csv` — rows with identical text across datasets
- `near_duplicate_candidates_WELFake_vs_ISOT.csv` — near-dup pairs + cosine score
- `near_dup_skipped_blocks_*.csv` — only appears if some prefix-block was too
  large to compare exhaustively (see method note below)

Your raw files are never modified — everything above is a new file.

## Near-duplicate method (before you run it)
1. **Exact duplicates**: SHA-256 hash of normalized (lowercased,
   whitespace-collapsed) text. Deterministic, no false positives.
2. **Near-duplicates**: articles are first grouped ("blocked") by their
   normalized 8-word prefix — only articles sharing that prefix are ever
   compared. Within a block, TF-IDF (word 1–2 grams) + cosine similarity is
   computed pairwise; pairs scoring ≥ `NEAR_DUP_COSINE_THRESHOLD` (default
   0.9) are reported with their exact score.
   - This is a **conservative, partial** check: it will miss near-duplicates
     that don't share an opening, e.g. one copy with a sentence prepended.
     Treat the count as a lower bound, not an exhaustive one.
   - Blocks larger than `MAX_BLOCK_SIZE_FOR_PAIRWISE` (default 3000) are
     skipped and logged separately rather than silently taking hours.

## What I did *not* do (per your rules)
- No model training, no hyperparameter tuning
- No column names or label meanings assumed — the script errors out until
  you confirm them from real inspection
- No silent deletion of duplicates/short articles — they're only flagged
  and exported for your review
- No fabricated statistics anywhere in this deliverable — every number in
  the README/script comments is either a documented method choice (with a
  reason) or a placeholder you fill in from your own data

## After you run it, send me back:
- The console output / `audit_report.json`
- Any `--discover` output if column names differed from what you expected

Then we'll go through: **A** dataset audit tables, **B** cross-dataset
overlap report, **C** data-quality warnings, **D** recommended next step —
together, from your real numbers.
