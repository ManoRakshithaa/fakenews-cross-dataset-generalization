# TF-IDF + Logistic Regression Baseline — Final Results

**Status: FINAL (v2).** These results use the corrected Reuters-dateline cleaning
(`processed_final/WELFake_final_clean.csv`, `processed_final/ISOT_final_clean.csv`).

> **Superseded v1 note:** An earlier run used an incomplete Reuters-dateline regex that missed several dateline formats (multi-city bylines, no-city leads, correction-note-prefixed datelines). That run produced near-identical results (max accuracy shift across all 4 directions: 0.19 percentage points) and is **not included below**. It is superseded by this corrected rerun and should not be cited.

---

## 1. Results table — all four directions

| Direction | Accuracy | Precision (FAKE) | Recall (FAKE) | F1 (FAKE) | Macro F1 | Weighted F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| WELFake → WELFake | 0.9452 | 0.9261 | 0.9509 | 0.9383 | 0.9445 | 0.9453 | not calculated |
| WELFake → ISOT | 0.7406 | 0.7453 | 0.6423 | 0.6900 | 0.7335 | 0.7379 | not calculated |
| ISOT → ISOT | 0.9814 | 0.9856 | 0.9728 | 0.9792 | 0.9812 | 0.9814 | not calculated |
| ISOT → WELFake | 0.5455 | 0.4898 | 0.8818 | 0.6298 | 0.5207 | 0.5073 | not calculated |

**ROC-AUC was not computed by the baseline scripts used for this run.** It is not included for any direction; no value has been estimated or substituted.

Precision/Recall/F1 above are for the **FAKE class (label=1)** as the positive class. Per-class (REAL and FAKE separately) values are in Section 5 (ISOT → WELFake) and in the full classification reports saved under `diagnostics_final/*.json`.

---

## 2. Confusion matrices (exact)

Rows = actual class, columns = predicted class, order [REAL(0), FAKE(1)].

**WELFake → WELFake** (test set: 20% stratified held-out split, n=4,781)
```
                 Predicted REAL   Predicted FAKE
Actual REAL           2526              159
Actual FAKE            103             1993
```

**WELFake → ISOT** (test set: full ISOT, n=38,478)
```
                 Predicted REAL   Predicted FAKE
Actual REAL          17393             3796
Actual FAKE           6184            11105
```

**ISOT → ISOT** (test set: 20% stratified held-out split, n=7,696)
```
                 Predicted REAL   Predicted FAKE
Actual REAL           4189               49
Actual FAKE             94             3364
```

**ISOT → WELFake** (test set: full WELFake, n=23,903)
```
                 Predicted REAL   Predicted FAKE
Actual REAL           3800             9624
Actual FAKE            1239            9240
```

---

## 3. Actual vs. predicted class distribution, per direction

| Direction | Actual REAL % | Actual FAKE % | Predicted REAL % | Predicted FAKE % | Predicted − Actual FAKE % |
|---|---|---|---|---|---|
| WELFake → WELFake | 56.16% | 43.84% | 54.99% | 45.01% | +1.17 pp |
| WELFake → ISOT | 55.07% | 44.93% | 61.27% | 38.73% | −6.20 pp |
| ISOT → ISOT | 55.07% | 44.93% | 55.65% | 44.35% | −0.58 pp |
| ISOT → WELFake | 56.16% | 43.84% | 21.08% | 78.92% | **+35.08 pp** |

---

## 4. Asymmetry / class-prediction imbalance — descriptive only

- **ISOT → WELFake** shows a pronounced shift toward predicting FAKE: predicted-FAKE share (78.92%) exceeds actual-FAKE share (43.84%) by 35.08 percentage points. REAL-class recall is 0.28 (the model misses 72% of actual REAL articles) while FAKE-class recall is 0.88.
- **WELFake → ISOT** shows a smaller, opposite-direction skew: predicted-FAKE share (38.73%) is 6.20 percentage points *below* actual-FAKE share (44.93%) — a mild lean toward predicting REAL, much smaller in magnitude than the ISOT → WELFake skew.
- **WELFake → WELFake** and **ISOT → ISOT** (both in-domain) show predicted/actual FAKE share within about 1 percentage point of each other.
- These are descriptive observations about prediction distribution only. **No claim is made here about what causes the ISOT → WELFake asymmetry** — that requires separate analysis (see the project's distribution-analysis and manual Reuters-residual audit for correlational findings, which are explicitly not causal either).

---

## 5. ISOT → WELFake — detailed breakdown

| Metric | Value |
|---|---|
| REAL precision | 0.75 |
| REAL recall | 0.28 |
| REAL F1 | 0.41 |
| FAKE precision | 0.4898 |
| FAKE recall | 0.8818 |
| FAKE F1 | 0.6298 |
| Actual FAKE % | 43.84% |
| Predicted FAKE % | 78.92% |

Confusion matrix (repeated from Section 2 for convenience):
```
                 Predicted REAL   Predicted FAKE
Actual REAL           3800             9624
Actual FAKE            1239            9240
```

*(REAL precision/recall/F1 are reported at 2 decimal places, as printed by the evaluation script's classification report; FAKE-class metrics are reported at 4 decimal places, as computed directly by the script. Higher-precision REAL-class values are available in `diagnostics_final/ISOT_to_WELFake_title_body_diagnostics.json` if needed.)*

---

## 6. Final cleaned dataset statistics

| Dataset | Final row count | REAL count | REAL % | FAKE count | FAKE % |
|---|---|---|---|---|---|
| WELFake | 23,903 | 13,424 | 56.16% | 10,479 | 43.84% |
| ISOT | 38,478 | 21,189 | 55.07% | 17,289 | 44.93% |

**Residual Reuters-marker counts in final cleaned data** (after corrected dateline-stripping):

| Dataset | Class | Leading tag still unstripped | Any "(Reuters)" mention remaining | % of class |
|---|---|---|---|---|
| WELFake | REAL | 0 | 13 | 0.10% |
| WELFake | FAKE | 0 | 6 | 0.06% |
| ISOT | REAL | 0 | 385 | 1.82% |
| ISOT | FAKE | 0 | 7 | 0.04% |

"Leading tag still unstripped" = 0 for every group confirms the corrected cleaning regex removed every leading dateline it was designed to catch. The small remaining "any mention" counts may include genuine mid-article citations, which the cleaning step was never intended to remove.

---

## 7. Experimental protocol (confirmed, all 4 directions)

- **Data source:** `processed_final/WELFake_final_clean.csv`, `processed_final/ISOT_final_clean.csv` (Reuters-regex-fixed cleaning; row counts confirmed against `final_clean_and_audit.py` output before each run)
- **Text input:** title + body concatenated (`title + " " + text`)
- **Vectorization:** TF-IDF, n-gram range (1, 2), `max_features=20000`, `stop_words="english"`. Vectorizer fit on training text only; test text transformed with the already-fitted vectorizer (no test-set leakage into vocabulary/IDF statistics).
- **Model:** Logistic Regression, `max_iter=1000`, `class_weight="balanced"`, `random_state=42`
- **Random seed:** `42`, used for both the train/test split and the model, throughout
- **In-domain directions** (WELFake → WELFake, ISOT → ISOT): 80/20 stratified train/test split on the single dataset (`stratify` on label), `test_size=0.2`, `random_state=42`
- **Cross-dataset directions** (WELFake → ISOT, ISOT → WELFake): trained on 100% of the cleaned source dataset; evaluated on 100% of the cleaned external dataset in full — no split applied to either side

---

## 8. Scope and limitations of this report

- Covers the TF-IDF + Logistic Regression baseline only, title+body text mode only. Title-only / body-only variants and DistilBERT are not included here.
- ROC-AUC is not available for any direction (see Section 1).
- This report does not draw conclusions about *why* ISOT → WELFake underperforms — only what the metrics and confusion matrices show.
- Full machine-readable results: `results_final/*.json`. Full diagnostics (per-class reports, confusion matrix plots, per-row predictions): `diagnostics_final/*.json`, `diagnostics_final/confusion_matrix_plots/*.png`, `diagnostics_final/predictions/*.csv`.