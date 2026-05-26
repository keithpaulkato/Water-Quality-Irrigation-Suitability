# Notebook Extension Design — Lecturer-Driven Additions

**Date:** 2026-05-18
**Project:** Water Quality Irrigation Suitability Prediction (Group 2)
**Target notebook:** `notebooks/Group_2_Water_Quality_Irrigation_Suitability.ipynb`

## 1. Motivation

The lecturer (Dr. Lillian Muyama) issued a handwritten requirements note:

1. A time series dataset
2. Do some prediction
3. Analysis by location (results per region)
4. Is the model biased given a particular location, and how is it handled
5. Interpretations based on results
6. What can be done to improve the quality

The current notebook is a strong static-classification pipeline (EDA → 12 ML + 2 NN → SHAP/LIME) but it does **not** address points 1, 3, 4, or 6 substantively. This design extends the notebook to close those gaps.

## 2. Dataset Reality Check (informs design choices)

Verified by directly inspecting `data/raw/citizen-science-water-quality-monitoring-wq-parameters_local_time.csv`:

- 480 rows × 23 columns
- Time span: 2024-03-13 → 2024-10-22 (116 unique dates, ~7 months)
- Biosphere distribution: Kruger 350, Vhembe 116, **Marico only 14**
- Per-site time series: median 2.5 obs/site, max 25 → **per-site LSTM is not viable**
- 5 columns 100% missing (Nitrate, Phosphate, Salinity ×2, TDS) → drop
- Target text labels: 360 "suitable" / 120 "NOT suitable" → 3:1 imbalance
- Label is derived from a pH rule (already flagged as label-leakage in Limitations)

**Implications:**
- Do **temporal trend + chronological split**, not LSTM forecasting.
- Leave-One-Biosphere-Out CV is valid but Marico-as-test will be noisy → caveat explicitly.

## 3. Scope

### In-scope (this design)
- Append new cells to the existing notebook addressing all 6 lecturer points.
- Reuse existing fitted models where possible — do not retrain from scratch unless required.
- Keep new sections self-contained so existing sections remain runnable.

### Out-of-scope
- LSTM/sequence forecasting (data too small).
- Restructuring the notebook to academic-report chapter style (deferred — separate user decision).
- Writing the formal `.docx` / PDF report.
- New raw data ingestion.

## 4. Architecture — Where New Cells Go

| New cell / subsection | Position in notebook | Lecturer point |
|---|---|---|
| **5.2 Temporal Feature Engineering** | inside Section 5 (Feature Engineering) | 1 |
| **6.1 Chronological vs Random Split** | inside/extends Section 6 | 1 |
| **23.x Cost-Sensitive Threshold Tuning** | inside Section 22 (Comprehensive Evaluation) | (bonus, fits eval) |
| **23.y Wilcoxon Signed-Rank Model Comparison** | inside Section 22 | (bonus, fits eval) |
| **23.z Overfitting Analysis Table** | inside Section 22 | (bonus, fits eval) |
| **24 Per-Location Performance & Error Analysis** | new section after Comprehensive Evaluation | 3, 5 |
| **25 Leave-One-Biosphere-Out CV (Bias Detection + Mitigation)** | new section after 24 | 4 |
| **27.6 SHAP per Biosphere** | inside Section 24 (SHAP) | 5 |
| **27.7 Cross-Region Feature Importance** | inside Section 24 | 5 |
| **30 Water-Quality Improvement Recommendations (FAO)** | new section after SHAP-vs-LIME | 6 |

Numbering after insertion will shift; the notebook markdown will use the new numbers throughout.

## 5. Component Specs

### 5.1 Temporal Feature Engineering (lecturer point 1)
- Parse `time` → `datetime`.
- Derived features: `month`, `day_of_year`, `day_of_week`, `hour`, `season` (wet=Oct–Mar / dry=Apr–Sep, South-Africa seasons), `is_weekend`.
- Per-site rolling features (where ≥3 obs at the site): rolling-3 mean of pH, EC, DO.
- Add these to the feature matrix and re-fit only the top model to demonstrate they help.

### 5.2 Chronological vs Random Split (lecturer point 1)
- Sort by `time`. Take first 80% chronologically as train, last 20% as test.
- Re-evaluate top model under chronological split.
- Compare metrics (Accuracy, F1, ROC-AUC) random-split vs chronological-split in a side-by-side table.
- Interpret: does performance drop under chronological split? (Indicates temporal drift if yes.)

### 5.3 Cost-Sensitive Threshold Tuning
- For top 3 models: sweep decision threshold ∈ [0.05, 0.95], plot precision/recall/F1/F2 vs threshold.
- Pick the threshold maximising F2 (recall-weighted, agriculturally safer — false "suitable" is costlier).
- Report new confusion matrices at chosen threshold.

### 5.4 Wilcoxon Signed-Rank Model Comparison
- Take 5-fold CV F1 scores per model (already available or compute).
- Pairwise Wilcoxon signed-rank test between top model and each other model.
- Report p-values in a heatmap.

### 5.5 Overfitting Analysis Table
- For each tuned model: `train_acc - test_acc`, `train_f1 - test_f1`.
- Tabulate; flag models with gap > 0.05 as overfit.
- One paragraph interpreting which regularisation choices worked.

### 5.6 Per-Location Performance & Error Analysis (lecturer point 3 + 5)
- Predict on the test set; join predictions back to `Biosphere`, `river`, `site`.
- Per-biosphere metrics table (Accuracy, Precision, Recall, F1, ROC-AUC, n).
- Per-biosphere confusion matrices (3-up subplot).
- Top-N most-misclassified sites table (`site`, `river`, `Biosphere`, error count).
- Scatter map (lat/lon) coloured by prediction-correct vs error.
- Narrative paragraph: which region/sites suffer most and why.

### 5.7 Leave-One-Biosphere-Out CV — Bias Detection + Mitigation (lecturer point 4)
- For each of {Kruger, Vhembe, Marico}: train on the other two, test on the held-out biosphere.
- Report F1, Accuracy, ROC-AUC per fold.
- **Mitigation experiments** (rerun LOBO CV for each):
  - (a) Drop `latitude`, `longitude`, `Altitude`, `Biosphere`, `river`, `site` → force chemistry-only model.
  - (b) Stratified resampling so each biosphere contributes equally.
- Compare baseline vs (a) vs (b) — table of F1 per held-out biosphere.
- Caveat: Marico n=14 → wide confidence interval; report but treat as indicative only.

### 5.8 SHAP per Biosphere + Cross-Region Importance (lecturer point 5)
- Compute SHAP values on test rows of each biosphere separately (using XGBoost — fastest, already trained).
- Per-biosphere SHAP bar plot (top 10).
- Cross-region table: feature × biosphere → mean |SHAP|.
- One paragraph per biosphere identifying its dominant driver and the environmental meaning.

### 5.9 Water-Quality Improvement Recommendations (lecturer point 6)
- For each common failure mode flagged by SHAP/per-region analysis, propose an FAO-aligned intervention:
  - High EC → dilution / drainage / leaching fraction
  - Low pH → liming
  - High pH → sulphur / acidifying amendments
  - Low DO → aeration / reduce organic load
- Per-region recommendation paragraph tied to the SHAP findings in 5.8.
- Reference: FAO Irrigation & Drainage Paper 29 thresholds (cite, do not fabricate numbers).

## 6. Data Flow

```
df (raw)
  → temporal features added (5.1)
  → existing cleaning + FE pipeline
  → chronological split alternative recorded (5.2)
  → existing model training reused
  → 5.3 / 5.4 / 5.5 wrap around existing comprehensive eval
  → 5.6 joins predictions to biosphere metadata
  → 5.7 retrains under LOBO + mitigation variants (separate from main models)
  → 5.8 uses existing XGBoost model + per-biosphere test slices
  → 5.9 narrative only, references 5.8 outputs
```

## 7. Error Handling / Edge Cases

- **Marico n=14**: explicit caveat box in 5.6 and 5.7; do not draw strong conclusions.
- **Sites with 1 obs**: rolling features become NaN → fill with site-level mean or biosphere-level mean.
- **Missing chemistry**: existing imputation pipeline already covers this — reuse, do not reinvent.
- **Time parse failures**: coerce to NaT, drop those rows from temporal analysis only (not from main pipeline).

## 8. Testing / Verification

- After implementation, restart kernel and run the entire notebook end-to-end.
- Confirm: no exceptions, all new figures render, all new tables produce values.
- Spot-check: per-biosphere metrics in 5.6 sum to overall test metrics from Section 22.

## 9. Deliverables

- Updated `notebooks/Group_2_Water_Quality_Irrigation_Suitability.ipynb` (new cells appended/inserted).
- This design doc, committed to `docs/superpowers/specs/`.
- No new modules, no `src/` files, no new dependencies (everything already in `requirements.txt`: scipy for Wilcoxon, sklearn for IsolationForest if used, matplotlib for maps).
