# Adversarial-Robust Fraud Detection

A fraud detection model built on the IEEE-CIS dataset, evaluated not just on accuracy
but on how well it holds up against a fraudster actively trying to evade it.

Most student fraud-detection projects stop at "here's my F1 score." This one goes
further: it identifies *why* the model fails where it fails, tries to fix it, and
honestly reports where the fix worked and where it didn't.

## Dataset

IEEE-CIS Fraud Detection (Kaggle), 354,324 transactions, 3.5% fraud rate.
Two tables: `train_transaction.csv` (394 features) and `train_identity.csv`
(41 features, covering ~24% of transactions), merged on `TransactionID`.

## 1. Exploratory Data Analysis

Key findings that shaped feature selection:

- **Time of day**: fraud rate peaks at hour 7 (~10.5%) vs a ~3.5% baseline —
  the strongest single pattern found in the raw data.
- **ProductCD**: category `C` has an 11.8% fraud rate vs 2.0% for the majority
  category `W` — a ~6x gap.
- **Card type**: credit cards show 2.7x the fraud rate of debit cards.
- **Identity data presence**: transactions with identity/device data attached have
  **4x** the fraud rate of those without (8.0% vs 2.1%) — likely because identity
  verification is triggered more often on suspicious transactions.
- **Device fingerprinting**: specific Android device-model strings (not generic
  OS labels) repeat disproportionately across fraud transactions — a device-farm
  signature.
- **V257** (one of 339 anonymized Vesta features): 100% missing for
  `ProductCD='W'`, present for other categories. Its *value*, not just its
  presence, carries fraud signal independent of ProductCD — confirmed later via
  model feature importance.

## 2. Baseline Model

XGBoost classifier, 11 engineered features (22 after one-hot encoding),
`scale_pos_weight` to handle the 96.5/3.5 class imbalance.

| Metric | Value |
|---|---|
| ROC-AUC | 0.837 |
| PR-AUC | 0.413 |

## 3. Connectivity Features (Phase 2)

Added graph-inspired features without a full graph library: shared-identifier
frequency counts (`card1_count`, `addr1_count`, `device_count`) and a
`risky_device_flag` built **only from training data** (to avoid leakage) that
flags device strings appearing frequently among fraud cases.

| Metric | Before | After |
|---|---|---|
| PR-AUC | 0.413 | **0.475** (+15%) |
| ROC-AUC | 0.837 | **0.880** |

## 4. Adversarial Red-Teaming (the differentiator)

Simulated a realistic fraudster: shrink the transaction amount, shift to an
off-peak hour — the two levers an attacker actually controls.

**Aggregate result**: 23.4% of previously-caught fraud evades detection under
this attack.

**Deeper finding**: this vulnerability is not uniform.

| Group | Evasion rate |
|---|---|
| High-confidence catches (n=914) | 10.2% |
| Low-confidence catches (n=239) | **74.1%** |

Root cause, confirmed by comparing feature values between groups: vulnerability
concentrates in fraud cases with a **weak `V257` signal** (median = 1). High-`V257`
fraud (median = 3–4) sits deep in "confident fraud" territory and is robust.
Low-`V257` fraud sits near the decision boundary and is easily nudged across it.

## 5. Hardening — Two Attempts, Honest Results

**Attempt 1**: adversarial training using one fixed perturbation
(amount × 0.7, hour = 14) on the vulnerable subgroup.
Evasion on that *same* pattern dropped to 1.3% — but on a genuinely new
perturbation, evasion was 21.7% (hardened) vs 19.8% (original). **The model had
memorized the specific attack, not learned general robustness.**

**Attempt 2**: adversarial training using a *randomized range* of perturbations
(amount × 0.5–0.9, six different hours). Results:

| Attack pattern | Original model | Hardened model |
|---|---|---|
| Within trained range | 23.4% | **3.3%** |
| Outside trained range (untested hour) | 7.8% | 15.2% |

**Conclusion**: adversarial training generalizes *within* the perturbation
space it was trained on, but does not extrapolate beyond it — a known,
expected property of adversarial training, not an implementation bug. Full
robustness would require covering the complete realistic attacker parameter
space (all 24 hours, not a curated subset), not a narrow augmentation set.

**Cost**: hardening reduced PR-AUC from 0.475 to 0.360 — a real
precision-for-robustness tradeoff, not a free win.

## 6. Explainability

SHAP values confirm and refine the feature-importance story: `TransactionAmt`
has the largest aggregate impact, while `V257` shows a distinctive pattern —
near-zero impact for most transactions but a long tail of strongly positive
impact when its value is high, consistent with the red-teaming findings above.

## 7. Dashboard

An interactive Streamlit app (`dashboard.py`) with four views: model
performance, feature importance, the adversarial red-teaming story, and a
live "test a transaction" tool.

## What I'd Do With More Time

- Full graph-based features (community detection for mule rings) instead of
  the simplified connectivity-count approximation used here
- Cover the full attacker parameter space during adversarial training rather
  than a curated subset
- An ensemble/second-stage model specifically for borderline (low-confidence)
  cases, since that's where the real vulnerability lives
- LLM-based document/transaction narrative analysis as a complementary signal

## Stack

Python, pandas, XGBoost, scikit-learn, SHAP, Streamlit, matplotlib
