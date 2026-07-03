# NovaMart Retail Intelligence — Project Portfolio

> End-to-end Advanced Analytics engagement · 36-month retail dataset · Python production stack
> Simulated consulting delivery inspired by BCG X Delivery standards

---

## Project Summary

NovaMart is a fictional Fortune 500 specialty retail chain with 800 stores across North America.
Leadership observed a sustained decline in gross margin over 18 months but could not identify the
root cause — pricing, product mix, store operations, customer behaviour, or marketing inefficiency.

This project delivers the full analytics answer: from raw data through statistical confirmation,
ML modelling, scenario simulation, and a ranked strategic recommendation agenda.

**Scale of analysis:**

| Dimension | Value |
|-----------|-------|
| Transactions | 504,038 |
| Stores | 800 |
| Customers | 315,996 loyalty members |
| Products | 5,000 SKUs |
| Time horizon | 36 months (Jan 2022 – Dec 2024) |
| Total Revenue analysed | $122.5M |
| Total Gross Profit | $47.4M |

---

## Dashboard KPIs

The Streamlit executive dashboard surfaces the following live KPIs computed directly from the
parquet data layer:

| KPI | Value | Period |
|-----|-------|--------|
| Total Revenue | **$122.5M** | 36-month cumulative |
| Gross Profit | **$47.4M** | 36-month cumulative |
| Average Gross Margin | **38.7%** | 36-month blended |
| Margin — Year 1 | **42.0%** | Months 1–12 |
| Margin — Year 3 | **31.8%** | Months 25–36 |
| Margin Drift | **−10.2pp** | Y1 → Y3 |
| Active Stores | **800** | |
| Loyalty Members | **316K** | |
| Top-10% stores revenue share | **30.5%** | Pareto concentration |
| Top-20% customers revenue share | **67.0%** | Power-law distribution |

---

## Key Analytical Findings

### 1. Margin Decline — Confirmed & Quantified

Gross margin declined from **42.0% → 31.8%** over 36 months (−24.3% relative).

- OLS regression: slope = −0.326pp/month, **R² = 0.917, p < 0.001**
- The trend is statistically indistinguishable from a straight line — systematic, not cyclical
- Root cause: COGS inflation beginning Month 13 (January 2023), ~0.75pp/month, with no
  corresponding price adjustment
- At current trajectory: operating profitability reaches zero within the medium-term horizon
  without structural intervention

### 2. Store Portfolio — Extreme Concentration

- Cluster A (140 stores, 17.5% of portfolio) → **43% of total profit**
- Cluster C (202 stores, 25.3% of portfolio) → **9% of total profit**
- Top 10% of stores (80 stores) → **30.5% of revenue** ($37.3M of $122.5M)
- Store revenue is highly right-skewed (skew = 1.80)
- Lifting bottom-quartile stores to the median = estimated **+$9.3M annual revenue**

### 3. Silver Tier Churn Crisis

- Silver loyalty customers show materially lower purchase frequency from Month 25 onward
- Trigger: loyalty fee hike, January 2024 (Month 25)
- Frequency decline: statistically confirmed — **p < 0.001, Cohen's d = 0.72 (large effect)**
- Revenue at risk: **$8–12M annually** if trend continues

### 4. Private Label Margin Opportunity

- Private label average gross margin: **50.4%** vs national brand: **28.8%** → **+21.7pp gap**
- Price-to-cost multiplier: **2.30×** (private label) vs **1.60×** (national brand)
- Private label penetration is *declining* despite this structural advantage
- 5pp share increase → estimated **+$4.2M incremental annual profit**

### 5. Marketing Mix Inefficiency

- Urban stores: digital ROI = **3.0× revenue per $1 spent** / TV = 0.8×
- Rural stores: TV ROI = **2.5×** / digital = 0.8×
- Current budget allocation ignores this segmentation — significant urban TV waste
- Redirecting urban TV → digital = **+$5.6M revenue impact** (Rec #5)

---

## Statistical Testing

5 pre-specified hypotheses tested at α = 0.05. All results include effect sizes.

### H1 — Cluster A Revenue vs B/C · One-Way ANOVA

**Question:** Is the revenue gap between store clusters statistically real or noise?

- **Test:** One-way ANOVA — 3 groups (A, B, C), annual store revenue
- **Why ANOVA:** 3+ groups → can't run multiple t-tests (inflates Type I error rate)
- **Assumptions:** normality per group, homogeneity of variance, independence
- **Result:** F(2, 797) = significant, **p < 0.001, η² = 0.31 (large effect)**
- **Interpretation:** η² = 0.31 means 31% of total variance in store revenue is explained by cluster
  membership — a business-meaningful segmentation, not a statistical artefact
- **Business conclusion:** Cluster A outperformance is real. Targeted support for B/C stores is
  justified by evidence.

---

### H2 — Silver Tier Churn · Welch's Two-Sample t-test (one-tailed)

**Question:** Did the Month-24 loyalty fee hike materially reduce Silver customer purchase frequency?

- **Test:** Welch's t-test, early (M1–24) vs late (M25–36) avg monthly transactions per customer
- **Why Welch (not Student):** doesn't assume equal variances between groups — always preferred
- **Why one-tailed:** directional hypothesis (frequency early > frequency late)
- **Assumptions:** approximate normality (CLT applies at n > 30), independence
- **Result:** **t significant, p < 0.001, Cohen's d = 0.72 (large effect)**
- **Interpretation:** d = 0.72 → the two distributions are separated by 0.72 pooled standard
  deviations. With n in the thousands, even a small p doesn't guarantee business significance —
  d = 0.72 confirms this difference *is* practically important.
- **Business conclusion:** The fee hike demonstrably damaged Silver engagement. Revenue at risk: $8–12M/year.

---

### H3 — Private Label Margin · Welch's t-test (two-tailed)

**Question:** Is the private label gross margin advantage statistically significant?

- **Test:** Welch's t-test on product-level gross margin %, private label vs national brand
- **Why two-tailed:** no prior directional assumption; testing whether margins differ
- **Result:** **p < 0.001, Cohen's d = 1.84 (very large effect)**
- **Interpretation:** d = 1.84 is exceptionally large — private label margin distributions barely
  overlap. The 21.7pp gap is not measurement noise; it is structural.
- **Business conclusion:** Private label is NovaMart's highest-leverage margin tool. Every 1pp
  penetration increase directly improves portfolio margin without requiring volume growth.

---

### H4 — Urban Digital ROI · Mann-Whitney U (one-tailed)

**Question:** Do urban stores achieve higher digital marketing ROI than rural stores?

- **Test:** Mann-Whitney U — non-parametric equivalent of t-test on ROI distributions
- **Why non-parametric:** ROI distributions are skewed (outlier stores with exceptional ROI);
  t-test would be biased. Mann-Whitney works on ranks, not raw values.
- **Effect size:** Rank-biserial correlation r
- **Assumptions:** independence, ordinal/continuous data, same distribution shape
- **Result:** **p < 0.001** — urban digital ROI stochastically dominates rural
- **Interpretation:** A randomly chosen urban store's digital ROI is significantly more likely to
  exceed a randomly chosen rural store's than would occur by chance
- **Business conclusion:** Channel ROI is format-dependent. Urban markets warrant higher digital
  allocation; rural markets warrant TV-first strategy.

---

### H5 — Discount Rate vs Gross Margin · Pearson Correlation

**Question:** Do deeper discounts erode gross margin at the transaction level?

- **Test:** Pearson r on discount_pct × gross_margin_pct (100K transaction sample, seed=42)
- **Why Pearson:** both variables continuous, linear relationship hypothesis, n large enough to
  satisfy normality via CLT
- **Assumptions:** linearity, bivariate normality, no extreme outliers, independence
- **Result:** **r = −0.41, p < 0.001** — significant negative correlation
- **Interpretation:** r = −0.41 is a medium-to-large effect. Each 1pp increase in average
  discount depth is associated with ~0.41pp of gross margin erosion. With n = 504K, any r ≠ 0
  would be significant — effect size matters more than p-value here.
- **Business conclusion:** Blanket discounting erodes margin without guaranteed volume offset.
  Targeted, elasticity-informed discounting is required.

---

## Machine Learning Models

4 production models trained with stratified cross-validation and SHAP explainability.

### Churn Prediction

| Item | Detail |
|------|--------|
| Business question | Which loyalty customers will disengage in the next 90 days? |
| Target | Binary: churned / active |
| Algorithms compared | Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost |
| Best model | **XGBoost** |
| AUC-ROC | **0.98** |
| Top SHAP features | Days since last purchase · loyalty tier · purchase frequency · discount dependency |

### Store Performance

| Item | Detail |
|------|--------|
| Business question | Which stores are underperforming relative to their potential? |
| Target | Continuous revenue / margin; cluster classification |
| Algorithms compared | Ridge, Random Forest, Gradient Boosting |
| Best model | **Random Forest** |
| R² | **0.87** |
| Top SHAP features | Performance cluster · marketing efficiency · manager tenure · digital spend share |

### Marketing Mix Model (MMM)

| Item | Detail |
|------|--------|
| Business question | What is the revenue return per dollar by channel and store format? |
| Method | Ridge regression with Adstock transformation (geometric decay) |
| Output | Channel ROI coefficients by urban/rural segment |
| Key finding | Digital ROI = 3.0× (urban), 0.8× (rural); TV ROI = 0.8× (urban), 2.5× (rural) |

### Price Elasticity

| Item | Detail |
|------|--------|
| Business question | Which categories support price increases without volume loss? |
| Method | Log-log OLS per category (ln(quantity) ~ ln(price)) |
| R² range | 0.65 – 0.82 per category |
| Most inelastic | Food & Beverage (ε = −0.45) — price increase recommended |
| Most elastic | Electronics (ε = −1.8) — price increase would reduce total revenue |

---

## Strategic Recommendations

**Total addressable opportunity: $32.4M in annual impact**

Recommendations are RICE-scored (Reach × Impact × Confidence / Effort):

| # | Action | Category | Timeline | Revenue Impact |
|---|--------|----------|----------|---------------|
| 1 | Targeted discount reduction for price-insensitive segments | Pricing | 0–30 days | Margin recovery |
| 2 | +5% price on inelastic categories (F&B, Health & Beauty) | Pricing | 1–3 months | $60K |
| 3 | Silver tier loyalty rescue programme | Retention | 0–30 days | $275K |
| 4 | Private label expansion — top-10 revenue categories | Mix | 3–6 months | $1.0M |
| 5 | Redirect urban TV budget → digital + email | Marketing | 1–3 months | $5.6M |
| 6 | Full urban digital budget reallocation | Marketing | 3–6 months | **$23.4M** |
| 7 | Manager development programme — mid-performing stores | Operations | 3–6 months | $2.5M |
| 8 | Close / reformat bottom 5% underperforming stores | Portfolio | 6–12 months | −$384K (cost save) |

---

## Technology Stack

| Layer | Stack |
|-------|-------|
| Language | Python 3.12 |
| Dashboard | Streamlit + Plotly (8 pages) |
| API | FastAPI (async, Pydantic, OpenAPI docs) |
| ML | scikit-learn · XGBoost · LightGBM · CatBoost |
| Explainability | SHAP TreeExplainer |
| Statistics | SciPy · Statsmodels · Pingouin |
| GenAI Copilot | Anthropic Claude API — tool-use agentic loop |
| Experiment tracking | MLflow (local) |
| Data format | Parquet (7 tables via DataLoader) |
| Config | Pydantic BaseSettings + .env |
| Tests | 117 unit tests (pytest) |
| Containers | Docker + Compose |
| CI/CD | GitHub Actions |

---

## GenAI Copilot

The dashboard includes an AI copilot powered by Anthropic Claude via the **tool-use pattern**:

- Claude receives the user's question + 7 analytics tool schemas
- It selects which tools to call and with what parameters
- Tools execute against live NovaMart parquet data → return structured JSON
- Claude synthesises a consulting-quality Markdown answer grounded in real figures
- The loop repeats up to 5 iterations for complex multi-part questions
- Every response shows which tools were called — full auditability, no hallucination

**Why tool-use over RAG:** The analytics layer returns structured, aggregated KPIs — vector
similarity retrieval would add noise on data that is already perfectly queryable.

Tools: `get_financial_summary` · `get_churn_analysis` · `get_store_performance` ·
`get_marketing_roi` · `get_pricing_analysis` · `get_recommendations` · `simulate_scenario`

---

*Analytics Simulation · BCG X–Inspired Methodology · Reproducible seed=42*
