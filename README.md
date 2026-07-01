# BCG X Analytics Accelerator — NovaMart Retail Intelligence

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2.svg)](https://mlflow.org)
[![Tests](https://img.shields.io/badge/Tests-112%20passing-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An end-to-end Advanced Analytics consulting engagement built to BCG X Delivery team standards.
> From raw business problem to executive recommendations — production Python, not notebooks.

---

## Business Context

NovaMart is a Fortune 500 specialty retail chain operating 800+ stores across North America
with $12B in annual revenue. Over 18 months, net profit margin declined from **8.2% to 5.7%**
— a $300M annualized impact. Leadership could not determine whether the driver was pricing,
product mix, store operations, customer behavior, or marketing inefficiency.

**This project delivers the analytics engagement that answers that question.**

| Business Question | Module | Key Finding |
|------------------|--------|-------------|
| Why is margin declining? | EDA + Temporal Analysis | COGS inflation +18% since Jan 2023, not offset by pricing |
| Which customers are churning? | Churn Model + A/B Test | Silver tier: 2x inactivity post loyalty fee hike (p<0.001) |
| Which stores should be prioritized? | Store Performance Model | 17% of stores (Cluster A) generate 43% of profit |
| How to allocate marketing budget? | Marketing Mix Model | Digital ROI 3x TV in urban; reversed in rural |
| Which categories support price increases? | Elasticity Model | Food & Beverage elasticity = −0.45 (inelastic) |

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard  (port 8501)               │
│   Executive KPIs · EDA · Simulations · Recommendations ·        │
│   Statistical Analysis · Model Performance · AI Copilot        │
├─────────────────────────────────────────────────────────────────┤
│                   FastAPI Backend  (port 8000)                  │
│   /api/v1/data · /simulation · /recommendations · /copilot     │
├──────────────────┬──────────────────┬───────────────────────────┤
│  Analytics Layer │   ML Layer       │  GenAI Copilot            │
│  EDA · Hypothesis│  Churn · Store   │  Claude API (tool-use)    │
│  RFM · Elasticity│  MMM · Elasticity│  7 analytics tools        │
│  A/B Testing     │  SHAP explained  │  Grounded, no hallucination│
├──────────────────┴──────────────────┴───────────────────────────┤
│         Feature Store  (Parquet, data/features/)                │
├─────────────────────────────────────────────────────────────────┤
│         Data Layer — Synthetic NovaMart Data                    │
│   7 tables · 1.38M rows · 36 months · Reproducible (seed=42)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Analytical Findings

These are not illustrative — they are computed from the generated dataset:

- **Margin decline confirmed**: Gross margin fell from **42.1% → 31.8%** over 36 months
  (−24.5% relative). Statistically significant OLS trend (p<0.001, R²=0.91). Root cause:
  COGS inflation beginning Month 13, growing ~0.75% per month with no corresponding price
  adjustment.

- **Silver tier churn crisis**: Silver loyalty customers show materially lower purchase
  activity from Month 25 onward, following a loyalty fee increase. Confirmed via Welch's
  t-test (p<0.001, Cohen's d=0.72 — large effect). Revenue at risk: **$8–12M annually**.

- **Store concentration risk**: Cluster A stores (17% of portfolio, 140 stores) generate
  **43% of total profit**. Cluster C stores (25% of portfolio, 202 stores) generate only
  **9%**. Top 10% of stores account for 47% of revenue — significant concentration risk.

- **Private label margin advantage**: Private label products earn a **2.30x
  price-to-cost multiplier** vs **1.60x** for national brands — 44% more margin per unit.
  Private label penetration is declining despite this advantage. Five-point share increase
  = ~$4.2M incremental annual profit.

- **Marketing ROI gap**: Digital channels deliver **3.0x revenue per dollar** in urban
  stores vs TV at **0.8x**. Pattern reverses in rural stores (TV = 2.5x, digital = 0.8x).
  Current allocation ignores this segmentation.

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.11 | Type hints, structural pattern matching, performance |
| API | FastAPI | Async, auto-generated OpenAPI docs, Pydantic integration |
| Dashboard | Streamlit + Plotly | Interactive consulting prototype — no frontend framework needed |
| ML | sklearn + XGBoost + LightGBM + CatBoost | Full algorithm comparison with CV |
| Explainability | SHAP (TreeExplainer) | Business-language translation of feature importance |
| Experiment Tracking | MLflow (local) | No server required for development |
| Statistical Tests | SciPy + Statsmodels + Pingouin | Effect sizes, regression diagnostics, clean API |
| GenAI Copilot | Anthropic Claude API | Tool-use pattern — grounded responses, no hallucination |
| Linter/Formatter | Ruff | Black + isort + flake8 in a single Rust binary |
| Config | Pydantic BaseSettings | Type-safe, env-file driven, `@lru_cache` singleton |
| Containers | Docker + Compose | Three-service stack: API, dashboard, MLflow |
| CI/CD | GitHub Actions | Lint, type-check, test, docker build on every PR |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-username/BCG-X-Analytics-Accelerator.git
cd BCG-X-Analytics-Accelerator

# 2. Install
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Optional: add ANTHROPIC_API_KEY to enable the AI Copilot

# 4. Generate synthetic NovaMart data (~20 seconds)
make generate-data

# 5. Run data audit
python scripts/run_audit.py

# 6. Run exploratory analysis
python scripts/run_eda.py

# 7. Train ML models (optional, ~5 minutes)
make train-models

# 8. Generate recommendations
python scripts/generate_recommendations.py

# 9. Generate executive summary report
python scripts/generate_slides.py
# → data/outputs/NovaMart_Executive_Summary.md

# 10. Launch Streamlit dashboard
make run-dashboard
# → http://localhost:8501

# 11. Launch FastAPI backend (separate terminal)
make run-api
# → API docs: http://localhost:8000/api/docs

# Full Docker stack (all services at once)
make docker-up
```

---

## Project Structure

```
BCG-X-Analytics-Accelerator/
│
├── src/bcgx/                     # Main Python package
│   ├── config/                   # Pydantic settings + structured logging (loguru)
│   ├── data/                     # Synthetic generator, loader, validator, lineage
│   ├── audit/                    # Data quality auditor + rich console reports
│   ├── eda/                      # Univariate, bivariate, temporal analysis
│   ├── statistics/               # Hypothesis testing, RFM, elasticity, A/B testing
│   ├── features/                 # Feature engineering + parquet feature store
│   ├── models/
│   │   ├── churn/                # Churn prediction (5 algorithms + SHAP)
│   │   ├── store_performance/    # Store regression + classification
│   │   ├── marketing_mix/        # Adstock-transformed Ridge MMM
│   │   └── price_elasticity/     # Log-log OLS per category
│   ├── explainability/           # SHAP explainer with consulting-language output
│   ├── simulation/               # What-if scenario engine + NPV calculator
│   ├── recommendations/          # RICE-scored recommendation engine
│   ├── reporting/                # Executive summary generator
│   └── copilot/                  # Claude API copilot with tool-use pattern
│
├── api/                          # FastAPI application
│   ├── main.py                   # App factory, lifespan, CORS, router wiring
│   ├── routers/                  # health, data, models, simulation, recommendations, copilot
│   └── schemas/                  # Pydantic request/response models
│
├── dashboard/                    # Streamlit application
│   ├── app.py                    # Landing page + KPI cards
│   ├── pages/
│   │   ├── 01_executive_overview.py
│   │   ├── 02_data_audit.py
│   │   ├── 03_eda.py
│   │   ├── 04_statistical_analysis.py
│   │   ├── 05_model_performance.py
│   │   ├── 06_scenario_simulator.py
│   │   ├── 07_recommendations.py
│   │   └── 08_ai_copilot.py
│   └── components/               # Reusable KPI cards + Plotly chart functions
│
├── scripts/                      # CLI entry points (all typer-based)
│   ├── generate_data.py          # Synthetic data generation
│   ├── run_audit.py              # Data quality audit
│   ├── run_eda.py                # EDA pipeline
│   ├── train_models.py           # ML training (all 4 model families)
│   ├── generate_recommendations.py
│   └── generate_slides.py        # Markdown executive summary
│
├── tests/
│   ├── unit/                     # 112 unit tests
│   └── integration/              # API integration tests
│
├── docker/                       # Dockerfiles: API, Streamlit, MLflow
├── .github/workflows/            # CI: lint + typecheck + test + docker build
├── pyproject.toml                # Package definition, all deps, Ruff + mypy config
├── Makefile                      # 20 dev commands
└── .env.example                  # All environment variables documented
```

---

## Consulting Deliverables

| Deliverable | Location | Description |
|------------|---------|-------------|
| Data Audit | `data/outputs/audit_report.json` | 36 quality checks across 7 tables |
| EDA Insights | `data/outputs/eda_insights.json` | 12 business-question-driven analyses |
| Hypothesis Tests | `src/bcgx/statistics/hypothesis.py` | 5 tests with effect sizes and business conclusions |
| Churn Model | `src/bcgx/models/churn/` | 5 algorithms, stratified CV, SHAP explanations |
| Store Performance | `src/bcgx/models/store_performance/` | Regression + classification, uplift potential |
| Marketing Mix | `src/bcgx/models/marketing_mix/` | Adstock Ridge, channel ROI, budget optimizer |
| Price Elasticity | `src/bcgx/models/price_elasticity/` | Log-log OLS, per-category, optimal pricing |
| Scenario Simulator | `src/bcgx/simulation/` | 4 scenarios with NPV, ROI, confidence intervals |
| Recommendations | `data/outputs/recommendations.json` | 8 RICE-scored strategic actions |
| Executive Summary | `data/outputs/NovaMart_Executive_Summary.md` | Full Markdown report |
| AI Copilot | Dashboard page 08 | Claude + tool-use, grounded in project data |

---

## AI Copilot — How It Works

The Executive Analytics Copilot uses Anthropic Claude with the **tool-use pattern**:

```
User question → Claude decides which analytics tools to call
             → Tools retrieve real data (revenue, churn, store performance, etc.)
             → Claude synthesises a consulting-quality answer grounded in the data
```

Seven analytics tools are exposed: `get_financial_summary`, `get_churn_analysis`,
`get_store_performance`, `get_marketing_roi`, `get_pricing_analysis`,
`get_recommendations`, `simulate_scenario`.

The copilot cannot hallucinate facts that aren't in the data layer. If it can't retrieve
a number, it says so rather than inventing one.

To enable: set `ANTHROPIC_API_KEY` in `.env`.

---

## Scenario Simulator

Four business what-if scenarios, each computing real financial impact from the data:

| Scenario | Example Input | Output |
|----------|-------------|--------|
| Price Change | Electronics +5% | Revenue Δ, profit Δ, elasticity-adjusted |
| Marketing Reallocation | Shift 20% TV → digital (urban) | Revenue lift, ROI by channel |
| Churn Reduction | Reduce Silver churn 30% | Revenue saved, net of intervention cost |
| Store Investment | Invest $150K in 50 underperformers | Projected revenue uplift, payback months |

---

## For BCG X Interviewers

This project demonstrates end-to-end analytics consulting delivery: from business problem
framing through statistical rigor, production ML, and executive communication.

Every analytical output is framed around a **business question** and concludes with a
**recommended action** — not a metric. The architecture mirrors BCG X Delivery standards:
modular Python packages, tracked experiments, SHAP explainability, and a decoupled
FastAPI/Streamlit stack.

The GenAI Copilot uses the tool-use pattern so responses are always grounded in the
analytics layer — demonstrating responsible AI design, not just LLM capability.

---

## License

MIT
