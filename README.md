# Private Equity Fund Dashboard

## Overview

This project bundles a synthetic private-equity dataset with a Streamlit application that presents the portfolio from both the general partner and limited partner perspectives. The dashboard surfaces geography-aware exposure, valuation history, forward-looking projections, company-level diagnostics, and capital deployment analytics. Supporting scripts keep the CSV inputs up to date, backfill valuation curves, and stage investor capital call schedules.

## Quick Start

1. Activate your Python environment (Python 3.10+ recommended) and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit experience from the project root:
   ```bash
   streamlit run dashboard.py
   ```
3. Optionally seed the CSVs with the latest synthetic refresh before starting Streamlit:
   ```bash
   python update_portfolio.py
   ```

All data files are expected to reside in the repository root alongside the application scripts.

## Dashboard Features

### Global Controls & Data Handling
- Sidebar multiselect filters narrow the portfolio by `Industry` and `Country`; every tab reuses the active subset.
- `st.cache_data` backed loaders provide memoised access to the three core CSVs and the public country GeoJSON.
- Helper utilities cleanse numeric columns, harmonise valuation year fields, and build a “latest month” snapshot by company.

### Manager View
- KPI header summarises active holdings, cumulative deployed capital, latest fund NAV, and an estimated dry powder buffer.
- Interactive `plotly` choropleth projects the fund’s global footprint, shading countries by current NAV and exposing company counts and invested dollars on hover.
- Historical NAV chart shows valuation progression; a companion projection chart extrapolates base/bull/bear scenarios five years out using log-linear regression.
- Industry performance bar chart ranks the latest average net income by sector for the filtered portfolio.
- Risk Watchlist evaluates leverage, profitability, and valuation momentum to score companies (`compute_company_risk`) and surfaces both a bar chart and detailed table.
- Follow-On Opportunities blends growth, margin, and leverage inputs (`compute_follow_on_candidates`) to highlight high potential reinvestment targets with a color-coded bar chart.
- ROE leaderboard tables celebrate the top and bottom performers based on most recent `ROE_%` values.

### Investor View
- Capital deployment area chart visualises the cumulative investment curve derived from acquisition dates and costs.
- Tabular NAV history exposes the trailing ten valuation points to support LP reporting.
- Latest valuation snapshot lists the top holdings for the most recent valuation year, easing “what’s driving NAV today?” conversations.

### Company Drilldown
- Single company selector reveals the most recent financial and return metrics for any holding, including revenue, net income, margin, ROE/ROA, and leverage ratios.

## Data Assets

| File | Description | Selected Columns |
|------|-------------|------------------|
| `company_profiles_150.csv` | Master entity registry for 150 portfolio companies. | `Company`, `Industry`, `Country` |
| `PE_Fund_timeline.csv` | Acquisition ledger with ownership and valuation history (2005‑2025). | `Acquisition_Month`, `Ownership_Percentage`, `Investment_Cost`, `Fund_Size_at_Acquisition`, `Valuation_of_company`, `Valuation_<YEAR>` |
| `synthetic_financial_data_150_companies.csv` | Monthly operating statements and leverage metrics. | `Month`, `Revenue`, `Net_Income`, `Assets`, `Equity`, `ROE_%`, `Debt/Equity`, … |
| `investor_capital_call_timeline.csv` | Generated capital-call detail by investor. | `Capital_Call_Date`, `Triggering_Investment`, `Investor`, `Investor_Contribution`, `Remaining_Commitment` |

Numeric values are stored without thousands separators to simplify ingestion. Divested holdings remain in the timeline (valuations zeroed) so historical NAV stays intact.

## Data Maintenance & Helper Scripts

- `update_portfolio.py` – Removes predefined divested companies, injects 20 new holdings across Nigeria, Sudan, South Africa, Brazil, India, Pakistan, and Canada, synthesises consistent valuations, and writes 36 months of bespoke financials per addition.
- `timeline_valuation.py` – Rebuilds the `Valuation_<YEAR>` columns by simulating year-over-year growth with industry- and country-specific shocks (2008 and 2020 included). Accepts optional CLI arguments for file paths and random seeds.
- `investors.py` – Models 25 LP commitments, stages onboarding cohorts, and produces the detailed `investor_capital_call_timeline.csv`, including cumulative and remaining obligations per investor.
- `file.py` – One-time factory that bootstraps the original 150-company dataset from a baseline Excel workbook and curated naming list. Useful when regenerating the entire synthetic universe.
- `st_test.py` – Streamlit sandbox showcasing map styling patterns used in the portfolio dashboard.

Each script reads and writes files in-place at the project root. Always close the Streamlit session (or let it auto-refresh) after running a data mutation script so cached data is invalidated.

## Typical Workflows

- **Refresh the synthetic portfolio:** `python update_portfolio.py` (optionally follow with `python timeline_valuation.py --seed 2025` for reproducible valuation curves).
- **Generate a fresh LP capital-call schedule:** `python investors.py`; inspect the summary in `investors_out.txt` and load the CSV in Excel or the dashboard as needed.
- **Operate the dashboard:** `streamlit run dashboard.py`, tweak sidebar filters, and explore the three tabs outlined above.

## Repository Layout

- `dashboard.py` – Primary Streamlit entry point wiring data ingestion, filtering logic, and tabbed visualisations.
- `update_portfolio.py`, `timeline_valuation.py`, `investors.py`, `file.py` – Data engineering utilities described above.
- `pages/` – Placeholder for optional multi-page Streamlit extensions.
- `requirements.txt` – Minimal dependency specification for reproduction.
- `*.csv` – Synthetic datasets consumed by the app and scripts.

## Notes & Customisation Tips

- The dashboard pulls world geometries from `https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson`; ensure outbound internet access when rendering maps.
- If you introduce new metrics or columns, update helper functions such as `clean_numeric`, `valuation_columns`, and downstream visualisations to recognise them.
- When integrating real data, revisit the pseudo-random assumptions in the maintenance scripts to avoid overwriting source-of-truth values.

