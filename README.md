# Private Equity Fund Dashboard

This repository contains a synthetic private equity dataset along with the Streamlit
application that renders a manager / LP dashboard. The bulk of the data is stored in
three CSV files under the repository root; a helper script keeps them in sync when the
portfolio changes.

## Data Files

| File | Purpose | Key Columns |
|------|---------|-------------|
| `company_profiles_150.csv` | Master list of portfolio companies. | `Company` (unique name), `Industry`, `Country` |
| `PE_Fund_timeline.csv` | One row per acquisition. Tracks historic valuations and invested capital. | `Acquisition_Month`, `Company`, `Ownership_Percentage`, `Investment_Cost`, `Fund_Size_at_Acquisition`, `Valuation_of_company`, `Valuation_<YEAR>` columns for 2005-2025 |
| `synthetic_financial_data_150_companies.csv` | Monthly operating metrics for each company. | `Company`, `Month`, `Revenue`, `Net_Income`, `Assets`, `Equity`, `ROE_%`, etc. |

All numeric columns are stored as plain numbers (no thousands separators). The timeline
file keeps prior acquisitions even if a holding is divested—rows are preserved and their
valuation fields are zeroed. The `Valuation_<YEAR>` columns are used to produce NAV
history, while `Valuation_of_company` mirrors the latest year.

## Application Structure

- `dashboard.py` – Streamlit app that loads the CSVs, applies sidebar filters, and renders
  three tabs (Manager View, Investor View, Company Drilldown). Notable components:
  - `load_*` helpers cache the CSVs.
  - Derived metrics include NAV aggregation, deployment curves, and per-country summaries.
  - Manager View features a pydeck choropleth using `GeoJsonLayer` to shade invested
    countries and display key metrics.

- `update_portfolio.py` – Script that removes divested companies and injects new holdings
  across all CSVs. It assigns random investment sizes, generates consistent valuations,
  and synthesizes 36 months of financials per company. Re-running the script is
  idempotent: it strips previously generated records before appending fresh ones.

## Running the Dashboard

Install dependencies (Streamlit, pandas, numpy, plotly, pydeck):

```bash
pip install -r requirements.txt  # or install the libraries individually
```

Launch the dashboard:

```bash
streamlit run dashboard.py
```

The app expects the three CSVs to live alongside `dashboard.py`. Any updates written by
`update_portfolio.py` are picked up on the next Streamlit rerun.

## Updating the Portfolio

Whenever the simulated fund adds or exits companies, run:

```bash
python update_portfolio.py
```

The script will:

1. Remove the three largest U.S. positions from the data and zero-out their valuations in
   the timeline.
2. Append 20 new companies (Nigeria, Sudan, South Africa, Brazil, India, Pakistan,
   Canada) with acquisitions spread across Aug–Nov 2025 and investments between $10M–$100M.
3. Generate monthly financial statements for the new holdings and update NAV figures.

After the script runs, restart Streamlit (or let it auto-rerun) to see the refreshed
charts and map.

## Notes

- The dataset is entirely synthetic and intended only for demonstration.
- If you modify the CSV schema, keep the derived functions in `dashboard.py` in sync
  (especially NAV aggregation and country summaries).
- The world map uses a remote GeoJSON file; the app requires internet access to load it.

