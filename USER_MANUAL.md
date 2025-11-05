# Private Equity Fund Dashboard – User Manual

This guide explains how to launch the Streamlit application and interact with each on-screen element. Use it as a quick reference for operators, analysts, and stakeholders who need to explore the simulated private-equity portfolio.

## 1. Launching the Application

1. From a terminal in the repository root, install dependencies if you have not already:
   ```bash
   pip install -r requirements.txt
   ```
2. Start Streamlit:
   ```bash
   streamlit run dashboard.py
   ```
3. After a few seconds your browser opens to `http://localhost:8501`, showing the dashboard.
4. (Optional) Run `python update_portfolio.py` beforehand to refresh the synthetic dataset, or other maintenance scripts if you want updated valuations or capital call timelines.

## 2. Page Layout Overview

The dashboard is a single Streamlit page with three tabs along the top: **Manager View**, **Investor View**, and **Company Drilldown**. The left sidebar contains interactive filters that control all content on the page.

### 2.1 Sidebar Filters

- **Industry** and **Country** multiselect controls default to “All”.
- Use the search box or checkboxes to narrow the universe. The header beneath the title confirms how many companies remain after filtering.
- Selected values appear as tags; scroll inside the selection area to manage long lists. Clear selections via the `x` icon in each tag or `⌫` / `Backspace` when focused.
- Every chart, table, and metric updates immediately when filters change. Streamlit’s caching keeps reloads fast.

### 2.2 Header Metrics

When a tab is active, summary metrics appear at the top of the tab. Hover the `ℹ️` tooltips for definitions. Values always reflect the currently filtered portfolio.

## 3. Manager View

The Manager View aggregates the portfolio across multiple dimensions to support fund-level oversight. Scroll within the tab to see all components.

### 3.1 KPI Panel
- **Active Portfolio**: number of holdings visible under current filters.
- **Deployed Capital**: sum of `Investment_Cost` for filtered companies.
- **Latest NAV**: most recent aggregated valuation (last `Valuation_<YEAR>` column).
- **Dry Powder (est)**: maximum `Fund_Size_at_Acquisition` minus deployed capital. Treat as an indicator of remaining capacity.
- Hover any metric to read additional context.

### 3.2 Global Investment Footprint Map
- Plotly choropleth shades each invested country by aggregated current NAV.
- Move your cursor over a country to see company count, invested capital, and NAV.
- Use the mouse wheel (or trackpad pinch) to zoom; click-and-drag to pan. Double-click resets zoom.
- Legend on the right shows the color scale. If a country is not shaded, it means no active holdings match the current filters.

### 3.3 Country Summary Table
- Appears below the map. Lists each included country with total companies, invested amount, and current net worth.
- Click column headers to sort (Streamlit interactive tables support single-column sorting via the UI overflow menu).
- Use the table options menu (three dots) to copy data or download as CSV.

### 3.4 Fund NAV Timeline
- Line chart showing historical NAV aggregated by year.
- Hover points for precise values; double-click legend items to isolate a trace (if multiple appear after modifications).

### 3.5 Projected NAV Outlook
- Displays historical NAV along with Base, Bear, and Bull forecasts derived from log-linear regression.
- Hover to compare scenario values year by year. Forecasts always extend five years beyond the latest historical point.

### 3.6 Industry Net Income Ranking
- Bar chart of average latest-month net income by industry.
- Hover for exact values; bars reflect only companies remaining after filters.

### 3.7 Risk Watchlist
- **Bar Chart**: Top 15 companies by composite risk score. Color encodes `High`, `Moderate`, or `Low` levels.
- **Details Table**: Full list with valuation CAGR, debt-to-equity, net income, and computed risk score. Use search or column sorting in the table menu for deeper inspection.
- Interpretation: Higher risk scores indicate leverage pressure, negative valuation momentum, or losses; target follow-up for high-risk companies.

### 3.8 Co-Invest & Follow-On Opportunities
- **Bar Chart**: Highlights companies with the strongest follow-on score (growth + margin + leverage blend). Colors map to `High`, `Moderate`, or `Watch` opportunity.
- **Details Table**: Full breakdown of scores and underlying metrics.
- Use these outputs to shortlist reinvestment candidates.

### 3.9 ROE Leaderboards
- Two tables showing the top and bottom 10 `ROE_%` values in the latest month.
- Compare net income alongside ROE to contextualise efficiency vs scale.

## 4. Investor View

Oriented toward LP reporting and capital pacing.

### 4.1 Capital Deployment Curve
- Area chart of cumulative investment by acquisition year.
- Hover to view total deployed capital at each point. Use it to communicate pace of deployment.

### 4.2 NAV History Table
- Tabular presentation of the most recent ten NAV values. Values appear once the timeline contains at least one valuation column post-filtering.
- Export via the table menu (three dots) if needed for offline analysis.

### 4.3 Latest Valuation Snapshot
- Table listing the top 15 companies by valuation in the most recent year.
- Useful for explaining short-term NAV drivers. Filter the list via sidebar settings to focus on regions or industries.

## 5. Company Drilldown

### 5.1 Company Selector
- Dropdown lists all companies in the filtered dataset (alphabetical).
- Start typing to search; select a company to reveal its profile table.

### 5.2 Metrics Table
- Displays the latest month’s values for revenue, net income, gross margin, ROE, ROA, and debt/ equity alongside company metadata.
- Use the table toolbar to copy or download metrics. When filters reduce the dataset significantly, this view becomes a quick audit of individual holdings.

## 6. Working With Filters and Data Refresh

- Filters are cumulative. If results disappear, click the `x` icon on tags or use the “Clear selection” option (three dots) to restore defaults.
- After running maintenance scripts (`update_portfolio.py`, `timeline_valuation.py`, or `investors.py`), refresh the browser tab or use the **Rerun** control in Streamlit (top-right) so cached data reloads.
- Streamlit auto-saves state when you modify filters; reloading the page retains your selections unless you restart the server.

## 7. Troubleshooting Tips

- **No data displayed:** Ensure the CSV files remain in the repository root and that the filters leave at least one company selected.
- **Map missing countries:** The GeoJSON file is fetched over the internet. Confirm you have an active connection.
- **Stale metrics after running scripts:** Trigger a rerun (`Ctrl+R` in the browser, or the circular arrow button in Streamlit) to refresh cached data.
- **Capital call output not visible:** Run `python investors.py` to regenerate `investor_capital_call_timeline.csv` and reload the dashboard.

Refer to `README.md` for background on data generation and project structure. This manual focuses solely on the application user experience.


