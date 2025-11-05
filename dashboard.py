"""Streamlit dashboard for the PE Fund manager and investors.

Loads the synthesized portfolio data stored in the Excel/CSV files under the
project root and presents role-specific views for managers and investors.
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Optional, cast
from urllib.request import urlopen

import numpy as np
import pandas as pd

px = cast(Any, import_module("plotly.express"))
st = cast(Any, import_module("streamlit"))


PROJECT_ROOT = Path(__file__).resolve().parent

PROFILES_PATH = PROJECT_ROOT / "company_profiles_150.csv"
TIMELINE_PATH = PROJECT_ROOT / "PE_Fund_timeline.csv"
FINANCIALS_PATH = PROJECT_ROOT / "synthetic_financial_data_150_companies.csv"

COUNTRY_ISO_A3 = {
    "USA": "USA",
    "UK": "GBR",
    "UAE": "ARE",
    "Qatar": "QAT",
    "Kuwait": "KWT",
    "Bahrain": "BHR",
    "Oman": "OMN",
    "Saudi Arabia": "SAU",
    "France": "FRA",
    "Germany": "DEU",
    "Netherlands": "NLD",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Spain": "ESP",
    "Nigeria": "NGA",
    "Sudan": "SDN",
    "South Africa": "ZAF",
    "Brazil": "BRA",
    "India": "IND",
    "Pakistan": "PAK",
    "Canada": "CAN",
}

COUNTRIES_GEOJSON_URL = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"

MAP_COLOR_SCALE = ["#1f0a0a", "#b91c1c", "#ef4444", "#fca5a5"]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert mixed-format numeric fields (with commas) to floats."""

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


@st.cache_data(show_spinner=False)
def load_profiles(path: Path) -> pd.DataFrame:
    profiles = pd.read_csv(path)
    return profiles


@st.cache_data(show_spinner=False)
def load_timeline(path: Path) -> pd.DataFrame:
    timeline = pd.read_csv(path, parse_dates=["Acquisition_Month"])
    timeline["Valuation_of_company"] = clean_numeric(timeline["Valuation_of_company"])
    for column in ("Investment_Cost", "Fund_Size_at_Acquisition"):
        if column in timeline.columns:
            timeline[column] = clean_numeric(timeline[column])
    return timeline


@st.cache_data(show_spinner=False)
def load_financials(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_countries_geojson(url: str = COUNTRIES_GEOJSON_URL) -> dict:
    with urlopen(url) as response:
        return json.load(response)


def inject_sidebar_styles(
    selected_container_height: int = 400, dropdown_height: int = 400
) -> None:
    st.markdown(
        f"""
        <style>
        /* Selected tag container inside the sidebar */
        div[data-testid="stSidebar"] div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:first-child {{
            max-height: {selected_container_height}px;
            overflow-y: auto;
        }}

        /* Dropdown list portal for multiselect options (renders outside sidebar) */
        .stMultiSelect [data-baseweb="select"] > div {{
            max-height: {dropdown_height}px;
            overflow-y: auto;
        }}
        .stMultiSelect [data-baseweb="popover"] {{
            max-height: {dropdown_height}px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------


def valuation_columns(columns: Iterable[str]) -> list[str]:
    valuation_prefix = "Valuation_"
    filtered = []
    for col in columns:
        if not col.startswith(valuation_prefix):
            continue
        suffix = col[len(valuation_prefix) :]
        if suffix.isdigit():
            filtered.append(col)
    return filtered


def latest_valuation_column(columns: Iterable[str]) -> Optional[str]:
    candidates: list[tuple[int, str]] = []
    for col in valuation_columns(columns):
        suffix = col.removeprefix("Valuation_")
        if suffix.isdigit():
            candidates.append((int(suffix), col))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def country_investment_summary(
    timeline: pd.DataFrame,
    latest: pd.DataFrame,
    companies: Iterable[str],
) -> pd.DataFrame:
    if not companies:
        return pd.DataFrame(columns=["Country", "Companies", "Invested_Amount", "Current_Net_Worth", "ISO_A3"])

    filtered = timeline[timeline["Company"].isin(companies)].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["Country", "Companies", "Invested_Amount", "Current_Net_Worth", "ISO_A3"])

    filtered["Investment_Cost"] = clean_numeric(filtered.get("Investment_Cost", pd.Series(dtype=float)))
    filtered["Investment_Cost"] = filtered["Investment_Cost"].fillna(0.0)

    valuation_cols = valuation_columns(filtered.columns)
    if valuation_cols:
        for col in valuation_cols:
            filtered[col] = clean_numeric(filtered[col])
        latest_col = latest_valuation_column(filtered.columns)
        if latest_col:
            filtered["Current_Net_Worth"] = filtered[latest_col]
        else:
            filtered["Current_Net_Worth"] = np.nan
    elif "Valuation_of_company" in filtered.columns:
        filtered["Current_Net_Worth"] = clean_numeric(filtered["Valuation_of_company"])
    else:
        filtered["Current_Net_Worth"] = np.nan

    filtered["Current_Net_Worth"] = filtered["Current_Net_Worth"].fillna(0.0)

    if "Country" in latest.columns:
        country_lookup = latest.set_index("Company")["Country"]
        filtered["Country"] = filtered["Company"].map(country_lookup)
    elif "Country" in filtered.columns:
        filtered["Country"] = filtered["Country"]
    else:
        filtered["Country"] = np.nan

    filtered = filtered.dropna(subset=["Country"])
    if filtered.empty:
        return pd.DataFrame(columns=["Country", "Companies", "Invested_Amount", "Current_Net_Worth", "ISO_A3"])

    summary = (
        filtered.groupby("Country")
        .agg(
            Companies=("Company", "nunique"),
            Invested_Amount=("Investment_Cost", "sum"),
            Current_Net_Worth=("Current_Net_Worth", "sum"),
        )
        .reset_index()
    )

    summary["Companies"] = summary["Companies"].astype(int)
    summary["Invested_Amount"] = summary["Invested_Amount"].fillna(0.0)
    summary["Current_Net_Worth"] = summary["Current_Net_Worth"].fillna(0.0)
    summary["ISO_A3"] = summary["Country"].map(COUNTRY_ISO_A3.get)
    summary.dropna(subset=["ISO_A3"], inplace=True)

    return summary


def latest_portfolio_snapshot(financials: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    latest = financials.sort_values("Month").groupby("Company").tail(1)
    latest = latest.merge(profiles, on="Company", how="left")
    latest.rename(columns={"Month": "Latest_Month"}, inplace=True)
    return latest


def aggregate_nav(timeline: pd.DataFrame, companies: Iterable[str]) -> pd.Series:
    valuations = valuation_columns(timeline.columns)
    filtered = timeline[timeline["Company"].isin(companies)]
    nav_series = filtered[valuations].sum(axis=0).rename("NAV")
    nav_series.index = nav_series.index.str.replace("Valuation_", "")
    nav_series = nav_series.loc[sorted(nav_series.index)]
    return nav_series


def compute_company_risk(latest: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    if latest.empty or timeline.empty:
        return pd.DataFrame()

    valuation_cols = valuation_columns(timeline.columns)
    if len(valuation_cols) < 2:
        return pd.DataFrame()

    valuation_cols_sorted = sorted(valuation_cols, key=lambda c: int(c.split("_")[1]))
    recent_cols = valuation_cols_sorted[-3:]

    timeline_subset = timeline[["Company"] + recent_cols + ["Ownership_Percentage"]].copy()

    def compute_cagr(row: pd.Series) -> float:
        values = row[recent_cols].astype(float).replace({0.0: np.nan}).dropna()
        if len(values) < 2:
            return np.nan
        start = values.iloc[0]
        end = values.iloc[-1]
        periods = len(values) - 1
        if start <= 0 or end <= 0 or periods == 0:
            return np.nan
        return (end / start) ** (1 / periods) - 1

    timeline_subset["Valuation_CAGR"] = timeline_subset.apply(compute_cagr, axis=1)

    merged = latest.merge(timeline_subset[["Company", "Valuation_CAGR"]], on="Company", how="left")

    risk_rows = []
    for _, row in merged.iterrows():
        company = row.get("Company")
        net_income = float(row.get("Net_Income", 0.0)) if pd.notna(row.get("Net_Income")) else 0.0
        debt_equity = row.get("Debt/Equity")
        try:
            debt_equity = float(debt_equity)
        except (TypeError, ValueError):
            debt_equity = np.nan
        cagr = row.get("Valuation_CAGR")

        leverage_score = 0.0 if pd.isna(debt_equity) else np.clip((debt_equity - 1.2) / 2.0, 0.0, 1.0)
        profitability_score = 0.0 if net_income >= 0 else np.clip(-net_income / max(abs(net_income), 1e6), 0.0, 1.0)
        growth_score = 0.0 if pd.notna(cagr) and cagr >= 0 else np.clip(abs(cagr) if pd.notna(cagr) else 0.3, 0.0, 1.0)

        risk_score = 0.4 * leverage_score + 0.35 * growth_score + 0.25 * profitability_score

        if risk_score >= 0.7:
            risk_level = "High"
        elif risk_score >= 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        risk_rows.append(
            {
                "Company": company,
                "Industry": row.get("Industry"),
                "Country": row.get("Country"),
                "Valuation_CAGR": cagr,
                "Net_Income": net_income,
                "Debt_Equity": debt_equity,
                "Risk_Score": risk_score,
                "Risk_Level": risk_level,
            }
        )

    df = pd.DataFrame(risk_rows)
    if df.empty:
        return df
    df["Valuation_CAGR"] = df["Valuation_CAGR"].round(3)
    df["Risk_Score"] = df["Risk_Score"].round(2)
    return df.sort_values("Risk_Score", ascending=False)


def compute_follow_on_candidates(latest: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    if latest.empty or timeline.empty:
        return pd.DataFrame()

    valuation_cols = valuation_columns(timeline.columns)
    if len(valuation_cols) < 2:
        return pd.DataFrame()

    valuation_cols_sorted = sorted(valuation_cols, key=lambda c: int(c.split("_")[1]))
    recent_cols = valuation_cols_sorted[-4:]

    valuations = timeline[["Company"] + recent_cols].copy()

    def growth_metric(row: pd.Series) -> float:
        values = row[recent_cols].astype(float).replace({0.0: np.nan}).dropna()
        if len(values) < 2:
            return np.nan
        start = values.iloc[0]
        end = values.iloc[-1]
        periods = len(values) - 1
        if start <= 0 or end <= 0 or periods == 0:
            return np.nan
        return (end / start) ** (1 / periods) - 1

    valuations["Valuation_CAGR"] = valuations.apply(growth_metric, axis=1)

    merged = latest.merge(valuations[["Company", "Valuation_CAGR"]], on="Company", how="left")

    rows = []
    for _, row in merged.iterrows():
        company = row.get("Company")
        revenue = row.get("Revenue", np.nan)
        net_income = row.get("Net_Income", np.nan)
        debt_equity = row.get("Debt/Equity", np.nan)
        cagr = row.get("Valuation_CAGR", np.nan)

        try:
            revenue = float(revenue)
        except (TypeError, ValueError):
            revenue = np.nan
        try:
            net_income = float(net_income)
        except (TypeError, ValueError):
            net_income = np.nan
        try:
            debt_equity = float(debt_equity)
        except (TypeError, ValueError):
            debt_equity = np.nan

        growth_score = 0.0 if pd.isna(cagr) else np.clip((cagr) * 5.0 + 0.5, 0.0, 1.0)

        if pd.isna(net_income) or pd.isna(revenue) or revenue == 0:
            profitability_score = 0.5
        else:
            margin = net_income / abs(revenue)
            profitability_score = np.clip(margin * 2 + 0.5, 0.0, 1.0)

        if pd.isna(debt_equity):
            leverage_score = 0.6
        else:
            leverage_score = np.clip(1.2 - (debt_equity - 1) / 2, 0.0, 1.0)

        follow_score = 0.5 * growth_score + 0.3 * profitability_score + 0.2 * leverage_score

        if follow_score >= 0.75:
            level = "High"
        elif follow_score >= 0.55:
            level = "Moderate"
        else:
            level = "Watch"

        rows.append(
            {
                "Company": company,
                "Industry": row.get("Industry"),
                "Country": row.get("Country"),
                "Valuation_CAGR": cagr,
                "Net_Income": net_income,
                "Revenue": revenue,
                "Debt_Equity": debt_equity,
                "Follow_On_Score": follow_score,
                "Opportunity_Level": level,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Follow_On_Score"] = df["Follow_On_Score"].round(2)
    df["Valuation_CAGR"] = df["Valuation_CAGR"].round(3)
    return df.sort_values("Follow_On_Score", ascending=False)


def forecast_nav(nav_series: pd.Series, horizon: int = 5) -> Optional[pd.DataFrame]:
    nav_series = nav_series.dropna()
    if nav_series.empty:
        return None

    df = nav_series.reset_index().rename(columns={"index": "Year", "NAV": "NAV"})
    try:
        df["Year"] = df["Year"].astype(int)
    except ValueError:
        return None

    df = df[df["NAV"] > 0]
    if len(df) < 2:
        return None

    years = df["Year"].to_numpy(dtype=float)
    logs = np.log(df["NAV"].to_numpy(dtype=float))

    slope, intercept = np.polyfit(years, logs, 1)

    last_year = int(df["Year"].max())
    future_years = np.arange(last_year + 1, last_year + horizon + 1)
    base = np.exp(intercept + slope * future_years)

    forecast_df = pd.DataFrame(
        {
            "Year": future_years,
            "Base": base,
            "Bear": base * 0.9,
            "Bull": base * 1.1,
        }
    )
    return forecast_df


def nav_forecast_chart(nav_series: pd.Series) -> Optional[px.line]:
    forecast_df = forecast_nav(nav_series)
    if forecast_df is None:
        return None

    history_df = nav_series.reset_index().rename(columns={"index": "Year", "NAV": "NAV"})
    history_df["Year"] = history_df["Year"].astype(int)
    history_df["Scenario"] = "Historical"

    melt_forecast = forecast_df.melt(id_vars="Year", var_name="Scenario", value_name="NAV")
    combined = pd.concat([history_df[["Year", "Scenario", "NAV"]], melt_forecast], ignore_index=True)

    fig = px.line(
        combined,
        x="Year",
        y="NAV",
        color="Scenario",
        title="Projected NAV Outlook",
        color_discrete_map={
            "Historical": "#f8fafc",
            "Base": "#ef4444",
            "Bear": "#7f1d1d",
            "Bull": "#fecaca",
        },
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#cbd5f5"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def deployment_curve(timeline: pd.DataFrame, companies: Iterable[str]) -> pd.DataFrame:
    filtered = timeline[timeline["Company"].isin(companies)].copy()
    filtered["Investment_Cost"] = clean_numeric(filtered["Investment_Cost"])
    filtered["Year"] = filtered["Acquisition_Month"].dt.year
    curve = (
        filtered.groupby("Year")["Investment_Cost"].sum().cumsum().rename("Cumulative_Investment")
    )
    return curve.reset_index()


def ownership_breakdown(timeline: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    latest_lookup = latest.set_index("Company")
    filtered = timeline.merge(
        latest_lookup[["Industry", "Country"]], on="Company", how="left"
    )
    breakdown = (
        filtered.groupby(["Industry", "Ownership_Percentage"])
        ["Valuation_of_company"]
        .sum()
        .reset_index()
    )
    return breakdown


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------


def render_sidebar(profiles: pd.DataFrame) -> dict:
    st.sidebar.header("Filters")
    industries = sorted(profiles["Industry"].dropna().unique())
    countries = sorted(profiles["Country"].dropna().unique())

    selected_industries = st.sidebar.multiselect(
        "Industry", options=industries, default=industries, help="Filter by industry."
    )
    selected_countries = st.sidebar.multiselect(
        "Country", options=countries, default=countries, help="Filter by headquarters country."
    )

    return {
        "industries": selected_industries,
        "countries": selected_countries,
    }


def apply_filters(profiles: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = profiles["Industry"].isin(filters["industries"]) & profiles["Country"].isin(
        filters["countries"]
    )
    return profiles[mask]


def render_manager_view(
    latest: pd.DataFrame,
    timeline: pd.DataFrame,
    nav_series: pd.Series,
    filtered_companies: list[str],
) -> None:
    st.subheader("Manager Overview")

    total_companies = len(filtered_companies)
    deployed_capital = timeline[timeline["Company"].isin(filtered_companies)][
        "Investment_Cost"
    ].sum()
    latest_nav = nav_series.iloc[-1] if not nav_series.empty else 0.0
    dry_powder = timeline["Fund_Size_at_Acquisition"].max() - deployed_capital

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Active Portfolio", total_companies, help="Number of companies matching current filters")
    col_b.metric(
        "Deployed Capital",
        f"${deployed_capital:,.0f}",
        help="Cumulative capital invested in filtered companies",
    )
    col_c.metric(
        "Latest NAV",
        f"${latest_nav:,.0f}",
        help="Latest aggregated valuation across the filtered portfolio",
    )
    col_d.metric(
        "Dry Powder (est)",
        f"${dry_powder:,.0f}",
        help="Headroom between fund size at acquisition and deployed capital",
    )

    st.caption("ℹ️ Hover over the map and charts to see per-country and per-company details.")

    country_summary = country_investment_summary(timeline, latest, filtered_companies)
    if not country_summary.empty:
        geojson = load_countries_geojson()
        feature_key = "properties.ISO3166-1-Alpha-3"

        map_fig = px.choropleth(
            country_summary,
            geojson=geojson,
            locations="ISO_A3",
            featureidkey=feature_key,
            color="Current_Net_Worth",
            hover_name="Country",
            projection="natural earth",
            color_continuous_scale=MAP_COLOR_SCALE,
        )
        map_fig.update_traces(
            customdata=country_summary[["Companies", "Invested_Amount", "Current_Net_Worth"]].to_numpy(),
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Companies: %{customdata[0]:.0f}<br>"
                "Invested: $%{customdata[1]:,.0f}<br>"
                "Current NAV: $%{customdata[2]:,.0f}<extra></extra>"
            ),
        )
        map_fig.update_geos(
            showframe=False,
            showcountries=True,
            countrycolor="rgba(255,255,255,0.3)",
            showcoastlines=True,
            coastlinecolor="rgba(148,163,184,0.3)",
            projection_scale=0.97,
            bgcolor="rgba(0,0,0,0)",
        )
        map_fig.update_layout(
            title={
                "text": "Global Investment Footprint",
                "x": 0.01,
                "xanchor": "left",
                "font": {"size": 24, "color": "#e2e8f0"},
            },
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#cbd5f5"),
            margin=dict(l=0, r=0, t=60, b=0),
            height=720,
            coloraxis_colorbar=dict(
                title="Current NAV",
                tickcolor="#cbd5f5",
                tickfont=dict(color="#cbd5f5"),
                outlinecolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(map_fig, width="stretch")

        st.dataframe(
            country_summary[
                ["Country", "Companies", "Invested_Amount", "Current_Net_Worth"]
            ].sort_values("Invested_Amount", ascending=False),
            width="stretch",
        )

    if not nav_series.empty:
        fig_nav = px.line(
            nav_series.reset_index(),
            x="index",
            y="NAV",
            title="Fund NAV Timeline",
            labels={"index": "Year", "NAV": "Net Asset Value"},
        )
        fig_nav.update_traces(mode="lines+markers")
        st.plotly_chart(fig_nav, width="stretch")

        forecast_fig = nav_forecast_chart(nav_series)
        if forecast_fig is not None:
            st.plotly_chart(forecast_fig, width="stretch")

    if not latest.empty:
        industry_perf = (
            latest.groupby("Industry")["Net_Income"].mean().sort_values(ascending=False)
        )
        fig_industry = px.bar(
            industry_perf.reset_index(),
            x="Industry",
            y="Net_Income",
            title="Average Net Income by Industry (Latest Month)",
        )
        st.plotly_chart(fig_industry, width="stretch")

    risk_df = compute_company_risk(latest, timeline)
    if not risk_df.empty:
        st.subheader("Risk Watchlist")
        st.caption("ℹ️ Score blends leverage, valuation momentum, and profitability to highlight companies needing attention.")
        fig_risk = px.bar(
            risk_df.head(15),
            x="Company",
            y="Risk_Score",
            color="Risk_Level",
            color_discrete_map={"High": "#b91c1c", "Moderate": "#f97316", "Low": "#22c55e"},
            title="Top Portfolio Risks",
        )
        fig_risk.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#cbd5f5"),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(fig_risk, width="stretch")

        st.dataframe(
            risk_df[[
                "Company",
                "Industry",
                "Country",
                "Risk_Level",
                "Risk_Score",
                "Valuation_CAGR",
                "Debt_Equity",
                "Net_Income",
            ]],
            width="stretch",
        )

    follow_df = compute_follow_on_candidates(latest, timeline)
    if not follow_df.empty:
        st.subheader("Co-Invest & Follow-On Opportunities")
        st.caption("ℹ️ Higher scores reflect strong growth, solid margins, and balanced leverage—prime for additional capital.")
        fig_follow = px.bar(
            follow_df.head(15),
            x="Company",
            y="Follow_On_Score",
            color="Opportunity_Level",
            color_discrete_map={"High": "#22c55e", "Moderate": "#facc15", "Watch": "#f97316"},
            title="Top Follow-On Candidates",
        )
        fig_follow.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#cbd5f5"),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(fig_follow, width="stretch")

        st.dataframe(
            follow_df[[
                "Company",
                "Industry",
                "Country",
                "Opportunity_Level",
                "Follow_On_Score",
                "Valuation_CAGR",
                "Revenue",
                "Net_Income",
                "Debt_Equity",
            ]],
            width="stretch",
        )

    st.caption("Top and bottom performers by ROE% in the most recent period.")
    if "ROE_%" in latest.columns:
        ranked = latest.sort_values("ROE_%", ascending=False)
        st.dataframe(ranked[["Company", "Industry", "Country", "ROE_%", "Net_Income"]].head(10))
        st.dataframe(ranked[["Company", "Industry", "Country", "ROE_%", "Net_Income"]].tail(10))


def render_investor_view(
    timeline: pd.DataFrame,
    nav_series: pd.Series,
    filtered_companies: list[str],
) -> None:
    st.subheader("Investor Overview")

    if filtered_companies:
        curve = deployment_curve(timeline, filtered_companies)
        fig_curve = px.area(
            curve,
            x="Year",
            y="Cumulative_Investment",
            title="Capital Deployment",
            labels={"Cumulative_Investment": "Cumulative Investment ($)", "Year": "Year"},
        )
        st.plotly_chart(fig_curve, width="stretch")

    if not nav_series.empty:
        nav_df = nav_series.reset_index().rename(columns={"index": "Year"})
        nav_df["Year"] = nav_df["Year"].astype(int)
        st.dataframe(nav_df.tail(10))

    valuation_cols = valuation_columns(timeline.columns)
    if valuation_cols:
        melt_df = timeline[timeline["Company"].isin(filtered_companies)][
            ["Company"] + valuation_cols
        ]
        melt_df = melt_df.melt(id_vars="Company", var_name="Year", value_name="Valuation")
        melt_df["Year"] = melt_df["Year"].str.replace("Valuation_", "").astype(int)
        latest_year = melt_df["Year"].max()
        latest_snapshot = (
            melt_df[melt_df["Year"] == latest_year].sort_values("Valuation", ascending=False)
        )
        st.caption(f"Valuation snapshot for {latest_year} (top 15 companies).")
        st.dataframe(latest_snapshot.head(15))


def render_company_drilldown(latest: pd.DataFrame) -> None:
    st.subheader("Company Drilldown")
    selected_company = st.selectbox("Choose a company", options=latest["Company"].sort_values())
    company_row = latest[latest["Company"] == selected_company]
    if company_row.empty:
        st.info("No data available for the selected company.")
        return

    metrics_cols = [
        "Revenue",
        "Net_Income",
        "Gross_Margin_%",
        "ROE_%",
        "ROA_%",
        "Debt/Equity",
    ]
    st.write(company_row[["Company", "Industry", "Country", "Latest_Month"] + metrics_cols])


# ---------------------------------------------------------------------------
# Main application entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="PE Fund Dashboard", layout="wide")
    inject_sidebar_styles()
    st.title("Private Equity Fund Dashboard")

    profiles = load_profiles(PROFILES_PATH)
    timeline = load_timeline(TIMELINE_PATH)
    financials = load_financials(FINANCIALS_PATH)

    filters = render_sidebar(profiles)
    filtered_profiles = apply_filters(profiles, filters)
    filtered_companies = filtered_profiles["Company"].tolist()

    st.write(
        f"Showing {len(filtered_companies)} companies (of {len(profiles)}) after applying filters."
    )

    if not filtered_companies:
        st.warning("No companies match the selected filters.")
        return

    latest = latest_portfolio_snapshot(financials, filtered_profiles)
    nav_series = aggregate_nav(timeline, filtered_companies)

    manager_tab, investor_tab, drilldown_tab = st.tabs(
        ["Manager View", "Investor View", "Company Drilldown"]
    )

    with manager_tab:
        render_manager_view(latest, timeline, nav_series, filtered_companies)

    with investor_tab:
        render_investor_view(timeline, nav_series, filtered_companies)

    with drilldown_tab:
        render_company_drilldown(latest)


if __name__ == "__main__":
    main()

