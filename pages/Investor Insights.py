from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, cast

import numpy as np
import pandas as pd

px = cast(Any, import_module("plotly.express"))
st = cast(Any, import_module("streamlit"))


DATA_ROOT = Path(__file__).resolve().parents[1]

TIMELINE_PATH = DATA_ROOT / "PE_Fund_Timeline.csv"
CAPITAL_CALLS_PATH = DATA_ROOT / "investor_capital_call_timeline.csv"
PROFILES_PATH = DATA_ROOT / "company_profiles_150.csv"

COLOR_SCALE = ["#1f0a0a", "#b91c1c", "#ef4444", "#fca5a5"]


@st.cache_data(show_spinner=False)
def load_timeline(path: Path, version: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in ("Investment_Cost", "Valuation_of_company"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    valuation_cols = [col for col in df.columns if col.startswith("Valuation_")]
    for col in valuation_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Acquisition_Month"] = pd.to_datetime(df["Acquisition_Month"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_capital_calls(path: Path, version: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Capital_Call_Date"] = pd.to_datetime(df["Capital_Call_Date"], errors="coerce")
    numeric_cols = [
        "Total_Capital_Called_for_Deal",
        "Investor_Contribution",
        "Cumulative_Contribution",
        "Total_Commitment",
        "Remaining_Commitment",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_profiles(path: Path, version: float) -> pd.DataFrame:
    return pd.read_csv(path)


def forecast_capital_calls(calls: pd.DataFrame, horizon_months: int = 24) -> Optional[pd.DataFrame]:
    if calls.empty:
        return None

    remaining_series = calls["Remaining_Commitment"].dropna()
    if remaining_series.empty:
        remaining = max(calls["Total_Commitment"].dropna().iloc[0] - calls["Investor_Contribution"].sum(), 0)
    else:
        remaining = float(remaining_series.iloc[-1])

    has_calls = calls[calls["Capital_Call_Date"].notna()].copy()
    has_calls["Capital_Call_Date"] = pd.to_datetime(has_calls["Capital_Call_Date"], errors="coerce")
    has_calls = has_calls.dropna(subset=["Capital_Call_Date"])

    if not has_calls.empty:
        has_calls["Year"] = has_calls["Capital_Call_Date"].dt.year
        has_calls["Quarter"] = has_calls["Capital_Call_Date"].dt.quarter
        quarterly = (
            has_calls.groupby(["Year", "Quarter"])["Investor_Contribution"].sum().reset_index().sort_values(["Year", "Quarter"])
        )
    else:
        quarterly = pd.DataFrame(columns=["Year", "Quarter", "Investor_Contribution"])

    if not quarterly.empty:
        quarterly["Period"] = pd.PeriodIndex.from_fields(year=quarterly["Year"], quarter=quarterly["Quarter"], freq="Q")
        quarterly.set_index("Period", inplace=True)
        tail = quarterly.tail(8)
        growth_rate = None
        if len(tail) >= 2 and tail["Investor_Contribution"].sum() > 0:
            idx = np.arange(len(tail))
            values = tail["Investor_Contribution"].to_numpy()
            if np.all(values > 0):
                slope, intercept = np.polyfit(idx, np.log(values), 1)
                growth_rate = slope
            avg_quarterly = values.mean()
        else:
            avg_quarterly = quarterly["Investor_Contribution"].mean() if quarterly["Investor_Contribution"].sum() > 0 else 0
        if np.isnan(avg_quarterly) or avg_quarterly <= 0:
            avg_quarterly = remaining / max(horizon_months / 3, 1)

        forecast_rows = []
        current_period = quarterly.index.max() if not quarterly.index.empty else pd.Period(pd.Timestamp.today(), freq="Q")
        current_period = current_period + 1
        rem = remaining
        for _ in range(max(int(horizon_months / 3), 1)):
            if rem <= 0:
                break
            periods_since_start = _ if growth_rate is None else _
            if growth_rate is not None:
                projected = avg_quarterly * np.exp(growth_rate * periods_since_start)
            else:
                projected = avg_quarterly
            projected = max(projected, 0)
            amount = min(projected, rem)
            forecast_rows.append({
                "Month": (current_period.to_timestamp() + pd.offsets.MonthEnd(0)).strftime("%Y-%m"),
                "Projected Call": amount,
            })
            rem -= amount
            current_period += 1
        if forecast_rows:
            forecast_df = pd.DataFrame(forecast_rows)
            forecast_df["Projected Call"] = forecast_df["Projected Call"].round(0)
            return forecast_df
        return None

    if remaining <= 0:
        return None

    start_date = calls["Capital_Call_Date"].max()
    if pd.isna(start_date):
        start_date = pd.Timestamp.today()
    next_month = (start_date + pd.offsets.MonthBegin(1)).normalize()

    rows = []
    rem = remaining
    avg = remaining / max(horizon_months, 1)
    for i in range(horizon_months):
        if rem <= 0:
            break
        date = next_month + pd.offsets.MonthBegin(i)
        amount = float(min(avg, rem))
        rows.append({"Month": date.strftime("%Y-%m"), "Projected Call": amount})
        rem -= amount

    if not rows:
        return None

    forecast_df = pd.DataFrame(rows)
    forecast_df["Projected Call"] = forecast_df["Projected Call"].round(0)
    return forecast_df


def forecast_distributions(
    deal_data: pd.DataFrame,
    horizon_months: int = 36,
    exit_years: int = 6,
) -> Optional[pd.DataFrame]:
    if deal_data.empty:
        return None

    exits = deal_data.copy()
    exits["Acquisition_Month"] = pd.to_datetime(exits["Acquisition_Month"], errors="coerce")
    exits = exits.dropna(subset=["Acquisition_Month", "Current_Value"])
    if exits.empty:
        return None

    exits["Expected_Distribution"] = exits["Current_Value"]
    exits["Distribution_Date"] = exits["Acquisition_Month"] + pd.DateOffset(years=exit_years)

    horizon_end = pd.Timestamp.today() + pd.DateOffset(months=horizon_months)
    upcoming = exits[(exits["Distribution_Date"] >= pd.Timestamp.today()) & (exits["Distribution_Date"] <= horizon_end)]
    if upcoming.empty:
        return None

    schedule = (
        upcoming.groupby(upcoming["Distribution_Date"].dt.to_period("M"))["Expected_Distribution"].sum().reset_index()
    )
    schedule["Month"] = schedule["Distribution_Date"].dt.strftime("%Y-%m")
    schedule = schedule[["Month", "Expected_Distribution"]].rename(
        columns={"Expected_Distribution": "Projected Distribution"}
    )
    schedule["Projected Distribution"] = schedule["Projected Distribution"].round(0)
    return schedule


def investor_metrics(
    calls: pd.DataFrame, timeline: pd.DataFrame, profiles: pd.DataFrame
) -> Dict[str, Any]:
    merged = calls.merge(
        timeline[["Company", "Investment_Cost", "Valuation_of_company", "Acquisition_Month"]],
        left_on="Triggering_Investment",
        right_on="Company",
        how="left",
    )

    merged = merged.merge(
        profiles[["Company", "Country"]],
        on="Company",
        how="left",
        suffixes=("", "_profile"),
    )

    merged["Investment_Share"] = merged["Investor_Contribution"] / merged["Total_Capital_Called_for_Deal"]
    merged["Investment_Share"] = merged["Investment_Share"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    merged["Current_Value"] = merged["Investment_Share"] * merged["Valuation_of_company"].fillna(0.0)

    invested_amount = merged["Investor_Contribution"].sum()
    nav = merged["Current_Value"].sum()

    evaluation_date = merged["Capital_Call_Date"].max()
    if pd.isna(evaluation_date):
        evaluation_date = timeline["Acquisition_Month"].max()

    irr_value = np.nan
    if invested_amount and not merged.empty and not pd.isna(evaluation_date):
        cash_dates = merged["Capital_Call_Date"].tolist()
        cash_flows = (-merged["Investor_Contribution"]).tolist()
        cash_dates.append(pd.to_datetime(evaluation_date))
        cash_flows.append(nav)
        irr_value = compute_xirr(cash_flows, cash_dates)

    performance_ratio = irr_value / 0.04 if irr_value and not np.isnan(irr_value) else np.nan

    company_view = (
        merged.groupby(["Company", "Country", "Acquisition_Month"], dropna=False)
        .agg(
            Invested_Amount=("Investor_Contribution", "sum"),
            Current_Value=("Current_Value", "sum"),
        )
        .reset_index()
    )
    company_view["Acquisition_Month"] = pd.to_datetime(company_view["Acquisition_Month"], errors="coerce")
    company_view["Investment_Start"] = company_view["Acquisition_Month"].dt.strftime("%Y-%m")
    company_view = company_view.drop(columns=["Acquisition_Month"]).sort_values("Current_Value", ascending=False)

    contributions = calls.sort_values("Capital_Call_Date").copy()
    contributions["Cumulative"] = contributions["Investor_Contribution"].cumsum()

    return {
        "invested_amount": invested_amount,
        "nav": nav,
        "irr": irr_value,
        "performance_ratio": performance_ratio,
        "company_view": company_view,
        "contributions": contributions,
        "call_forecast": forecast_capital_calls(calls),
        "distribution_forecast": forecast_distributions(merged),
        "company_distribution": merged,
    }


def compute_xirr(cash_flows: Iterable[float], dates: Iterable[pd.Timestamp]) -> float:
    amounts = np.array(list(cash_flows), dtype=float)
    date_list = [pd.to_datetime(d) for d in dates]
    if len(amounts) < 2 or any(pd.isna(date) for date in date_list):
        return np.nan

    day_diffs = np.array([(d - date_list[0]).days / 365.25 for d in date_list], dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1 + rate) ** day_diffs))

    def d_npv(rate: float) -> float:
        return float(np.sum(-day_diffs * amounts / (1 + rate) ** (day_diffs + 1)))

    guess = 0.1
    for _ in range(100):
        if guess <= -0.9999:
            guess = -0.9999 + 1e-6
        f = npv(guess)
        fp = d_npv(guess)
        if abs(fp) < 1e-9:
            break
        new_guess = guess - f / fp
        if abs(new_guess - guess) < 1e-7:
            return float(new_guess)
        guess = new_guess
    return np.nan


def main() -> None:
    st.set_page_config(page_title="Investor Insights", layout="wide")
    st.title("Investor Capital & Portfolio Overview")

    timeline = load_timeline(TIMELINE_PATH, TIMELINE_PATH.stat().st_mtime)
    calls = load_capital_calls(CAPITAL_CALLS_PATH, CAPITAL_CALLS_PATH.stat().st_mtime)
    profiles = load_profiles(PROFILES_PATH, PROFILES_PATH.stat().st_mtime)

    investors = sorted(calls["Investor"].dropna().unique())
    selected_investor = st.selectbox("Select Investor", investors)

    investor_calls = calls[calls["Investor"] == selected_investor].copy()

    metrics = investor_metrics(investor_calls, timeline, profiles)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Invested Capital",
        f"${metrics['invested_amount']:,.0f}",
        help="Total capital contributed by the selected investor",
    )
    col2.metric(
        "Estimated NAV",
        f"${metrics['nav']:,.0f}",
        help="Current value of holdings attributed to the investor",
    )
    irr_value = metrics["irr"]
    ratio_value = metrics["performance_ratio"]
    ratio_display = "—" if np.isnan(ratio_value) else f"{ratio_value:,.2f}×"
    irr_display = "—" if np.isnan(irr_value) else f"{irr_value*100:,.2f}%"
    col3.metric(
        "IRR vs 4% Hurdle",
        ratio_display,
        irr_display,
        help="Shows projected IRR relative to the 4% preferred return; delta displays absolute IRR",
    )

    if not investor_calls.empty:
        chart = px.line(
            metrics["contributions"],
            x="Capital_Call_Date",
            y="Cumulative",
            title=f"Capital Call Timeline — {selected_investor}",
            labels={"Capital_Call_Date": "Date", "Cumulative": "Cumulative Contribution"},
            markers=True,
        )
        chart.update_traces(line=dict(color="#ef4444", width=3))
        chart.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#cbd5f5"),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(chart, width="stretch")
        st.caption("ℹ️ Cumulative historical capital calls for the investor.")

    company_table = metrics["company_view"].rename(
        columns={
            "Company": "Portfolio Company",
            "Country": "Country",
            "Investment_Start": "Investment Start",
            "Invested_Amount": "Invested Amount",
            "Current_Value": "Current Value",
        }
    )
    company_table["Invested Amount"] = company_table["Invested Amount"].map(lambda x: f"${x:,.0f}")
    company_table["Current Value"] = company_table["Current Value"].map(lambda x: f"${x:,.0f}")

    st.subheader("Portfolio Snapshot")
    st.dataframe(company_table, width="stretch")
    st.caption("ℹ️ Table shows the investor's share of each portfolio company, including current valuation.")

    st.subheader("Scenario Sandbox")
    st.caption("ℹ️ Adjust valuation haircuts, exit timing, and reinvestment assumptions to stress-test future distributions.")
    sandbox_cols = st.columns(3)
    value_haircut = sandbox_cols[0].slider("Valuation Adjustment", -50, 50, 0, format="%d%%")
    timing_shift = sandbox_cols[1].slider("Exit Timing Shift (months)", -24, 24, 0)
    reinvestment_rate = sandbox_cols[2].slider("Reinvestment Rate", 0, 40, 10, format="%d%%")

    base_data = metrics["company_distribution"]
    if not base_data.empty:
        adjusted = base_data.copy()
        adjusted["Acquisition_Month"] = pd.to_datetime(adjusted["Acquisition_Month"], errors="coerce")
        adjusted = adjusted.dropna(subset=["Acquisition_Month"])

        growth_factor = (1 + value_haircut / 100)
        adjusted["Adjusted_Value"] = adjusted["Current_Value"] * growth_factor
        adjusted["Adjusted_Value"] *= (1 + reinvestment_rate / 100)
        adjusted["Adjusted_Distribution_Date"] = (
            adjusted["Acquisition_Month"] + pd.DateOffset(years=6) + pd.DateOffset(months=timing_shift)
        )

        adjusted_future = adjusted[
            adjusted["Adjusted_Distribution_Date"] >= pd.Timestamp.today()
        ]

        if not adjusted_future.empty:
            scenario_schedule = (
                adjusted_future.groupby(adjusted_future["Adjusted_Distribution_Date"].dt.to_period("M"))["Adjusted_Value"].sum().reset_index()
            )
            scenario_schedule["Month"] = scenario_schedule["Adjusted_Distribution_Date"].dt.strftime("%Y-%m")
            scenario_table = scenario_schedule[["Month", "Adjusted_Value"]].rename(
                columns={"Adjusted_Value": "Scenario Distribution"}
            )
            scenario_table["Scenario Distribution"] = scenario_table["Scenario Distribution"].round(0)
            st.dataframe(scenario_table, width="stretch")
        else:
            st.info("No distributions projected under current scenario.")

    call_forecast = metrics["call_forecast"]
    distribution_forecast = metrics["distribution_forecast"]

    if call_forecast is not None or distribution_forecast is not None:
        st.subheader("Cash Flow Outlook")
        col_a, col_b = st.columns(2)
        if call_forecast is not None:
            col_a.metric(
                "Projected Calls (24M)",
                f"${call_forecast['Projected Call'].sum():,.0f}",
                help="Estimated capital required over the next two years",
            )
            col_a.dataframe(call_forecast, width="stretch")
        else:
            col_a.info("No future capital calls projected.")

        if distribution_forecast is not None:
            col_b.metric(
                "Projected Distributions (36M)",
                f"${distribution_forecast['Projected Distribution'].sum():,.0f}",
                help="Expected cash back to the investor across the next three years",
            )
            col_b.dataframe(distribution_forecast, width="stretch")
        else:
            col_b.info("No expected distributions in the next 3 years.")


if __name__ == "__main__":
    main()

