from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROFILES_PATH = ROOT / "company_profiles_150.csv"
TIMELINE_PATH = ROOT / "PE_Fund_timeline.csv"
FINANCIALS_PATH = ROOT / "synthetic_financial_data_150_companies.csv"


REMOVED_COMPANIES = [
    "Voltmotor British",
    "Riyadh Pharmatechnologie",
    "Britannia Markets Gas",
]


NEW_COMPANIES = [
    {"Company": "Lagos Horizon Energy", "Industry": "Energy", "Country": "Nigeria"},
    {"Company": "Niger Delta BioHealth", "Industry": "Biotech", "Country": "Nigeria"},
    {"Company": "Abuja Quantum Logistics", "Industry": "Logistics", "Country": "Nigeria"},
    {"Company": "Khartoum Desert Industries", "Industry": "Industrial", "Country": "Sudan"},
    {"Company": "Nile Frontier Agritech", "Industry": "Tech", "Country": "Sudan"},
    {"Company": "Cape Stellar Retail", "Industry": "Retail", "Country": "South Africa"},
    {"Company": "Johannesburg Fintech Nexus", "Industry": "Software", "Country": "South Africa"},
    {"Company": "Durban Green Utilities", "Industry": "Utilities", "Country": "South Africa"},
    {"Company": "Rio Nova Consumer", "Industry": "Consumer Goods", "Country": "Brazil"},
    {"Company": "Amazonia Agro Ventures", "Industry": "Services", "Country": "Brazil"},
    {"Company": "Sao Paulo Data Systems", "Industry": "Software", "Country": "Brazil"},
    {"Company": "Mumbai NextGen Pharma", "Industry": "Pharma", "Country": "India"},
    {"Company": "Bangalore Aero Dynamics", "Industry": "Aerospace", "Country": "India"},
    {"Company": "Chennai Solar Holdings", "Industry": "Energy", "Country": "India"},
    {"Company": "Delhi Quantum Services", "Industry": "Services", "Country": "India"},
    {"Company": "Karachi Harbor Logistics", "Industry": "Logistics", "Country": "Pakistan"},
    {"Company": "Islamabad Precision Tech", "Industry": "Tech", "Country": "Pakistan"},
    {"Company": "Toronto Arctic Capital", "Industry": "Services", "Country": "Canada"},
    {"Company": "Vancouver Coastal Energy", "Industry": "Energy", "Country": "Canada"},
    {"Company": "Montreal Quantum Medics", "Industry": "Biotech", "Country": "Canada"},
]

NEW_COMPANY_NAMES = [entry["Company"] for entry in NEW_COMPANIES]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles = pd.read_csv(PROFILES_PATH)
    timeline = pd.read_csv(TIMELINE_PATH)
    financials = pd.read_csv(FINANCIALS_PATH)
    return profiles, timeline, financials


def update_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    updated = profiles[
        ~profiles["Company"].isin(REMOVED_COMPANIES + NEW_COMPANY_NAMES)
    ].copy()
    new_entries = pd.DataFrame(NEW_COMPANIES)
    return pd.concat([updated, new_entries], ignore_index=True)


def update_timeline(timeline: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    updated = timeline[~timeline["Company"].isin(NEW_COMPANY_NAMES)].copy()
    valuation_cols = [
        col
        for col in updated.columns
        if col.startswith("Valuation_") and col != "Valuation_of_company"
    ]
    mask = updated["Company"].isin(REMOVED_COMPANIES)
    if mask.any():
        valuation_cols_sorted = sorted(valuation_cols, key=lambda c: int(c.split("_")[1]))
        cols_to_zero = ["Valuation_of_company"]
        if valuation_cols_sorted:
            cols_to_zero.append(valuation_cols_sorted[-1])
        updated.loc[mask, cols_to_zero] = 0

    rng = np.random.default_rng(2025)
    acquisition_choices = pd.date_range("2025-08-31", "2025-11-30", freq="ME")
    acquisition_months = rng.choice(acquisition_choices, size=len(NEW_COMPANIES), replace=True)
    valuation_cols_sorted = sorted(valuation_cols, key=lambda c: int(c.split("_")[1]))

    investment_lookup: dict[str, float] = {}
    new_rows = []
    for company, acquired in zip(NEW_COMPANIES, acquisition_months):
        investment_cost = float(rng.uniform(1.0e7, 1.0e8))
        fund_size = float(rng.uniform(4.5e10, 1.5e11))
        ownership = rng.choice(["50%", "75%", "100%"])

        valuations: dict[str, float] = {}
        for col in valuation_cols_sorted:
            year = int(col.split("_")[1])
            if year <= 2024:
                value = 0.0
            else:
                value = investment_cost * rng.uniform(1.05, 1.35)
            valuations[col] = float(value)

        row = {
            "Acquisition_Month": pd.to_datetime(acquired).strftime("%Y-%m-%d"),
            "Company": company["Company"],
            "Ownership_Percentage": ownership,
            "Investment_Cost": round(investment_cost, 2),
            "Fund_Size_at_Acquisition": round(fund_size, 2),
            "Valuation_of_company": round(valuations["Valuation_2025"], 2),
        }
        row.update({col: round(valuations[col], 2) for col in valuation_cols_sorted})
        new_rows.append(row)
        investment_lookup[company["Company"]] = investment_cost

    if new_rows:
        updated = pd.concat([updated, pd.DataFrame(new_rows)], ignore_index=True)

    return updated, investment_lookup


def make_financial_rows(company: str, investment: float, rng: np.random.Generator) -> list[dict[str, float]]:
    months = pd.date_range("2022-01-31", "2024-12-31", freq="ME")
    records: list[dict[str, float]] = []
    scale = np.clip(investment / 5.0e7, 0.4, 1.8)
    for month in months:
        revenue = max(rng.normal(1.35e6, 2.2e5) * scale, 3.5e5)
        cogs = revenue * rng.uniform(0.46, 0.63)
        gross_profit = revenue - cogs
        gross_margin_pct = (gross_profit / revenue) * 100 if revenue else 0.0
        sga = gross_profit * rng.uniform(0.22, 0.35)
        rd = revenue * rng.uniform(0.06, 0.12)
        ebit = gross_profit - sga - rd
        ebit = max(ebit, revenue * 0.045)
        interest = revenue * rng.uniform(0.01, 0.018)
        taxable = max(ebit - interest, 0.0)
        tax = taxable * rng.uniform(0.18, 0.24)
        net_income = ebit - interest - tax
        net_income = max(net_income, revenue * 0.02)

        assets = max(investment * rng.uniform(0.78, 1.12), 1.0)
        debt = assets * rng.uniform(0.32, 0.55)
        equity = max(assets - debt, assets * 0.35)

        roe = (net_income * 12 / equity * 100) if equity else 0.0
        roa = (net_income * 12 / assets * 100) if assets else 0.0
        ebitda = ebit + sga * 0.25 + rd * 0.15
        ebitda_annual = max(ebitda * 12, 1e5)
        debt_ebitda = debt / ebitda_annual
        debt_equity = debt / equity if equity else math.inf

        record = {
            "Company": company,
            "Month": month.strftime("%Y-%m-%d"),
            "Revenue": int(round(revenue)),
            "COGS": int(round(cogs)),
            "Gross_Profit": int(round(gross_profit)),
            "Gross_Margin_%": round(gross_margin_pct, 2),
            "SG&A": int(round(sga)),
            "R&D": int(round(rd)),
            "Operating_Income_(EBIT)": int(round(ebit)),
            "Interest_Expense": int(round(interest)),
            "Tax_Expense": int(round(tax)),
            "Net_Income": int(round(net_income)),
            "Assets": int(round(assets)),
            "Debt": int(round(debt)),
            "Equity": int(round(equity)),
            "ROE_%": round(roe, 2),
            "ROA_%": round(roa, 2),
            "Debt/EBITDA": round(debt_ebitda, 2),
            "Debt/Equity": round(debt_equity, 2),
        }
        records.append(record)

    return records


def update_financials(financials: pd.DataFrame, investments: dict[str, float]) -> pd.DataFrame:
    updated = financials[~financials["Company"].isin(NEW_COMPANY_NAMES)].copy()
    rng = np.random.default_rng(3105)
    new_records: list[dict[str, float]] = []
    for company, investment in investments.items():
        new_records.extend(make_financial_rows(company, investment, rng))
    if new_records:
        updated = pd.concat([updated, pd.DataFrame(new_records)], ignore_index=True)
    return updated


def main() -> None:
    profiles, timeline, financials = load_data()

    profiles_updated = update_profiles(profiles)
    timeline_updated, investment_lookup = update_timeline(timeline)
    financials_updated = update_financials(financials, investment_lookup)

    profiles_updated.to_csv(PROFILES_PATH, index=False)
    timeline_updated.to_csv(TIMELINE_PATH, index=False)
    financials_updated.to_csv(FINANCIALS_PATH, index=False)


if __name__ == "__main__":
    main()

