import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = list(range(2005, 2026))
DEFAULT_GROWTH_RANGE = (-0.10, 0.20)

INDUSTRY_SHOCKS_2008 = {
    "Aerospace": (-0.35, 0.05),
    "Automotive": (-0.40, 0.10),
    "Industrial": (-0.32, 0.08),
    "Logistics": (-0.30, 0.08),
    "Retail": (-0.35, 0.08),
    "Energy": (-0.28, 0.12),
    "Consumer Goods": (-0.25, 0.10),
    "Utilities": (-0.18, 0.08),
    "Services": (-0.30, 0.10),
    "Media": (-0.28, 0.12),
    "Tech": (-0.18, 0.18),
    "Software": (-0.12, 0.22),
    "E-commerce": (-0.20, 0.25),
    "Biotech": (-0.15, 0.22),
    "Pharma": (-0.12, 0.20),
}

INDUSTRY_SHOCKS_2020 = {
    "Aerospace": (-0.50, 0.08),
    "Automotive": (-0.40, 0.10),
    "Industrial": (-0.32, 0.12),
    "Logistics": (-0.25, 0.20),
    "Retail": (-0.45, 0.08),
    "Consumer Goods": (-0.28, 0.15),
    "Energy": (-0.35, 0.12),
    "Utilities": (-0.15, 0.12),
    "Services": (-0.38, 0.10),
    "Media": (-0.22, 0.20),
    "Tech": (-0.05, 0.28),
    "Software": (0.00, 0.35),
    "E-commerce": (0.05, 0.40),
    "Biotech": (0.08, 0.40),
    "Pharma": (0.05, 0.35),
}

COUNTRY_SHOCKS_2008 = {
    "USA": (-0.07, 0.02),
    "UK": (-0.08, 0.02),
    "France": (-0.06, 0.03),
    "Germany": (-0.06, 0.03),
    "Italy": (-0.08, 0.02),
    "Spain": (-0.08, 0.02),
    "Netherlands": (-0.05, 0.03),
    "Ireland": (-0.05, 0.03),
    "Qatar": (-0.03, 0.05),
    "UAE": (-0.03, 0.05),
    "Saudi Arabia": (-0.03, 0.05),
    "Kuwait": (-0.03, 0.05),
    "Oman": (-0.03, 0.05),
    "Bahrain": (-0.03, 0.05),
}

COUNTRY_SHOCKS_2020 = {
    "USA": (-0.12, 0.05),
    "UK": (-0.14, 0.04),
    "France": (-0.12, 0.05),
    "Germany": (-0.10, 0.06),
    "Italy": (-0.15, 0.04),
    "Spain": (-0.15, 0.04),
    "Netherlands": (-0.10, 0.06),
    "Ireland": (-0.08, 0.07),
    "Qatar": (-0.05, 0.12),
    "UAE": (-0.05, 0.12),
    "Saudi Arabia": (-0.06, 0.10),
    "Kuwait": (-0.06, 0.10),
    "Oman": (-0.06, 0.10),
    "Bahrain": (-0.06, 0.10),
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(min(value, maximum), minimum)


def clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
        .replace({'': np.nan, 'nan': np.nan})
    )
    return pd.to_numeric(cleaned, errors='coerce')


def get_growth(year: int, industry: str | None, country: str | None) -> float:
    industry = industry if isinstance(industry, str) else "Unknown"
    country = country if isinstance(country, str) else "Unknown"

    if year == 2008:
        base_min, base_max = INDUSTRY_SHOCKS_2008.get(industry, (-0.22, 0.12))
        growth = random.uniform(base_min, base_max)
        adj_min, adj_max = COUNTRY_SHOCKS_2008.get(country, (0.0, 0.0))
        if adj_min != 0.0 or adj_max != 0.0:
            growth += random.uniform(adj_min, adj_max)
    elif year == 2020:
        base_min, base_max = INDUSTRY_SHOCKS_2020.get(industry, (-0.25, 0.20))
        growth = random.uniform(base_min, base_max)
        adj_min, adj_max = COUNTRY_SHOCKS_2020.get(country, (0.0, 0.0))
        if adj_min != 0.0 or adj_max != 0.0:
            growth += random.uniform(adj_min, adj_max)
    else:
        growth = random.uniform(*DEFAULT_GROWTH_RANGE)

    return clamp(growth, -0.60, 0.50)


def project_company(row: pd.Series) -> list[float]:
    start_value = row.get('Valuation_of_company')
    if pd.isna(start_value):
        return [np.nan] * len(YEARS)

    acquisition_month = row.get('Acquisition_Month')
    acquisition_year = acquisition_month.year if not pd.isna(acquisition_month) else YEARS[0]

    industry = row.get('Industry')
    country = row.get('Country')

    value = float(start_value)
    projections: list[float] = []

    for year in YEARS:
        if year < acquisition_year:
            projections.append(np.nan)
        elif year == acquisition_year:
            projections.append(round(value, 2))
        else:
            growth = get_growth(year, industry, country)
            value = max(value * (1 + growth), 0)
            projections.append(round(value, 2))

    return projections


def augment_timeline(timeline_path: Path, profiles_path: Path) -> None:
    if not timeline_path.exists():
        raise FileNotFoundError(f"Timeline file not found at '{timeline_path}'")
    if not profiles_path.exists():
        raise FileNotFoundError(f"Company profiles file not found at '{profiles_path}'")

    timeline_df = pd.read_csv(timeline_path, parse_dates=['Acquisition_Month'])
    profiles_df = pd.read_csv(profiles_path)

    timeline_df['Valuation_of_company'] = clean_numeric(timeline_df['Valuation_of_company'])
    for column in ('Investment_Cost', 'Fund_Size_at_Acquisition'):
        if column in timeline_df.columns:
            timeline_df[column] = clean_numeric(timeline_df[column])

    timeline_df = timeline_df.merge(
        profiles_df[['Company', 'Industry', 'Country']],
        on='Company',
        how='left'
    )

    projection_columns = [f'Valuation_{year}' for year in YEARS]
    existing_projection_cols = [col for col in projection_columns if col in timeline_df.columns]
    if existing_projection_cols:
        timeline_df.drop(columns=existing_projection_cols, inplace=True)

    projections = timeline_df.apply(project_company, axis=1, result_type='expand')
    projections.columns = projection_columns

    timeline_df = pd.concat([timeline_df, projections], axis=1)

    missing_profiles = timeline_df['Industry'].isna().sum()
    if missing_profiles:
        print(f"Warning: Missing industry/country data for {missing_profiles} companies; using default growth ranges.")

    timeline_df.drop(columns=['Industry', 'Country'], inplace=True, errors='ignore')

    timeline_df.to_csv(timeline_path, index=False)
    print(f"Augmented '{timeline_path.name}' with valuation projections for 2005-2025.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment PE fund timeline with valuation projections.")
    parser.add_argument(
        "--timeline",
        type=Path,
        default=Path("PE_Fund_timeline.csv"),
        help="Path to the PE fund timeline CSV (default: PE_Fund_timeline.csv).",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("company_profiles_150.csv"),
        help="Path to the company profiles CSV (default: company_profiles_150.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible valuations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    augment_timeline(args.timeline, args.profiles)


if __name__ == "__main__":
    main()

