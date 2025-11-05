import pandas as pd
import numpy as np

# --- 1. Load Prerequisite Data ---
# This script assumes 'PE_Fund_Timeline.csv' (the 150-row acquisition history) exists.
try:
    df_timeline = pd.read_csv("PE_Fund_Timeline.csv")
except FileNotFoundError:
    print("Error: 'PE_Fund_Timeline.csv' not found.")
    print("Please ensure this file (the 150-row acquisition history) is in the same directory.")
    exit()

print(f"Loaded 'PE_Fund_Timeline.csv' successfully. Found {len(df_timeline)} investments.")

# --- 2. Clean and Prepare Data ---
# Ensure 'Investment_Cost' is numeric
df_timeline['Investment_Cost'] = pd.to_numeric(df_timeline['Investment_Cost'], errors='coerce')
df_timeline = df_timeline.dropna(subset=['Investment_Cost'])
# Ensure 'Acquisition_Month' is datetime and in order
df_timeline['Acquisition_Month'] = pd.to_datetime(df_timeline['Acquisition_Month'])
df_timeline = df_timeline.sort_values(by='Acquisition_Month')

# --- 3. Define Investor Commitments ---
print("Defining investor commitments...")

# 12 Individuals @ $5B each
individual_total_commitment = 12 * 5_000_000_000
# 12 Family Offices @ $5B each
family_office_total_commitment = 12 * 5_000_000_000
# Total from these 24 investors
total_from_24_investors = individual_total_commitment + family_office_total_commitment

# Calculate total fund size from the timeline data (should be ~$200B)
total_fund_size_from_file = df_timeline['Investment_Cost'].sum()

# Calculate "The First's" commitment (the remaining amount)
the_first_commitment = total_fund_size_from_file - total_from_24_investors

if the_first_commitment < 0:
    print(f"Warning: The 24 investors' commitments (${total_from_24_investors:,.0f}) already exceed the total fund size (${total_fund_size_from_file:,.0f}).")
    print("'The First' commitment will be zero and commitments will not be met.")
    the_first_commitment = 0 # Prevent negative commitment

# --- 4. Create Investor Dictionaries ---
investor_names_ordered = [
    "Amelia Chen",
    "Noah Sullivan",
    "Priya Deshmukh",
    "Luca Moretti",
    "Isabella Vargas",
    "Ethan Gallagher",
    "Aisha Mensah",
    "Kenji Watanabe",
    "Sofia Petrovic",
    "Mateo Alvarez",
    "Fatima Al-Sayeed",
    "Julian Becker",
    "Harborview Legacy Partners",
    "Silvercrest Stewardship",
    "Carraway Heritage Capital",
    "Everpine Holdings",
    "BlueMesa Family Investments",
    "Marigold Continuum Group",
    "Northbridge Dynasty Trust",
    "Aurora Ridge Capital",
    "Saffron Gate Assets",
    "Redwood Haven Ventures",
    "Celestia Heritage Fund",
    "Stonewell Family Capital",
    "FirstLight Institutional Partners",
]

# Cohort onboarding schedule (cumulative count of active investors by year)
cohort_schedule = [
    (pd.Timestamp("2005-01-01"), 8),   # first 8 investors
    (pd.Timestamp("2010-01-01"), 16),  # next 8 join (total 16)
    (pd.Timestamp("2015-01-01"), 24),  # all 24 LPs
    (pd.Timestamp("2020-01-01"), 25),  # FirstLight joins
]

investor_commitments = {name: 5_000_000_000 for name in investor_names_ordered[:-1]}
investor_commitments["FirstLight Institutional Partners"] = the_first_commitment

print(f"Total investors: {len(investor_names_ordered)}")

# Calculate the *actual* total commitment (this should match 'total_fund_size_from_file')
total_commitment_calculated = sum(investor_commitments.values())

# Calculate pro-rata percentage for each investor
# This is the % of *total capital* each investor is responsible for.
investor_pro_rata = {
    investor: commit / total_commitment_calculated
    for investor, commit in investor_commitments.items()
}


def active_investors_for_date(date: pd.Timestamp) -> list[str]:
    active_count = 0
    for cohort_date, cumulative_count in sorted(cohort_schedule, key=lambda x: x[0]):
        if date >= cohort_date:
            active_count = cumulative_count
    return investor_names_ordered[:active_count]

print("\n--- Fund Structure Verification ---")
print(f"Total Fund Size (from file): ${total_fund_size_from_file:,.0f}")
print(f"Individuals (12x $5B):       ${individual_total_commitment:,.0f}")
print(f"Family Offices (12x $5B):    ${family_office_total_commitment:,.0f}")
print(f"The First's Commitment:      ${the_first_commitment:,.0f}")
print(f"Total Calculated Commitment: ${total_commitment_calculated:,.0f}")
print("----------------------------------\n")

# --- 5. Create the Capital Call Timeline ---
print("Generating capital call timeline with staged onboarding...")
timeline_rows = []

# Iterate through each deal in the original timeline
for index, deal in df_timeline.iterrows():
    total_capital_called = deal['Investment_Cost']
    deal_date = deal['Acquisition_Month']
    active_investors = active_investors_for_date(deal_date)

    active_commitment = sum(investor_commitments[inv] for inv in active_investors)
    if active_commitment <= 0:
        continue

    for investor in active_investors:
        share = investor_commitments[investor] / active_commitment
        contribution = total_capital_called * share

        timeline_rows.append({
            'Capital_Call_Date': deal['Acquisition_Month'],
            'Triggering_Investment': deal['Company'],
            'Total_Capital_Called_for_Deal': total_capital_called,
            'Investor': investor,
            'Investor_Contribution': contribution
        })

df_capital_calls = pd.DataFrame(timeline_rows)

# --- 6. Add Cumulative Totals ---
print("Calculating cumulative and remaining commitments...")
# Calculate the cumulative contribution *per investor*
df_capital_calls['Cumulative_Contribution'] = df_capital_calls.groupby('Investor')['Investor_Contribution'].cumsum()

# Map the total commitment for each investor to the table
df_capital_calls['Total_Commitment'] = df_capital_calls['Investor'].map(investor_commitments)

# Calculate their remaining commitment
df_capital_calls['Remaining_Commitment'] = df_capital_calls['Total_Commitment'] - df_capital_calls['Cumulative_Contribution']

# --- 7. Save and Display Results ---
# Format for readability before saving
df_capital_calls['Capital_Call_Date'] = df_capital_calls['Capital_Call_Date'].dt.strftime('%Y-%m-%d')

# Round the currency columns
cols_to_round = [
    'Total_Capital_Called_for_Deal', 
    'Investor_Contribution', 
    'Cumulative_Contribution', 
    'Total_Commitment', 
    'Remaining_Commitment'
]
df_capital_calls[cols_to_round] = df_capital_calls[cols_to_round].round(0)

# Save the full timeline
output_filename = "investor_capital_call_timeline.csv"
df_capital_calls.to_csv(output_filename, index=False)

print(f"\nSuccessfully generated and saved '{output_filename}'")
print("\n--- Script Finished ---")