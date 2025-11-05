import pandas as pd
import numpy as np
import random

# --- 0. Configuration ---
original_file_name = r"D:\PEDummy\GreenFoods_Manufacturing_FullMonitoring.xlsx"
names_file_name = r"D:\PEDummy\Names.csv"
profiles_output_filename = "company_profiles_150.csv"
financials_output_filename = "synthetic_financial_data_150_companies.csv"
num_companies = 150

# Use a reduced std dev for noise to prevent wild negative swings
noise_reduction_factor = 4.0
# Add a small epsilon to denominator to avoid division by zero
epsilon = 1e-6

# --- 1. Define Company Profiles (Industries & Countries) ---

# Generate 150 unique company codes and map to provided names
company_codes = [f"GlobalCorp_{i+1:03d}" for i in range(num_companies)]

try:
    names_df = pd.read_csv(names_file_name, header=None, names=['Original', 'New'])
except FileNotFoundError:
    print(f"Error: Names file not found at '{names_file_name}'")
    exit()

name_map = dict(zip(names_df['Original'], names_df['New']))
company_names = [name_map.get(code, code) for code in company_codes]

missing_names = [code for code in company_codes if code not in name_map]
if missing_names:
    preview = ', '.join(missing_names[:5])
    suffix = '...' if len(missing_names) > 5 else ''
    print(f"Warning: Missing custom names for {len(missing_names)} companies: {preview}{suffix}")

# --- Define Industries ---
# 15 base industries
base_industries = [
    "Aerospace", "Biotech", "Software", "Logistics", "Retail", 
    "Energy", "Media", "Automotive", "Tech", "Consumer Goods", 
    "Utilities", "Services", "Pharma", "E-commerce", "Industrial"
]
# Repeat the list 10 times to get 150 industries
industries = (base_industries * 10)

# --- Define Countries (with specified density) ---
# Define country lists
countries_me = ["UAE", "Saudi Arabia", "Qatar", "Kuwait", "Oman", "Bahrain"]
countries_uk = ["UK"]
countries_eu = ["Germany", "France", "Italy", "Spain", "Netherlands", "Ireland"]
countries_usa = ["USA"]

# Create the weighted pool of 150 countries
# ~40% Middle-East (60 companies)
country_pool = countries_me * 10  
# ~30% UK (45 companies)
country_pool += countries_uk * 45 
# ~20% EU (30 companies)
country_pool += countries_eu * 5   
# ~10% USA (15 companies)
country_pool += countries_usa * 15 

# Shuffle the country list to randomly assign them
random.shuffle(country_pool)
countries = country_pool

# --- Define Industry Financial Profile "Multipliers" ---
# These will be applied to the baseline (GreenFoods) ratios.
industry_profiles = {
    # Baseline (Original File)
    "Manufacturing":   {'cogs_mult': 1.0, 'sga_mult': 1.0, 'rd_mult': 1.0, 'debt_asset_mult': 1.0, 'int_rate_mult': 1.0},
    
    # Industry-specific profiles
    "Aerospace":       {'cogs_mult': 1.1, 'sga_mult': 0.8, 'rd_mult': 2.0, 'debt_asset_mult': 1.5, 'int_rate_mult': 1.0},
    "Biotech":         {'cogs_mult': 0.4, 'sga_mult': 1.4, 'rd_mult': 4.0, 'debt_asset_mult': 1.0, 'int_rate_mult': 1.2},
    "Software":        {'cogs_mult': 0.3, 'sga_mult': 1.6, 'rd_mult': 2.2, 'debt_asset_mult': 0.7, 'int_rate_mult': 0.9},
    "Logistics":       {'cogs_mult': 1.2, 'sga_mult': 1.1, 'rd_mult': 0.1, 'debt_asset_mult': 1.3, 'int_rate_mult': 1.1},
    "Retail":          {'cogs_mult': 1.2, 'sga_mult': 1.3, 'rd_mult': 0.2, 'debt_asset_mult': 1.1, 'int_rate_mult': 1.1},
    "Energy":          {'cogs_mult': 1.1, 'sga_mult': 0.8, 'rd_mult': 0.3, 'debt_asset_mult': 1.8, 'int_rate_mult': 0.9},
    "Media":           {'cogs_mult': 0.7, 'sga_mult': 1.4, 'rd_mult': 0.8, 'debt_asset_mult': 1.1, 'int_rate_mult': 1.1},
    "Automotive":      {'cogs_mult': 1.2, 'sga_mult': 0.9, 'rd_mult': 1.2, 'debt_asset_mult': 1.4, 'int_rate_mult': 1.0},
    "Tech":            {'cogs_mult': 0.6, 'sga_mult': 1.4, 'rd_mult': 2.5, 'debt_asset_mult': 0.8, 'int_rate_mult': 0.9},
    "Consumer Goods":  {'cogs_mult': 1.1, 'sga_mult': 1.2, 'rd_mult': 0.5, 'debt_asset_mult': 1.0, 'int_rate_mult': 1.0},
    "Utilities":       {'cogs_mult': 1.0, 'sga_mult': 0.7, 'rd_mult': 0.1, 'debt_asset_mult': 2.0, 'int_rate_mult': 0.8},
    "Services":        {'cogs_mult': 0.2, 'sga_mult': 2.0, 'rd_mult': 0.1, 'debt_asset_mult': 0.5, 'int_rate_mult': 1.0},
    "Pharma":          {'cogs_mult': 0.5, 'sga_mult': 1.3, 'rd_mult': 3.0, 'debt_asset_mult': 0.9, 'int_rate_mult': 0.9},
    "E-commerce":      {'cogs_mult': 0.8, 'sga_mult': 1.5, 'rd_mult': 1.5, 'debt_asset_mult': 1.0, 'int_rate_mult': 1.0},
    "Industrial":      {'cogs_mult': 1.1, 'sga_mult': 0.9, 'rd_mult': 0.7, 'debt_asset_mult': 1.3, 'int_rate_mult': 1.1}
}

# --- Create the company profiles DataFrame ---
company_profiles_df = pd.DataFrame({
    'Company': company_names,
    'Industry': industries,
    'Country': countries
})

# Save the profiles to CSV
company_profiles_df.to_csv(profiles_output_filename, index=False)
print(f"Successfully created '{profiles_output_filename}'")
print(f"Country distribution check:\n{company_profiles_df['Country'].value_counts()}")

# --- 2. Load Original Data and Get Baseline Parameters ---
try:
    df = pd.read_excel(original_file_name)
except FileNotFoundError:
    print(f"Error: Original file not found at '{original_file_name}'")
    exit()

df['Month'] = df['Month'].astype(str)
original_months = df['Month'].values
num_months = len(original_months)

# Base value means
mean_revenue = df['Revenue'].mean()
mean_assets = df['Assets'].mean()

# Noise/Variation percentages
std_revenue_percent = df['Revenue'].std() / mean_revenue
std_assets_percent = df['Assets'].std() / mean_assets

# Replace 0s with epsilon to avoid division by zero errors
df['Revenue'] = df['Revenue'].replace(0, epsilon)
df['Assets'] = df['Assets'].replace(0, epsilon)
df['Debt'] = df['Debt'].replace(0, epsilon)
df['Equity'] = df['Equity'].replace(0, epsilon)
df['Operating_Income_(EBIT)'] = df['Operating_Income_(EBIT)'].replace(0, epsilon)

# Calculate BASELINE ratios
base_mean_cogs_ratio = (df['COGS'] / df['Revenue']).mean()
base_std_cogs_ratio = (df['COGS'] / df['Revenue']).std()

base_mean_sga_ratio = (df['SG&A'] / df['Revenue']).mean()
base_std_sga_ratio = (df['SG&A'] / df['Revenue']).std()

base_mean_rd_ratio = (df['R&D'] / df['Revenue']).mean()
base_std_rd_ratio = (df['R&D'] / df['Revenue']).std()

base_mean_debt_to_assets_ratio = (df['Debt'] / df['Assets']).mean()
base_std_debt_to_assets_ratio = (df['Debt'] / df['Assets']).std()

base_mean_interest_rate_on_debt = (df['Interest_Expense'] / df['Debt']).mean()
base_std_interest_rate_on_debt = (df['Interest_Expense'] / df['Debt']).std()

# Calculate tax rate
positive_ebit_df = df[df['Operating_Income_(EBIT)'] > 0]
if not positive_ebit_df.empty:
    mean_tax_rate = (positive_ebit_df['Tax_Expense'] / positive_ebit_df['Operating_Income_(EBIT)']).mean()
    std_tax_rate = (positive_ebit_df['Tax_Expense'] / positive_ebit_df['Operating_Income_(EBIT)']).std()
else:
    mean_tax_rate = 0.2  # Fallback
    std_tax_rate = 0.05

# Calculate EBITDA relationship
df['Debt/EBITDA'] = df['Debt/EBITDA'].replace(0, epsilon)
implied_EBITDA = df['Debt'] / df['Debt/EBITDA']
implied_EBITDA = implied_EBITDA.replace(0, epsilon)
mean_ebit_to_ebitda_ratio = (df['Operating_Income_(EBIT)'] / implied_EBITDA).mean()
std_ebit_to_ebitda_ratio = (df['Operating_Income_(EBIT)'] / implied_EBITDA).std()

# --- 3. Generate Industry-Specific Financial Data Loop ---
all_dfs = []

print(f"Generating financial data for {num_companies} companies...")

for index, row in company_profiles_df.iterrows():
    company_name = row['Company']
    industry = row['Industry']
    
    # Get the financial profile for this company's industry
    profile = industry_profiles.get(industry, industry_profiles['Manufacturing'])
    
    # Apply industry multipliers to baseline ratios
    current_mean_cogs_ratio = base_mean_cogs_ratio * profile['cogs_mult']
    current_std_cogs_ratio = base_std_cogs_ratio * profile['cogs_mult']
    
    current_mean_sga_ratio = base_mean_sga_ratio * profile['sga_mult']
    current_std_sga_ratio = base_std_sga_ratio * profile['sga_mult']
    
    current_mean_rd_ratio = base_mean_rd_ratio * profile['rd_mult']
    current_std_rd_ratio = base_std_rd_ratio * profile['rd_mult']
    
    current_mean_debt_to_assets_ratio = base_mean_debt_to_assets_ratio * profile['debt_asset_mult']
    current_std_debt_to_assets_ratio = base_std_debt_to_assets_ratio * profile['debt_asset_mult']
    
    current_mean_interest_rate_on_debt = base_mean_interest_rate_on_debt * profile['int_rate_mult']
    current_std_interest_rate_on_debt = base_std_interest_rate_on_debt * profile['int_rate_mult']

    # --- Start Generating Data ---
    new_df = pd.DataFrame()
    new_df['Month'] = original_months
    new_df['Company'] = company_name
    
    # Each company has a different base scale
    company_scale_factor = np.random.uniform(0.3, 2.5)
    
    # Generate base columns
    rev_noise = np.random.normal(1, std_revenue_percent / noise_reduction_factor, num_months)
    new_df['Revenue'] = np.clip(mean_revenue * company_scale_factor * rev_noise, a_min=0, a_max=None)

    assets_noise = np.random.normal(1, std_assets_percent / noise_reduction_factor, num_months)
    new_df['Assets'] = np.clip(mean_assets * company_scale_factor * assets_noise, a_min=0, a_max=None)

    # Generate derived columns based on INDUSTRY-SPECIFIC ratios
    cogs_ratio = np.random.normal(current_mean_cogs_ratio, current_std_cogs_ratio / noise_reduction_factor, num_months)
    new_df['COGS'] = new_df['Revenue'] * cogs_ratio.clip(0.1, 0.95) # Clip to reasonable bounds

    sga_ratio = np.random.normal(current_mean_sga_ratio, current_std_sga_ratio / noise_reduction_factor, num_months)
    new_df['SG&A'] = new_df['Revenue'] * sga_ratio.clip(0.05, 0.5) # Clip

    rd_ratio = np.random.normal(current_mean_rd_ratio, current_std_rd_ratio / noise_reduction_factor, num_months)
    new_df['R&D'] = new_df['Revenue'] * rd_ratio.clip(0.0, 0.4) # Clip (some firms have 0 R&D)
    
    debt_assets_ratio = np.random.normal(current_mean_debt_to_assets_ratio, current_std_debt_to_assets_ratio / noise_reduction_factor, num_months)
    new_df['Debt'] = new_df['Assets'] * debt_assets_ratio.clip(0.1, 0.9)
    
    new_df['Equity'] = new_df['Assets'] - new_df['Debt']
    
    interest_rate = np.random.normal(current_mean_interest_rate_on_debt, current_std_interest_rate_on_debt / noise_reduction_factor, num_months)
    new_df['Interest_Expense'] = new_df['Debt'] * interest_rate.clip(0.01, 0.1)

    # Calculate P&L
    new_df['Gross_Profit'] = new_df['Revenue'] - new_df['COGS']
    new_df['Operating_Income_(EBIT)'] = new_df['Gross_Profit'] - new_df['SG&A'] - new_df['R&D']
    
    # Tax
    tax_rate_noise = np.random.normal(mean_tax_rate, std_tax_rate / noise_reduction_factor, num_months)
    applied_tax_rates = tax_rate_noise.clip(0.1, 0.4)
    new_df['Tax_Expense'] = new_df['Operating_Income_(EBIT)'] * applied_tax_rates
    new_df.loc[new_df['Operating_Income_(EBIT)'] < 0, 'Tax_Expense'] = 0 # No tax credits on losses

    new_df['Net_Income'] = new_df['Operating_Income_(EBIT)'] - new_df['Interest_Expense'] - new_df['Tax_Expense']

    # --- Calculate Ratios ---
    new_df['Revenue_e'] = new_df['Revenue'].replace(0, epsilon)
    new_df['Equity_e'] = new_df['Equity'].replace(0, epsilon)
    new_df['Assets_e'] = new_df['Assets'].replace(0, epsilon)

    new_df['Gross_Margin_%'] = (new_df['Gross_Profit'] / new_df['Revenue_e']) * 100
    new_df['ROE_%'] = (new_df['Net_Income'] / new_df['Equity_e']) * 100
    new_df['ROA_%'] = (new_df['Net_Income'] / new_df['Assets_e']) * 100
    new_df['Debt/Equity'] = new_df['Debt'] / new_df['Equity_e']

    # Calculate Debt/EBITDA
    ebit_to_ebitda_ratio_noise = np.random.normal(mean_ebit_to_ebitda_ratio, std_ebit_to_ebitda_ratio / noise_reduction_factor, num_months)
    ebit_to_ebitda_ratio_noise = ebit_to_ebitda_ratio_noise.clip(0.5, 1.5)
    
    new_EBITDA = new_df['Operating_Income_(EBIT)'] / ebit_to_ebitda_ratio_noise
    new_EBITDA.loc[new_EBITDA <= 0] = epsilon # Handle 0 or negative EBITDA
    
    new_df['Debt/EBITDA'] = new_df['Debt'] / new_EBITDA

    # Clean up infinities and helper columns
    new_df = new_df.replace([np.inf, -np.inf], 0)
    new_df = new_df.drop(columns=['Revenue_e', 'Equity_e', 'Assets_e'])
    
    all_dfs.append(new_df)

# --- 4. Combine, Finalize, and Save Financials ---
final_df = pd.concat(all_dfs, ignore_index=True)

# Reorder columns to match original
original_cols_no_month = [col for col in df.columns if col != 'Month' and col in final_df.columns]
final_cols = ['Company', 'Month'] + original_cols_no_month
final_df = final_df[final_cols]

# Round and set data types
float_cols = final_df.select_dtypes(include=['float64']).columns
final_df[float_cols] = final_df[float_cols].round(2)

int_like_cols = ['Revenue', 'COGS', 'Gross_Profit', 'SG&A', 'R&D', 
                 'Operating_Income_(EBIT)', 'Interest_Expense', 'Tax_Expense', 
                 'Net_Income', 'Assets', 'Debt', 'Equity']
for col in int_like_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].astype(np.int64)

# Save the main financial data CSV
final_df.to_csv(financials_output_filename, index=False)
print(f"\nSuccessfully created '{financials_output_filename}'")

print(f"Total rows in financial data: {len(final_df)}")
print("\nHead of the new 150-company financial data:")
print(final_df.head())