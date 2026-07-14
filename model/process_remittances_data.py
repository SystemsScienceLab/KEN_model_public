# =============================================================================
# Lead model conceptualization and development (theory, methodology, and
# implementation), software architecture & model coding (all components),
# data acquisition & processing, team supervision: Michael Miess
# Initial software architecture development, first coding, technical consulting:
# Joel Foramitti
# Lead of energy modeling: Ansir Ilyas
# Lead of water modeling: Dan Wang
# Model consulting & support: Asjad Naqvi
# Project conceptualization, funding & supervision: Yoshihide Wada
# =============================================================================

import pandas as pd
import os

# Set working directory to project root
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

##############################################################################
# Process Remittances Data for Saudi Arabia
# Source: World Development Indicators (IMF/World Bank)
# File: input_data/MM_Excels_Working/Remittances_IMF_EXCEL_BUTDOWNLOAD_CSV_fromHP_forINPUT.xls
# Sheet: Data
# Indicator: Personal remittances, paid (current US$) [BM.TRF.PWKR.CD.DT]
##############################################################################

# --- Step 1: Locate and read the Excel file ---
excel_filename = "Remittances_IMF_EXCEL_BUTDOWNLOAD_CSV_fromHP_forINPUT.xls"
search_dir = os.path.join("input_data", "MM_Excels_Working")
filepath = os.path.join(search_dir, excel_filename)

if not os.path.exists(filepath):
    # Search recursively in case the file was moved
    found = False
    for root, dirs, files in os.walk(search_dir):
        if excel_filename in files:
            filepath = os.path.join(root, excel_filename)
            found = True
            break
    if not found:
        raise FileNotFoundError(
            f"Could not find '{excel_filename}' in '{search_dir}' or its subdirectories."
        )

print(f"Reading: {filepath}")

# --- Step 2: Read the 'Data' sheet ---
# Structure: Row 0-2 = metadata, Row 3 = headers (Country Name, Country Code,
# Indicator Name, Indicator Code, 1960, 1961, ..., 2025), Row 4+ = data
df = pd.read_excel(filepath, sheet_name='Data', header=None)

# --- Step 3: Extract Saudi Arabia row ---
saudi_mask = df.iloc[:, 0].astype(str).str.contains('Saudi Arabia', case=False, na=False)
saudi_rows = df[saudi_mask]

if len(saudi_rows) == 0:
    raise ValueError("No 'Saudi Arabia' row found in the Data sheet.")

saudi_row = saudi_rows.iloc[0]
print(f"Country: {saudi_row.iloc[0]} ({saudi_row.iloc[1]})")
print(f"Indicator: {saudi_row.iloc[2]}")
print(f"Indicator Code: {saudi_row.iloc[3]}")

# --- Step 4: Build a year-indexed Series from 1971 onwards ---
year_headers = df.iloc[3, 4:].values  # year floats from row 3
values = saudi_row.iloc[4:].values

# Build dict {year: value} for years >= 1971 with non-NaN values
remittances = {}
for y, v in zip(year_headers, values):
    if pd.notna(y):
        yr = int(y)
        if yr >= 1971:
            remittances[yr] = v if pd.notna(v) else None

ts_remittances_usd = pd.Series(remittances, name='Remittances_paid_USD')
ts_remittances_usd.index.name = 'Year'

# --- Step 5: Convert from USD to 1000 SAR ---
# Conversion rate: 1 USD = 3.75 SAR
USD_TO_SAR = 3.75
ts_remittances = ts_remittances_usd.apply(
    lambda v: (v * USD_TO_SAR) / 1000.0 if v is not None else None
)
ts_remittances.name = 'Remittances_paid_1000SAR'
ts_remittances.index.name = 'Year'

# --- Step 6: Print the extracted data ---
print(f"\n{'='*70}")
print(f"  Saudi Arabia — Personal Remittances Paid (1000 SAR)")
print(f"  Converted from current US$ at 1 USD = {USD_TO_SAR} SAR")
print(f"  Source: World Development Indicators, {excel_filename}")
print(f"{'='*70}")
print(f"{'Year':>6}  {'Remittances (1000 SAR)':>25}  {'Remittances (Bn SAR)':>22}")
print(f"{'-'*6}  {'-'*25}  {'-'*22}")

for yr, val in ts_remittances.items():
    if val is not None:
        print(f"{yr:>6}  {val:>25,.0f}  {val/1e6:>22.2f}")
    else:
        print(f"{yr:>6}  {'NaN':>25}  {'NaN':>22}")

n_valid = ts_remittances.dropna().count()
print(f"\nTotal years with data: {n_valid} (from {ts_remittances.dropna().index.min()} to {ts_remittances.dropna().index.max()})")
if n_valid > 0:
    print(f"Latest value ({ts_remittances.dropna().index.max()}): {ts_remittances.dropna().iloc[-1]:,.0f} (1000 SAR) = {ts_remittances.dropna().iloc[-1]/1e6:.2f} Bn SAR")

# --- Step 7: Save to CSV for calibration.py ---
output_file = 'input_data/ts_remittances.csv'
ts_out = ts_remittances.dropna()
ts_out.to_csv(output_file)
print(f"\nProcessed remittances time series saved to {output_file}")
print(f"Shape: {len(ts_out)} years, units: 1000 SAR")

print("\nDone.")
