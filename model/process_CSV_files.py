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
import csv

## Process the CSV raw data

# Read the raw CSV file
input_file = 'input_data/ts_exports_raw.csv'
df = pd.read_csv(input_file)

# Remove any leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

# Remove any leading/trailing whitespace from SITC Section values
df['SITC Section'] = df['SITC Section'].str.strip()

# Remove quotes from column names and values
df.columns = df.columns.str.replace('"', '')
df = df.replace('"', '', regex=True)

# Convert the data to numeric, handling commas and coercing errors
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

# Pivot the dataframe to have years as rows and SITC Sections as columns
df_pivoted = df.set_index('SITC Section').transpose()


# Write the processed data to a new CSV file
output_file = 'input_data/ts_exports_automatic.csv'
df_pivoted.to_csv(output_file, index=False)

print(f"Processed data has been saved to {output_file}")