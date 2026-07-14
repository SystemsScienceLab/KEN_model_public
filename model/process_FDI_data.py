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
import numpy as np
import os
import csv

# Set working directory to project root
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

##############################################################################
# Process the Foreign Direct Investment (FDI) Bulletin 2023 Excel data - FDI NET INFLOW
# Adapted from process_CSV_files.py blueprint
# Source: input_data/FDI_foreign_direct_investment_data/Foreign-Direct-Investment-Bulletin-2023-Excel.xlsx
# Sheet: 2-2
##############################################################################

# --- Step 1: Create the processed FDI CSV from the raw data ---
# The raw data is provided as text from the FDI Bulletin.
# We create it directly as a DataFrame to avoid Excel reading issues.

# FDI sectors as they appear in the data
fdi_sectors = [
    "Manufacturing",
    "Financial and Insurance Activities",
    "Construction",
    "Wholesale And Retail Trade",
    "Professional, Scientific and Technical Activities",
    "Information and Communication",
    "Mining and Quarrying",
    "Real Estate Activities",
    "Administrative and Support Service Activities",
    "Arts, entertainment and recreation",
    "Accommodation And Food Service Activities",
    "Water supply; sewerage, waste management and remediation activities",
    "Human Health and Social Work Activities",
    "Electricity, Gas, Steam And Air Conditioning Supply",
    "Education",
    "Other service activities",
    "Agriculture, forestry and fishing",
    "Transportation and Storage",
    "Total"
]

# Raw FDI NET INFLOW data by year (in SAR thousands)
fdi_data = {
    2015: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    2016: [49660186, 1263044, 6454952, 4714444, 1917810, -852666, 1485596, 7622730, 964734, 1958, 589538, 4413471, -20551, 1852738, 26265, 11129, 108558, 2116691, 82330627],
    2017: [2742542, -3059348, -541908, 1032163, -140934, -1281606, 2716882, 2881417, -598495, 17414, -759301, -560186, 277019, 1133027, 67627, -12799, -100107, -10671, 3802734],
    2018: [7834422, -25760, 6862076, 8299438, 1868279, 3107228, 5037513, 7760216, -466746, -9796, 1889736, 462686, 1079476, 761881, 279372, -215663, 409181, 595669, 45529208],
    2019: [-14293672, -2734702, 3563302, 11030888, -1045877, 13118140, -389456, -306640, -37800, -38313, 2054261, -1550052, 1142258, -3222245, 397232, -680324, 49596, 4490468, 11547065],
    2020: [10544436, -373371, -3372142, 1313361, -518094, 880492, -1218587, 375164, -209725, 29028, -187088, 100578, -383317, -120312, 35045, -82969, -96300, -636458, 6079741],
    2021: [28945760, 3649829, 5322989, 12361821, 1961959, 1527562, 889986, -289534, 749193, 37412, -71456, 673422, 195228, 402564, 120431, 160591, 108448, 49567696, 106313900],
    2022: [13139483, 15807081, 7536847, 1494611, 4778115, 540041, 1455993, 422697, 1400725, 75894, 905239, 886368, 224527, 1404102, 50453, 79318, 159028, 49802210, 100162733],
    2023: [32101829, 14677093, 10854632, 10455630, 5245063, 3382104, 3024699, 2509512, 1937344, 1045157, 962221, 537512, 369611, 217518, 171217, 155415, 153922, -2287359, 85513118],
}

# Create DataFrame with years as rows and FDI sectors as columns
df = pd.DataFrame(fdi_data, index=fdi_sectors).T
df.index.name = 'Year'

# Remove leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

# Write the processed FDI NET INFLOW time series to CSV (years as rows, sectors as columns)
output_file = 'input_data/ts_fdi.csv'
df.to_csv(output_file)
print(f"Processed FDI NET INFLOW time series data saved to {output_file}")
print(f"Shape: {df.shape} (years x sectors)")
print(f"Years: {list(df.index)}")
print(f"Sectors: {list(df.columns)}")
print()
print("FDI NET INFLOW data (SAR):")
print(df.to_string())


##############################################################################
# Process the FDI STOCK data
# Source: Foreign Direct Investment Bulletin 2023
##############################################################################

# FDI STOCK sectors as they appear in the data (different column order from net flows)
fdi_stock_sectors = [
    "Manufacturing",
    "Wholesale And Retail Trade",
    "Financial and Insurance Activities",
    "Transportation and Storage",
    "Construction",
    "Mining and Quarrying",
    "Information and Communication",
    "Professional, Scientific and Technical Activities",
    "Administrative and Support Service Activities",
    "Real Estate Activities",
    "Electricity, Gas, Steam And Air Conditioning Supply",
    "Water supply; sewerage, waste management and remediation activities",
    "Accommodation And Food Service Activities",
    "Human Health and Social Work Activities",
    "Education",
    "Agriculture, forestry and fishing",
    "Arts, entertainment and recreation",
    "Other service activities",
    "Total"
]

# Raw FDI STOCK data by year (in SAR)
fdi_stock_data = {
    2015: [113672819, 61046498, 62852665, 9649029, 30947185, 16697612, 16726358, 11054899, 15405922, 36213633, 12434145, 5664324, 12609172, 1684803, 244750, 521374, 234945, 569592, 408229726],
    2016: [159255099, 65549398, 59997548, 13181064, 44612603, 17637338, 15251019, 14361268, 18573276, 47156061, 14369858, 8980115, 13094191, 1765052, 790747, 710559, 357707, 598568, 496241470],
    2017: [157435162, 69738324, 55954503, 12421043, 46669671, 19268888, 13981326, 16451629, 17310655, 51218502, 16170794, 8585676, 11706592, 2264364, 1059531, 459467, 366245, 751207, 501813581],
    2018: [168985822, 76301457, 66544492, 13331933, 54019169, 20222789, 15209418, 16439423, 17247698, 56195793, 16486536, 9590039, 13111662, 3458548, 1334070, 798870, 355379, 444029, 550077127],
    2019: [191811769, 73133345, 63993391, 14582836, 50965292, 17391465, 17107514, 14911430, 16215607, 55161008, 15231217, 8809019, 13724943, 3739148, 1440712, 809764, 404398, -30694, 559402164],
    2020: [189810796, 83107831, 58565391, 14251481, 51746274, 17946507, 21876592, 14542126, 17513951, 55475722, 15795620, 9101086, 14224834, 4027298, 1584249, 751929, 505846, -107402, 570720131],
    2021: [200183347, 123071334, 70147399, 63513539, 61619436, 24964341, 24705088, 20549255, 25008604, 18749374, 14079753, 12894266, 10983845, 4751469, 3063378, 1865499, 569356, 25226, 680744508],
    2022: [214405134, 124358837, 94362944, 113691001, 68977162, 27925089, 24725187, 25152342, 26318766, 19146735, 15471190, 13764892, 11889841, 4969922, 3109515, 2023123, 645088, 99203, 791035970],
    2023: [258737185, 134755131, 112129546, 111256304, 79785614, 33224432, 31710238, 30407786, 28141031, 21655657, 15673084, 14291381, 12843145, 5337023, 3276642, 2176052, 1698260, 248748, 897347258],
}

# Create DataFrame
df_stock = pd.DataFrame(fdi_stock_data, index=fdi_stock_sectors).T
df_stock.index.name = 'Year'
df_stock.columns = df_stock.columns.str.strip()

# Write the processed FDI STOCK time series to CSV
output_file_stock = 'input_data/ts_fdi_stock.csv'
df_stock.to_csv(output_file_stock)
print(f"\nProcessed FDI STOCK time series data saved to {output_file_stock}")
print(f"Shape: {df_stock.shape} (years x sectors)")
print(f"Years: {list(df_stock.index)}")
print(f"Sectors: {list(df_stock.columns)}")
print()
print("FDI STOCK data (SAR):")
print(df_stock.to_string())


##############################################################################
# Process the FDI GROSS INFLOW data
# Source: Foreign Direct Investment Bulletin 2023
##############################################################################

# FDI GROSS INFLOW sectors as they appear in the data (different column order)
fdi_gross_sectors = [
    "Manufacturing",
    "Financial and Insurance Activities",
    "Construction",
    "Wholesale And Retail Trade",
    "Professional, Scientific and Technical Activities",
    "Mining and Quarrying",
    "Information and Communication",
    "Real Estate Activities",
    "Administrative and Support Service Activities",
    "Arts, entertainment and recreation",
    "Accommodation And Food Service Activities",
    "Water supply; sewerage, waste management and remediation activities",
    "Human Health and Social Work Activities",
    "Electricity, Gas, Steam And Air Conditioning Supply",
    "Education",
    "Other service activities",
    "Agriculture, forestry and fishing",
    "Transportation and Storage",
    "Total"
]

# Raw FDI GROSS INFLOW data by year (in SAR)
fdi_gross_data = {
    2015: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    2016: [54310229, 4345066, 12794412, 6018827, 2670433, 4786712, 3098087, 8493234, 1953466, 1958, 589503, 4903577, -20550, 3521904, 173911, 20560, 108551, 2308469, 110078350],
    2017: [6921769, -1223084, 6847650, 2650766, 662938, 6988405, 522294, 3146078, -431298, 17285, -753677, 97913, 289059, 1832159, 188935, 22154, -90822, 403510, 28092034],
    2018: [15324758, 1961533, 13921326, 10002886, 2501488, 7297574, 4420619, 8073178, 49481, -9792, 1889100, 1098716, 1345772, 1229284, 280067, -190692, 409034, 706537, 70310869],
    2019: [-9211095, -902127, 9539771, 12387166, -586708, 2155626, 13902057, 344357, 866366, -38314, 2065892, -1269144, 1231767, -3125333, 433600, -668170, 49598, 4650420, 31825729],
    2020: [17634019, 1171949, 2505374, 3515185, 34907, 680740, 2072017, 619752, 539822, 28991, -186726, 200147, 177095, 842522, 100835, -64654, 9076, -17891, 29863160],
    2021: [33528371, 3821513, 7793650, 14247912, 3275368, 1355089, 3300931, 492506, 1052643, 47070, 825363, 725083, 367293, 775445, 122138, 176456, 125718, 49757262, 121789813],
    2022: [21407846, 16114122, 10755470, 4150635, 5515861, 3036032, 1255563, 685562, 1902723, 75896, 933932, 963978, 248468, 1585130, 63011, 108684, 175388, 49962861, 118941162],
    2023: [34439778, 14860140, 13385272, 12574361, 5926334, 4211062, 3813011, 2609079, 2230652, 1062094, 1004643, 566961, 380345, 265404, 173342, 162401, 154922, -1836481, 95983319],
}

# Create DataFrame
df_gross = pd.DataFrame(fdi_gross_data, index=fdi_gross_sectors).T
df_gross.index.name = 'Year'
df_gross.columns = df_gross.columns.str.strip()

# Write the processed FDI GROSS INFLOW time series to CSV
output_file_gross = 'input_data/ts_fdi_gross.csv'
df_gross.to_csv(output_file_gross)
print(f"\nProcessed FDI GROSS INFLOW time series data saved to {output_file_gross}")
print(f"Shape: {df_gross.shape} (years x sectors)")
print(f"Years: {list(df_gross.index)}")
print(f"Sectors: {list(df_gross.columns)}")
print()
print("FDI GROSS INFLOW data (SAR):")
print(df_gross.to_string())


# --- Step 2: Create the FDI-to-model-sector correspondence mapping ---
# The FDI data has 18 sectors (excluding Total), the model has 85 sectors.
# We map each model sector to the corresponding FDI aggregate sector using the
# NACE classification, following the same logic as in postprocessing.py (lines 281-302).
#
# The mapping uses investment shares (dI) within each NACE group to disaggregate
# the aggregate FDI flows into model-sector-level flows.

# Model sector names (85 sectors, indices 0-84, as produced by calibration.py)
model_sectors = [
    "Crop and animal production, hunting and related service activities",            # 0  A
    "Forestry and logging",                                                         # 1  A
    "Fishing and aquaculture",                                                      # 2  A
    "Mining of coal and lignite",                                                   # 3  B
    "Extraction of crude petroleum and natural gas",                                # 4  B
    "Mining of metal ores",                                                         # 5  B
    "Other mining and quarrying activities",                                        # 6  B
    "Mining support service activities",                                            # 7  B
    "Manufacture of food products",                                                 # 8  C
    "Manufacture of beverages",                                                     # 9  C
    "Manufacture of tobacco products",                                              # 10 C
    "Manufacture of textiles",                                                      # 11 C
    "Manufacture of wearing apparel",                                               # 12 C
    "Manufacture of leather and related products",                                  # 13 C
    "Manufacture of woods, wood products and cork except furniture",                 # 14 C
    "Manufacture of paper and paper products",                                      # 15 C
    "Printing and reproduction of recorded media",                                  # 16 C
    "Manufacture of coke and refined petroleum products",                           # 17 C
    "Manufacture of chemicals and chemical products",                               # 18 C
    "Manufacture of basic pharmaceutical products and pharmaceutical preparations", # 19 C
    "Manufacture of rubber and plastics products",                                  # 20 C
    "Manufacture of other non-metallic mineral products",                           # 21 C
    "Manufacture of basic metals",                                                  # 22 C
    "Manufacture of fabricated metal products, except machinery and equipment",      # 23 C
    "Manufacture of computer, electronic and optical products",                     # 24 C
    "Manufacture of electrical equipment",                                          # 25 C
    "Manufacture of machinery and equipment n.e.c.",                                # 26 C
    "Manufacture of motor vehicles, trailers and semi-trailers",                    # 27 C
    "Manufacture of other transport equipment",                                     # 28 C
    "Manufacture of furniture",                                                     # 29 C
    "Other manufacturing",                                                          # 30 C
    "Repair and installation of machinery and equipment",                           # 31 C
    "Electricity, gas, steam and air conditioning supply",                          # 32 D
    "Water collection, treatment and supply",                                       # 33 E
    "Sewerage",                                                                     # 34 E
    "Waste collection, treatment and disposal activities; materials recovery",       # 35 E
    "Remediation activities and other waste management services",                   # 36 E
    "Construction of buildings",                                                    # 37 F
    "Civil engineering",                                                            # 38 F
    "Specialized construction activities",                                          # 39 F
    "Wholesale and retail trade and repair of motor vehicles and motorcycles",       # 40 G
    "Wholesale trade, except of motor vehicles and motorcycles",                    # 41 G
    "Retail trade, except of motor vehicles and motorcycles",                       # 42 G
    "Land transport and transport via pipelines",                                   # 43 H
    "Water transport",                                                              # 44 H
    "Air transport",                                                                # 45 H
    "Warehousing and support activities for transportation",                        # 46 H
    "Postal and courier activities",                                                # 47 H
    "Accommodation",                                                                # 48 I
    "Food and beverage service activities",                                         # 49 I
    "Publishing activities",                                                        # 50 J
    "Motion picture, video and television programme production, sound recording and music publishing activities", # 51 J
    "Programming and broadcasting activities",                                      # 52 J
    "Telecommunications",                                                           # 53 J
    "Computer programming, consultancy and related activities",                     # 54 J
    "Information service activities",                                               # 55 J
    "Financial service activities, except insurance and pension funding",            # 56 K
    "Insurance, reinsurance and pension funding, except compulsory social security", # 57 K
    "Activities auxiliary to financial service and insurance activities",            # 58 K
    "Real estate activities",                                                       # 59 L
    "Legal and accounting activities",                                              # 60 M
    "Activities of head offices; management consultancy activities",                 # 61 M
    "Architectural and engineering activities; technical testing and analysis",      # 62 M
    "Scientific research and development",                                          # 63 M
    "Advertising and market research",                                              # 64 M
    "Other professional, scientific and technical activities",                       # 65 M
    "Veterinary activities",                                                        # 66 M
    "Rental and leasing activities",                                                # 67 N
    "Employment activities",                                                        # 68 N
    "Travel agency, tour operator, reservation service and related activities",     # 69 N
    "Security and investigation activities",                                        # 70 N
    "Services to buildings and landscape activities",                               # 71 N
    "Office administrative, office support and other business support activities",   # 72 N
    "Public administration and defence; compulsory social security",                 # 73 O
    "Education",                                                                    # 74 P
    "Human health activities",                                                      # 75 Q
    "Residential care activities",                                                  # 76 Q
    "Social work activities without accommodation",                                 # 77 Q
    "Creative, arts and entertainment activities",                                  # 78 R
    "Libraries, archives, museums and other cultural activities",                    # 79 R
    "Sports activities and amusement and recreation activities",                     # 80 R
    "Activities of membership organizations",                                       # 81 S+T
    "Repair of computers and personal and household goods",                         # 82 S+T
    "Other personal service activities",                                            # 83 S+T
    "Desalination",                                                                 # 84 Desal
]

# Mapping from model sector index to FDI aggregate sector name
# Based on NACE classification used in postprocessing.py (lines 281-302)
# Note: sectors with no FDI match (Public administration=73, Desalination=84) get None
def get_fdi_sector_for_model_index(idx):
    """Map model sector index (0-based) to FDI aggregate sector name."""
    if 0 <= idx <= 2:                                                        # A
        return "Agriculture, forestry and fishing"
    elif 3 <= idx <= 7:                                                      # B
        return "Mining and Quarrying"
    elif 8 <= idx <= 31:                                                     # C
        return "Manufacturing"
    elif idx == 32:                                                          # D
        return "Electricity, Gas, Steam And Air Conditioning Supply"
    elif 33 <= idx <= 36:                                                    # E
        return "Water supply; sewerage, waste management and remediation activities"
    elif 37 <= idx <= 39:                                                    # F
        return "Construction"
    elif 40 <= idx <= 42:                                                    # G
        return "Wholesale And Retail Trade"
    elif 43 <= idx <= 47:                                                    # H
        return "Transportation and Storage"
    elif 48 <= idx <= 49:                                                    # I
        return "Accommodation And Food Service Activities"
    elif 50 <= idx <= 55:                                                    # J
        return "Information and Communication"
    elif 56 <= idx <= 58:                                                    # K
        return "Financial and Insurance Activities"
    elif idx == 59:                                                          # L
        return "Real Estate Activities"
    elif 60 <= idx <= 66:                                                    # M
        return "Professional, Scientific and Technical Activities"
    elif 67 <= idx <= 72:                                                    # N
        return "Administrative and Support Service Activities"
    elif idx == 73:                                                          # O
        return None  # Public administration - no FDI
    elif idx == 74:                                                          # P
        return "Education"
    elif 75 <= idx <= 77:                                                    # Q
        return "Human Health and Social Work Activities"
    elif 78 <= idx <= 80:                                                    # R
        return "Arts, entertainment and recreation"
    elif 81 <= idx <= 83:                                                    # S + T
        return "Other service activities"
    elif idx == 84:                                                          # Desal
        return None  # Desalination - no FDI equivalent
    else:
        return None

# Build the mapping dictionary: model_sector_index -> FDI_sector_name
model_to_fdi_mapping = {}
for idx in range(len(model_sectors)):
    fdi_sector = get_fdi_sector_for_model_index(idx)
    model_to_fdi_mapping[idx] = fdi_sector

# Save the correspondence mapping to CSV
mapping_df = pd.DataFrame({
    'model_sector_index': list(model_to_fdi_mapping.keys()),
    'model_sector_name': model_sectors,
    'fdi_sector_name': [model_to_fdi_mapping[i] for i in range(len(model_sectors))]
})
mapping_output = 'input_data/FDI_foreign_direct_investment_data/fdi_to_model_sector_mapping.csv'
mapping_df.to_csv(mapping_output, index=False)
print(f"\nSector correspondence mapping saved to {mapping_output}")
print(f"\nMapping summary:")
for fdi_sec in fdi_sectors[:-1]:  # Exclude 'Total'
    matched = [i for i, v in model_to_fdi_mapping.items() if v == fdi_sec]
    print(f"  {fdi_sec}: model sectors {matched} ({len(matched)} sectors)")
unmatched = [i for i, v in model_to_fdi_mapping.items() if v is None]
print(f"  No FDI match: model sectors {unmatched} ({len(unmatched)} sectors)")

print(f"\nDone. FDI data and mapping are ready for loading in calibration.py")
