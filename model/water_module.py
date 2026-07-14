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
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .calibration import ParametersCalibrated
    from .model_classes import ModelConfig, ModelVariables, ModelResults, ModelParameters

# ==========================================
# 1. Configuration & Constants
# ==========================================

# Input Directory
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input_data" / "input_data_water_dan"

# Irrigation Efficiency (Constant across years/scenarios)
EFFICIENCY = {
    "Surface": 0.55,
    "Sprinkler": 0.70,
    "Drip": 0.85,
    "Rainfed": 0.0  # Rainfed uses no blue water withdrawal
}

# ==========================================
# Cost Parameters
# ==========================================
INVEST_UNIT = {"Surface": 0.00575, "Sprinkler": 0.0121875, "Drip": 0.02625}
OM_UNIT = {"Surface": 0.008625, "Sprinkler": 0.012375, "Drip": 0.02625}

SUBSIDY_RATE = {"BAU": 0.45, "2030 Vision": 0.60, "Transformation": 0.65}

# Crop Types for Greenhouse Split
VEGETABLES_TO_SPLIT = [
    "Cucumber", "Eggplant", "Melon", "Okra", "Onion", 
    "Other vegetables", "Potatoes", "Tomatoes", "Watermelon", "Zucchini"
]

# Common Input Files
FILE_PER_CAPITA = INPUT_DIR / "calibrated_crop_consumption_per_capita_basedon_2015_2021.xlsx"
FILE_YIELD = INPUT_DIR / "crop_prodution_per_m2_2021.xlsx"
FILE_BWF = INPUT_DIR / "BWF_coefficiency.xlsx"
FILE_OUTPUT_VALUE = INPUT_DIR / "crop_total_output_per_ton_2021.xlsx"

# Scenarios Definition
SCENARIOS = {
    "BAU": {
        "ssr": "SSR_of_BAU_scenario_2021_2060.xlsx",
        "gh_pct": "BAU_GH_production_percentage_2021_2060.xlsx",
        "tech_pct": "irrigation_area_pct_BAU_2021_2060.xlsx"
    },
    "2030 Vision": {
        "ssr": "SSR_of_2030_vision_scenario_2021_2060.xlsx",
        "gh_pct": "2030_vision_GH_production_percentage_2021_2060.xlsx",
        "tech_pct": "irrigation_area_pct_2030_vision_2021_2060.xlsx"
    },
    "Transformation": {
        "ssr": "SSR_of_sustainability_scenario_2021_2060.xlsx",
        "gh_pct": "sustainability_GH_production_percentage_2021_2060.xlsx",
        "tech_pct": "irrigation_area_pct_sustainability_2021_2060.xlsx"
    }
}

# ==========================================
# 2. Data Loading Helpers
# ==========================================

def generate_population(start_year, end_year, growth_rate):
    """
    Generates population data based on 2021 baseline and growth rate.
    Returns dictionary: {Year: Persons}
    """
    base_year = 2021
    base_pop = 30.78 * 1_000_000  # 30.78 Million
    
    pop_data = {}
    
    # Calculate directly: P_t = P_base * (1 + g)^(t - base)
    for year in range(start_year, end_year + 1):
        # Calculate relative to base year
        # If year < base_year, the exponent is negative, which correctly "reverse grows" the population
        pop_data[year] = base_pop * ((1 + growth_rate) ** (year - base_year))
            
    return pop_data

def load_per_capita(path):
    """Loads per capita consumption. Returns dict: {Crop: Ton/Person}."""
    df = pd.read_excel(path, sheet_name=0)
    # Assuming first two columns are [Crop, Value]
    return dict(zip(df.iloc[:, 0].str.strip(), df.iloc[:, 1]))

def load_parameter_map(path, value_col_index=1):
    """Generic loader for simple Key-Value pairs (Yield, BWF, Output Value)."""
    df = pd.read_excel(path, sheet_name=0)
    return dict(zip(df.iloc[:, 0].str.strip(), df.iloc[:, value_col_index]))

def get_value_with_fallback(year, crop, matrix_df):
    """
    Helper to get value from a matrix (Crop x Year).
    Implements Forward Fill: if year > max_year in matrix, use max_year.
    """
    if matrix_df is None or crop not in matrix_df.index:
        return 0.0
    
    available_years = [c for c in matrix_df.columns if isinstance(c, int)]
    if not available_years:
        return 0.0
        
    max_year = max(available_years)
    
    if year in matrix_df.columns:
        return matrix_df.loc[crop, year]
    elif year > max_year:
        # Forward fill for future years
        return matrix_df.loc[crop, max_year]
    else:
        return 0.0

def load_year_crop_matrix(path):
    """
    Loads a matrix where Rows=Crops, Cols=Years. 
    Returns a DataFrame: index=Crop, columns=Years (integers).
    """
    try:
        df = pd.read_excel(path, sheet_name="Data")
    except:
        df = pd.read_excel(path, sheet_name=0)
    
    df.set_index(df.columns[0], inplace=True) # First column is Crop
    
    # Identify integer year columns
    new_cols = {}
    for c in df.columns:
        if str(c).strip().isdigit():
            new_cols[c] = int(str(c).strip())
            
    df = df[list(new_cols.keys())].rename(columns=new_cols)
    return df

def load_tech_pct_matrix(path):
    """
    Loads irrigation technology percentages.
    Returns dict: {Technology: DataFrame(index=Crop, cols=Years)}.
    """
    tech_data = {}
    for tech in EFFICIENCY.keys():
        try:
            df = pd.read_excel(path, sheet_name=tech, index_col=0)
            # Identify integer year columns
            new_cols = {}
            for c in df.columns:
                if str(c).strip().isdigit():
                    new_cols[c] = int(str(c).strip())
            
            df = df[list(new_cols.keys())].rename(columns=new_cols)
            tech_data[tech] = df
        except Exception as e:
            tech_data[tech] = pd.DataFrame()
    return tech_data

# ==========================================
# 3. Calculation Logic Steps
# ==========================================

def step1_calculate_consumption(population, per_capita_map, start_year, end_year, is_sustainability=False):
    """
    Calculates Total Consumption = Population * Per Capita.
    """
    consumption_records = []
    
    for year in range(start_year, end_year + 1):
        pop = population.get(year, 0)
        
        # Calculate decay factor for Transformation scenario
        decay = 1.0
        if is_sustainability:
            if year <= 2035:
                # Linear decay to 75% from 2021 to 2035
                t = (year - 2021) / (2035 - 2021)
                decay = 1.0 - 0.25 * t
            else:
                decay = 0.75
        
        for crop, base_pc in per_capita_map.items():
            pc = base_pc * decay
            total_cons = pop * pc
            consumption_records.append({
                "Year": year,
                "Crop": crop,
                "Consumption_ton": total_cons
            })
            
    return pd.DataFrame(consumption_records)

def step2_calculate_production(df_consumption, df_ssr):
    """
    Calculates Production = Consumption * SSR.
    """
    results = []
    for _, row in df_consumption.iterrows():
        year = row["Year"]
        crop = row["Crop"]
        cons = row["Consumption_ton"]
        
        # Get SSR (Forward Fill)
        ssr = get_value_with_fallback(year, crop, df_ssr)
        
        prod = cons * ssr
        results.append({
            "Year": year,
            "Crop": crop,
            "Production_ton": prod
        })
    return pd.DataFrame(results)

def step3_split_gh_production(df_production, df_gh_pct):
    """
    Splits production of vegetables into GH and Non-GH varieties.
    """
    new_rows = []
    
    for _, row in df_production.iterrows():
        year = row["Year"]
        crop = row["Crop"]
        prod = row["Production_ton"]
        
        if crop in VEGETABLES_TO_SPLIT:
            gh_key = f"GH {crop}"
            gh_percent = get_value_with_fallback(year, gh_key, df_gh_pct)
            
            # Create two new rows
            new_rows.append({
                "Year": year, "Crop": f"GH {crop}", "Production_ton": prod * gh_percent
            })
            new_rows.append({
                "Year": year, "Crop": f"Non-GH {crop}", "Production_ton": prod * (1.0 - gh_percent)
            })
        else:
            # Keep original
            new_rows.append(row.to_dict())
            
    return pd.DataFrame(new_rows)

def step4_calculate_metrics(df_production_split, yield_map, bwf_map, output_value_map, tech_pct_data):
    """
    Calculates Water Withdrawal, Total Output (1000 SAR), and Water Use Intensity (m3/1000 SAR).
    Also calculates Irrigation Investment (SAR), O&M Costs (SAR), and Subsidies (SAR).
    """
    results = []
    
    # The subsidy rate for the scenario is passed in as the 'subsidy_rate' argument.
    
    for _, row in df_production_split.iterrows():
        year = row["Year"]
        crop = row["Crop"]
        prod = row["Production_ton"]
        
        # 1. Parameter Lookup
        base_name = crop.replace("GH ", "").replace("Non-GH ", "")
        
        crop_yield = yield_map.get(crop, yield_map.get(base_name, 0))
        bwf = bwf_map.get(crop, bwf_map.get(base_name, 0))
        output_price = output_value_map.get(crop, output_value_map.get(base_name, 0))
        
        if crop_yield <= 0: 
            # If yield is missing, we can't calculate Area/Withdrawal
            # But we can calculate Output Value
            total_output_sar = prod * output_price
            total_output_1000sar = total_output_sar / 1000.0
            
            results.append({
                "Year": year, "Crop": crop, "Technology": "Unknown",
                "Production_ton": prod,
                "Total_Output_1000SAR": total_output_1000sar,
                "Withdrawal_m3": 0,
                "Water_Use_Intensity_m3_per_1000SAR": 0
            })
            continue
        
        # 2. Calculate Total Area & Output Value
        total_area = prod / crop_yield
        total_output_sar = prod * output_price
        
        # 3. Iterate Technologies
        for tech, eff in EFFICIENCY.items():
            tech_df = tech_pct_data.get(tech)
            
            # Lookup Tech Percentage
            tech_pct = 0.0
            if tech_df is not None:
                if crop in tech_df.index:
                    tech_pct = get_value_with_fallback(year, crop, tech_df)
                elif base_name in tech_df.index:
                    tech_pct = get_value_with_fallback(year, base_name, tech_df)
            
            if tech_pct <= 0:
                continue

            # Apportion Production and Output to this Technology
            prod_tech = prod * tech_pct
            output_tech_sar = total_output_sar * tech_pct
            output_tech_1000sar = output_tech_sar / 1000.0
            
            # Calculate Withdrawal
            # Withdrawal = (Production * BWF) / Eff
            water_need = prod_tech * bwf
            withdrawal = 0.0
            if eff > 0:
                withdrawal = water_need / eff
            
                intensity = withdrawal / output_tech_1000sar
            
            # 4. Calculate Costs (SAR)
            # Area = Production / Yield
            area_m2 = 0.0
            if crop_yield > 0:
                area_m2 = prod_tech / crop_yield
            
            # Unit Costs
            unit_invest = INVEST_UNIT.get(tech, 0.0)
            unit_om = OM_UNIT.get(tech, 0.0)
            
            invest_cost_sar = area_m2 * unit_invest
            om_cost_sar = area_m2 * unit_om
            
            # Subsidy (Only for Drip)
            # Subsidy = Investment * Rate
            subsidy_sar = 0.0
            if tech == "Drip":
                # REFINED LOGIC: 
                # 2021 uses historical subsidy rate (0.45) for all scenarios.
                # 2022+ uses scenario-specific rate.
                if year == 2021:
                    subsidy_rate = 0.45
                else:
                    subsidy_rate = tech_pct_data.get("subsidy_rate", 0.0)
                    
                subsidy_sar = invest_cost_sar * subsidy_rate

            results.append({
                "Year": year,
                "Crop": crop,
                "Technology": tech,
                "Production_ton": prod_tech,
                "Total_Output_1000SAR": output_tech_1000sar,
                "Withdrawal_m3": withdrawal,
                "Water_Use_Intensity_m3_per_1000SAR": intensity,
                "Investment_Cost_SAR": invest_cost_sar,
                "OM_Cost_SAR": om_cost_sar,
                "Subsidy_SAR": subsidy_sar
            })
            
    return pd.DataFrame(results)

# ==========================================
# 4. Main Execution Controller
# ==========================================

def run_scenario(scenario_name, start_year, end_year,
                 path_ssr, path_gh_pct, path_tech_pct,
                 population_growth_rate=0.0113, save_excel=True, verbose=False):

    if verbose:
        print(f"--- Running Scenario: {scenario_name} (Years: {start_year}-{end_year}) ---")
    
    # 1. Load Common Data
    pop_map = generate_population(start_year, end_year, population_growth_rate)
    pc_map = load_per_capita(FILE_PER_CAPITA)
    yield_map = load_parameter_map(FILE_YIELD)
    bwf_map = load_parameter_map(FILE_BWF)
    output_val_map = load_parameter_map(FILE_OUTPUT_VALUE)
    
    # 2. Load Scenario-Specific Data
    ssr_matrix = load_year_crop_matrix(INPUT_DIR / path_ssr)
    gh_pct_matrix = load_year_crop_matrix(INPUT_DIR / path_gh_pct)
    tech_pct_data = load_tech_pct_matrix(INPUT_DIR / path_tech_pct)
    
    # Add Subsidy Rate to tech_pct_data for convenience
    # Handle case-insensitive scenario name lookup for SUBSIDY_RATE
    # "2030 Vision" vs "2030 vision"
    mapped_scen_name = None
    for k in SUBSIDY_RATE.keys():
        if k.lower() == scenario_name.lower():
            mapped_scen_name = k
            break
    
    if mapped_scen_name:
        tech_pct_data["subsidy_rate"] = SUBSIDY_RATE[mapped_scen_name]
    else:
        tech_pct_data["subsidy_rate"] = 0.0 # Default if not found
        print(f"Warning: No subsidy rate found for '{scenario_name}'. using 0.0")

    # 3. Execute Pipeline
    is_sust = (scenario_name == "Transformation")
    
    # Step 1: Consumption
    df_cons = step1_calculate_consumption(pop_map, pc_map, start_year, end_year, is_sust)
    
    # Step 2: Production
    df_prod = step2_calculate_production(df_cons, ssr_matrix)
    
    # Step 3: GH Split
    df_prod_split = step3_split_gh_production(df_prod, gh_pct_matrix)
    
    # Step 4: Metrics (Withdrawal, Output, Intensity)
    df_result = step4_calculate_metrics(
        df_prod_split, yield_map, bwf_map, output_val_map, tech_pct_data
    )
    
    if save_excel:
        # 4. Save to Excel
        filename = f"Results_{scenario_name.replace(' ', '_')}_{start_year}_{end_year}.xlsx"
        
        print(f"Saving results to {filename}...")
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Detailed Data
            df_result.to_excel(writer, sheet_name="Detailed_Data", index=False)
            
            # Sheet 2: Summary by Crop & Year (Aggregated across technologies)
            summary_crop = df_result.groupby(["Year", "Crop"])[
                ["Withdrawal_m3", "Total_Output_1000SAR", "Production_ton", 
                 "Investment_Cost_SAR", "OM_Cost_SAR", "Subsidy_SAR"]
            ].sum().reset_index()
            
            summary_crop["Avg_Intensity_m3_per_1000SAR"] = summary_crop.apply(
                lambda x: x["Withdrawal_m3"] / x["Total_Output_1000SAR"] if x["Total_Output_1000SAR"] > 0 else 0, 
                axis=1
            )
            summary_crop.to_excel(writer, sheet_name="Summary_By_Crop", index=False)
            
            # Sheet 3: Summary by Sector (Aggregated across all crops)
            summary_sector = df_result.groupby(["Year"])[
                ["Withdrawal_m3", "Total_Output_1000SAR", "Production_ton",
                 "Investment_Cost_SAR", "OM_Cost_SAR", "Subsidy_SAR"]
            ].sum().reset_index()
            
            summary_sector["Sector_Intensity_m3_per_1000SAR"] = summary_sector.apply(
                lambda x: x["Withdrawal_m3"] / x["Total_Output_1000SAR"] if x["Total_Output_1000SAR"] > 0 else 0,
                axis=1
            )
            summary_sector.to_excel(writer, sheet_name="Summary_Sector_Level", index=False)
        
    if verbose:
        print("Done.\n")
    return df_result

def get_agr_data(scenario_name, start_year=2021, end_year=2065, population_growth_rate=0.0113, verbose=False):
    """
    Runs the water module for the specified scenario and returns aggregated time series 
    for Total Output (agr_X) and Water Use Intensity (agr_water_use_intensity).
    
    Returns:
        crop_X (dict): {Year: Total_Output_1000SAR}
        agr_water_use_intensity (dict): {Year: Water_Use_Intensity_m3_per_1000SAR}
        agr_irrigation_invest (dict): {Year: Investment_Cost_SAR}
        agr_irrigation_om (dict): {Year: OM_Cost_SAR}
        agr_irrigation_subsidy (dict): {Year: Subsidy_SAR}
    """
    if scenario_name not in SCENARIOS:
        print(f"Warning: Scenario '{scenario_name}' not found. Defaulting to BAU.")
        scenario_name = "BAU"
        
    sc_config = SCENARIOS[scenario_name]
    
    df_result = run_scenario(
        scenario_name, start_year, end_year,
        sc_config["ssr"], sc_config["gh_pct"], sc_config["tech_pct"],
        population_growth_rate=population_growth_rate,
        save_excel=False,
        verbose=verbose
    )
    
    # Aggregate by Year
    summary_sector = df_result.groupby(["Year"])[
        ["Withdrawal_m3", "Total_Output_1000SAR", 
         "Investment_Cost_SAR", "OM_Cost_SAR", "Subsidy_SAR"]
    ].sum()
    
    
    # --- New Logic for Livestock Water Use ---
    # 1. Calculate Crops Data (Bottom-up)
    crop_X_dict = summary_sector["Total_Output_1000SAR"].to_dict() # {Year: Value}
    crop_water_dict = summary_sector["Withdrawal_m3"].to_dict()   # {Year: Value}

    # 2. Determine 2021 Livestock Intensity (Fixed)
    # Given: Total Agr Water (2021) = 10.8 billion m3
    TOTAL_AGR_WATER_2021 = 10.8 * 1e9 
    
    # Get 2021 Crop Values
    crop_water_2021 = crop_water_dict.get(2021, 0.0)
    crop_output_2021 = crop_X_dict.get(2021, 0.0)
    
    # Calculate 2021 Livestock Values
    # Livestock Output = Crop Output / 0.34 * (1 - 0.34) = Crop Output / 0.34 * 0.66
    # Or simply: Total Output = Crop / 0.34 -> Livestock = Total - Crop
    total_output_2021 = crop_output_2021 / 0.34 if crop_output_2021 > 0 else 0
    livestock_output_2021 = total_output_2021 - crop_output_2021
    
    livestock_water_2021 = TOTAL_AGR_WATER_2021 - crop_water_2021
    
    # Calculate Fixed Livestock Intensity (m3 / 1000 SAR)
    if livestock_output_2021 > 0:
        LIVESTOCK_INTENSITY = livestock_water_2021 / livestock_output_2021
    else:
        LIVESTOCK_INTENSITY = 0.0
        
    # 3. Calculate Total Agr Water & Intensity for ALL years
    agr_water_use_intensity = {}
    
    # Note: Total Agr Output = Crop + Livestock, but the returned `crop_X` is Crop Output only.
    # The return value `crop_X` is used in calibration.py to set `self.crop_X` and then derive `AGR_X`.
    # So `crop_X` MUST remain Crop Output.
    # The INTENSITY however, must be (Total Water / Total Output).
    
    for year, crop_output in crop_X_dict.items():
        crop_water = crop_water_dict.get(year, 0.0)
        
        # Calculate derived Livestock values for this year
        total_output = crop_output / 0.34 if crop_output > 0 else 0
        livestock_output = total_output - crop_output
        
        # Livestock Water = Output * Fixed Intensity
        livestock_water = livestock_output * LIVESTOCK_INTENSITY
        
        # Total Agr Water
        total_agr_water = crop_water + livestock_water
        
        # Total Agr Intensity = Total Water / Total Output
        if total_output > 0:
            agr_water_use_intensity[year] = total_agr_water / total_output
        else:
            agr_water_use_intensity[year] = 0.0

    # New Cost Outputs (Dictionaries {Year: Value})
    agr_irrigation_invest = summary_sector["Investment_Cost_SAR"].to_dict()
    agr_irrigation_om = summary_sector["OM_Cost_SAR"].to_dict()
    agr_irrigation_subsidy = summary_sector["Subsidy_SAR"].to_dict()
    
    return crop_X_dict, agr_water_use_intensity, agr_irrigation_invest, agr_irrigation_om, agr_irrigation_subsidy

if __name__ == "__main__":
    
    START = 2021
    END = 2065  # Extended to 2065
    
    # Example: Run all scenarios
    for sc_name, sc_config in SCENARIOS.items():
        run_scenario(
            sc_name, START, END,
            sc_config["ssr"], sc_config["gh_pct"], sc_config["tech_pct"]
        )

# ==========================================
# 5. Water Supply Mix & Allocation Logic
# ==========================================

def get_supply_mix_config(scenario_config: dict):
    """
    Returns the base and target allocation shares for water supply mix.
    
    The keys represent the **DEMAND DEPENDENCY SHARE** of a sector.
    They represent the share of a sector's demand met by a specific source.
    
    Example:
    - "desal_hh": 0.69 means 69% of Household DEMAND is met by Desalination.
    
    Therefore, the sum of shares for a specific SECTOR (e.g., *_hh) MUST sum to 1.0.
    """
    # Base Year (2021) Allocation Shares (Demand Structure)
    base = {
        # Household (GW 31%, Desal 69%, WW 0%)
        "groundwater_hh": 0.31, "desal_hh": 0.69, "wwater_hh": 0.00,
        
        # Services (GW 31%, Desal 69%, WW 0%)
        "groundwater_ser": 0.31, "desal_ser": 0.69, "wwater_ser": 0.00,
        
        # Agriculture (GW 97%, Desal 0%, WW 3%)
        "groundwater_agr": 0.97, "desal_agr": 0.00, "wwater_agr": 0.03,
        
        # Industry (GW 45%, Desal 51%, WW 4%)
        "groundwater_ind": 0.45, "desal_ind": 0.51, "wwater_ind": 0.04
    }
    
    # Target Shares (Default to Base if not specified)
    target = base.copy()
    target_year = 2060
    
    return base, target, target_year

def calculate_water_demand(v, p, pc, t):
    """
    Step 1 & 2: Update intensities and calculate total and sectoral water use.
    """
    import numpy as np
    
    # 1. Update Water Intensities
    v.water_use_intensities_[t] = pc.water_use_intensities_.copy()
    v.water_use_intensities_[t][pc.agr_sector] = pc.agr_water_use_intensity[t-1]
    
    # FIX: Zero out other agricultural sectors (Forestry & Fishing) to prevent double counting
    # The calibrated total water (10.8 b) is assigned to pc.agr_sector (0).
    # Summing indices 0-3 (Crop, Forestry, Fishing) would otherwise exceed the target.
    agr_indices = [0, 1, 2]
    for idx in agr_indices:
        if idx != pc.agr_sector:
            v.water_use_intensities_[t][idx] = 0.0

    # 2. Calculate Total Water Use
    # Household water use
    v.water_use_hh[t] = pc.water_use_hh_2021 * (1 + p.wage_population_growth_adjustment)**(t-1)

    # Sectoral water use
    v.water_use_sectoral_[t] = v.X_[t] * v.water_use_intensities_[t]
    v.water_use_agr[t] = np.sum(v.water_use_sectoral_[t][0:3])
    v.water_use_ind[t] = np.sum(v.water_use_sectoral_[t][3:40])
    v.water_use_ser[t] = np.sum(v.water_use_sectoral_[t][40:])
    
    # Aggregate total
    v.water_use_total_national[t] = np.sum(v.water_use_sectoral_[t]) + v.water_use_hh[t]
    
    return v.water_use_sectoral_[t]

def calculate_aggregate_supply_mix(v, pc, t):
    """
    Step 3: Calculate aggregate Desalination, Wastewater, and Groundwater totals.
    Handles capacity constraints and the 'zero groundwater' policy.
    """
    import numpy as np
    
    # Calculate potentials
    v.pot_desal[t] = v.K_[t][pc.sector_desal] * pc.eK_desal * pc.u_desal / pc.p_desal_m3
    v.pot_wwater[t] = v.K_[t][pc.sector_wwater] * pc.eK_wwater * pc.u_wwater / pc.p_wwater_m3

    # Initial Allocation (capped by total demand)
    if (v.pot_desal[t] + v.pot_wwater[t]) > v.water_use_total_national[t]:
        v.water_use_scale_factor[t] = v.water_use_total_national[t] / (v.pot_desal[t] + v.pot_wwater[t])
        v.water_use_desal_tot[t] = v.pot_desal[t] * v.water_use_scale_factor[t]
        v.water_use_wwater_tot[t] = v.pot_wwater[t] * v.water_use_scale_factor[t]
    else:
        v.water_use_desal_tot[t] = v.pot_desal[t]
        v.water_use_wwater_tot[t] = v.pot_wwater[t]

    # Groundwater as residual
    v.water_use_gw_tot[t] = np.maximum(v.water_use_total_national[t] - v.water_use_desal_tot[t] - v.water_use_wwater_tot[t], 0)

    # Constraint: Once groundwater supply becomes 0, it stays 0
    if t > 1:
        if v.water_use_gw_tot[t-1] <= 0.1:
            v.water_use_gw_tot[t] = 0
            
            # Recalculate Desal and Wastewater to meet total demand
            v.share_desal[t] = v.pot_desal[t] / (v.pot_desal[t] + v.pot_wwater[t])
            v.share_wwater[t] = v.pot_wwater[t] / (v.pot_desal[t] + v.pot_wwater[t])
                
            v.water_use_desal_tot[t] = v.water_use_total_national[t] * v.share_desal[t]
            v.water_use_wwater_tot[t] = v.water_use_total_national[t] * v.share_wwater[t]

def get_current_allocation_shares(pc, t):
    """
    Step 4: Returns current year's target demand shares.
    Since demand structure is assumed constant (Base == Target), interpolation is skipped.
    """
    base_shares, _, _ = get_supply_mix_config(None)
    return base_shares



def distribute_sectoral_allocations(v, pc, t, water_use_sectoral, allocs, target_var_array):
    """
    Helper: Distributes aggregated sector allocations (e.g., 'ind') to sub-sectors (array indices).
    """
    import numpy as np
    
    target_var_array[t] = np.zeros(pc.S)
    
    # Agriculture (0-3)
    demands_agr = v.water_use_agr[t]
    if demands_agr > 0:
        target_var_array[t][0:3] = (water_use_sectoral[0:3] / demands_agr) * allocs["agr"]
        
    # Industry (3-40)
    demands_ind = v.water_use_ind[t]
    if demands_ind > 0:
        target_var_array[t][3:40] = (water_use_sectoral[3:40] / demands_ind) * allocs["ind"]
        
    # Services (40+)
    demands_ser = v.water_use_ser[t]
    if demands_ser > 0:
        target_var_array[t][40:] = (water_use_sectoral[40:] / demands_ser) * allocs["ser"]

def ras_balance(matrix, row_totals, col_totals, max_iter=50, tol=1e-4):
    """
    Bi-proportional Matrix Balancing (RAS / Iterative Proportional Fitting).
    Adjusts the matrix such that row sums equal row_totals and col sums equal col_totals.
    Preserves the structure (zeros remain zeros) of the initial matrix.
    """
    import numpy as np
    m = matrix.copy()
    
    # Handle zero totals to avoid division by zero
    # If a row/col total is 0, the corresponding row/col in m will naturally become 0 or stay 0
    
    for _ in range(max_iter):
        # 1. Row Scaling (Match Demand)
        current_row_sums = m.sum(axis=1)
        r_factors = np.divide(row_totals, current_row_sums, out=np.ones_like(row_totals), where=current_row_sums!=0)
        m = m * r_factors[:, np.newaxis]
        
        # 2. Col Scaling (Match Supply)
        current_col_sums = m.sum(axis=0)
        c_factors = np.divide(col_totals, current_col_sums, out=np.ones_like(col_totals), where=current_col_sums!=0)
        m = m * c_factors
        
        # Check convergence (optional, simple max_diff check)
        if np.allclose(m.sum(axis=1), row_totals, atol=tol) and np.allclose(m.sum(axis=0), col_totals, atol=tol):
            break
            
    return m

def water_accounting(v: "ModelVariables", p: "ModelParameters", pc: "ParametersCalibrated", t: int, verbose: bool = False):
    """
    Unified water module to handle water use accounting and supply mix calculation.
    Transferred from model.py to maintain modularity.
    Refactored to use RAS (Bi-proportional) allocation based on Demand Structure Targets.
    """
    import numpy as np
    
    # 1. Calculate Demand
    water_use_sectoral = calculate_water_demand(v, p, pc, t)
    
    # 2. Calculate Aggregate Supply Mix (Fixes Column Totals)
    calculate_aggregate_supply_mix(v, pc, t)
    
    # 3. Prepare RAS Inputs
    # 3.1 Row Totals (Sectoral Demands)
    demands_dict = {
        "hh": v.water_use_hh[t],
        "agr": v.water_use_agr[t],
        "ind": v.water_use_ind[t],
        "ser": v.water_use_ser[t]
    }
    sectors_order = ["hh", "agr", "ind", "ser"]
    row_totals = np.array([demands_dict[k] for k in sectors_order])
    
    # 3.2 Column Totals (Source Supplies)
    # Note: GW is calculated as residual in calculate_aggregate_supply_mix, 
    # ensuring sum(Supplies) == sum(Demands)
    supplies_dict = {
        "desal": v.water_use_desal_tot[t],
        "wwater": v.water_use_wwater_tot[t],
        "gw": v.water_use_gw_tot[t]
    }
    sources_order = ["desal", "wwater", "gw"]
    col_totals = np.array([supplies_dict[k] for k in sources_order])
    
    # 3.3 Initial Matrix (Based on Target Demand Structure)
    shares = get_current_allocation_shares(pc, t)
    
    # Construct Initial Matrix [Sectors x Sources]
    # Structure:
    #       Desal   WW      GW
    # HH    ...     ...     ...
    # Agr   ...     ...     ...
    # Ind   ...     ...     ...
    # Ser   ...     ...     ...
    
    initial_matrix = np.zeros((4, 3))
    epsilon = 1.0 # Small value (1.0 m3) to allow RAS to allocate water even if target share is 0
    
    for i, sec in enumerate(sectors_order):
        # Get target shares for this sector
        # Keys are like "desal_hh", "wwater_agr" etc.
        s_desal = shares.get(f"desal_{sec}", 0.0)
        s_ww = shares.get(f"wwater_{sec}", 0.0)
        s_gw = shares.get(f"groundwater_{sec}", 0.0)
        
        # Initial guess = Demand * TargetShare (+ Epsilon)
        demand = demands_dict[sec]
        
        initial_matrix[i, 0] = demand * s_desal + epsilon
        initial_matrix[i, 1] = demand * s_ww + epsilon
        initial_matrix[i, 2] = demand * s_gw + epsilon
        
    # 4. Perform RAS Balancing
    # Adjusts the matrix to match Supply and Demand totals while preserving structure
    final_matrix = ras_balance(initial_matrix, row_totals, col_totals, max_iter=1000, tol=1e-4)
    
    # Final enforcement: Ensure Row Totals (Sector Demands) are exact
    # We prioritize Demand matching over Supply matching if there is a tiny residual error.
    current_row_sums = final_matrix.sum(axis=1)
    r_factors = np.divide(row_totals, current_row_sums, out=np.ones_like(row_totals), where=current_row_sums!=0)
    final_matrix = final_matrix * r_factors[:, np.newaxis]
    
    # Validation: Check if RAS converged on Column Totals (Supply)
    # Row totals are enforced above, so we check if this enforcement caused drift in Supply.
    if verbose:
        computed_col_sums = final_matrix.sum(axis=0)
        # Check mismatch > 1.0 m3
        if not np.allclose(computed_col_sums, col_totals, atol=1.0):
             print(f"[Warning] RAS Supply Mismatch Year {2021+(t-1)}: Target {col_totals} vs Result {computed_col_sums}")
    
    # 5. Distribute Results back to Variables
    
    # Helper to extract column for a source and map to dict
    def get_alloc_dict(source_idx):
        return {sec: final_matrix[i, source_idx] for i, sec in enumerate(sectors_order)}
        
    # 5.1 Desalination Distribution
    allocs_desal = get_alloc_dict(0) # Col 0
    v.water_use_desal_hh[t] = allocs_desal["hh"]
    distribute_sectoral_allocations(v, pc, t, water_use_sectoral, allocs_desal, v.water_use_desal_)
    
    # 5.2 Wastewater Distribution
    allocs_ww = get_alloc_dict(1) # Col 1
    v.water_use_wwater_hh[t] = allocs_ww["hh"]
    distribute_sectoral_allocations(v, pc, t, water_use_sectoral, allocs_ww, v.water_use_wwater_)
    
    # 5.3 Groundwater Distribution
    allocs_gw = get_alloc_dict(2) # Col 2
    v.water_use_gw_hh[t] = allocs_gw["hh"]
    distribute_sectoral_allocations(v, pc, t, water_use_sectoral, allocs_gw, v.water_use_gw_)

    if verbose:
        current_year = 2021 + (t - 1)
        print (f"Year {current_year}: Total Water Use = {v.water_use_total_national[t]/1e9:.2f} bln m3")
        print (f"Supply Mix: Desal={v.water_use_desal_tot[t]/1e9:.2f}, WW={v.water_use_wwater_tot[t]/1e9:.2f}, GW={v.water_use_gw_tot[t]/1e9:.2f}")


# ==========================================
# 6. Replace X of agriculture sector with the AGR_X
# ==========================================
def replace_agr_X(v: "ModelVariables", pc: "ParametersCalibrated", t: int):
    """
    Replace the original X of agriculture sector with the agr_X (from water-crop model).
    AND ensure global consistency by recalculating Intermediate Sales and balancing Imports for ALL sectors.
    """
    # 1. Update X for agriculture sector
    # Use the pre-calculated Total Agriculture Output (AGR_X) from calibration
    v.X_[t][pc.agr_sector] = pc.AGR_X[t-1]


def apply_irrigation_costs(v, pc, t):
    """
    Applies irrigation costs (Investment, O&M, Subsidy) to the model variables for the current time step.
    
    Args:
        v: ModelVariables object
        pc: ParametersCalibrated object (containing cost arrays)
        t: Current simulation time step (int)
    """
    # Retrieve costs for the current year t (mapped from start_sim_year)
    # Simulation starts at t=1 (2021). Arrays in pc are aligned such that index i corresponds to t=i if start_year matches.
    # pc.agr_irrigation_* arrays were created with size max_t, where index 0 is 2021.
    # In model.py, t starts at 1 for 2021. So we use t-1 to access the correct year.
    
    # Safety check for index bounds
    if t-1 < len(pc.agr_irrigation_invest):
        # 1. Irrigation Investment (Add to Agriculture Investment)
        # Convert from SAR to 1000 SAR (Model Unit)
        inv_irrigation = pc.agr_irrigation_invest[t-1] / 1000.0
        v.I_[t][pc.agr_sector] += inv_irrigation
        
        # Update Total Investment Demand components
        v.I_demand_[t][pc.agr_sector] += inv_irrigation
        v.I_total[t] += inv_irrigation
        v.i_total[t] = v.I_total[t] / v.deflator_gdp[t] # Re-calculate real investment

        # 2. Irrigation O&M Cost (Subtract from Agriculture Profits)
        # Convert from SAR to 1000 SAR
        om_irrigation = pc.agr_irrigation_om[t-1] / 1000.0
        v.P_[t][pc.agr_sector] -= om_irrigation
        
        # 3. Irrigation Subsidy (Add to Agriculture Subsidies)
        # Convert from SAR to 1000 SAR
        sub_irrigation = pc.agr_irrigation_subsidy[t-1] / 1000.0
        v.Sub_production_[t][pc.agr_sector] += sub_irrigation
        
        # Re-calculate Profits after Subsidy and O&M adjustments
        # Note: v.P_ was already calculated before this function check in model.py.
        # We need to explicitly update it to reflect the changes in Sub_production and the direct O&M cost deduction.
        # Logic: P_new = P_old - O&M + Subsidy_new_part
        # Subsidy is part of Revenue for the firm.
        v.P_[t][pc.agr_sector] += sub_irrigation 
