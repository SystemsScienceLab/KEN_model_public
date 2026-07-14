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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import textwrap
from IPython.display import display as IPdisplay

from .model_classes import ModelConfig, ModelResults, ModelParameters
from .calibration import ParametersCalibrated
from .utils import string_to_value

# Note: endyear must match the value set in the run configuration.
startyear = 1
endyear = 41

def get_srange(config: ModelConfig):
    return [str(i) for i in range(1, len(config.pc.sectors)+1)]


def plot_primary_energy_mix(results: 'ModelResults'):
    """
    Plots the total primary energy mix of the economy (Gas, Oil, Renewables) in Exajoules.
    """
    # --- 1. Setup ---
    pc = results.config.pc
    macro = results.macro
    years = list(range(2021, 2021 + len(macro)))
    GJ_TO_EJ = 1e-9

    # --- 2. Get Data ---
    gas_primary_ej = macro["gas_use_total"] * GJ_TO_EJ
    oil_primary_ej = macro["oil_use_total"] * GJ_TO_EJ
    re_primary_ej = macro["renewable_energy_generation"] * GJ_TO_EJ

    # --- 3. Create Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    stack_labels = ["Gas", "Oil", "Renewables"]
    stack_colors = ['#FF9900', '#333333', '#009E73'] # Orange, Black, Green
    
    ax.stackplot(years, gas_primary_ej, oil_primary_ej, re_primary_ej, labels=stack_labels, colors=stack_colors, alpha=0.8)
    
    ax.set_title("1. Total Primary Energy Mix", fontsize=16)
    ax.set_xlabel("Year")
    ax.set_ylabel("Energy Consumption (Exajoules - EJ)")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_xlim(years[0], years[-1])
    
    plt.tight_layout()
    plt.show()



def plot_electricity_generation_mix(results: 'ModelResults'):
    """
    Plots the generation mix (TWh) and annual RE investment (Billion SAR).
    Includes 5-year renewable share annotations (e.g., 0.7%) and
    investment bubbles.
    """
    # --- 1. Setup ---
    pc = results.config.pc
    macro = results.macro
    sectoral = results.sectoral
    years = list(range(2021, 2021 + len(macro)))
    GJ_TO_TWH = (1 / 3600) / 1000  # GJ -> GWh -> TWh

    # --- 2. Get Data ---
    elec_idx = pc.electricity_sector_idx
    
    # Get the historical/forced generation values for Oil and Gas
    gas_for_elec_gj = sectoral["gas_use_final"][elec_idx]
    oil_for_elec_gj = sectoral["oil_use_final"][elec_idx]
    
    # Get the actual renewable generation (this is the one growing at 10%)
    re_for_elec_gj = macro["renewable_energy_generation"]
    
    # Get RE investment data and convert from Thousand SAR to Billion SAR
    re_investment_bsar = macro["I_RE"] / 1_000_000

    # --- 3. Create Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    stack_labels = ["Gas", "Oil", "Renewables"]
    stack_colors = ['#FF9900', '#333333', '#009E73'] # Orange, Black, Green
    
    # --- Plot 1: Stackplot for Generation (Primary Y-axis) ---
    artists = ax.stackplot(years,
                             gas_for_elec_gj * GJ_TO_TWH,
                             oil_for_elec_gj * GJ_TO_TWH,
                             re_for_elec_gj * GJ_TO_TWH,
                             labels=stack_labels,
                             colors=stack_colors,
                             alpha=0.8)
    
    ax.set_title("2. Electricity Generation Mix & RE Investment", fontsize=16)
    ax.set_xlabel("Year")
    ax.set_ylabel("Electricity Generation (Terawatt-hours - TWh)")
    ax.grid(True, linestyle=':', linewidth=0.5, which='both')
    ax.set_xlim(years[0], years[-1])

    # --- Plot 2: Bubbles for Investment (Secondary Y-axis) ---
    ax2 = ax.twinx() # Create a second y-axis
    
    # The line plot from ax2.plot() has been REMOVED as requested.
    
    ax2.set_ylabel("Annual RE Investment (Billion SAR)", color='red')
    ax2.tick_params(axis='y', labelcolor='red') # Make tick labels red
    ax2.set_ylim(0, re_investment_bsar.max() * 1.15) # Give it some space

    # --- 4. Combine Legends from both axes ---
    
    # Create a proxy artist (an invisible scatter plot point)
    # just for the legend, so "RE Investment" is included.
    bubble_legend_proxy = ax2.scatter([], [],
                                      color='red',
                                      s=100,
                                      edgecolors='black',
                                      alpha=0.7,
                                      label="RE Investment (Billion SAR)")
    
    ax.legend(artists + [bubble_legend_proxy],
              stack_labels + [bubble_legend_proxy.get_label()],
              loc='upper left')

    # --- 5. Add Annotations and Plot Bubbles ---
    
    # Add 10% growth rate text
    ax.text(0.97, 0.97,
            "",
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=10,
            style='italic',
            color='gray')

    print("\n--- 5-Year Renewable Energy Investment & Share ---")
    
    # Loop through years to add annotations and print
    for i, year in enumerate(years):
        if year >= 2025 and year % 5 == 0:
            
            # --- A. Calculate Share for Annotation ---
            gas_gj = gas_for_elec_gj.iloc[i]
            oil_gj = oil_for_elec_gj.iloc[i]
            re_gj = re_for_elec_gj.iloc[i]
            total_gen_gj = gas_gj + oil_gj + re_gj

            re_share_pct = 0.0
            if total_gen_gj > 0:
                re_share_pct = (re_gj / total_gen_gj) * 100
                
                # Get y position (top of the stack)
                y_pos_twh = total_gen_gj * GJ_TO_TWH
                
                # *** UPDATED FORMATTING HERE ***
                # Add text to the plot, 2% above the stack
                ax.text(year, y_pos_twh * 1.02,
                        f"{re_share_pct:.1f}%", # Changed from .0f to .1f
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        color='black') # Ensure text is visible

            # --- B. Get Investment Value ---
            investment_bsar_val = re_investment_bsar.iloc[i]
            
            # --- C. Plot Investment Bubble (NEW) ---
            # Plot the actual bubble on the secondary axis
            ax2.scatter(year,
                        investment_bsar_val,
                        color='red',
                        s=100,  # size of bubble
                        edgecolors='black',
                        alpha=0.7,
                        zorder=10) # Make sure it's on top
            
            # --- D. Print to Console ---
            print(f"Year {year}: Share = {re_share_pct:.1f}%, Annual Investment = {investment_bsar_val:.2f} Billion SAR")
    
    plt.tight_layout()
    plt.show()



def plot_energy_investments(results: 'ModelResults'):
    """
    Plots the energy-related investments in Billion SAR.
    - I_RE: Priority investment for new renewable capacity.
    - I_[elec]: General investment in the electricity sector.
    - Total: The sum of both.
    """
    # --- 1. Setup ---
    pc = results.config.pc
    macro = results.macro
    sectoral = results.sectoral
    years = list(range(2021, 2021 + len(macro)))
    THOUSAND_TO_BILLION = 1e-6     # Model investment is in Thousand SAR

    # --- 2. Get Data ---
    elec_idx = pc.electricity_sector_idx
    
    # Investment for new RE capacity (Solar, Wind, etc.)
    re_investment_bil = macro["I_RE"] * THOUSAND_TO_BILLION
    
    # General investment for the "Electricity, Gas, Steam" sector (grid, etc.)
    elec_sector_investment_bil = sectoral["I"][elec_idx] * THOUSAND_TO_BILLION
    
    # Calculate the total investment
    total_elec_investment_bil = re_investment_bil + elec_sector_investment_bil

    # --- 3. Create Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot the total investment first
    ax.plot(years, total_elec_investment_bil, label="Total Electricity-Related Investment", color='purple', marker='x', linewidth=3, linestyle='-')
    
    # Plot the components
    ax.plot(years, re_investment_bil, label="New Renewable Capacity (I_RE)", color='#009E73', marker='o', linewidth=2, linestyle='--')
    ax.plot(years, elec_sector_investment_bil, label="General Electricity Sector (I_[elec])", color='#0072B2', marker='s', linewidth=2, linestyle='--')
    
    ax.set_title("3. Energy-Related Investments (Trend Scenario)", fontsize=16)
    ax.set_xlabel("Year")
    ax.set_ylabel("Investment (Billion SAR)")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
    
    plt.tight_layout()
    plt.show()

def create_water_graphs(results: ModelResults):
    """Create various plots based on model results regarding water use."""

    pc = results.config.pc
    x_desal = results.sectoral["X"][pc.sector_desal]
    x_water = results.sectoral["X"][pc.sector_water]
    x_desal_real = results.sectoral["x"][pc.sector_desal]
    x_water_real = results.sectoral["x"][pc.sector_water]
    water_use_desal_biophysical = results.macro["water_use_desal_tot"]
    df = results.macro.copy()
    df_ = results.sectoral.copy()
    pc = results.config.pc
    # Set years for x-axis labels
    years = list(range(2021, 2021 + endyear))  # 2021 to 2060 inclusive

    water_user_emp = results.macro["water_use"] + 1.825 * 1e9
    # Or with small demographic projections


    # Plot physical water uses and desalination
    _, axs = plt.subplots(1, 3, figsize=(12, 4))
    ax1, ax2, ax3 = axs
    ax1.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    ax1.set_xticklabels(years[::5], rotation=45)  # label them as years
    ax1.set_title("Water use - physical quantities")
    ax1.plot(water_user_emp, label="Water use total (Output * intensities), m3")
    ax1.plot(water_use_desal_biophysical, label="Desalinated water, m3")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Cubic meters")
    ax1.legend()

    ax2.set_title("Share desalinated water biophysical")
    ax2.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    ax2.set_xticklabels(years[::5], rotation=45)  # label them as years
    ax2.plot(water_use_desal_biophysical / water_user_emp * 100, label="Desalinated water use share (%)")
    ax2.set_ylabel("Percentage %")
    ax2.set_xlabel("Time")
    ax2.legend()

    ax3.set_title("Desalination: Investment in comparison")
    ax3.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    ax3.set_xticklabels(years[::5], rotation=45)  # label them as years
    ax3.plot(results.sectoral["I"][pc.sector_desal],
             label="Investment desalination ")
    ax3.plot(results.macro["I_public"],
             label="Government investment")
    ax3.set_ylabel("Thousand SAR")
    ax3.set_xlabel("Time")
    ax3.legend()

    prices = results.sectoral["p"].copy()

    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import textwrap

def create_energy_graphs(results: 'ModelResults'):
    """Create plots to visualize energy consumption results in Exajoules (EJ)."""

    pc = results.config.pc
    years = list(range(2021, 2021 + len(results.macro)))

    # --- Conversion factor from GJ to EJ (1 EJ = 10^9 GJ) ---
    GJ_TO_EJ = 1e9

    # --- Create figure with two subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Energy Consumption Analysis', fontsize=16)

    # --- Subplot 1: Total Energy Use Over Time (in EJ) ---
    ax1.set_title("Total Energy Consumption by Type")
    # Convert data from GJ to EJ for plotting
    ax1.plot(years, results.macro["gas_use_total"] / GJ_TO_EJ, label="Total Gas Use (EJ)", color='orange', marker='o', linestyle='-')
    ax1.plot(years, results.macro["electricity_use_total"] / GJ_TO_EJ, label="Total Electricity Use (EJ)", color='blue', marker='s', linestyle='-')
    ax1.plot(years, results.macro["oil_use_total"] / GJ_TO_EJ, label="Total Oil Use (EJ)", color='black', marker='^', linestyle='-')
    
    ax1.set_xlabel("Year")
    # Update Y-axis label to EJ
    ax1.set_ylabel("Energy Consumption (EJ)")
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Subplot 2: Sectoral Energy Use in Final Year (in EJ) ---
    final_year_index = len(results.macro) - 1
    
    # Convert sectoral data from GJ to EJ first
    gas_use_ej = results.sectoral["gas_use"] / GJ_TO_EJ
    electricity_use_ej = results.sectoral["electricity_use"] / GJ_TO_EJ
    oil_use_ej = results.sectoral["oil_use"] / GJ_TO_EJ

    # Combine all energy types (now in EJ) for ranking sectors
    total_energy_use_by_sector = (
        gas_use_ej.iloc[final_year_index] +
        electricity_use_ej.iloc[final_year_index] +
        oil_use_ej.iloc[final_year_index]
    )
    
    # Get top 10 energy consuming sectors
    top_10_sectors = total_energy_use_by_sector.nlargest(10)
    top_10_indices = top_10_sectors.index
    
    # Get data (in EJ) for these top sectors
    gas_use_top10 = gas_use_ej[top_10_indices].iloc[final_year_index]
    electricity_use_top10 = electricity_use_ej[top_10_indices].iloc[final_year_index]
    oil_use_top10 = oil_use_ej[top_10_indices].iloc[final_year_index]
    
    sector_labels = [textwrap.fill(pc.sectors[i], 15) for i in top_10_indices] # Wrap long labels
    
    # Create stacked bar chart
    ax2.set_title(f"Top 10 Energy Consuming Sectors (Year {years[-1]})")
    ax2.bar(sector_labels, gas_use_top10, label='Gas Use', color='orange')
    ax2.bar(sector_labels, electricity_use_top10, bottom=gas_use_top10, label='Electricity Use', color='blue')
    ax2.bar(sector_labels, oil_use_top10, bottom=gas_use_top10 + electricity_use_top10, label='Oil Use', color='black')
    
    ax2.set_xlabel("Sector")
    # Update Y-axis label to EJ
    ax2.set_ylabel("Energy Consumption (EJ)")
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()

# ... (other imports) ...

# In postprocessing.py, add this new function at the end of the file

# In postprocessing.py, add this new function at the end of the file

# In postprocessing.py, replace the entire function with this one.

# In postprocessing.py, replace the entire function with this one.

def create_renewable_energy_graphs(results: ModelResults):
    """
    Creates plots to visualize the primary energy mix, renewable adoption, and impact on emissions.
    """
    
    df_macro = results.macro.copy()
    pc = results.config.pc
    p = results.config.p
    years = list(range(2021, 2021 + len(df_macro)))
    
    # --- Conversion factor from GJ to Exajoules (EJ) for better readability ---
    GJ_TO_EJ = 1e-9

    # --- Create Figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
    # Set main title based on the scenario run
    scenario_title = "Vision 2030 Scenario" if p.vision_2030_renewable_target else "Trend Scenario"
    fig.suptitle(f"Energy and Emissions Analysis: {scenario_title}", fontsize=16)

    # --- Plot 1: Total Primary Energy Consumption by Source (Corrected) ---
    ax1.set_title("Primary Energy Mix by Source")
    
    # Convert final primary energy data to EJ for plotting
    gas_final_ej = df_macro["gas_use_final"] * GJ_TO_EJ
    oil_final_ej = df_macro["oil_use_final"] * GJ_TO_EJ
    renewable_ej = df_macro["renewable_energy_generation"] * GJ_TO_EJ

    # Create a stacked plot showing only the primary energy sources
    ax1.stackplot(years, oil_final_ej, gas_final_ej, renewable_ej,
                  labels=['Oil (Primary Use)', 'Gas (Primary Use)', 'Renewables'],
                  colors=['#2d2d2d', '#ff7f0e', '#2ca02c'])
    
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Primary Energy Consumption (EJ)")
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # --- Plot 2: Share of Renewable Energy (Unchanged) ---
    ax2.set_title("Share of Renewables in Total Energy Demand")
    ax2.plot(years, df_macro["renewable_energy_share"] * 100,
             label="Renewable Energy Share (%)", color='green', marker='.')
    
    if p.vision_2030_renewable_target:
        ax2.axhline(y=50, color='r', linestyle='--', label='50% Vision 2030 Target')
        ax2.axvline(x=2030, color='grey', linestyle=':', label='2030 Target Year')

    ax2.set_xlabel("Year")
    ax2.set_ylabel("Share (%)")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, linestyle='--', linewidth=0.5)

    # --- Plot 3: Total Greenhouse Gas Emissions (Unchanged) ---
    ax3.set_title("Total GHG Emissions by Source")
    ax3.stackplot(years, df_macro["oil_emissions_total"], df_macro["gas_emissions_total"], df_macro["electricity_emissions_total"],
                  labels=['Oil Emissions', 'Gas Emissions', 'Electricity Emissions'],
                  colors=['#2d2d2d', '#ff7f0e', '#1f77b4'])

    ax3.set_xlabel("Year")
    ax3.set_ylabel("Emissions (MtCO2)")
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # In postprocessing.py, add this entire new function at the end of the file.

def plot_renewable_energy_assumptions(config: ModelConfig, T: int):
    """
    Plots the underlying assumptions for renewable energy generation for both scenarios.
    This function visualizes the model's "input data" for renewables.
    """
    pc = config.pc
    p = config.p
    years = np.arange(2021, 2021 + T)
    GJ_TO_EJ = 1e-9 # Conversion factor from Gigajoules to Exajoules

    # --- 1. Calculate the Trend Scenario Pathway ---
    # This simulates the simple exponential growth based on the trend rate parameter.
    trend_generation = np.zeros(T)
    if T > 0:
        trend_generation[0] = pc.initial_RE_generation
        for t in range(1, T):
            trend_generation[t] = trend_generation[t-1] * (1 + p.gRE_trend_rate)

    # --- 2. Calculate the Vision 2030 Share Pathway ---
    # This visualizes the linear target path for the share of renewables until 2030.
    vision_share = np.zeros(T)
    vision_timing_index = p.vision_2030_timing - 1  # e.g., year 2030 is t=10, index=9
    
    if T > vision_timing_index:
        # Ramp-up phase until 2030
        for t in range(vision_timing_index + 1):
             start_share = pc.initial_RE_share
             end_share = 0.50
             vision_share[t] = start_share + (end_share - start_share) * (t / vision_timing_index)
        # After 2030, the logic follows the trend growth rate. We'll show this in the text.
        # For the plot, we'll set the rest to NaN so only the target path is drawn.
        vision_share[vision_timing_index + 1:] = np.nan


    # --- 3. Create the Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Model Input Assumptions for Renewable Energy Pathways", fontsize=16)

    # --- Subplot 1: Trend Scenario Generation ---
    ax1.plot(years, trend_generation * GJ_TO_EJ, marker='.', linestyle='--', color='teal')
    ax1.set_title("Scenario 1: 'Trend' Assumption")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Assumed RE Generation (EJ)")
    ax1.text(0.05, 0.95, f"Based on {p.gRE_trend_rate*100:.0f}% annual growth from a 2021 baseline.",
             transform=ax1.transAxes, verticalalignment='top', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))
    ax1.grid(True, linestyle=':', linewidth=0.5)

    # --- Subplot 2: Vision 2030 Share Target ---
    ax2.plot(years, vision_share * 100, marker='.', linestyle='-', color='green', linewidth=2)
    ax2.set_title("Scenario 2: 'Vision 2030' Assumption")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Target RE Share of Total Energy (%)")
    ax2.axhline(50, color='r', linestyle='--', label='50% Target')
    ax2.axvline(2030, color='grey', linestyle=':', label='Target Year (2030)')
    ax2.set_ylim(0, 60)
    ax2.legend()
    ax2.text(0.05, 0.95, "A linear path to a 50% share by 2030.\nPost-2030 growth follows the trend rate.",
             transform=ax2.transAxes, verticalalignment='top', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', fc='honeydew', alpha=0.8))
    ax2.grid(True, linestyle=':', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
def create_sectoral_energy_transformation_plot(results: ModelResults, target_year: int = 2040):
    """
    Creates a plot showing the energy mix transformation for the top energy-consuming sectors
    in a specific target year for the Vision 2030 scenario.
    """
    p = results.config.p
    pc = results.config.pc
    
    if not p.vision_2030_renewable_target:
        print("This plot is designed for the Vision 2030 scenario. Please run the model with `vision_2030_renewable_target = True`.")
        return

    # --- Data Preparation ---
    # Find the index for the target year (e.g., year 2040 is t=20)
    time_index = target_year - 2021 + 1
    if time_index >= len(results.macro):
        print(f"Target year {target_year} is beyond the simulation horizon.")
        return
    
    # Get sectoral energy data for the target year
    oil_original = results.sectoral['oil_use'].iloc[time_index]
    gas_original = results.sectoral['gas_use'].iloc[time_index]
    oil_final = results.sectoral['oil_use_final'].iloc[time_index]
    gas_final = results.sectoral['gas_use_final'].iloc[time_index]

    # Calculate total original fossil fuel demand to find top sectors
    total_ff_demand = oil_original + gas_original
    top_10_sectors_indices = total_ff_demand.nlargest(10).index

    # Calculate how much renewable energy each of the top sectors "consumed"
    oil_substituted = oil_original[top_10_sectors_indices] - oil_final[top_10_sectors_indices]
    gas_substituted = gas_original[top_10_sectors_indices] - gas_final[top_10_sectors_indices]
    re_consumed_by_sector = oil_substituted + gas_substituted

    # Get the final fossil fuel use for the plot
    oil_final_top10 = oil_final[top_10_sectors_indices]
    gas_final_top10 = gas_final[top_10_sectors_indices]
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(18, 8))
    sector_labels = [textwrap.fill(pc.sectors[i], 15) for i in top_10_sectors_indices]
    indices = np.arange(len(top_10_sectors_indices))
    bar_width = 0.6

    # Create the stacked bar chart
    ax.bar(indices, oil_final_top10, bar_width, label='Oil Use (Final)', color='#2d2d2d')
    ax.bar(indices, gas_final_top10, bar_width, bottom=oil_final_top10, label='Gas Use (Final)', color='#ff7f0e')
    ax.bar(indices, re_consumed_by_sector, bar_width, bottom=oil_final_top10 + gas_final_top10, label='Renewable Energy', color='#2ca02c')

    ax.set_title(f'Energy Mix Transformation for Top 10 Sectors in {target_year} (Vision 2030)', fontsize=16)
    ax.set_ylabel('Energy Consumption (GJ)')
    ax.set_xlabel('Sector')
    ax.set_xticks(indices)
    ax.set_xticklabels(sector_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # --- Print Share Analysis ---
    print(f"\n--- Renewable Energy Share Analysis for Top Sectors in {target_year} (Vision 2030) ---")
    for i, sector_idx in enumerate(top_10_sectors_indices):
        total_energy_sector = oil_final_top10.iloc[i] + gas_final_top10.iloc[i] + re_consumed_by_sector.iloc[i]
        if total_energy_sector > 0:
            re_share = (re_consumed_by_sector.iloc[i] / total_energy_sector) * 100
            print(f"- Sector {sector_idx} ({pc.sectors[sector_idx]}): {re_share:.1f}% renewable.")
        else:
            print(f"- Sector {sector_idx} ({pc.sectors[sector_idx]}): 0% renewable (no energy use).")
def create_emission_graphs(results: ModelResults):
    """Create plots to visualize emission results."""

    pc = results.config.pc
    years = list(range(2021, 2021 + len(results.macro)))

    # --- Create figure with two subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Emission Analysis', fontsize=16)

    # --- Subplot 1: Total Emissions Over Time ---
    ax1.set_title("Total Emissions by Source")
    ax1.plot(years, results.macro["gas_emissions_total"], label="Total Gas Emissions (MtCO2)", color='orange', marker='o', linestyle='-')
    ax1.plot(years, results.macro["electricity_emissions_total"], label="Total Electricity Emissions (MtCO2)", color='blue', marker='s', linestyle='-')
    ax1.plot(years, results.macro["oil_emissions_total"], label="Total Oil Emissions (MtCO2)", color='black', marker='^', linestyle='-')
    
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Emissions (MtCO2)")
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Subplot 2: Sectoral Emissions in Final Year ---
    # Combine all emissions for ranking sectors
    final_year_index = len(results.macro) - 1
    total_emissions_by_sector = (
        results.sectoral["gas_emissions"].iloc[final_year_index] +
        results.sectoral["electricity_emissions"].iloc[final_year_index] +
        results.sectoral["oil_emissions"].iloc[final_year_index]
    )
    
    # Get top 10 emitting sectors
    top_10_sectors = total_emissions_by_sector.nlargest(10)
    top_10_indices = top_10_sectors.index
    
    # Get data for these top sectors
    gas_emissions_top10 = results.sectoral["gas_emissions"][top_10_indices].iloc[final_year_index]
    electricity_emissions_top10 = results.sectoral["electricity_emissions"][top_10_indices].iloc[final_year_index]
    oil_emissions_top10 = results.sectoral["oil_emissions"][top_10_indices].iloc[final_year_index]
    
    sector_labels = [textwrap.fill(pc.sectors[i], 15) for i in top_10_indices] # Wrap long labels
    
    # Create stacked bar chart
    ax2.set_title(f"Top 10 Emitting Sectors (Year {years[-1]})")
    ax2.bar(sector_labels, gas_emissions_top10, label='Gas Emissions', color='orange')
    ax2.bar(sector_labels, electricity_emissions_top10, bottom=gas_emissions_top10, label='Electricity Emissions', color='blue')
    ax2.bar(sector_labels, oil_emissions_top10, bottom=gas_emissions_top10 + electricity_emissions_top10, label='Oil Emissions', color='black')
    
    ax2.set_xlabel("Sector")
    ax2.set_ylabel("Emissions (MtCO2)")
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()
# --- START OF NEW CODE BLOCK ---
# In postprocessing.py, replace the entire function with this corrected one.

def create_transition_analysis_plots(results: ModelResults):
    """
    Creates plots to visualize renewable energy investment and the net cost of transition.
    """
    df_macro = results.macro.copy()
    df_sectoral = results.sectoral
    pc = results.config.pc
    p = results.config.p
    years = list(range(2021, 2021 + len(df_macro)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    scenario_title = "Vision 2030" if p.vision_2030_renewable_target else "Trend"
    fig.suptitle(f"Investment and Cost of Energy Transition ({scenario_title} Scenario)", fontsize=16)

    # --- Plot 1: Annual Investment in Energy ---
    ax1.set_title("Annual Investment in Energy Capacity")
    # Investment in renewables
    ax1.plot(years, df_macro["I_RE"], label="Renewable Energy Investment", color='green', linewidth=2)
    # Investment in the conventional electricity sector
    elec_sector_investment = df_sectoral["I"][pc.electricity_sector_idx]
    ax1.plot(years, elec_sector_investment, label="Electricity Sector Investment (Conventional)", color='grey', linestyle='--')
    
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Investment (Thousand SAR)")
    ax1.legend()
    ax1.grid(True, linestyle=':', linewidth=0.5)

    # --- Plot 2: Net Cost of Transition for Top Sectors ---
    target_year = 2035
    time_index = target_year - 2021 + 1
    
    if time_index < len(df_macro):
        net_costs_in_year = df_sectoral['net_cost_of_transition'].iloc[time_index]
        # Find sectors with largest costs (positive or negative)
        top_impacted_indices = net_costs_in_year.abs().nlargest(10).index
        top_costs = net_costs_in_year[top_impacted_indices].sort_values(ascending=False)
        
        sector_labels = [textwrap.fill(pc.sectors[i], 15) for i in top_costs.index]
        colors = ['#28a745' if c < 0 else '#dc3545' for c in top_costs]

        ax2.bar(sector_labels, top_costs, color=colors)
        ax2.set_ylabel("Net Cost / Saving (Thousand SAR)")
        ax2.set_title(f"Net Cost of Transition for Top Sectors ({target_year})")
        
        ax2.tick_params(axis='x', rotation=45)
        
        ax2.grid(axis='y', linestyle=':', linewidth=0.5)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.text(0.02, 0.02, "Negative bars = Cost Savings\nPositive bars = Additional Cost",
                 transform=ax2.transAxes, fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
# In postprocessing.py, add this entire new function at the end of the file.

# In postprocessing.py, replace the entire function with this corrected one.

def create_net_zero_analysis_plots(results: ModelResults):
    """
    Creates a 2x2 dashboard to analyze the Net Zero 2060 scenario,
    covering emissions, investment, energy mix, and sectoral costs.
    """
    p = results.config.p
    if not p.net_emission_reduction_green_investments_exports_transformation:
        print("This plot is designed for the Net Zero 2060 scenario. Please run with `net_emission_reduction_green_investments_exports_transformation = True`.")
        return

    df_macro = results.macro.copy()
    df_sectoral = results.sectoral
    pc = results.config.pc
    years = list(range(2021, 2021 + len(df_macro)))
    GJ_TO_EJ = 1e-9

    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Analysis of the Saudi Net Zero 2060 Scenario", fontsize=20)

    # --- Plot 1: Emission Reduction Pathway ---
    ax1 = axs[0, 0]
    ax1.set_title("GHG Emission Reduction Pathway")
    ax1.stackplot(years, df_macro["oil_emissions_total"], df_macro["gas_emissions_total"], df_macro["electricity_emissions_total"],
                  labels=['Oil Emissions', 'Gas Emissions', 'Electricity Emissions'],
                  colors=['#2d2d2d', '#ff7f0e', '#1f77b4'])
    
    start_emissions = df_macro["total_emissions"].iloc[0]
    target_path = np.linspace(start_emissions, 0, num=40)
    ax1.plot(years[:40], target_path, color='red', linestyle='--', linewidth=2, label='Linear Target Path to Zero')

    ax1.set_ylabel("Emissions (MtCO2)")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', linewidth=0.5)

    # --- Plot 2: Required Investment ---
    ax2 = axs[0, 1]
    ax2.set_title("Annual Investment Profile")
    ax2.bar(years, df_macro["I_RE"], color='green', label='Renewable Energy Investment')
    total_other_investment = df_macro["I_total"] - df_macro["I_RE"]
    ax2.bar(years, total_other_investment, bottom=df_macro["I_RE"], color='grey', label='All Other Investment')
    
    ax2.set_ylabel("Investment (Thousand SAR)")
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', linewidth=0.5)

    # --- Plot 3: Energy Mix Profile ---
    ax3 = axs[1, 0]
    ax3.set_title("Primary Energy Mix Transformation")
    
    gas_final_ej = df_macro["gas_use_final"] * GJ_TO_EJ
    oil_final_ej = df_macro["oil_use_final"] * GJ_TO_EJ
    renewable_ej = df_macro["renewable_energy_generation"] * GJ_TO_EJ
    
    ax3.stackplot(years, oil_final_ej, gas_final_ej, renewable_ej,
                  labels=['Oil', 'Gas', 'Renewables'],
                  colors=['#2d2d2d', '#ff7f0e', '#2ca02c'])

    ax3.set_xlabel("Year")
    ax3.set_ylabel("Primary Energy Consumption (EJ)")
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', linewidth=0.5)

    # --- Plot 4: Sectoral Cost of Transition ---
    ax4 = axs[1, 1]
    target_year = 2045
    time_index = target_year - 2021
    
    if time_index < len(df_macro):
        net_costs_in_year = df_sectoral['net_cost_of_transition'].iloc[time_index]
        top_impacted_indices = net_costs_in_year.abs().nlargest(10).index
        top_costs = net_costs_in_year[top_impacted_indices].sort_values(ascending=True)
        
        sector_labels = [textwrap.fill(pc.sectors[i], 20) for i in top_costs.index]
        colors = ['#28a745' if c < 0 else '#dc3545' for c in top_costs]
        
        ax4.barh(sector_labels, top_costs, color=colors)
        ax4.set_title(f"Net Cost/Saving of Transition for Top Sectors ({target_year})")
        ax4.set_xlabel("Net Annual Cost / Saving (Thousand SAR)")
        ax4.axvline(0, color='black', linewidth=0.8)
        ax4.grid(axis='x', linestyle=':', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
def create_multi_year_energy_plot(results: ModelResults):
    """Creates a 2x2 subplot showing top 10 energy consuming sectors for specified years."""
    
    pc = results.config.pc
    target_years = [2025, 2035, 2045, 2055]
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Top 10 Energy Consuming Sectors Over Time', fontsize=16, y=0.98)
    
    # Flatten the 2x2 axes array for easy iteration
    axs = axs.flatten()

    for i, year in enumerate(target_years):
        ax = axs[i]
        
        # Find the row index corresponding to the target year
        # The simulation runs from t=1 to 41, corresponding to years 2021 to 2061.
        # So, year 2025 corresponds to index t=5 (since index starts at 1)
        time_index = year - 2020

        # Check if the calculated index is within the bounds of the results
        if time_index > len(results.macro) or time_index < 1:
            ax.set_title(f"Year {year} - Data not available")
            ax.text(0.5, 0.5, 'Simulation data not available for this year.',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue
            
        # iloc uses 0-based indexing, so we subtract 1 from our 1-based time_index
        row_iloc = time_index - 1

        # Combine all energy types for ranking sectors for the specific year
        total_energy_use_by_sector = (
            results.sectoral["gas_use"].iloc[row_iloc] +
            results.sectoral["electricity_use"].iloc[row_iloc] +
            results.sectoral["oil_use"].iloc[row_iloc]
        )
        
        # Get top 10 energy consuming sectors for that year
        top_10_sectors = total_energy_use_by_sector.nlargest(10)
        top_10_indices = top_10_sectors.index
        
        # Get data for these top sectors for that year
        gas_use_top10 = results.sectoral["gas_use"][top_10_indices].iloc[row_iloc]
        electricity_use_top10 = results.sectoral["electricity_use"][top_10_indices].iloc[row_iloc]
        oil_use_top10 = results.sectoral["oil_use"][top_10_indices].iloc[row_iloc]
        
        sector_labels = [textwrap.fill(pc.sectors[s_idx], 15) for s_idx in top_10_indices]
        
        # Create stacked bar chart
        ax.set_title(f"Top 10 Energy Consuming Sectors (Year {year})")
        ax.bar(sector_labels, gas_use_top10, label='Gas Use', color='orange')
        ax.bar(sector_labels, electricity_use_top10, bottom=gas_use_top10, label='Electricity Use', color='blue')
        ax.bar(sector_labels, oil_use_top10, bottom=gas_use_top10 + electricity_use_top10, label='Oil Use', color='black')
        
        ax.set_ylabel("Energy Consumption (GJ)")
        ax.tick_params(axis='x', rotation=60, labelsize=9)
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def create_multi_year_emission_plot(results: ModelResults):
    """Creates a 2x2 subplot showing top 10 emitting sectors for specified years."""
    
    pc = results.config.pc
    target_years = [2025, 2035, 2045, 2055]
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Top 10 Emitting Sectors Over Time', fontsize=16, y=0.98)
    
    # Flatten the 2x2 axes array for easy iteration
    axs = axs.flatten()

    for i, year in enumerate(target_years):
        ax = axs[i]
        time_index = year - 2020

        if time_index > len(results.macro) or time_index < 1:
            ax.set_title(f"Year {year} - Data not available")
            ax.text(0.5, 0.5, 'Simulation data not available for this year.',
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue
        
        row_iloc = time_index - 1

        # Combine all emissions for ranking sectors for the specific year
        total_emissions_by_sector = (
            results.sectoral["gas_emissions"].iloc[row_iloc] +
            results.sectoral["electricity_emissions"].iloc[row_iloc] +
            results.sectoral["oil_emissions"].iloc[row_iloc]
        )
        
        # Get top 10 emitting sectors for that year
        top_10_sectors = total_emissions_by_sector.nlargest(10)
        top_10_indices = top_10_sectors.index
        
        # Get data for these top sectors for that year
        gas_emissions_top10 = results.sectoral["gas_emissions"][top_10_indices].iloc[row_iloc]
        electricity_emissions_top10 = results.sectoral["electricity_emissions"][top_10_indices].iloc[row_iloc]
        oil_emissions_top10 = results.sectoral["oil_emissions"][top_10_indices].iloc[row_iloc]
        
        sector_labels = [textwrap.fill(pc.sectors[s_idx], 15) for s_idx in top_10_indices]
        
        # Create stacked bar chart
        ax.set_title(f"Top 10 Emitting Sectors (Year {year})")
        ax.bar(sector_labels, gas_emissions_top10, label='Gas Emissions', color='orange')
        ax.bar(sector_labels, electricity_emissions_top10, bottom=gas_emissions_top10, label='Electricity Emissions', color='blue')
        ax.bar(sector_labels, oil_emissions_top10, bottom=gas_emissions_top10 + electricity_emissions_top10, label='Oil Emissions', color='black')
        
        ax.set_ylabel("Emissions (MtCO2)")
        ax.tick_params(axis='x', rotation=60, labelsize=9)
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- END OF NEW CODE BLOCK ---
def create_combined_water_graphs(results: ModelResults):
    """Create three consolidated subplots for water use, source shares, and investments."""
    pc = results.config.pc

     
    pc = results.config.pc
    x_wwater = results.sectoral["X"][pc.sector_wwater]
    x_wwater = results.sectoral["X"][pc.sector_wwater]
    x_wwater_real = results.sectoral["x"][pc.sector_wwater]
    x_wwater_real = results.sectoral["x"][pc.sector_wwater]
    water_use_wwater_biophysical = results.macro["water_use_wwater_tot"]
    #water_use_wwater_biophysical = results.macro["water_use_wwater_tot"]

    # --- Define all required variables ---
    water_use_desal = results.macro["water_use_desal_tot"]

    # Ensure wastewater total is defined
    if "water_use_wwater_tot" not in results.macro:
        results.macro["water_use_wwater_tot"] = results.sectoral["X"][pc.sector_wwater]
    
    water_use_wwater = results.macro["water_use_wwater_tot"]
    x_wwater = results.sectoral["X"][pc.sector_wwater]
    water_user_emp = results.macro["water_use"] + 1.825 * 1e9  # Including household use

    desal_share = water_use_desal / water_user_emp * 100
    wwater_share = water_use_wwater / water_user_emp * 100

    # --- Create figure ---
    _, axs = plt.subplots(1, 3, figsize=(16, 5))
    ax1, ax2, ax3 = axs

    # Subplot 1: Physical quantities
    ax1.set_title("Water use – physical quantities")
    ax1.plot(water_user_emp, label="Total water use (m³)")
    ax1.plot(water_use_desal, label="Desalinated water (m³)")
    ax1.plot(x_wwater, label="Treated wastewater output (m³)")
    ax1.plot(water_use_wwater, label="Wastewater use (m³)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Cubic meters")
    ax1.legend()

    # Subplot 2: Shares
    ax2.set_title("Share of water sources")
    ax2.plot(desal_share, label="Desalinated water share (%)")
    ax2.plot(wwater_share, label="Wastewater reuse share (%)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Percentage (%)")
    ax2.legend()

    # Subplot 3: Investment
    ax3.set_title("Water infrastructure investments")
    ax3.plot(results.sectoral["I"][pc.sector_desal], label="Desalination investment")
    ax3.plot(results.sectoral["I"][pc.sector_wwater], label="Wastewater investment")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Thousand SAR")
    ax3.legend()

    plt.tight_layout()
    plt.show()




def create_macro_graphs(results: ModelResults, displayPlots: bool = True):
    """Create various plots based on model results."""

    df = results.macro.copy()
    df_ = results.sectoral.copy()
    pc = results.config.pc
    # Set years for x-axis labels
    years = list(range(2021, 2021 + endyear))  # 2021 to 2060 inclusive


    # Plot Main macro variables in one big plot
    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    (ax1) = axs
    ax1.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    ax1.set_xticklabels(years[::5], rotation=0)  # label them as years
    df.plot(y=['Y','C','I_total','Gov_exp','P','EX', 'IM', 'YD_wage'],
            title='Key macro variables', xlabel='T', ylabel='Thousand SAR', ax=ax1)
    ax1.legend(labels=[ "Final output GDP Y",  "Consumption C", "Investment total",\
         "Government Spending Gov Exp","Profits P", "Exports EX", "Imports IM", "Disposable Wage income YD_wage"])

    ###########################################################################
    # TIMESERIES PLOT: plot the main time series that we feed into the model
    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    (ax1) = axs
    # ax1.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    # ax1.set_xticklabels(years[::5], rotation=0)  # label them as years
    ax1.set_title("Key times series for KSA economy fed into model")
    ax1.plot(pc.ts_national['Gross Domestic Product (4)'], label="Output Y GDP")
    ax1.plot(pc.ts_national['Gross Fixed Capital Formation (4)'], label="Investments")
    ax1.plot(pc.ts_government['Total Expenditures (5)'], label="Government total expenditures")
    ax1.plot(pc.ts_national['Exports of Goods & Services (4)'], label="Exports")
    ax1.plot(pc.ts_national['Oil Sectors (4)'], label="Oil sectors")
    #ax1.plot(pc.ts_national['Oil Activities (4)'], label="Oil ACTIVITIES used to calibrate oil export growth rates")
    ax1.plot(pc.ts_national['Non-Oil Sector (4)'], label="NON-Oil sectors")
    ax1.plot(pc.ts_national['Imports of Goods & Services (4)'], label="Imports")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("THOUSAND SAR")
    ax1.legend()
    
    

    ###############################
    # Plot Main macro variables with some more details
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    (ax1) = axs
    ax1.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    ax1.set_xticklabels(years[::5], rotation=0)  # label them as years
    df.plot(y=['X', 'Y','Y_production','Y_distribution','Q_d' ,'unmet_demand_total', 'C', 'I_private', 'I_public','IM', 'Gov_exp', 'P', 'EX_oil', 'EX_non_oil'],
            title='Demand Details', xlabel='T', ylabel='Thousand SAR', ax=ax1, style=line_styles)
    ax1.legend(labels=["Total Output X", "Final demand Y (expenditure)","Final demand Y (production)","Final demand Y (distribution)", "Desired demand Q_d","Unmet demand total", "Consumption C", "Investment private", "Investment public","Imports",
               "Government Spending", "Profits P", "Oil Exports", " Non-oil Exports"])



    # Multiple Plots for comparison
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axs
    ax1.set_xticks(df.index[::10])               # ticks at each model step according to years (5-year intervalls)
    ax1.set_xticklabels(years[::10], rotation=0)  # label them as years
    ax2.set_xticks(df.index[::10])               # ticks at each model step according to years (5-year intervalls)
    ax2.set_xticklabels(years[::10], rotation=0)  # label them as years
    ax3.set_xticks(df.index[::10])               # ticks at each model step according to years (5-year intervalls)
    ax3.set_xticklabels(years[::10], rotation=0)  # label them as years
    ax4.set_xticks(df.index[::10])               # ticks at each model step according to years (5-year intervalls)
    ax4.set_xticklabels(years[::10], rotation=0)  # label them as years
    ax5.set_xticks(df.index[::10])               # ticks at each model step according to years (5-year intervalls)
    ax5.set_xticklabels(years[::10], rotation=0)  # label them as years
    ax6.set_xticks(df.index[::10])               # ticks at each model step according to years (5-year intervalls)
    ax6.set_xticklabels(years[::10], rotation=0)  # label them as years

    df.plot(y=['X', 'Y', 'C', 'I_private', 'I_public', 'Gov_exp', 'P', 'EX', 'IM', 'W'],
            title='Demand', xlabel='T', ylabel='Thousand SAR', ax=ax1)
    ax1.legend(labels=["Total Output X", "GDP demand Y", "Consumption", "Investment private",
               "Investment public", "Government Spending", "Profits", "Exports", "Imports", "Wages"])

    df.plot(y=['YD_wage', 'YD_profit'],
            title='Households', xlabel='T', ylabel='Thousand SAR', ax=ax2)
    ax2.legend(labels=["Income wages", "Income profits"])

    df['Net_trade'] = df['EX'] - df['IM']
    df.plot(y=['Net_trade', 'EX', 'IM', 'EX_oil', 'EX_non_oil','Y'],
            title='Trade', xlabel='T', ylabel='Thousand SAR', ax=ax3)
    ax3.legend(labels=["Net trade Exports - Imports", "Exports",
               "Imports", "Oil Exports", " Non oil Exports","GDP demand Y"])

    df['Expenditure'] = df['Gov_exp']
    df['Revenue'] = df['Gov_rev']
    df['GP'] = df['GP']
    df['TH'] = df['TH']
    df['Gov_net_wealth'] = df['Gov_net_wealth']
    df['Gov_net_wealth_plus_aramco'] = df['Gov_net_wealth_plus_aramco']

    df.plot(y=['Expenditure', 'Revenue', 'GP', 'TH'], title='Government Flows',
            xlabel='T', ylabel='Thousand SAR', ax=ax4)

    df.plot(y=['K', 'Gov_net_wealth', 'V', 'L', ], title='Stocks',
            xlabel='T', ylabel='Thousand SAR', ax=ax5)
    ax5.legend(labels=["Capital", 'Gov net wealth',
               "HH Savings", "Firm Loans",])

    df['IntP'] = df['IntP']
    df.plot(y=['X', 'Y', 'P', 'W', 'IntP'], title='Firms',
            xlabel='T', ylabel='Thousand SAR', ax=ax6)
    ax6.legend(labels=["Total output X",
               "Total Income GDP Y", "Profits", "Wage Costs", "Intermediate Inputs"])

    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    # Stocks large view
    df['Gov_deficit'] = df['Gov_rev']-df['Gov_exp']
    (ax1) = axs
    ax1.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    ax1.set_xticklabels(years[::5], rotation=0)  # label them as years
    df.plot(y=['Y','Gov_rev','Gov_exp','Gov_deficit'], title='Government stocks and flows',
            xlabel='T', ylabel='Thousand SAR', ax=ax1)
    ax1.legend(labels=['GDP','Gov revenue','Gov expenditure','Gov deficit'])
    # Taken out: 'Gov_net_wealth','Government net wealth',,'Gov_net_wealth_plus_aramco' ,'Government net wealth incl. Saudi Aramco'


    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    # Plot growth rates
    (ax1) = axs
    ax1.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    ax1.set_xticklabels(years[::5], rotation=0)  # label them as years

    df['gY'] = df['Y'].pct_change()*100
    df['gC'] = df['C'].pct_change()*100
    df['gI'] = df['I'].pct_change()*100
    df['gGov'] = df['GY'].pct_change()*100
    df['gEX'] = df['EX'].pct_change()*100
    df['gIM'] = df['IM'].pct_change()*100
    df['gW'] = df['W'].pct_change()*100
    df['gP'] = df['P'].pct_change()*100
    df.plot(y=['gY', 'gC', 'gI', 'gGov', 'gP', 'gEX', 'gIM', 'gW'],
            title='Growth rates of key macro variables', xlabel='T', ylabel='% (percent)', ax=ax1)
    ax1.legend(labels=["Final demand gowth", "Consumption growth",
               "Investment growth", "Gov Spending growth", "Profits growth", "Exports growth", "Imports growth", "Wages growth"])

    # Plot inflation
    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    (ax1) = axs
    ax1.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    ax1.set_xticklabels(years[::5], rotation=0)  # label them as years
    df['inflation'] = df['inflation']*100
    df.plot(y=['inflation'],
            title='Prices', xlabel='T', ylabel='Inflation in %', ax=ax1)
    ax1.legend(labels=["Inflation"])


    



 

    if displayPlots:
        plt.tight_layout()
        plt.show()
    return fig, axs





def create_electricity_graph_twh(results: 'ModelResults'):
    """Create a separate plot for total electricity consumption in TWh."""

    pc = results.config.pc
    years = list(range(2021, 2021 + len(results.macro)))

    # --- Conversion factor from GJ to TWh ---
    # 1 TWh = 1,000 GWh = 3,600,000 GJ (using 1 GWh = 3600 GJ)
    # If electricity_use_total is in GWh: 1 TWh = 1,000 GWh
    
    # Assuming "electricity_use_total" is in GJ, and using your model's
    # GJ_PER_GWH = 3600 conversion:
    # GJ -> GWh -> TWh
    GJ_PER_GWH = 3600
    GWH_PER_TWH = 1000
    
    # Convert GJ to TWh
    electricity_twh = (results.macro["electricity_use_total"] / GJ_PER_GWH) / GWH_PER_TWH

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Total Electricity Consumption Over Time', fontsize=16)

    # --- Plot: Total Electricity Use Over Time (in TWh) ---
    ax.plot(years, electricity_twh, label="Total Electricity Use (TWh)", color='blue', marker='s', linestyle='-')
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Electricity Consumption (TWh)")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.show()
def plot_re_generation_and_investment(results: 'ModelResults'):
    """
    Plots Renewable Energy Generation (TWh) for Solar, Wind, and Bioenergy.
    - VISUALIZES ELECTRIFICATION IMPACT: Shows the specific portion of RE built for industry.
    - Annotates Solar, Wind, and Bioenergy lines with their required Annual Investment (Billion SAR).
    - Displays the Vision 2030 Target Mix parameters.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # --- 1. Data Setup ---
    p = results.config.p
    pc = results.config.pc
    years = np.array(range(2021, 2021 + len(results.macro)))
    
    # Conversions
    GJ_TO_TWH = (1 / 3600) / 1000  # GJ -> GWh -> TWh
    TO_BILLION = 1_000_000.0       # Thousand SAR -> Billion SAR

    # --- 2. Extract Generation Data (TWh) ---
    gen_total = results.macro["renewable_energy_generation"].values
    gen_solar = results.macro["re_gen_solar"].values
    gen_wind = results.macro["re_gen_wind"].values
    gen_bio = results.macro["re_gen_bioenergy"].values
    
    # Extract Electrification Specific Load (if available)
    if 'electrification_added_load' in results.macro:
        elec_load_gj = results.macro['electrification_added_load'].values
    else:
        elec_load_gj = np.zeros_like(gen_total)
    
    twh_total = gen_total * GJ_TO_TWH
    twh_solar = gen_solar * GJ_TO_TWH
    twh_wind = gen_wind * GJ_TO_TWH
    twh_bio = gen_bio * GJ_TO_TWH
    twh_elec_load = elec_load_gj * GJ_TO_TWH

    # Calculate "Base" Generation (Total - Electrification Load)
    twh_base = np.maximum(0, twh_total - twh_elec_load)

    # --- 3. Extract Specific Investments (Billion SAR) ---
    # UPDATED: Use the new segregated variables directly
    # Check if keys exist (backward compatibility), otherwise fallback to 0
    if "I_solar" in results.macro:
        inv_solar_bn = results.macro["I_solar"].values / TO_BILLION
        inv_wind_bn = results.macro["I_wind"].values / TO_BILLION
        inv_bio_bn = results.macro["I_bio"].values / TO_BILLION
    else:
        # Fallback if model wasn't updated (estimate based on shares)
        total_re_inv = results.macro["I_RE"].values
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_solar_bn = (total_re_inv * np.where(gen_total>0, gen_solar/gen_total, 0)) / TO_BILLION
            inv_wind_bn = (total_re_inv * np.where(gen_total>0, gen_wind/gen_total, 0)) / TO_BILLION
            inv_bio_bn = (total_re_inv * np.where(gen_total>0, gen_bio/gen_total, 0)) / TO_BILLION

    # --- 4. Plotting ---
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # A. Stackplot for Context (Base vs Electrification)
    ax.fill_between(years, 0, twh_base, color='#e5f5e0', alpha=0.5, label='Base Grid Demand (Vision 2030)')
    ax.fill_between(years, twh_base, twh_total, color='#a1d99b', alpha=0.8, label='Added for Industrial Electrification')
    
    # B. Main Generation Lines
    ax.plot(years, twh_total, label="Total RE Generation", color='green', linewidth=3)
    ax.plot(years, twh_solar, label="Solar PV", color='#FFD700', linewidth=2.5, linestyle='--')
    ax.plot(years, twh_wind, label="Wind", color='#87CEEB', linewidth=2.5, linestyle='--')
    ax.plot(years, twh_bio, label="Bioenergy", color='#8B4513', linewidth=2, linestyle=':')

    # --- 5. Annotations: Investment Costs at 5-Year Steps ---
    milestones = [y for y in years if y % 5 == 0 and y >= 2025]
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)

    for year in milestones:
        if year in years:
            idx = np.where(years == year)[0][0]
            
            # 1. Solar Investment Annotation
            val_solar = inv_solar_bn[idx]
            if val_solar > 0.05:
                ax.plot(year, twh_solar[idx], marker='o', color='#FFD700', markersize=8)
                ax.annotate(f"{val_solar:.1f} B",
                            xy=(year, twh_solar[idx]),
                            xytext=(0, 15), textcoords='offset points',
                            ha='center', color='#B8860B', fontweight='bold',
                            bbox=bbox_props)

            # 2. Wind Investment Annotation
            val_wind = inv_wind_bn[idx]
            if val_wind > 0.05:
                ax.plot(year, twh_wind[idx], marker='o', color='#87CEEB', markersize=8)
                ax.annotate(f"{val_wind:.1f} B",
                            xy=(year, twh_wind[idx]),
                            xytext=(0, -20), textcoords='offset points',
                            ha='center', color='#4682B4', fontweight='bold',
                            bbox=bbox_props)
            
            # 3. Bioenergy Investment Annotation (NEW)
            val_bio = inv_bio_bn[idx]
            if val_bio > 0.05:
                ax.plot(year, twh_bio[idx], marker='o', color='#8B4513', markersize=8)
                # Offset slightly to the right/left if it overlaps, or just below
                ax.annotate(f"{val_bio:.1f} B",
                            xy=(year, twh_bio[idx]),
                            xytext=(0, -15), textcoords='offset points',
                            ha='center', color='#8B4513', fontweight='bold',
                            bbox=bbox_props)

    # --- 6. Display Target Shares ---
    tgt_solar = getattr(p, 'target_solar_share_2030', 0.75)
    tgt_wind = getattr(p, 'target_wind_share_2030', 0.20)
    tgt_bio = getattr(p, 'target_bio_share_2030', 0.05)
    
    info_text = (
        f"2030 TARGET MIX:\n"
        f"☀ Solar: {tgt_solar:.0%}\n"
        f"༄ Wind: {tgt_wind:.0%}\n"
        f"♻ Bio: {tgt_bio:.0%}"
    )
    
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

    # --- 7. Formatting ---
    ax.set_title("Renewable Energy Generation (TWh) & Required Investment", fontsize=16, fontweight='bold')
    ax.set_ylabel("Energy Generation (TWh)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Custom Legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.8), title="Components")
    
    ax.text(0.99, 0.02, "Labels: Annual Investment in Billion SAR",
            transform=ax.transAxes, ha='right', fontsize=10, style='italic', color='gray')

    plt.tight_layout()
    plt.show()


def plot_electrification_target_sectors(results: 'ModelResults'):
    """
    Visualizes the Sector Selection Logic for the Electrification Strategy.
    Ranks sectors by Fossil Fuel Intensity (Oil + Gas) and highlights the Top 15 targets.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    pc = results.config.pc
    
    # --- 1. Data Prep ---
    # Calculate Total Fossil Intensity (GJ per Thousand SAR Output)
    fossil_intensity = pc.gas_intensities_ + pc.oil_intensities_
    
    # Create a DataFrame for easier sorting
    df = pd.DataFrame({
        'Sector': pc.sectors,
        'Gas Intensity': pc.gas_intensities_,
        'Oil Intensity': pc.oil_intensities_,
        'Total Fossil Intensity': fossil_intensity
    })
    
    # Exclude the Electricity Sector itself (as per logic)
    # We filter strings that contain "Electricity"
    df = df[~df['Sector'].str.contains("Electricity, gas, steam")]
    
    # Sort by Total Intensity (Descending)
    df_sorted = df.sort_values(by='Total Fossil Intensity', ascending=True) # Ascending for horizontal bar plot
    
    # Top 25 sectors shown for legibility
    # The 'Top 15' targets will be at the bottom of this slice (highest values)
    df_plot = df_sorted.tail(25)
    
    # Identify which ones are actually in the "Target List"
    # We re-derive the indices to match the calibration logic exactly
    target_indices = list(pc.electrification_target_sectors)
    target_names = [pc.sectors[i] for i in target_indices]
    
    # Create Color Array
    colors = []
    for sector in df_plot['Sector']:
        if sector in target_names:
            colors.append('#ff7f0e') # Orange for Targets
        else:
            colors.append('#d9d9d9') # Grey for others
            
    # --- 2. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Horizontal Bar Chart (Stacked Oil vs Gas)
    # We plot Gas first, then Oil on top
    # Note: We use the sorted dataframe indices for plotting
    
    y_pos = np.arange(len(df_plot))
    
    p1 = ax.barh(y_pos, df_plot['Gas Intensity'], color='#1f77b4', alpha=0.6, label='Gas Intensity')
    p2 = ax.barh(y_pos, df_plot['Oil Intensity'], left=df_plot['Gas Intensity'], color='#2ca02c', alpha=0.6, label='Oil Intensity')
    
    # Add outlines to highlighted sectors
    for i, sector in enumerate(df_plot['Sector']):
        if sector in target_names:
            # Re-draw the bar with a bold border to highlight it
            total_val = df_plot.iloc[i]['Total Fossil Intensity']
            ax.barh(y_pos[i], total_val, color='none', edgecolor='#ff7f0e', linewidth=2)

    # --- 3. Formatting ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['Sector'], fontsize=10)
    
    ax.set_xlabel("Energy Intensity (GJ per Thousand SAR of Output)", fontsize=12, fontweight='bold')
    ax.set_title(f"Sector Selection: Top 15 Fossil-Intensive Industries\n(Targeted for Electrification)", fontsize=14, fontweight='bold')
    
    # Add Legend for the Highlights
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='#1f77b4', lw=4, alpha=0.6),
        Line2D([0], [0], color='#2ca02c', lw=4, alpha=0.6),
        Line2D([0], [0], color='white', markeredgecolor='#ff7f0e', markeredgewidth=2, marker='s', markersize=10)
    ]
    ax.legend(custom_lines, ['Gas Intensity', 'Oil Intensity', 'Targeted Sector'], loc='lower right')
    
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Annotation explaining the logic
    ax.text(0.98, 0.02,
            "Selection Logic:\nTop 15 sectors with highest combined\nOil & Gas intensity (excluding Power Gen).",
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()
def plot_electrification_impact(results: 'ModelResults'):
    """
    Visualizes the Industrial Electrification Strategy.
    
    Primary Axis (Left): Cumulative Energy Shifted (TWh) - This represents the
    amount of Oil/Gas demand that has been permanently replaced by Electricity.
    
    Secondary Axis (Right): Annual Investment (Billion SAR) required to
    install the electrical equipment for that year's new shift.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    p = results.config.p
    # Check if strategy is enabled
    if not getattr(p, 'enable_industrial_electrification', False):
        print("Electrification Strategy is OFF. Skipping plot.")
        return

    # --- 1. Data Extraction & Calculation ---
    years = np.array(range(2021, 2021 + len(results.macro)))
    
    # Get Investment Flow (Thousand SAR)
    inv_electrification = results.macro['I_electrification_total'].values
    
    # Get Cost Parameter (SAR/GJ)
    # We use getattr in case the parameter is missing (defaults to 300)
    cost_per_gj = getattr(p, 'electrification_cost_per_gj', 300.0)
    
    # Back-calculate the Physical Energy Shifted
    # Investment (Thousand SAR) = (GJ_Switched * Cost_per_GJ) / 1000
    # Therefore: GJ_Switched = (Investment * 1000) / Cost_per_GJ
    
    # This gives us the *New* capacity added each year
    annual_gj_newly_switched = (inv_electrification * 1000.0) / cost_per_gj
    
    # The *Total* energy electrified in a given year is the cumulative sum of all past switches
    # (Since a furnace switched in 2025 is still electric in 2030)
    total_gj_electrified_in_year = np.cumsum(annual_gj_newly_switched)
    
    # Conversions
    TO_TWH = (1 / 3600) / 1000  # GJ -> TWh
    TO_BILLION = 1_000_000.0    # Thousand SAR -> Billion SAR
    
    data_twh = total_gj_electrified_in_year * TO_TWH
    data_cost_bn = inv_electrification / TO_BILLION

    # --- 2. Plotting ---
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # A. Area Chart: Cumulative Energy Shifted (The "Wedge")
    color_energy = '#1f77b4' # Blue
    ax1.fill_between(years, 0, data_twh, color=color_energy, alpha=0.3, label='Fossil Fuel Displaced (TWh)')
    ax1.plot(years, data_twh, color=color_energy, linewidth=2)
    
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Cumulative Energy Shifted to Electricity (TWh)", color=color_energy, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_energy)
    ax1.set_ylim(bottom=0)
    
    # B. Bar Chart: Annual Investment (The Cost)
    ax2 = ax1.twinx()
    color_cost = '#ff7f0e' # Orange
    
    # Plot bars
    bars = ax2.bar(years, data_cost_bn, color=color_cost, alpha=0.6, width=0.6, label='Annual Retrofit Investment (Bn SAR)')
    
    ax2.set_ylabel("Annual Investment (Billion SAR)", color=color_cost, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_cost)
    ax2.set_ylim(0, max(data_cost_bn) * 1.5) # Give headroom for labels

    # --- 3. Annotations ---
    # Annotate total TWh shifted at the end
    final_year = years[-1]
    final_twh = data_twh[-1]
    if final_twh > 0:
        ax1.annotate(f"Total Shift:\n{final_twh:.1f} TWh/yr",
                     xy=(final_year, final_twh),
                     xytext=(-40, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold', color=color_energy,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_energy, alpha=0.9))

    # Annotate max investment year
    if max(data_cost_bn) > 0:
        max_inv_idx = np.argmax(data_cost_bn)
        max_inv_year = years[max_inv_idx]
        max_inv_val = data_cost_bn[max_inv_idx]
        
        ax2.annotate(f"Peak Invest:\n{max_inv_val:.1f} B",
                     xy=(max_inv_year, max_inv_val),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9, color='#cc5500', fontweight='bold')

    # --- 4. Final Layout ---
    plt.title("Industrial Electrification: Energy Shift vs. Investment Cost", fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()


def plot_electrification_emission_benefit(results: 'ModelResults'):
    """
    Visualizes the Net Emission Reduction from the Electrification Strategy.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    p = results.config.p
    pc = results.config.pc
    
    if not getattr(p, 'enable_industrial_electrification', False):
        print("Electrification Strategy is OFF. Skipping plot.")
        return

    years = np.array(range(2021, 2021 + len(results.macro)))
    
    # --- 1. Reconstruct the "Counterfactual" Baseline ---
    rate = p.electrification_annual_rate
    shift_pct_curve = np.array([1.0 - (1.0 - rate)**(t) for t in range(len(years))])
    
    target_indices = pc.electrification_target_sectors
    
    # --- CORRECTION: REMOVED TRAILING UNDERSCORES FROM KEYS ---
    actual_oil_use = results.sectoral['oil_use_final'][target_indices].sum(axis=1).values
    actual_gas_use = results.sectoral['gas_use_final'][target_indices].sum(axis=1).values
    
    # Reconstruct BASELINE
    divisor = 1.0 - shift_pct_curve
    divisor[divisor < 0.01] = 0.01
    
    baseline_oil_use = actual_oil_use / divisor
    baseline_gas_use = actual_gas_use / divisor
    
    oil_removed_gj = baseline_oil_use - actual_oil_use
    gas_removed_gj = baseline_gas_use - actual_gas_use
    
    # --- 2. Calculate Emission Flows (MtCO2) ---
    saved_oil_ems = oil_removed_gj * pc.oil_emission_factor
    saved_gas_ems = gas_removed_gj * pc.gas_emission_factor
    total_direct_savings = saved_oil_ems + saved_gas_ems
    
    # Indirect Added (Grid)
    cop = getattr(p, 'electrification_cop', 2.5)
    elec_added_gj = (oil_removed_gj + gas_removed_gj) / cop
    
    grid_ems_total = results.macro['electricity_emissions_total'].values
    grid_gen_total = results.macro['electricity_use_total'].values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        grid_intensity = np.where(grid_gen_total > 0, grid_ems_total / grid_gen_total, 0)
    
    total_indirect_added = elec_added_gj * grid_intensity
    net_reduction = total_direct_savings - total_indirect_added

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.fill_between(years, 0, total_direct_savings, color='#2ca02c', alpha=0.3, label='Direct Abatement (Avoided Fossil)')
    ax.plot(years, total_direct_savings, color='#2ca02c', linestyle='--')
    
    ax.fill_between(years, 0, -total_indirect_added, color='#d62728', alpha=0.3, label='Indirect Emissions (Grid Load)')
    ax.plot(years, -total_indirect_added, color='#d62728', linestyle=':')
    
    ax.plot(years, net_reduction, color='black', linewidth=3, marker='o', markersize=4, label='NET Emission Reduction')
    
    #txt_stats = (f"Strategy Details:\n"
    #             f"• Annual Shift Rate: {rate:.1%}\n"
    #             f"• Efficiency (COP): {cop:.1f}\n"
    #             f"• Grid Strategy: 100% RE Replacement")
    
    #ax.text(0.02, 0.95, txt_stats, transform=ax.transAxes, va='top', fontsize=11,
    #        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9))

    ax.set_title("Electrification Strategy: Emission Benefit Analysis", fontsize=16, fontweight='bold')
    ax.set_ylabel("Annual Emissions Impact (MtCO2e)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_sectoral_fuel_mix_transition(results: 'ModelResults'):
    """
    Shows the evolution of the energy mix for the sectors targeted by Electrification.
    Demonstrates the physical shift from molecules (Oil/Gas) to electrons (Electricity).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    p = results.config.p
    pc = results.config.pc
    
    if not getattr(p, 'enable_industrial_electrification', False):
        print("Electrification is OFF.")
        return

    years = np.array(range(2021, 2021 + len(results.macro)))
    target_indices = pc.electrification_target_sectors
    
    # Scale to Million GJ (or PJ) for readability
    scale = 1_000_000
    
    # --- CORRECTION: REMOVED TRAILING UNDERSCORES FROM KEYS ---
    use_oil = results.sectoral['oil_use_final'][target_indices].sum(axis=1).values / scale
    use_gas = results.sectoral['gas_use_final'][target_indices].sum(axis=1).values / scale
    use_elec = results.sectoral['electricity_use'][target_indices].sum(axis=1).values / scale
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Stackplot
    pal = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue (Gas), Orange (Oil), Green (Elec)
    labels = ["Natural Gas", "Oil / Liquid Fuels", "Electricity (Low Carbon)"]
    
    ax.stackplot(years, use_gas, use_oil, use_elec, labels=labels, colors=pal, alpha=0.85)
    
    # Annotate the Electricity wedge growth
    start_elec = use_elec[0]
    end_elec = use_elec[-1]
    growth = (end_elec - start_elec)
    
    if growth > 0:
        ax.annotate(f"Electrification Growth:\n+{growth:.1f} Million GJ",
                    xy=(years[-1], end_elec + use_oil[-1] + use_gas[-1]),
                    xytext=(-60, -40), textcoords='offset points',
                    arrowprops=dict(facecolor='white', shrink=0.05),
                    color='white', fontweight='bold', ha='center')

    ax.set_title(f"Energy Transition in Target Industrial Sectors (Top {len(target_indices)})", fontsize=16, fontweight='bold')
    ax.set_ylabel("Total Energy Consumption (Million GJ)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_xlim(years[0], years[-1])
    ax.legend(loc='upper left', title="Energy Source")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()

def plot_oil_and_gas_switching(results: 'ModelResults'):
    """
    Plots the amount of oil use reduced AND gas use increased over time,
    stacked by the key categories (Power, Industry, Agriculture) that are
    targeted by the 'enable_targeted_oil_reduction' policy.
    """
    
    pc = results.config.pc
    years = list(range(2021, 2021 + len(results.macro)))
    
    # Conversion factor from Gigajoules to Terawatt-hours
    GJ_TO_TWH = (1 / 3600) / 1000

    # 1. Get the sectoral dataframes
    try:
        df_oil_original = results.sectoral['oil_use']
        df_oil_final = results.sectoral['oil_use_final']
        df_gas_original = results.sectoral['gas_use']
        df_gas_final = results.sectoral['gas_use_final']
    except KeyError as e:
        print(f"Error: Missing {e} not in results.sectoral.")
        return

    # 2. Calculate the *change* in TWh
    # Oil reduction should be positive (Original > Final)
    df_oil_reduction_twh = (df_oil_original - df_oil_final) * GJ_TO_TWH
    # Gas increase should be positive (Final > Original)
    df_gas_increase_twh = (df_gas_final - df_gas_original) * GJ_TO_TWH

    # 3. Identify the indices for each of the 3 target categories
    power_desal_names = [
        'Electricity, gas, steam and air conditioning supply',
        'Desalination'
    ]
    power_desal_indices = [pc.sectors.index(s) for s in power_desal_names if s in pc.sectors]

    agri_names = ['Crop and animal production, hunting and related service activities']
    agri_indices = [pc.sectors.index(s) for s in agri_names if s in pc.sectors]

    all_targeted_indices = set(pc.oil_reduction_sectors_idx)
    power_agri_indices = set(power_desal_indices + agri_indices)
    industry_indices = list(all_targeted_indices - power_agri_indices)

    # 4. Sum the reductions/increases for each category
    # Oil Reductions
    oil_power_reduction = df_oil_reduction_twh[power_desal_indices].sum(axis=1)
    oil_agri_reduction = df_oil_reduction_twh[agri_indices].sum(axis=1)
    oil_industry_reduction = df_oil_reduction_twh[industry_indices].sum(axis=1)
    
    # Gas Increases
    gas_power_increase = df_gas_increase_twh[power_desal_indices].sum(axis=1)
    gas_agri_increase = df_gas_increase_twh[agri_indices].sum(axis=1)
    gas_industry_increase = df_gas_increase_twh[industry_indices].sum(axis=1)

    # 5. Create the plotting DataFrames
    plot_df_oil = pd.DataFrame({
        'Power & Desalination': oil_power_reduction,
        'Industry': oil_industry_reduction,
        'Agriculture': oil_agri_reduction
    }, index=years)
    
    plot_df_gas = pd.DataFrame({
        'Power & Desalination': gas_power_increase,
        'Industry': gas_industry_increase,
        'Agriculture': gas_agri_increase
    }, index=years)

    # 6. Create the plot (side-by-side subplots)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=True)
    
    # --- Plot 1: Oil Reduction ---
    plot_df_oil.plot(
        kind='bar',
        stacked=True,
        ax=ax1,
        color=['#3498DB', '#E74C3C', '#2ECC71'] # Blue, Red, Green
    )
    ax1.set_title('Targeted Oil Reduction by Category (Oil-to-Gas Switch)', fontsize=16)
    ax1.set_ylabel('Annual Oil Use Reduction (TWh)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend(title='Sector Categories', loc='upper left')

    # --- Plot 2: Gas Increase ---
    plot_df_gas.plot(
        kind='bar',
        stacked=True,
        ax=ax2,
        color=['#3498DB', '#E74C3C', '#2ECC71'] # Blue, Red, Green
    )
    ax2.set_title('Resulting Gas Increase by Category (Oil-to-Gas Switch)', fontsize=16)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Annual Gas Use Increase (TWh)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set x-axis ticks to show years
    ax2.set_xticklabels(years, rotation=45)
    ax2.legend().set_visible(False) # Hide redundant legend
    
    plt.tight_layout()
    plt.show()


def create_renewable_energy_graphs(results: ModelResults):
    """
    Creates plots to visualize the primary energy mix, renewable adoption, and impact on emissions.
    """
    
    df_macro = results.macro.copy()
    pc = results.config.pc
    p = results.config.p
    years = list(range(2021, 2021 + len(df_macro)))
    
    # --- Conversion factor from GJ to Exajoules (EJ) for better readability ---
    GJ_TO_EJ = 1e-9

    # --- Create Figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
    # Set main title based on the scenario run
    scenario_title = "Vision 2030 Scenario" if p.vision_2030_renewable_target else "Trend Scenario"
    fig.suptitle(f"Energy and Emissions Analysis: {scenario_title}", fontsize=16)

    # --- Plot 1: Total Primary Energy Consumption by Source (Corrected) ---
    ax1.set_title("Primary Energy Mix by Source")
    
    # Convert final primary energy data to EJ for plotting
    gas_final_ej = df_macro["gas_use_final"] * GJ_TO_EJ
    oil_final_ej = df_macro["oil_use_final"] * GJ_TO_EJ
    renewable_ej = df_macro["renewable_energy_generation"] * GJ_TO_EJ

    # Create a stacked plot showing only the primary energy sources
    ax1.stackplot(years, oil_final_ej, gas_final_ej, renewable_ej,
                  labels=['Oil (Primary Use)', 'Gas (Primary Use)', 'Renewables'],
                  colors=['#2d2d2d', '#ff7f0e', '#2ca02c'])
    
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Primary Energy Consumption (EJ)")
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # --- Plot 2: Share of Renewable Energy (Unchanged) ---
    ax2.set_title("Share of Renewables in Total Energy Demand")
    ax2.plot(years, df_macro["renewable_energy_share"] * 100,
             label="Renewable Energy Share (%)", color='green', marker='.')
    
    if p.vision_2030_renewable_target:
        ax2.axhline(y=50, color='r', linestyle='--', label='50% Vision 2030 Target')
        ax2.axvline(x=2030, color='grey', linestyle=':', label='2030 Target Year')

    ax2.set_xlabel("Year")
    ax2.set_ylabel("Share (%)")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, linestyle='--', linewidth=0.5)

    # --- Plot 3: Total Greenhouse Gas Emissions (Unchanged) ---
    ax3.set_title("Total GHG Emissions by Source")
    ax3.stackplot(years, df_macro["oil_emissions_total"], df_macro["gas_emissions_total"], df_macro["electricity_emissions_total"],
                  labels=['Oil Emissions', 'Gas Emissions', 'Electricity Emissions'],
                  colors=['#2d2d2d', '#ff7f0e', '#1f77b4'])

    ax3.set_xlabel("Year")
    ax3.set_ylabel("Emissions (MtCO2)")
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

import os

def create_energy_efficiency_plot(results: ModelResults):
    """
    Creates a plot to visualize the energy reduction factor with 
    sector names and a summary table.
    """
    
    pc = results.config.pc
    p = results.config.p  
    df_efficiency = results.sectoral['energy_reduction_factor']
    years = list(range(2021, 2021 + len(df_efficiency)))
    final_year = years[-1]
    
    fig, ax = plt.subplots(figsize=(14, 10)) # Increased height for table space
    
    # 1. Plotting and Annotating Tiers
    # --- Top 20 (Red) ---
    ax.plot(years, df_efficiency[pc.top_20_energy_sectors[0]], color='red', alpha=0.9, 
            label=f'Top 20 Intensive Sectors ({p.efficiency_top_tier_annual_gain*100}%/yr)')
    ax.text(final_year + 0.5, df_efficiency[pc.top_20_energy_sectors[0]].iloc[-1], 
            'Petrochemicals, Steel, Cement', color='red', fontweight='bold', va='center')
    for sector_index in pc.top_20_energy_sectors[1:]:
        ax.plot(years, df_efficiency[sector_index], color='red', alpha=0.15)

    # --- Mid 20 (Orange) ---
    ax.plot(years, df_efficiency[pc.mid_20_energy_sectors[0]], color='orange', alpha=0.9, 
            label=f'Mid 20 Intensive Sectors ({p.efficiency_mid_tier_annual_gain*100}%/yr)')
    ax.text(final_year + 0.5, df_efficiency[pc.mid_20_energy_sectors[0]].iloc[-1], 
            'Construction, Food Processing', color='orange', fontweight='bold', va='center')
    for sector_index in pc.mid_20_energy_sectors[1:]:
        ax.plot(years, df_efficiency[sector_index], color='orange', alpha=0.15)

    # --- Low (Green) ---
    ax.plot(years, df_efficiency[pc.low_energy_sectors[0]], color='green', alpha=0.9, 
            label=f'Low Intensive Sectors ({p.efficiency_low_tier_annual_gain*100}%/yr)')
    ax.text(final_year + 0.5, df_efficiency[pc.low_energy_sectors[0]].iloc[-1], 
            'Finance, IT, Tourism', color='green', fontweight='bold', va='center')
    for sector_index in pc.low_energy_sectors[1:]:
        ax.plot(years, df_efficiency[sector_index], color='green', alpha=0.15)

    # 2. Add Summary Table on top (North)
    table_data = [
        ["Top 20", "High-heat, chemical processes, utilities", "1.0%"],
        ["Mid 20", "Manufacturing, heavy assembly", "0.5%"],
        ["Low", "Services, offices, light technology", "0.25%"]
    ]
    col_labels = ["Efficiency Tier", "Industry Profile", "Annual Gain"]
    
    # Position the table at the top of the axes
    the_table = ax.table(cellText=table_data, colLabels=col_labels, 
                         loc='top', cellLoc='center', colWidths=[0.15, 0.5, 0.15])
    the_table.scale(1, 1.8)
    the_table.set_fontsize(10)

    # 3. Formatting
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Cumulative Reduction (Share of Demand)", fontsize=12)
    ax.set_ylim(-0.05, 1.0) 
    ax.set_xlim(2021, 2060 + 10) # Extended X-axis to fit sector names
    ax.grid(True, linestyle='--', alpha=0.4)
    
    ax.axhline(y=p.efficiency_max_reduction_pct, color='black', linestyle='--', 
               label=f'Max Reduction Cap ({p.efficiency_max_reduction_pct*100}%)')

    # Legend at North (just below table)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.92]) # Make space for table at top
    
    figures_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
    os.makedirs(figures_path, exist_ok=True)
    plt.savefig(os.path.join(figures_path, "energy_efficiency_with_names.pdf"),
                format='pdf', dpi=300, bbox_inches='tight')
    
    plt.show()
# --- END OF NEW FUNCTION ---


def plot_vision_2030_results(results: ModelResults):
    """
    Plots the key outcomes of the new Vision 2030 energy policy.
    
    Assumes 'results' is the ModelResults object from running the model.
    """
    
    # --- Helper data ---
    pc = results.config.pc
    p = results.config.p
    macro = results.macro
    sectoral = results.sectoral
    
    # Get sector names
    sectors = pc.sectors
    elec_idx = pc.electricity_sector_idx
    elec_sector_name = sectors[elec_idx]
    
    # Get indices for oil reduction
    oil_reduce_indices = pc.oil_reduction_sectors_idx
    oil_reduce_names = [sectors[i] for i in oil_reduce_indices]
    
    # Create a time index (e.g., 2021, 2022, ...)
    start_year = 2021
    time_index = np.arange(start_year, start_year + len(macro))
    
    # Conversion factor for plotting (GJ to Terajoules)
    GJ_TO_TJ = 1 / 1000

    # --- 1. Plot Electricity Generation Mix ---
    
    # Get data
    elec_demand_total = macro['electricity_use_total'] * GJ_TO_TJ
    re_gen_total = macro['renewable_energy_generation'] * GJ_TO_TJ
    
    # Get FF inputs to the electricity sector
    gas_input_elec = sectoral['gas_use_final'].iloc[:, elec_idx] * GJ_TO_TJ
    oil_input_elec = sectoral['oil_use_final'].iloc[:, elec_idx] * GJ_TO_TJ
    
    # Calculate the "traditional" generation
    # We assume the RE portion directly meets demand
    re_met_demand = np.minimum(elec_demand_total, re_gen_total)
    traditional_met_demand = elec_demand_total - re_met_demand
    
    # Calculate RE share of electricity
    re_share_elec = (re_met_demand / elec_demand_total) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True)
    
    # Stack plot for generation mix
    ax1.stackplot(time_index,
                  re_met_demand, traditional_met_demand,
                  labels=['Renewable Generation', 'Traditional (Gas) Generation'],
                  colors=['#2ca02c', '#8c564b'])
    ax1.set_title('Electricity Generation Mix by Source (Vision 2030)')
    ax1.set_ylabel('Electricity Generation (TJ)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Line plot for RE share
    ax1b = ax1.twinx()
    ax1b.plot(time_index, re_share_elec, color='blue', linestyle='--', marker='o', label='Renewable Share of Electricity')
    ax1b.set_ylabel('Renewable Share (%)')
    ax1b.set_ylim(0, 100)
    ax1b.legend(loc='upper right')

    # --- 2. Plot Oil Use in Targeted Sectors ---
    
    # Get oil use for targeted sectors
    oil_use_targeted = sectoral['oil_use_final'].iloc[:, oil_reduce_indices].sum(axis=1) * GJ_TO_TJ
    oil_use_all_other = (macro['oil_use_final'] * GJ_TO_TJ) - oil_use_targeted
    
    ax2.stackplot(time_index,
                  oil_use_targeted, oil_use_all_other,
                  labels=['Targeted Sectors (Industry, Power, etc.)', 'All Other Sectors'],
                  colors=['#d62728', '#aec7e8'])
    ax2.set_title('Total Oil Consumption')
    ax2.set_ylabel('Oil Use (TJ)')
    ax2.set_xlabel('Year')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()



def create_sectoral_graphs(results: ModelResults, displayPlots: bool = True):
    """Create various plots based on model results."""

    df = results.macro.copy()
    df_ = results.sectoral.copy()
    pc = results.config.pc
    # Set years for x-axis labels
    years = list(range(2021, 2021 + endyear))  # 2021 to 2060 inclusive

    ####################################################################
    # Create one-letter NACE codes for each sector
    ###################################################################
    # Full section names for NACE sections
    # Full section names for NACE sections
    nace_section_full_names = {
        "A": "A - Agriculture, forestry and fishing",
        "B": "B - Mining and quarrying",
        "C": "C - Manufacturing",
        "D": "D - Electricity, gas, steam and air conditioning supply",
        "E": "E - Water supply; sewerage, waste management and remediation activities",
        "F": "F - Construction",
        "G": "G - Wholesale and retail trade; repair of motor vehicles and motorcycles",
        "H": "H - Transportation and storage",
        "I": "I - Accommodation and food service activities",
        "J": "J - Information and communication",
        "K": "K - Financial and insurance activities",
        "L": "L - Real estate activities",
        "M": "M - Professional, scientific and technical activities",
        "N": "N - Administrative and support service activities",
        "O": "O - Public administration and defence; compulsory social security",
        "P": "P - Education",
        "Q": "Q - Human health and social work activities",
        "R": "R - Arts, entertainment and recreation",
        "S + T": "S (Other service activities) + T (Activities of households as employers of domestic personnel)",
        "Desal": "Desalination"
    }

    def get_nace_section_letter_by_number(idx):
        # Map sector index (0-based) to a NACE code (1-based)
        # MM changed these codes by hand to adjust for missing sectors.
        # See e.g. here for the sectoral classification in KSA: https://www.stats.gov.sa/en/w/%D8%A7%D9%84%D8%AA%D8%B5%D9%86%D9%8A%D9%81-%D8%A7%D9%84%D9%88%D8%B7%D9%86%D9%8A-%D9%84%D9%84%D8%A3%D9%86%D8%B4%D8%B7%D8%A9-%D8%A7%D9%84%D8%A7%D9%82%D8%AA%D8%B5%D8%A7%D8%AF%D9%8A%D8%A9?id=2812935
        # The correspondence made by MM can be found on GITHUB:
        # C:\Users\MIESSMG\SFC_CORE3\input_data\MM_Excels_Working\National Classification for the Economic Activities(SSIC)_Jan_2025_En_MMKENModified
        code_int = idx
        if 0 <= code_int <= 2: return "A"
        elif 3 <= code_int <= 7: return "B"
        elif 8 <= code_int <= 31: return "C"
        elif 32 == code_int: return "D"
        elif 33 <= code_int <= 36: return "E"
        elif 37 <= code_int <= 39: return "F"
        elif 40 <= code_int <= 42: return "G"
        elif 43 <= code_int <= 47: return "H"
        elif 48 <= code_int <= 49: return "I"
        elif 50 <= code_int <= 55: return "J"
        elif 56 <= code_int <= 58: return "K"
        elif 59 == code_int: return "L"
        elif 60 <= code_int <= 66: return "M"
        elif 67 <= code_int <= 72: return "N"
        elif 73 == code_int: return "O"
        elif 74 == code_int: return "P"
        elif 75 <= code_int <= 77: return "Q"
        elif 78 <= code_int <= 80: return "R"
        elif 81 <= code_int <= 83: return "S + T"
        elif 84 == code_int: return "Desal"
        else: return "Other"

    # Build mapping from sector index to full section name
    sector_number_to_nace_section_full = {
        idx: nace_section_full_names.get(get_nace_section_letter_by_number(idx), "Other")
        for idx in range(85)
    }

    # If you have a DataFrame with sector columns 0..84 (e.g. results.sectoral['Y'])
    # You can aggregate it:
    def aggregate_to_nace_sections(df):
        section_labels = pd.Series(sector_number_to_nace_section_full)
        return df.groupby(section_labels, axis=1).sum()

    

    # For checking the mapping:



 
    # Iterate over the sectors and their indices
    sector_notes = [f"Sector {index}: {sector}" for index, sector in enumerate(pc.sectors)]

    ################################################################################
    # Sectoral plots
    ################################################################################
    # Define sectors according to columns in X, which is takes as representative for all sectoral vars
    sectors = results.sectoral['X'].columns
    # Convert imports to panda time series to be able to plot according to size (indices issue)!
    pc.IM_ = pd.Series(pc.IM_, index=sectors)

    #############################################################################################################################
    # Plot Sectoral Output Y_
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    axs.set_xticklabels(years[::5], rotation=0)  # label them as years
    sector = results.sectoral['Y'].columns
    axs.set_title("Sectoral OUTPUT Y_s (GDP) - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['Y'], label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Thousand SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['Y'].index[-1]
    top_sectors_start = results.sectoral['Y'].loc[1].nlargest(5).index  # Top 10 sectors by output at start
    top_sectors_end = results.sectoral['Y'].loc[end_period].nlargest(5).index  # Top 10 sectors by output by the end
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    # Combine the notes into one joint string
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    #############################################################################################################################
    # Plot Sectoral TOTAL OUTPUT X_
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    axs.set_xticklabels(years[::5], rotation=0)  # label them as years
    sector = results.sectoral['X'].columns
    axs.set_title("Sectoral TOTAL OUTPUT X_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['X'], label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Thousand SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['X'].index[-1]
    top_sectors_start = results.sectoral['X'].loc[1].nlargest(5).index  # Top 10 sectors by output at start
    top_sectors_end = results.sectoral['X'].loc[end_period].nlargest(5).index  # Top 10 sectors by output by the end
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    # Combine the notes into one joint string
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    #############################################################################################################################
    # Plot Sectoral INVESTMENT I_
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    axs.set_xticklabels(years[::5], rotation=0)  # label them as years
    sector = results.sectoral['I'].columns
    axs.set_title("Sectoral INVESTMENT I_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['I'], label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Thousand SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['I'].index[-1]
    top_sectors_start = results.sectoral['I'].loc[1].nlargest(5).index  # Top 10 sectors by output at start
    top_sectors_end = results.sectoral['I'].loc[end_period].nlargest(5).index  # Top 10 sectors by output by the end
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    # Combine the notes into one joint string
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)



    #############################################################################################################################
    # Plot Sectoral WAGES W_
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    axs.set_xticklabels(years[::5], rotation=0)  # label them as years
    sector = results.sectoral['W'].columns
    axs.set_title("Sectoral WAGES W_ - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['W'], label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Thousand SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['W'].index[-1]
    top_sectors_start = results.sectoral['W'].loc[1].nlargest(5).index  # Top 10 sectors by output at start
    top_sectors_end = results.sectoral['W'].loc[end_period].nlargest(5).index  # Top 10 sectors by output by the end
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    # Combine the notes into one joint string
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)




    #############################################################################################################################
    # Plot AGGREGATE Sectoral Output Y_onedigit_
    Y_onedigit_ = aggregate_to_nace_sections(results.sectoral['Y'])
    fig, axs = plt.subplots(1, 1, figsize=(16, 10))
    axs.set_xticks(df.index[::5])
    axs.set_xticklabels(years[::5], rotation=0)  # label them as years

    # Use aggregated Y DataFrame
    sector = Y_onedigit_.columns
    axs.set_title("Sectoral OUTPUT Y_s (GDP) by NACE 1-digit section")
    axs.plot(Y_onedigit_, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Thousand SAR")
    # axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
    # Wrap legend labels to a fixed width (e.g., 25 chars)
    wrapped_labels = [textwrap.fill(str(s), width=28) for s in sector]
    axs.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10, title="NACE Section", labels=wrapped_labels)
    plt.subplots_adjust(right=0.78, bottom=0.14)  # Make room for the legend and notes


    end_period = Y_onedigit_.index[-1]
    top_sectors_start = Y_onedigit_.loc[1].nlargest(5).index  # Top 5 sections by output at start
    top_sectors_end = Y_onedigit_.loc[end_period].nlargest(5).index  # Top 5 sections by output at end

    # For sections, just use their names as labels
    sector_notes_start = [f"Start - Section {section}" for section in top_sectors_start]
    sector_notes_end = [f"End - Section {section}" for section in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, 0, notes, wrap=True, horizontalalignment='center', fontsize=9)

    # Define sectors according to columns in X, which is takes as representative for all sectoral vars
    sectors = results.sectoral['X'].columns
    # Convert imports to panda time series to be able to plot according to size (indices issue)!
    pc.IM_ = pd.Series(pc.IM_, index=sectors)

    #############################################################################################################################
    # Plot Sectoral IMPORTS IM_
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    axs.set_xticklabels(years[::5], rotation=0)  # label them as years
    sector = results.sectoral['IM'].columns
    axs.set_title("Sectoral IMPORTS model CALCULATED IM_s (GDP) - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['IM'], label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Thousand SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['IM'].index[-1]
    top_sectors_start = results.sectoral['IM'].loc[1].nlargest(5).index  # Top 10 sectors by output at start
    top_sectors_end = results.sectoral['IM'].loc[end_period].nlargest(5).index  # Top 10 sectors by output by the end
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    # Combine the notes into one joint string
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

 
    #############################################################################################################################
    # Plot Sectoral Capital Stock K_
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    axs.set_xticklabels(years[::5], rotation=0)  # label them as years
    sector = results.sectoral['K'].columns
    axs.set_title("Sectoral Capital Stock K_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['K'], label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Thousand SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['K'].index[-1]
    top_sectors_start = results.sectoral['K'].loc[1].nlargest(5).index  # Top 10 sectors by output at start
    top_sectors_end = results.sectoral['K'].loc[end_period].nlargest(5).index  # Top 10 sectors by output by the end
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    # Combine the notes into one joint string
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)


    #############################################################################################################################
    # Plot Sectoral INTERMEDIATE INPUT PURCHASES IntP_
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    axs.set_xticklabels(years[::5], rotation=0)  # label them as years
    sector = results.sectoral['IntP'].columns
    axs.set_title("Sectoral INTERMEDIATE INPUT IntP_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['IntP'], label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Thousand SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['IntP'].index[-1]
    top_sectors_start = results.sectoral['IntP'].loc[1].nlargest(5).index  # Top 10 sectors by output at start
    top_sectors_end = results.sectoral['IntP'].loc[end_period].nlargest(5).index  # Top 10 sectors by output by the end
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    # Combine the notes into one joint string
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    #############################################################################################################################
    # Plot Sectoral EXPORTS EX_
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    axs.set_xticklabels(years[::5], rotation=0)  # label them as years
    sector = results.sectoral['EX'].columns
    axs.set_title("Sectoral EXPORTS EX_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['EX'], label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Thousand SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['EX'].index[-1]
    top_sectors_start = results.sectoral['EX'].loc[1].nlargest(5).index  # Top 10 sectors by output at start
    top_sectors_end = results.sectoral['EX'].loc[end_period].nlargest(5).index  # Top 10 sectors by output by the end
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    # Combine the notes into one joint string
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    #############################################################################################################################
    # Plot Sectoral Output Y_ for the top 10 sectors at the end of the modeling period
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
    axs.set_xticklabels(years[::5], rotation=0)  # label them as years
    # Identify the top 10 sectors in terms of output at the end of the modeling period
    end_period = results.sectoral['Y'].index[-1]  # Get the last time period
    top_10_sectors = results.sectoral['Y'].loc[end_period].nlargest(10).index  # Top 10 sectors by output
    # Filter the data for the top 10 sectors
    Y_top_10 = results.sectoral['Y'][top_10_sectors]
    # Plot the data for the top 10 sectors
    for sector in top_10_sectors:
        axs.plot(Y_top_10.index, Y_top_10[sector], label=f"Sector {sector}: {pc.sectors[sector]}")
    # Add labels, title, and legend
    axs.set_title("Sectoral Output Y_s (GDP) - Top 10 Sectors in last modeling period")
    axs.set_xlabel("Time")
    axs.set_ylabel("Thousand SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    # Update notes to display the names of the top 10 sectors


    # Plot investment public to inspect
    # fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    # (ax1) = axs

    #############################################################################################################################
    # Plot sectoral decomposition of components of X total output, in relation Y value added GDP and to Imports!
    # Extract data for period 1 START CALIBRATION
    sectors = results.sectoral['X'].columns
    X_ = results.sectoral['X'].loc[1]
    IntP_ = results.sectoral['IntP'].loc[1]
    W_ = results.sectoral['W'].loc[1]
    P_ = results.sectoral['P'].loc[1]
    Y_ = results.sectoral['Y'].loc[1]
    IM_ = results.sectoral['IM'].loc[1]
    Q_d_ = results.sectoral['Q_d'].loc[1]
    EX_ = results.sectoral['EX'].loc[1]
    GY_ = results.sectoral['GY'].loc[1]
    C_ = results.sectoral['C'].loc[1]
    # Calculate other components of Y
    Other_Y_ = Y_ - W_ - P_
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 8))
    # Bar width
    bar_width = 0.25
    indices = np.arange(len(sectors))
    # Plot X decomposition
    ax.bar(indices - bar_width/2, IntP_, bar_width, label='Intermediate Inputs (X)')
    ax.bar(indices - bar_width/2, W_, bar_width, bottom=IntP_, label='Wages (X)')
    ax.bar(indices - bar_width/2, P_, bar_width, bottom=IntP_ + W_, label='Profits (X)')
    # Plot Y decomposition
    ax.bar(indices + bar_width/2, Y_, bar_width,color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width, IM_, bar_width, label='Imports model calculated')
    ax.bar(indices + bar_width*1.5, pc.IM_, bar_width, label='Imports calibrated')
    # Add labels, title, and legend
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('START Sectoral Decomposition of Total Output X in comparison to Value Added Y and Imports model calculated and calibrated, Period 1')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)



    #############################################################################################################################
    # AGREGATE Plot sectoral decomposition of components of X total output, in relation Y value added GDP and to Imports!
    # START IN PERIOD 1
    # Aggregated data for calibration period (period 1)
    IntP_agg = aggregate_to_nace_sections(results.sectoral['IntP']).loc[1]
    W_agg    = aggregate_to_nace_sections(results.sectoral['W']).loc[1]
    P_agg    = aggregate_to_nace_sections(results.sectoral['P']).loc[1]
    Y_agg    = aggregate_to_nace_sections(results.sectoral['Y']).loc[1]
    IM_agg   = aggregate_to_nace_sections(results.sectoral['IM']).loc[1]
    Qd_agg   = aggregate_to_nace_sections(results.sectoral['Q_d']).loc[1]
    EX_agg   = aggregate_to_nace_sections(results.sectoral['EX']).loc[1]
    GY_agg   = aggregate_to_nace_sections(results.sectoral['GY']).loc[1]
    C_agg    = aggregate_to_nace_sections(results.sectoral['C']).loc[1]
    # For calibrated imports, you may need to aggregate as well:
    IM_calib_agg = aggregate_to_nace_sections(pd.DataFrame(pc.IM_).T).iloc[0]
    # Calculate other components of Y
    Other_Y_agg = Y_agg - W_agg - P_agg
    # Mapping from long name → short letter
    long_to_short = {v: k for k, v in nace_section_full_names.items()}
    sections_long = Y_agg.index
    sections_short = [long_to_short.get(name, name) for name in sections_long]
    bar_width = 0.25
    indices = np.arange(len(sections_short))
    # Suppose your DataFrame looks like this after aggregation:
    # Its columns are long names, e.g. "A - Agriculture, forestry and fishing"
    columns_long = Y_onedigit_.columns
    legend_labels = [long_to_short.get(col, col) for col in columns_long]
    fig, ax = plt.subplots(figsize=(15, 8))
    # Plot X decomposition
    ax.bar(indices - bar_width/2, IntP_agg, bar_width, label='Intermediate Inputs (X)')
    ax.bar(indices - bar_width/2, W_agg, bar_width, bottom=IntP_agg, label='Wages (X)')
    ax.bar(indices - bar_width/2, P_agg, bar_width, bottom=IntP_agg + W_agg, label='Profits (X)')
    # Plot GDP Value Added, Imports model/calibrated
    ax.bar(indices + bar_width/2, Y_agg, bar_width, color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width, IM_agg, bar_width, label='Imports model calculated')
    ax.bar(indices + bar_width*1.5, IM_calib_agg, bar_width, label='Imports calibrated')
    # Add labels, title, and legend
    ax.set_xlabel('NACE 1-digit Section')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('START: Decomposition of Total Output X (NACE 1-digit) vs. Value Added Y and Imports')
    ax.set_xticks(indices)
    ax.set_xticklabels(sections_short, rotation=0)
    # Place the legend directly under the main graph, spanning the width, with no extra space below
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.23),  # Just below the axes, minimal distance
        ncol=3,
        frameon=True,
        fontsize=11
    )
    # Prepare the long section names as notes, using a wider wrap for full-width notes
    wrapped_section_names = [textwrap.fill(f"{short}: {long}", width=60)
                            for short, long in zip(sections_short, sections_long)]
    # Arrange wrapped names in rows of 4 for broader lines
    lines = ["; ".join(wrapped_section_names[i:i+4]) for i in range(0, len(wrapped_section_names), 4)]
    notes = "\n".join(lines)
    # Adjust bottom margin to fit the notes, but keep the legend close to the plot
    plt.subplots_adjust(bottom=0.22)  # Less than before, since legend is right under plot
    # Place notes under the legend, spanning the full width
    plt.figtext(0.5, 0.04, notes, wrap=True, ha='center', va='top', fontsize=9)
        
    


    #############################################################################################################################
    # Plot sectoral decomposition of components of X total output, in relation Y value added GDP and to Imports!
    # Extract data for LAST period END of model
    sectors = results.sectoral['X'].columns
    end_period = results.sectoral['Y'].index[-1]
    X_ = results.sectoral['X'].loc[end_period]
    IntP_ = results.sectoral['IntP'].loc[end_period]
    W_ = results.sectoral['W'].loc[end_period]
    P_ = results.sectoral['P'].loc[end_period]
    Y_ = results.sectoral['Y'].loc[end_period]
    IM_ = results.sectoral['IM'].loc[end_period]
    Q_d_ = results.sectoral['Q_d'].loc[end_period]
    EX_ = results.sectoral['EX'].loc[end_period]
    GY_ = results.sectoral['GY'].loc[end_period]
    C_ = results.sectoral['C'].loc[end_period]
    # Calculate other components of Y
    Other_Y_ = Y_ - W_ - P_
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 8))
    # Bar width
    bar_width = 0.25
    indices = np.arange(len(sectors))
    # Plot X decomposition
    ax.bar(indices - bar_width/2, IntP_, bar_width, label='Intermediate Inputs (X)')
    ax.bar(indices - bar_width/2, W_, bar_width, bottom=IntP_, label='Wages (X)')
    ax.bar(indices - bar_width/2, P_, bar_width, bottom=IntP_ + W_, label='Profits (X)')
    # Plot Y decomposition
    ax.bar(indices + bar_width/2, Y_, bar_width,color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width, IM_, bar_width, label='Imports model calculated')
    ax.bar(indices + bar_width*1.5, pc.IM_, bar_width, label='Imports calibrated')
    # Add labels, title, and legend
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('END Sectoral Decomposition of Total Output X in comparison to Value Added Y and Imports model calculated and calibrated, LAST PERIOD')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    #############################################################################################################################
    # LAST PERIOD AGREGATE Plot sectoral decomposition of components of X total output, in relation Y value added GDP and to Imports!
    # LAST PERIOD
    # Aggregated data for calibration period (period 1)
    IntP_agg = aggregate_to_nace_sections(results.sectoral['IntP']).loc[1]
    W_agg    = aggregate_to_nace_sections(results.sectoral['W']).loc[end_period]
    P_agg    = aggregate_to_nace_sections(results.sectoral['P']).loc[end_period]
    Y_agg    = aggregate_to_nace_sections(results.sectoral['Y']).loc[end_period]
    IM_agg   = aggregate_to_nace_sections(results.sectoral['IM']).loc[end_period]
    Qd_agg   = aggregate_to_nace_sections(results.sectoral['Q_d']).loc[end_period]
    EX_agg   = aggregate_to_nace_sections(results.sectoral['EX']).loc[end_period]
    GY_agg   = aggregate_to_nace_sections(results.sectoral['GY']).loc[end_period]
    C_agg    = aggregate_to_nace_sections(results.sectoral['C']).loc[end_period]
    # For calibrated imports, you may need to aggregate as well:
    IM_calib_agg = aggregate_to_nace_sections(pd.DataFrame(pc.IM_).T).iloc[0]

    # Calculate other components of Y
    Other_Y_agg = Y_agg - W_agg - P_agg

    # Mapping from long name → short letter
    long_to_short = {v: k for k, v in nace_section_full_names.items()}

    sections_long = Y_agg.index
    sections_short = [long_to_short.get(name, name) for name in sections_long]

    bar_width = 0.25
    indices = np.arange(len(sections_short))


    # Suppose your DataFrame looks like this after aggregation:
    # Its columns are long names, e.g. "A - Agriculture, forestry and fishing"
    columns_long = Y_onedigit_.columns
    legend_labels = [long_to_short.get(col, col) for col in columns_long]


    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot X decomposition
    ax.bar(indices - bar_width/2, IntP_agg, bar_width, label='Intermediate Inputs (X)')
    ax.bar(indices - bar_width/2, W_agg, bar_width, bottom=IntP_agg, label='Wages (X)')
    ax.bar(indices - bar_width/2, P_agg, bar_width, bottom=IntP_agg + W_agg, label='Profits (X)')

    # Plot GDP Value Added, Imports model/calibrated
    ax.bar(indices + bar_width/2, Y_agg, bar_width, color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width, IM_agg, bar_width, label='Imports model calculated')
    ax.bar(indices + bar_width*1.5, IM_calib_agg, bar_width, label='Imports calibrated')

    # Add labels, title, and legend
    ax.set_xlabel('NACE 1-digit Section')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('LAST PERIOD: Decomposition of Total Output X (NACE 1-digit) vs. Value Added Y and Imports')
    ax.set_xticks(indices)
    ax.set_xticklabels(sections_short, rotation=0)
    # Place the legend directly under the main graph, spanning the width, with no extra space below
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.23),  # Just below the axes, minimal distance
        ncol=3,
        frameon=True,
        fontsize=11
    )
    # Prepare the long section names as notes, using a wider wrap for full-width notes
    wrapped_section_names = [textwrap.fill(f"{short}: {long}", width=60)
                            for short, long in zip(sections_short, sections_long)]
    # Arrange wrapped names in rows of 4 for broader lines
    lines = ["; ".join(wrapped_section_names[i:i+4]) for i in range(0, len(wrapped_section_names), 4)]
    notes = "\n".join(lines)
    # Adjust bottom margin to fit the notes, but keep the legend close to the plot
    plt.subplots_adjust(bottom=0.22)  # Less than before, since legend is right under plot
    # Place notes under the legend, spanning the full width
    plt.figtext(0.5, 0.04, notes, wrap=True, ha='center', va='top', fontsize=8)



    # Reset data for period 1 so that other plots are still for calibration period
    sectors = results.sectoral['X'].columns
    X_ = results.sectoral['X'].loc[1]
    IntP_ = results.sectoral['IntP'].loc[1]
    W_ = results.sectoral['W'].loc[1]
    P_ = results.sectoral['P'].loc[1]
    Y_ = results.sectoral['Y'].loc[1]
    IM_ = results.sectoral['IM'].loc[1]
    Q_d_ = results.sectoral['Q_d'].loc[1]
    EX_ = results.sectoral['EX'].loc[1]
    GY_ = results.sectoral['GY'].loc[1]
    C_ = results.sectoral['C'].loc[1]

    #############################################################################################################################
    # Plot Y model calculated vs. Y calibrated sectoral
    fig, ax = plt.subplots(figsize=(15, 8))
    # Bar width
    bar_width = 0.25
    indices = np.arange(len(sectors))
    ax.bar(indices - bar_width/2, Y_, bar_width,color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width/2, pc.Y_, bar_width, label='Y_ sectoral CALIBRATED ')
    # Add labels, title, and legend
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('Sectoral Value Added Y_ MODEL CALCULATED vs. Y_ sectoral CALIBRATED Period 1')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    #############################################################################################################################
    # Plot Y model calculated vs. Y calibrated sectoral
    fig, ax = plt.subplots(figsize=(15, 8))
    # Bar width
    bar_width = 0.25
    indices = np.arange(len(sectors))
    ax.bar(indices - bar_width/2, results.sectoral['Y'].loc[2], bar_width,color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width/2, pc.Y_, bar_width, label='Y_ sectoral CALIBRATED ')
    # Add labels, title, and legend
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('Period 2 Sectoral Value Added Y_ MODEL CALCULATED vs. Y_ sectoral CALIBRATED Period 2')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)



    #############################################################################################################################
    if displayPlots:
        plt.tight_layout()
        plt.show()

    return fig, ax





    #######################################################################################################################################################################################################################################################################################################################################################################################
    #######################################################################################################################################################################################################################################################################################################################################################################################
    # HERE START THE DETAILED SECTORAL GRAPHS NOT ALWAYS NEEDED
def create_detailed_sectoral_graphs(results: ModelResults, displayPlots: bool = True):
    #############################################################################################################################

    df = results.macro.copy()
    df_ = results.sectoral.copy()
    pc = results.config.pc
    # Set years for x-axis labels
    years = list(range(2021, 2021 + endyear))  # 2021 to 2060 inclusive

    ####################################################################
    # Create one-letter NACE codes for each sector
    ###################################################################
    # Full section names for NACE sections
    # Full section names for NACE sections
    nace_section_full_names = {
        "A": "A - Agriculture, forestry and fishing",
        "B": "B - Mining and quarrying",
        "C": "C - Manufacturing",
        "D": "D - Electricity, gas, steam and air conditioning supply",
        "E": "E - Water supply; sewerage, waste management and remediation activities",
        "F": "F - Construction",
        "G": "G - Wholesale and retail trade; repair of motor vehicles and motorcycles",
        "H": "H - Transportation and storage",
        "I": "I - Accommodation and food service activities",
        "J": "J - Information and communication",
        "K": "K - Financial and insurance activities",
        "L": "L - Real estate activities",
        "M": "M - Professional, scientific and technical activities",
        "N": "N - Administrative and support service activities",
        "O": "O - Public administration and defence; compulsory social security",
        "P": "P - Education",
        "Q": "Q - Human health and social work activities",
        "R": "R - Arts, entertainment and recreation",
        "S + T": "S (Other service activities) + T (Activities of households as employers of domestic personnel)",
        "Desal": "Desalination"
    }

    def get_nace_section_letter_by_number(idx):
        # Map sector index (0-based) to a NACE code (1-based)
        # MM changed these codes by hand to adjust for missing sectors.
        # See e.g. here for the sectoral classification in KSA: https://www.stats.gov.sa/en/w/%D8%A7%D9%84%D8%AA%D8%B5%D9%86%D9%8A%D9%81-%D8%A7%D9%84%D9%88%D8%B7%D9%86%D9%8A-%D9%84%D9%84%D8%A3%D9%86%D8%B4%D8%B7%D8%A9-%D8%A7%D9%84%D8%A7%D9%82%D8%AA%D8%B5%D8%A7%D8%AF%D9%8A%D8%A9?id=2812935
        # The correspondence made by MM can be found on GITHUB:
        # C:\Users\MIESSMG\SFC_CORE3\input_data\MM_Excels_Working\National Classification for the Economic Activities(SSIC)_Jan_2025_En_MMKENModified
        code_int = idx
        if 0 <= code_int <= 2: return "A"
        elif 3 <= code_int <= 7: return "B"
        elif 8 <= code_int <= 31: return "C"
        elif 32 == code_int: return "D"
        elif 33 <= code_int <= 36: return "E"
        elif 37 <= code_int <= 39: return "F"
        elif 40 <= code_int <= 42: return "G"
        elif 43 <= code_int <= 47: return "H"
        elif 48 <= code_int <= 49: return "I"
        elif 50 <= code_int <= 55: return "J"
        elif 56 <= code_int <= 58: return "K"
        elif 59 == code_int: return "L"
        elif 60 <= code_int <= 66: return "M"
        elif 67 <= code_int <= 72: return "N"
        elif 73 == code_int: return "O"
        elif 74 == code_int: return "P"
        elif 75 <= code_int <= 77: return "Q"
        elif 78 <= code_int <= 80: return "R"
        elif 81 <= code_int <= 83: return "S + T"
        elif 84 == code_int: return "Desal"
        else: return "Other"

    # Build mapping from sector index to full section name
    sector_number_to_nace_section_full = {
        idx: nace_section_full_names.get(get_nace_section_letter_by_number(idx), "Other")
        for idx in range(85)
    }
    # If you have a DataFrame with sector columns 0..84 (e.g. results.sectoral['Y'])
    # You can aggregate it:
    def aggregate_to_nace_sections(df):
        section_labels = pd.Series(sector_number_to_nace_section_full)
        return df.groupby(section_labels, axis=1).sum()

    #############################################################################################################################
    # Plot sectoral decomposition of components of X total output, in relation Y value added GDP and to Imports!
    # Extract data for period 1 START CALIBRATION
    sectors = results.sectoral['X'].columns
    X_ = results.sectoral['X'].loc[1]
    IntP_ = results.sectoral['IntP'].loc[1]
    W_ = results.sectoral['W'].loc[1]
    P_ = results.sectoral['P'].loc[1]
    Y_ = results.sectoral['Y'].loc[1]
    IM_ = results.sectoral['IM'].loc[1]
    Q_d_ = results.sectoral['Q_d'].loc[1]
    EX_ = results.sectoral['EX'].loc[1]
    GY_ = results.sectoral['GY'].loc[1]
    C_ = results.sectoral['C'].loc[1]
    # Calculate other components of Y
    Other_Y_ = Y_ - W_ - P_
    # Create figure and axes

    # For checking the mapping:



 
    # Iterate over the sectors and their indices
    sector_notes = [f"Sector {index}: {sector}" for index, sector in enumerate(pc.sectors)]

    ################################################################################
    # Sectoral plots
    ################################################################################
    # Define sectors according to columns in X, which is takes as representative for all sectoral vars
    sectors = results.sectoral['X'].columns
    # Convert imports to panda time series to be able to plot according to size (indices issue)!
    pc.IM_ = pd.Series(pc.IM_, index=sectors)
    
    
    
    # Plot Y vs. Q_d
    fig, ax = plt.subplots(figsize=(15, 8))
    # Bar width
    bar_width = 0.25
    indices = np.arange(len(sectors))
    ax.bar(indices - bar_width/2, Y_, bar_width,color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width/2, Q_d_, bar_width, label='Sectoral demand Q_d_')
    ax.bar(indices + bar_width*1.5,pc.Y_, bar_width, label='Y_ sectoral CALIBRATED')
    # Add labels, title, and legend
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('Sectoral Value Added Y_ & calibrated Y vs. Sectoral Demand Q_d_ Period 1')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    #############################################################################################################################
    # Plot IntP model calculated vs. IntP calibrated sectoral
    fig, ax = plt.subplots(figsize=(15, 8))
    # Bar width
    bar_width = 0.25
    indices = np.arange(len(sectors))
    ax.bar(indices - bar_width/2, IntP_, bar_width,color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width/2, pc.IntP_, bar_width, label='Y_ sectoral CALIBRATED ')
    # Add labels, title, and legend
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('Sectoral INT. INPUT IntP_ model calculated MODEL CALCULATED vs. IntP_ sectoral CALIBRATED Period 1')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    #############################################################################################################################
     # Determine the largest sectors in terms of total output X and plot only those
    top_sectors = X_.nlargest(10).index
    X_top = X_.loc[top_sectors]
    Y_top = Y_.loc[top_sectors]
    Q_d_top = Q_d_.loc[top_sectors]
    IntP_top = IntP_.loc[top_sectors]
    W_top = W_.loc[top_sectors]
    P_top = P_.loc[top_sectors]
    IM_top = IM_.loc[top_sectors]
    pc.IM_top = pc.IM_.loc[top_sectors]
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.25
    indices = np.arange(len(top_sectors))
    top_sector_notes = [f"Sector {index}: {pc.sectors[index]}" for index in top_sectors]
    # Plot X decomposition for top 15 sectors
    ax.bar(indices - bar_width/2, IntP_top, bar_width, label='Intermediate Inputs (X)')
    ax.bar(indices - bar_width/2, W_top, bar_width, bottom=IntP_top, label='Wages (X)')
    ax.bar(indices - bar_width/2, P_top, bar_width, bottom=IntP_top + W_top, label='Profits (X)')
    # Plot imports
    ax.bar(indices + bar_width/2, Y_top, bar_width, color='skyblue', alpha=0.7, label='Model calculated Output Y')
    ax.bar(indices + bar_width, Q_d_top, bar_width, color='purple', label='Aggregate sectoral demand Q_d_')
    ax.bar(indices + bar_width*1.5, IM_top, bar_width, label='Imports model calculated')
    ax.bar(indices + bar_width*2, pc.IM_top, bar_width, label='Imports calibrated')
    # Add labels, title, and legend
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('Sectoral Decomposition of Total Output X TOP Sectors + Output Y + imports model and calibrated, period 1')
    ax.set_xticks(indices)
    ax.set_xticklabels(top_sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    notes = "\n".join(["; ".join(top_sector_notes[i:i+3]) for i in range(0, len(top_sector_notes), 3)])
    # Adjust the figure layout to make space for the notes
    plt.subplots_adjust(bottom=0.3)  # Increase bottom margin to fit notes
    plt.figtext(0.5, 0.02, notes, wrap=True, horizontalalignment='center', fontsize=7)

    #############################################################################################################################
    # Determine the largest sectors in terms of total GDP Value added Yand plot only those
    top_sectors = Y_.nlargest(10).index
    X_top = X_.loc[top_sectors]
    Y_top = Y_.loc[top_sectors]
    IntP_top = IntP_.loc[top_sectors]
    W_top = W_.loc[top_sectors]
    P_top = P_.loc[top_sectors]
    IM_top = IM_.loc[top_sectors]
    pc.IM_top = pc.IM_.loc[top_sectors]
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.25
    indices = np.arange(len(top_sectors))
    top_sector_notes = [f"Sector {index}: {pc.sectors[index]}" for index in top_sectors]
    # Plot X decomposition for top 15 sectors
    ax.bar(indices - bar_width/2, IntP_top, bar_width, label='Intermediate Inputs (X)')
    ax.bar(indices - bar_width/2, W_top, bar_width, bottom=IntP_top, label='Wages (X)')
    ax.bar(indices - bar_width/2, P_top, bar_width, bottom=IntP_top + W_top, label='Profits (X)')
    # Plot imports
    ax.bar(indices + bar_width/2, Y_top, bar_width, color='skyblue', alpha=0.7, label='Model calculated Output Y')
    ax.bar(indices + bar_width, IM_top, bar_width, label='Imports model calculated')
    ax.bar(indices + bar_width*1.5, pc.IM_top, bar_width, label='Imports calibrated')
    # Add labels, title, and legend
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('Sectoral Decomposition of Total GDP Value added Y TOP Sectors + Output Y + imports model and calibrated, period 1')
    ax.set_xticks(indices)
    ax.set_xticklabels(top_sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    notes = "\n".join(["; ".join(top_sector_notes[i:i+3]) for i in range(0, len(top_sector_notes), 3)])
    # Adjust the figure layout to make space for the notes
    plt.subplots_adjust(bottom=0.3)  # Increase bottom margin to fit notes
    plt.figtext(0.5, 0.02, notes, wrap=True, horizontalalignment='center', fontsize=7)


    #############################################################################################################################
    # Determine the 10 largest sectors in terms of MODEL CALCULATED IMPORTS
    top_sectors = IM_.nlargest(10).index
    X_top = X_.loc[top_sectors]
    Y_top = Y_.loc[top_sectors]
    IntP_top = IntP_.loc[top_sectors]
    W_top = W_.loc[top_sectors]
    P_top = P_.loc[top_sectors]
    IM_top = IM_.loc[top_sectors]
    Q_d_top = Q_d_.loc[top_sectors]
    pc.IM_top = pc.IM_.loc[top_sectors]
    EX_top = EX_.loc[top_sectors]
    GY_top = GY_.loc[top_sectors]
    C_top = C_.loc[top_sectors]
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.1
    indices = np.arange(len(top_sectors))
    top_sector_notes = [f"Sector {index}: {pc.sectors[index]}" for index in top_sectors]
    # Plot X decomposition for top 15 sectors
    ax.bar(indices - bar_width, IntP_top, bar_width, label='Intermediate Inputs (X)')
    ax.bar(indices - bar_width, W_top, bar_width, bottom=IntP_top, label='Wages (X)')
    ax.bar(indices - bar_width, P_top, bar_width, bottom=IntP_top + W_top, label='Profits (X)')
    # Plot imports
    ax.bar(indices , Y_top, bar_width, color='skyblue', alpha=0.7, label='Model calculated Output Y')
    ax.bar(indices + bar_width, Q_d_top, bar_width, color='purple', label='Aggregate sectoral demand Q_d_')
    ax.bar(indices + bar_width*2, IM_top, bar_width, label='Imports model calculated')
    ax.bar(indices + bar_width*3, pc.IM_top, bar_width, label='Imports calibrated')
    ax.bar(indices + bar_width*4, EX_top, bar_width, label='Exports model calculated')
    ax.bar(indices + bar_width*5, GY_top, bar_width, label='Government expenditure model calculated')
    ax.bar(indices + bar_width*6, C_top, bar_width, label='Consumption model calculated')
    # Add labels, title, and legend
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('Sectoral Decomposition of top sectors IMPORTS MODEL CALCULATED, period 1')
    ax.set_xticks(indices)
    ax.set_xticklabels(top_sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    notes = "\n".join(["; ".join(top_sector_notes[i:i+3]) for i in range(0, len(top_sector_notes), 3)])
    # Adjust the figure layout to make space for the notes
    plt.subplots_adjust(bottom=0.3)  # Increase bottom margin to fit notes
    plt.figtext(0.5, 0.02, notes, wrap=True, horizontalalignment='center', fontsize=7)

    #############################################################################################################################
    # 10 top model calculated imports sectors: Create a stacked area chart to track the evolution of variables over time
    fig, ax = plt.subplots(figsize=(15, 8))
    # Extract data for the top sectors over the entire modeling horizon
    IntP_top = results.sectoral['IntP'][top_sectors]
    W_top = results.sectoral['W'][top_sectors]
    P_top = results.sectoral['P'][top_sectors]
    Y_top = results.sectoral['Y'][top_sectors]
    IM_top = results.sectoral['IM'][top_sectors]
    # Plot the stacked area chart
    ax.stackplot(
        results.sectoral['X'].index,  # Time index
        IntP_top.T,  # Intermediate Inputs (X)
        W_top.T,     # Wages (X)
        P_top.T,     # Profits (X)
        labels=['Intermediate Inputs (X)', 'Wages (X)', 'Profits (X)']
    )
    # Overlay the evolution of imports and output
    ax.plot(results.sectoral['X'].index, Y_top.sum(axis=1), label='Model calculated Output Y', color='skyblue', linewidth=2)
    ax.plot(results.sectoral['X'].index, IM_top.sum(axis=1), label='Imports model calculated', color='orange', linewidth=2)
    # Add labels, title, and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('Evolution of Top 10 Sectors for TOP IMPORTS MODEL CALCULATED over time')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    #################################################
    # Top 10 sectors according to the metric defined here, used by the loop below to create separate charts
    #################################################
    top_sectors = Y_.nlargest(10).index  # Select the top 10 sectors according to the chosen metric
    time_index = results.sectoral['X'].index  # Time index

    # Loop through the top 5 sectors (according to metric defined here) and create a separate chart for each
    for sector in top_sectors:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.set_xticks(df.index[::5])               # ticks at each model step according to years (5-year intervalls)
        ax.set_xticklabels(years[::5], rotation=0)  # label them as years
        # Extract data for the current sector
        IntP_sector = results.sectoral['IntP'][sector]
        W_sector = results.sectoral['W'][sector]
        P_sector = results.sectoral['P'][sector]
        Y_sector = results.sectoral['Y'][sector]
        IM_sector = results.sectoral['IM'][sector]
        # Plot the stacked area chart for the current sector
        ax.stackplot(
            time_index,  # Time index
            IntP_sector,  # Intermediate Inputs (X)
            W_sector,     # Wages (X)
            P_sector,     # Profits (X)
            labels=['Intermediate Inputs (X)', 'Wages (X)', 'Profits (X)']
        )
        # Overlay the evolution of imports and output for the current sector
        ax.plot(time_index, Y_sector, label=f'Output Y ({sector})', linewidth=2, color='blue')
        ax.plot(time_index, IM_sector, label=f'Imports ({sector})', linewidth=2, linestyle='--', color='orange')
        # Add labels, title, and legend
        ax.set_xlabel('Time')
        ax.set_ylabel('Thousand SAR')
        ax.set_title(f'Evolution of Sector {sector} Over Time')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        # Display only the current sector in the notes
        current_sector_note = f"Sector {sector}: {pc.sectors[sector]}"
        notes = current_sector_note
        # Adjust the figure layout to make space for the notes
        plt.figtext(0.5, 0.02, notes, wrap=True, horizontalalignment='center', fontsize=12)

    ##########################################################################################################################################################################################################################################################
    # Determine the 10 largest sectors in terms of CALIBRATED IMPORTS
    top_sectors = pc.IM_.nlargest(10).index
    X_top = X_.loc[top_sectors]
    IntP_top = IntP_.loc[top_sectors]
    W_top = W_.loc[top_sectors]
    P_top = P_.loc[top_sectors]
    IM_top = IM_.loc[top_sectors]
    pc.IM_top = pc.IM_.loc[top_sectors]
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.25
    indices = np.arange(len(top_sectors))
    top_sector_notes = [f"Sector {index}: {pc.sectors[index]}" for index in top_sectors]
    # Plot X decomposition for top 15 sectors
    ax.bar(indices - bar_width/2, IntP_top, bar_width, label='Intermediate Inputs (X)')
    ax.bar(indices - bar_width/2, W_top, bar_width, bottom=IntP_top, label='Wages (X)')
    ax.bar(indices - bar_width/2, P_top, bar_width, bottom=IntP_top + W_top, label='Profits (X)')
    # Plot imports
    ax.bar(indices + bar_width/2, IM_top, bar_width, label='Imports model calculated')
    ax.bar(indices + bar_width, pc.IM_top, bar_width, label='Imports calibrated')
    # Add labels, title, and legend
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('Sectoral Decomposition of top sectors IMPORTS CALIBRATED')
    ax.set_xticks(indices)
    ax.set_xticklabels(top_sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    notes = "\n".join(["; ".join(top_sector_notes[i:i+3]) for i in range(0, len(top_sector_notes), 3)])
    # Adjust the figure layout to make space for the notes
    plt.subplots_adjust(bottom=0.3)  # Increase bottom margin to fit notes
    plt.figtext(0.5, 0.02, notes, wrap=True, horizontalalignment='center', fontsize=7)


    #############################################################################################################################
    # Plot Imports CALIBRATED and Output Y side by side
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.35
    indices = np.arange(len(sectors))
    ax.bar(indices - bar_width / 2, results.sectoral['Y'].loc[1], bar_width, color='skyblue', alpha=0.7, label='Model calculated Output Y period 1')
    ax.bar(indices + bar_width / 2, pc.IM_, bar_width, color='orange', alpha=0.7, label='Calibrated Imports')
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Imports (Thousand SAR)')
    ax.set_title('Model calculated output GDP Y period 1 vs. CALIBRATED Sectoral IMPORTS')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    #############################################################################################################################
    # Plot Imports in the model (calculated) and Output Y side by side
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.35
    indices = np.arange(len(sectors))
    ax.bar(indices - bar_width / 2, results.sectoral['Y'].loc[1], bar_width, color='skyblue', alpha=0.7, label='Model calculated Output Y period 1')
    ax.bar(indices + bar_width / 2, results.sectoral['IM'].loc[1], bar_width, color='orange', alpha=0.7, label='Imports model CALCULATED')
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Imports (Thousand SAR)')
    ax.set_title('mMdel calculated output GDP Y vs. MODEL CALCULATED Sectoral IMPORTS period 1')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    #############################################################################################################################
    # Plot Imports in the model (calculated) and calibrated imports side by side
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.35
    indices = np.arange(len(sectors))
    ax.bar(indices - bar_width / 2, results.sectoral['IM'].loc[1], bar_width, color='skyblue', alpha=0.7, label='Model Calculated Imports')
    ax.bar(indices + bar_width / 2, pc.IM_, bar_width, color='orange', alpha=0.7, label='Calibrated Imports')
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Imports (Thousand SAR)')
    ax.set_title('Sectoral Imports in the First Period: Model vs. Calibrated')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)


    #############################################################################################################################
    # Plot Y and Imports side by side, where Imports are larger than output model calculated period 1
    fig, ax = plt.subplots(figsize=(15, 8))
    # Filter sectors where imports are larger than output
    sectors_with_high_imports = results.sectoral['IM'].loc[1] > results.sectoral['Y'].loc[1]
    high_import_sectors = results.sectoral['IM'].loc[1][sectors_with_high_imports]
    high_output_sectors = results.sectoral['Y'].loc[1][sectors_with_high_imports]
    # Get the sector numbers
    sector_numbers = high_import_sectors.index
    # Bar width and positions
    bar_width = 0.35
    indices = np.arange(len(sector_numbers))
    # Plot Y and Imports side by side
    ax.bar(indices - bar_width / 2, high_output_sectors, bar_width, label='Output Y', alpha=0.7)
    ax.bar(indices + bar_width / 2, high_import_sectors, bar_width, label='Imports', alpha=0.7)
    # Add labels, title, and legend
    ax.set_xlabel('Sector Number')
    ax.set_ylabel('Thousand SAR')
    ax.set_title('Sectors with Imports model calculated larger than Output Y model calculated period 1')
    ax.set_xticks(indices)
    ax.set_xticklabels(sector_numbers, rotation=90)
    ax.legend()
    # Add sector notes as a legend or text

    
    # Plot the different versions of Y in model and data for period 1/calibratioin
    # fig, ax = plt.subplots(figsize=(15, 8))
    # ax.bar(sector, results.sectoral['Y'].loc[1], color='green', alpha=0.7)
    # ax.set_xlabel('Sectors')
    # ax.set_ylabel('Output Y GDP (Thousand SAR)')
    # ax.set_title('Output Y GDP MODEL CORRECTED in the First Period bar chart')
    # indices = np.arange(len(sector))
    # ax.set_xticks(indices)
    # ax.set_xticklabels(sector, rotation=90)

    #############################################################################################################################
    # Plot all Y Value Added GDP bars side by side
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.2
    indices = np.arange(len(sectors))
    ax.bar(indices - 1.5 * bar_width, results.sectoral['Y'].loc[1], bar_width, color='green', alpha=0.7, label='Model-calculated GDP Y')
    ax.bar(indices - 0.5 * bar_width, pc.Y_, bar_width, color='red', alpha=0.7, label='Calibrated Y (Production Approach)')
    ax.bar(indices + 0.5 * bar_width, pc.Y_expenditure_, bar_width, color='skyblue', alpha=0.7, label='Calibrated Y (Expenditure Approach)')
    ax.bar(indices + 1.5 * bar_width, pc.Y_finaldemand_, bar_width, color='purple', alpha=0.7, label='Calibrated Final Demand (Including Imports)')
    # Add labels, title, and legend
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Output Y GDP (Thousand SAR)')
    ax.set_title('Comparison of Sectoral Output Y GDP in the First Period')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    # Add notes as legend
    notes = (
        "Notes:\n"
        "- Model-calculated GDP Y: Results from the model for period 1.\n"
        "- Calibrated Y (Production Approach): Calculated as self.Y_ = self.X_ - self.IntS_.\n"
        "- Calibrated Y (Expenditure Approach): Calculated as self.Y_expenditure_ = self.C_ + self.G_ + self.dI * self.I_total + self.EX_ - self.IM_.\n"
        "- Calibrated Final Demand: Directly from IOT, including imports."
    )
    plt.figtext(0.5, -0.2, notes, wrap=True, horizontalalignment='center', fontsize=10)

    # Calculate the absolute differences between model-calculated and calibrated final demand including imports
    differences = abs(results.sectoral['Y'].loc[1] - pc.Y_finaldemand_)
    # Identify the 15 sectors with the largest differences
    top_15_sectors = differences.nlargest(15).index
    # Extract data for the top 15 sectors
    Y_model_top = results.sectoral['Y'].loc[1, top_15_sectors]
    Y_calibrated_production_top = pc.Y_[top_15_sectors]
    Y_calibrated_expenditure_top = pc.Y_expenditure_[top_15_sectors]
    Y_calibrated_final_demand_top = pc.Y_finaldemand_[top_15_sectors]
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.2
    indices = np.arange(len(top_15_sectors))
    # Plot the bars
    ax.bar(indices - 1.5 * bar_width, Y_model_top, bar_width, color='green', alpha=0.7, label='Model-calculated GDP Y')
    ax.bar(indices - 0.5 * bar_width, Y_calibrated_production_top, bar_width, color='red', alpha=0.7, label='Calibrated Y (Production Approach)')
    ax.bar(indices + 0.5 * bar_width, Y_calibrated_expenditure_top, bar_width, color='skyblue', alpha=0.7, label='Calibrated Y (Expenditure Approach)')
    ax.bar(indices + 1.5 * bar_width, Y_calibrated_final_demand_top, bar_width, color='purple', alpha=0.7, label='Calibrated Final Demand (Including Imports)')
    # Add labels, title, and legend
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Output Y GDP (Thousand SAR)')
    ax.set_title('Comparison of Sectoral Output Y GDP for Top 15 Sectors with Largest Differences')
    ax.set_xticks(indices)
    ax.set_xticklabels(top_15_sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    # Add notes as legend
    notes = (
        "Notes:\n"
        "- Model-calculated GDP Y: Results from the model for period 1.\n"
        "- Calibrated Y (Production Approach): Calculated as self.Y_ = self.X_ - self.IntS_.\n"
        "- Calibrated Y (Expenditure Approach): Calculated as self.Y_expenditure_ = self.C_ + self.G_ + self.dI * self.I_total + self.EX_ - self.IM_.\n"
        "- Calibrated Final Demand: Directly from IOT, including imports."
    )
    plt.figtext(0.5, -0.2, notes, wrap=True, horizontalalignment='center', fontsize=10)


 


    #############################################################################################################################
    if displayPlots:
        plt.tight_layout()
        plt.show()

    return fig, ax


# Changes from original:
# - Using sum over all sectors instead of first sector only
def create_balance_sheet(results: ModelResults, t: int = -1, display: bool = True):
    """Return two pandas dataframes with symbols and values."""

    dt = results.macro.iloc[t].copy()  # Data at time t
    columns = ["Unemployed", "Households", "Firms",
               "Banks", "Govt", "CB", "Ene-Dirty", "Ene-Clean"]
    rows = ["Capital", "Deposits", "Loans"]
    symbols = [
        ["", "",  "+K", "", "", "", "", ""],
        ["",  "+V", "", "", "", "-V", "", ""],
        ["",  "", "-L", "+L", "", "", "", ""]
    ]
    data = [[string_to_value(s, dt) for s in row] for row in symbols]

    balance_sheet_symbols = pd.DataFrame(symbols, columns=columns, index=rows)
    balance_sheet = pd.DataFrame(data, columns=columns, index=rows)

    # Create sums of rows and columns
    balance_sheet.loc['Total'] = balance_sheet.sum()
    balance_sheet['Total'] = balance_sheet.sum(axis=1)

    if display:
        IPdisplay(balance_sheet_symbols)
        IPdisplay(balance_sheet.round(2))

    return balance_sheet, balance_sheet_symbols


def plot_heatmap_of_correlation(results: ModelResults, title="Heatmap of Input-Output Matrix Correlation (Period 1)"):
    """
    Plots a heatmap of the correlation of the input-output matrix.

    Parameters:
    matrix (pd.DataFrame or np.array): The input-output matrix.
    title (str): Title of the heatmap.
    """
    # Calculate the correlation matrix
    correlation_matrix = pd.DataFrame(results.iomatrix['A'][1]).corr()
    
    # Create the heatmap using seaborn
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True, square=True,
                linewidths=.5, xticklabels=True, yticklabels=True)
    plt.title(title, fontsize=16)
    plt.xlabel("Sectors", fontsize=12)
    plt.ylabel("Sectors", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot the heatmap
    plot_heatmap_of_correlation(pd.DataFrame(results.iomatrix['A'][1]), title="Heatmap of Input-Output Matrix Correlation (Period 1)")


def create_transition_matrix(results: pd.DataFrame, c: ModelConfig, t: int = -1, display: bool = True):
    """Return two pandas dataframes with symbols and values."""

    srange = get_srange(c)

    # Add parameters to dataframe
    df = results.copy()
    for var in ["dC", "dI", "dG"]:
        for i, sector in enumerate(srange):
            df[var + sector] = getattr(c.p, var)[:, i][1:]
    for ix, x in enumerate(srange):
        for iy, y in enumerate(srange):
            df[f"a{x}{y}"] = c.pc.A[ix, iy]

    dt = df.iloc[t].copy()  # Data at time t
    dtp = df.iloc[t-1].copy()  # Data at time t-1

    # Prepare data

    for X in srange:
        dt['C'+X] = dt["dC"+X] * (dt['CK'] + dt['CW'])
        dt['G'+X] = dt["dG"+X] * dt['G']
        dt['dI'+X+'.Inv'] = dt["dI"+X] * dt["Inv"]

        dt['ρL'+X] = c.p.ρ * dt['Ld'+X]
        dt['rl.L'+X+'(-1)'] = c.p.rl * dtp['Ld'+X]
        dt['ΔL'+X] = dt["Ld"+X] - dtp["Ld"+X]

        for Y in srange:
            pass

            # dt['IntFF'] = dt['IntSF'] - (dt['aFF']*dt['XF'])
            # dt['IntAF'] = (dt['aFA']*dt['XA'])
            # dt['IntEF'] = (dt['aFE']*dt['XE'])
            # dt['IntFA'] = (dt['aAF']*dt['XF'])
            # dt['IntAA'] = dt['IntSA'] - (dt['aAA']*dt['XA'])
            # dt['IntEA'] = (dt['aAE']*dt['XE'])
            # dt['IntFE'] = (dt['aEF']*dt['XF'])
            # dt['IntAE'] = (dt['aEA']*dt['XA'])
            # dt['IntEE'] = dt['IntSE'] - (dt['aEE']*dt['XE'])

    dt['WB'] = sum([dt['WB'+X] for X in srange])
    dt['Pi'] = sum([dt['P'+X] for X in srange])
    dt['T'] = dt["TW"] + dt["TK"] + sum([dt['T'+X] for X in srange])
    dt['rm.Vk(-1)'] = c.p.rm * dtp['VK']
    dt['rm.Vw(-1)'] = c.p.rm * dtp['VW']
    dt['rm.V(-1)'] = c.p.rm * (dt['VK'] + dt['VW'])

    dt['rl.L(-1)'] = sum([dt['rl.L'+X+'(-1)'] for X in srange])
    dt['ΔVk'] = dt["VK"] - dtp["VK"]
    dt['ΔVw'] = dt["VW"] - dtp["VW"]
    dt['ΔV'] = dt['ΔVk'] + dt['ΔVw']

    dt['ΔL'] = sum([dt['ΔL'+X] for X in srange])
    dt['ΔB'] = dt["Bond"] - dtp["Bond"]

    columns = ["Capitalists", "Workers", "F Cur", "F Cap", "A Cur",
               "A Cap", "E Cur", "E Cap", "Bank Cur", "Bank Cap", "Govt"]

    rows = ["Consumption", "Govt. Expenditure", "Investment", "Loan repayment", "Wages", "Int F", "Int A", "Int E",
            "Firm profits", "Bank profits", "Taxes", "i Deposits", "i Loans", "Δ Deposits", "Δ Loans", "Δ Bond"]

    symbols = [
        ["-CK", "-CW", "+CF", "", "+CA", "", "+CE", "", "", "", ""],
        ["", "", "+GF", "", "+GA", "", "+GE", "", "", "", "-G"],
        ["", "", "+dIF.Inv", "-IF", "+dIA.Inv",
            "-IA", "+dIE.Inv", "-IE", "", "", ""],
        ["", "", "-ρLF", "+ρLF", "-ρLA", "+ρLA", "-ρLE", "+ρLE", "", "", ""],
        ["", "+WB", "-WBF", "", "-WBA", "", "-WBE", "", "", "", ""],
        ["", "", "+IntFF", "", "-IntAF",  "",  "-IntEF",  "", "", "", ""],
        ["", "", "-IntFA", "", "+IntAA",  "",  "-IntEA", "", "", "", ""],
        ["", "", "-IntFE", "", "-IntAE",  "",  "+IntEE", "",  "", "", ""],
        ["+Pi", "", "-PF", "", "-PA", "", "-PE", "", "", "", ""],
        ["+BB", "", "", "", "", "", "", "", "-BB", "", ""],
        ["-TK", "-TW", "-TF", "", "-TA", "", "-TE", "", "", "", "+T"],
        ["+rm.Vk(-1)", "+rm.Vw(-1)", "", "", "",
         "", "", "", "-rm.V(-1)", "", ""],
        ["", "", "-rl.Lf(-1)", "", "-rl.La(-1)", "",
         "-rl.Le(-1)", "", "+rl.L(-1)", "", ""],
        ["-ΔVk", "-ΔVw", "", "", "", "", "", "",     "", "+ΔV", ""],
        ["", "", "", "+ΔLF", "", "+ΔLA", "", "+ΔLE", "", "-ΔL", ""],
        ["", "", "", "", "", "", "", "", "", "+ΔB", "-ΔB"]
    ]

    # Create dataframes
    data = [[string_to_value(s, dt) for s in row] for row in symbols]
    transition_matrix_symbols = pd.DataFrame(
        symbols, columns=columns, index=rows)
    transition_matrix = pd.DataFrame(data, columns=columns, index=rows)

    # Add sums of rows and columns
    transition_matrix.loc['Total'] = transition_matrix.sum()
    transition_matrix['Total'] = transition_matrix.sum(axis=1)

    if display:
        IPdisplay(transition_matrix_symbols)
        IPdisplay(transition_matrix.round(2))

    return transition_matrix, transition_matrix_symbols
# --- Add this helper function (ideally near the top with other imports) ---
def _safe_pct_change(policy_val, base_val):
    """Calculates (policy - base) / base, handling division by zero."""
    # (policy - base) / base
    # Use np.where to avoid division by zero
    change = np.where(
        base_val != 0,
        (policy_val - base_val) / base_val,
        0 # If base is 0, show 0 change
    )
    # Handle cases where both are 0 (which is 0 change, not nan)
    change = np.nan_to_num(change, nan=0.0, posinf=0.0, neginf=0.0)
    return change * 100 # Return as percentage

# --- This is the updated plotting function ---

def plot_scenario_percentage_change(results_base: 'ModelResults', results_policy: 'ModelResults', title_suffix=""):
    """
    Plots the percentage change between a baseline and policy scenario
    for key energy, emissions, investment, and GDP variables as bar charts
    for specific sampled years.
    """
    base_macro = results_base.macro
    policy_macro = results_policy.macro
    
    if len(base_macro) != len(policy_macro):
        print("Error: Scenarios have different time lengths.")
        return
        
    years = list(range(2021, 2021 + len(base_macro)))
    
    # --- 1. Calculate % Change Data for ALL years ---
    
    # Energy Mix
    pct_oil = _safe_pct_change(policy_macro['oil_use_total'], base_macro['oil_use_total'])
    pct_gas = _safe_pct_change(policy_macro['gas_use_total'], base_macro['gas_use_total'])
    pct_elec = _safe_pct_change(policy_macro['electricity_use_total'], base_macro['electricity_use_total'])
    # NEW: Total Energy
    base_total_energy = base_macro['oil_use_total'] + base_macro['gas_use_total'] + base_macro['electricity_use_total']
    policy_total_energy = policy_macro['oil_use_total'] + policy_macro['gas_use_total'] + policy_macro['electricity_use_total']
    pct_total_energy = _safe_pct_change(policy_total_energy, base_total_energy)

    # Emissions
    pct_emis_oil = _safe_pct_change(policy_macro['oil_emissions_total'], base_macro['oil_emissions_total'])
    pct_emis_gas = _safe_pct_change(policy_macro['gas_emissions_total'], base_macro['gas_emissions_total'])
    pct_emis_elec_indirect = _safe_pct_change(policy_macro['electricity_emissions_total'], base_macro['electricity_emissions_total'])
    # NEW: Total Emissions (including indirect)
    base_total_emissions = base_macro['oil_emissions_total'] + base_macro['gas_emissions_total'] + base_macro['electricity_emissions_total']
    policy_total_emissions = policy_macro['oil_emissions_total'] + policy_macro['gas_emissions_total'] + policy_macro['electricity_emissions_total']
    pct_total_emissions = _safe_pct_change(policy_total_emissions, base_total_emissions)

    # Investment
    pct_ire = _safe_pct_change(policy_macro['I_RE'], base_macro['I_RE'])
    pct_re_gen = _safe_pct_change(policy_macro['renewable_energy_generation'], base_macro['renewable_energy_generation'])

    # NEW: GDP
    pct_gdp = _safe_pct_change(policy_macro['Y'], base_macro['Y'])
    
    # --- 2. Filter Data for Sample Years ---
    # Create a DataFrame with all % change data
    pct_data = pd.DataFrame({
        'Year': years,
        '% Change in Oil': pct_oil,
        '% Change in Gas': pct_gas,
        '% Change in Electricity': pct_elec,
        '% Change in Total Energy': pct_total_energy,
        '% Change Oil Emissions': pct_emis_oil,
        '% Change Gas Emissions': pct_emis_gas,
        '% Change Elec. Emissions': pct_emis_elec_indirect,
        '% Change in Total Emissions': pct_total_emissions,
        '% Change RE Investment': pct_ire,
        '% Change RE Generation': pct_re_gen,
        '% Change in GDP': pct_gdp
    }).set_index('Year')

    # Define the years to sample (2021, 2025, 2030, ..., 2060)
    sample_years = [2021] + list(range(2025, 2061, 5))
    
    # Filter the data to only these years
    available_sample_years = [year for year in sample_years if year in pct_data.index]
    pct_data_sampled = pct_data.loc[available_sample_years]
    
    # --- 3. Create Plots (3x1 layout) ---
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 20), sharex=True) # Share x-axis
    fig.suptitle(f"Policy Impact: % Change ({title_suffix})", fontsize=22, y=1.03) # FONTSIZE
    
    # --- Plot 1: Energy Mix & GDP % Change (Bar + Scatter) ---
    df_energy = pct_data_sampled[['% Change in Total Energy', '% Change in Oil', '% Change in Gas', '% Change in Electricity']]
    df_energy.plot(
        kind='bar',
        ax=ax1,
        color={'% Change in Total Energy': '#636EFA', '% Change in Oil': 'black', '% Change in Gas': 'orange', '% Change in Electricity': 'blue'}
    )
    ax1.set_title("Percentage Change in Primary Energy Use and GDP", fontsize=20) # FONTSIZE
    ax1.set_ylabel("% Change from Baseline (Bars)", fontsize=16) # FONTSIZE
    ax1.axhline(0, color='grey', linewidth=0.5)
    ax1.grid(True, linestyle=':', alpha=0.7, axis='y')
    
    # --- Y-AXIS LIMITS FOR ENERGY ---
    ax1.set_ylim(-50, 155)
    # --- END ---

    ax1.set_xlabel('')
    ax1.tick_params(axis='x', labelrotation=0, labelsize=14) # FONTSIZE
    ax1.tick_params(axis='y', labelsize=14) # FONTSIZE

    # --- Add Secondary Y-Axis for GDP Bubbles ---
    ax1_twin = ax1.twinx()
    gdp_data = pct_data_sampled['% Change in GDP']
    
    # Get the x-positions from the bar chart's axes
    x_positions = np.arange(len(gdp_data))

    # Plot the GDP bubbles
    gdp_bubbles = ax1_twin.scatter(
        x_positions,
        gdp_data,
        color='#AB63FA', # Purple
        s=150,  # size of bubble
        edgecolor='black',
        zorder=10,
        label='% Change in GDP (Bubbles)'
    )
    
    # Add annotations (the text labels) for the bubbles
    for i, val in enumerate(gdp_data):
        ax1_twin.annotate(
            f'{val:.1f}%',
            (x_positions[i], val),
            textcoords="offset points",
            xytext=(0, 15), # 15 points above the bubble center
            ha='center',
            fontsize=12, # FONTSIZE
            fontweight='bold',
            color='#AB63FA'
        )
    
    # Set axis properties for the twin axis
    ax1_twin.set_ylabel("% Change in GDP (Bubbles)", color='#AB63FA', fontsize=16) # FONTSIZE
    ax1_twin.tick_params(axis='y', labelcolor='#AB63FA', labelsize=14) # FONTSIZE
    #ax1_twin.yaxis.set_major_formatter(PercentFormatter())
    
    # Set ylim for twin axis to make sure bubbles and labels are visible
    min_gdp = gdp_data.min()
    max_gdp = gdp_data.max()
    # Add padding + a fixed amount for the text label
    padding = max(np.abs(min_gdp*0.2), np.abs(max_gdp*0.2), 2.0)
    ax1_twin.set_ylim((min_gdp - padding), (max_gdp + padding))

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=14) # FONTSIZE

    # --- Plot 2: Emissions % Change (Bar Chart) ---
    df_emis = pct_data_sampled[['% Change in Total Emissions', '% Change Oil Emissions', '% Change Gas Emissions', '% Change Elec. Emissions']]
    df_emis.plot(
        kind='bar',
        ax=ax2,
        color={'% Change in Total Emissions': '#EF553B', '% Change Oil Emissions': 'black', '% Change Gas Emissions': 'orange', '% Change Elec. Emissions': 'blue'}
    )
    ax2.set_title("Percentage Change in Total Emissions (Direct + Indirect)", fontsize=20) # FONTSIZE
    ax2.set_ylabel("% Change from Baseline", fontsize=16) # FONTSIZE
    ax2.axhline(0, color='grey', linewidth=0.5)
    ax2.grid(True, linestyle=':', alpha=0.7, axis='y')
    
    # --- Y-AXIS LIMITS FOR EMISSIONS ---
    ax2.set_ylim(-70, 155)
    # --- END ---
    
    ax2.set_xlabel('')
    ax2.legend(fontsize=14) # FONTSIZE
    ax2.tick_params(axis='x', labelrotation=0, labelsize=14) # FONTSIZE
    ax2.tick_params(axis='y', labelsize=14) # FONTSIZE

    # --- Plot 3: RE Investment & Generation % Change (Bar Chart) ---
    # This is now the bottom plot, so it gets the x-axis label
    df_re = pct_data_sampled[['% Change RE Investment', '% Change RE Generation']]
    df_re.plot(
        kind='bar',
        ax=ax3,
        color={'% Change RE Investment': '#006400', '% Change RE Generation': '#2ca02c'} # Dark/Light Green
    )
    ax3.set_title("Percentage Change in Renewable Energy", fontsize=20) # FONTSIZE
    ax3.set_xlabel("Year", fontsize=16) # FONTSIZE
    ax3.set_ylabel("% Change from Baseline", fontsize=16) # FONTSIZE
    ax3.axhline(0, color='grey', linewidth=0.5)
    ax3.grid(True, linestyle=':', alpha=0.7, axis='y')
    ax3.tick_params(axis='x', rotation=0, labelsize=14) # FONTSIZE
    ax3.tick_params(axis='y', labelsize=14) # FONTSIZE
    ax3.legend(fontsize=14) # FONTSIZE
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    
def plot_efficiency_savings(results: 'ModelResults'):
    """
    Plots only the Annual Value of Avoided Energy Bills (The Benefit).
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    pc = results.config.pc
    p = results.config.p
    
    if not p.enable_energy_efficiency_gains:
        print("NOTE: Efficiency gains disabled. No plots.")
        return

    # --- 1. Setup Data ---
    # Check for necessary keys
    # We look for energy reduction factors to determine timeline
    if 'energy_reduction_factor' in results.sectoral:
        num_steps = len(results.sectoral['energy_reduction_factor'])
    elif 'energy_reduction_factor_' in results.sectoral:
        num_steps = len(results.sectoral['energy_reduction_factor_'])
    else:
        print("CRITICAL ERROR: Energy reduction factors not found in results. Check variable names.")
        return

    years = list(range(2021, 2021 + num_steps))
    TO_BILLION = 1_000_000

    # Identify energy rows in A matrix for cost calculation
    idx_oil_crude = pc.sectors.index('Extraction of crude petroleum and natural gas')
    idx_oil_refined = pc.sectors.index('Manufacture of coke and refined petroleum products')
    idx_elec = pc.electricity_sector_idx
    
    # Sum of energy coefficients (The share of energy cost in total production)
    # Using initial A matrix as proxy for energy cost structure
    energy_cost_share = pc.A[idx_oil_crude, :] + pc.A[idx_oil_refined, :] + pc.A[idx_elec, :]
    
    total_savings_bn = []
    
    # --- 2. Calculate Annual Savings ---
    for t_idx, year in enumerate(years):
        # robustly get X
        if 'X' in results.sectoral:
             X_vals = results.sectoral['X'].iloc[t_idx].values
        else:
             X_vals = results.sectoral['X_'].iloc[t_idx].values 

        # robustly get Reduction Factor
        if 'energy_reduction_factor' in results.sectoral:
            Red_vals = results.sectoral['energy_reduction_factor'].iloc[t_idx].values
        elif 'energy_reduction_factor_' in results.sectoral:
             Red_vals = results.sectoral['energy_reduction_factor_'].iloc[t_idx].values
        else:
             Red_vals = np.zeros_like(X_vals)

        # Logic: Output (SAR) * Energy Share (%) * % Saved = Amount Saved (SAR)
        avoided_cost_thousands = X_vals * energy_cost_share * Red_vals
        
        # Sum and convert to Billions
        total_savings_bn.append(np.sum(avoided_cost_thousands) / TO_BILLION)

    df_savings = pd.DataFrame({'Total Value of Energy Saved': total_savings_bn}, index=years)

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Area plot for visual weight
    df_savings.clip(lower=0).plot(kind='area', ax=ax, color='#1f77b4', alpha=0.4)
    # Line plot for sharp edge
    df_savings.plot(kind='line', ax=ax, color='#1f77b4', linewidth=3, legend=False) 
    
    # Styling
    ax.set_title("The Benefit: Annual Value of Avoided Energy Bills", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Billion SAR per Year", fontsize=12, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Annotation
    max_val = df_savings['Total Value of Energy Saved'].max()
    ax.text(0.02, 0.9, 
            f"Peak Annual Savings: {max_val:.1f} Bn SAR\n(Permanent cumulative gain)", 
            transform=ax.transAxes, fontsize=11, 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#1f77b4'))

    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np

def plot_net_zero_emissions(results: 'ModelResults'):
    """
    Visualizes the path to Net Zero with the new Electrification Wedge.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    pc = results.config.pc
    p = results.config.p
    
    if not getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False):
        print("Net Zero Target is OFF. Skipping plot.")
        return

    years = list(range(2021, 2021 + len(results.macro)))
    
    # --- 1. Data Extraction ---
    gross_emissions = results.macro['emissions_gross_total'].values
    net_emissions = results.macro['emissions_net_total'].values
    
    ccu_flow = results.macro['flow_reuse_recycle'].values
    ccs_flow = results.macro['flow_remove_ccs'].values
    nature_flow = results.macro['flow_remove_nature'].values
    
    # --- 2. Calculate Avoided Emissions ---
    
    # A. Hydrogen
    if 'hydrogen_domestic_use' in results.macro:
        h2_use_gj = results.macro['hydrogen_domestic_use'].values
        abated_h2 = (h2_use_gj * 0.06) / 1_000_000.0
    else:
        abated_h2 = np.zeros_like(gross_emissions)

    # B. Electrification (Net Benefit)
    abated_elec_net = np.zeros_like(gross_emissions)
    
    if getattr(p, 'enable_industrial_electrification', False):
        try:
            rate = p.electrification_annual_rate
            target_indices = pc.electrification_target_sectors
            
            actual_oil = results.sectoral['oil_use_final'][target_indices].sum(axis=1).values
            actual_gas = results.sectoral['gas_use_final'][target_indices].sum(axis=1).values
            
            # Reconstruct Baseline
            shift_pct_curve = np.array([1.0 - (1.0 - rate)**(t) for t in range(len(years))])
            divisor = 1.0 - shift_pct_curve
            divisor[divisor < 0.001] = 0.001
            
            base_oil = actual_oil / divisor
            base_gas = actual_gas / divisor
            
            # Savings
            direct_saved = ((base_oil - actual_oil) * pc.oil_emission_factor) + \
                           ((base_gas - actual_gas) * pc.gas_emission_factor)
            
            # Costs (Grid emissions added)
            cop = getattr(p, 'electrification_cop', 2.5)
            elec_added_gj = ((base_oil - actual_oil) + (base_gas - actual_gas)) / cop
            
            grid_ems = results.macro['electricity_emissions_total'].values
            grid_gen = results.macro['electricity_use_total'].values
            grid_intensity = np.zeros_like(grid_ems)
            np.divide(grid_ems, grid_gen, out=grid_intensity, where=grid_gen > 0)
            
            indirect_added = elec_added_gj * grid_intensity
            
            # Net Abatement
            abated_elec_net = np.maximum(0, direct_saved - indirect_added)
            
        except Exception as e:
            print(f"Plot Warning: Electrification calc error: {e}")

    # --- 3. Stack Data ---
    df_emissions = pd.DataFrame({
        'Net Remaining': net_emissions,
        'Nature (SGI)': nature_flow,
        'CCS': ccs_flow,
        'CCU': ccu_flow,
        'Hydrogen': abated_h2,
        'Electrification': abated_elec_net  # <--- New Wedge
    }, index=years)
    
    # --- 4. Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))
    colors_ems = ['#555555', '#2ca02c', '#1f77b4', '#9467bd', '#e377c2', '#ff7f0e']
    
    df_emissions.plot(kind='area', stacked=True, ax=ax, color=colors_ems, alpha=0.85, linewidth=0)
    
    # True Baseline Line
    true_baseline = gross_emissions + abated_h2 + abated_elec_net
    ax.plot(years, true_baseline, color='black', linestyle='--', linewidth=2.5, label='Baseline (No Action)')
    
    ax.set_title("Abatement Strategy: Pillars of Net Zero", fontsize=16, fontweight='bold')
    ax.set_ylabel("Emissions (MtCO2e)", fontsize=12)
    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_net_zero_investment(results: 'ModelResults'):
    """
    Shows annual investment in Net Zero technologies.
    Updated to include Industrial Electrification costs.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    pc = results.config.pc
    p = results.config.p
    if not p.net_emission_reduction_green_investments_exports_transformation: return

    years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000
    
    # --- Data Extraction ---
    # Sectoral Sums
    inv_ccu = results.sectoral['I_netzero_ccu'].sum(axis=1).values / TO_BILLION
    inv_ccs = results.sectoral['I_netzero_ccs'].sum(axis=1).values / TO_BILLION
    inv_nature = results.sectoral['I_netzero_nature'].sum(axis=1).values / TO_BILLION
    
    # Macro Variable (New Electrification Investment)
    if 'I_electrification_total' in results.macro:
        inv_elec = results.macro['I_electrification_total'].values / TO_BILLION
    else:
        inv_elec = np.zeros_like(inv_ccu)
    
    # --- Plotting ---
    df_investment = pd.DataFrame({
        'Nature (SGI)': inv_nature,
        'CCS Infrastructure': inv_ccs,
        'CCU / Recycling': inv_ccu,
        'Ind. Electrification': inv_elec # <--- New Bar
    }, index=years)
    
    # Calculate % of GDP
    gdp_nominal = results.macro['Y'].values / TO_BILLION
    total_nz_inv = df_investment.sum(axis=1)
    inv_as_pct_gdp = (total_nz_inv / gdp_nominal) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors: Green, Blue, Purple, Orange
    colors_inv = ['#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e']
    
    df_investment.plot(kind='bar', stacked=True, ax=ax, width=0.8, color=colors_inv, alpha=0.9)
    
    ax.set_title("Economic Contribution: Net Zero Investment Composition", fontsize=14, fontweight='bold')
    ax.set_ylabel("Investment (Billion SAR)", fontsize=12)
    
    # GDP Line
    ax_twin = ax.twinx()
    ax_twin.plot(years, inv_as_pct_gdp, color='red', linewidth=3, label='% of GDP')
    ax_twin.set_ylabel("Share of GDP (%)", color='red')
    ax_twin.tick_params(axis='y', labelcolor='red')
    ax_twin.set_ylim(0, max(inv_as_pct_gdp)*1.2)
    
    plt.tight_layout()
    plt.show()

def plot_carbon_market_cost(results: 'ModelResults'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    p = results.config.p
    
    # --- Safety Check 1: Is the policy on? ---
    if not p.net_emission_reduction_green_investments_exports_transformation:
        print("NOTE: Net Zero Target is FALSE. Skipping plot.")
        return
        
    # --- Safety Check 2: Is the Market Enabled? ---
    if not hasattr(p, 'enable_carbon_market') or not p.enable_carbon_market:
        print("NOTE: 'enable_carbon_market' is FALSE. No trading occurring.")
        return

    years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000
    
    # --- Data Extraction with Diagnostics ---
    try:
        # 1. COST (Buyers)
        # Ensure variable exists. If sums are 0, it means Net Emissions were <= 0.
        cost_credits = results.sectoral['I_carbon_credits'].sum(axis=1).values / TO_BILLION
        
        # 2. REVENUE (Sellers)
        # This key often causes errors if model_classes.py wasn't updated.
        if 'I_carbon_credit_revenue' in results.sectoral:
            rev_credits = results.sectoral['I_carbon_credit_revenue'].sum(axis=1).values / TO_BILLION
        else:
            print("WARNING: 'I_carbon_credit_revenue' missing from results. Did you add it to ModelVariables?")
            rev_credits = np.zeros_like(cost_credits)

    except KeyError as e:
        print(f"Critical Data Error: {e}")
        return

    # --- Diagnostic Print ---
    total_market_vol = np.sum(cost_credits)
    if total_market_vol == 0:
        print("!!! ALERT: Market Volume is ZERO. !!!")
        print("Reason: 'Net Emissions' reached 0 physically, so no 'Hard-to-Abate' gap remained.")
        print("Fix: Ensure the 'Hard-to-Abate' logic (scaling down physical removal) is active in model.py")
        # Return to avoid plotting empty axes, or continue to show empty plot as proof
        # return
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot Revenue (Positive Green Bars)
    ax.bar(years, rev_credits, color='#27AE60', alpha=0.8, label='Credit Revenue\n(Earned by CCS/Nature Operators)')
    
    # Plot Cost (Negative Red Bars)
    ax.bar(years, -cost_credits, color='#C0392B', alpha=0.8, label='Compliance Cost\n(Paid by Heavy Emitters)')
    
    ax.set_title("3. Market Mechanism: Financial Flows of Carbon Credits", fontsize=16, fontweight='bold')
    ax.set_ylabel("Billion SAR (+ Revenue / - Cost)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.axhline(0, color='black', linewidth=1.5)
    
    # Dynamic Info Box
    # Calculate the implied gap based on the last year
    try:
        net_em = results.macro['emissions_net_total'].values[-1]
        gross_em = results.macro['emissions_gross_total'].values[-1]
        gap_pct = (net_em / gross_em * 100) if gross_em > 0 else 0
    except:
        gap_pct = 0

    info_text = (f"Market Dynamics (2060):\n"
                 f"• Hard-to-Abate Gap: {gap_pct:.1f}%\n"
                 f"• Market Price: {p.carbon_credit_price_floor} SAR/t\n"
                 f"• Annual Volume: {cost_credits[-1]:.1f} Bn SAR")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray')
    ax.text(0.02, 0.82, info_text, transform=ax.transAxes, fontsize=11, bbox=props, zorder=5)
    
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Annotate the flow
    if total_market_vol > 0:
        ax.annotate('Wealth Transfer',
                    xy=(years[-5], 0),
                    xytext=(years[-8], max(rev_credits)*0.5),
                    arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=.2"),
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_net_zero_balance_check(results: 'ModelResults'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    p = results.config.p
    if not p.net_emission_reduction_green_investments_exports_transformation or not p.enable_carbon_market:
        print("Net Zero or Market is OFF. Cannot plot balance.")
        return

    years = list(range(2021, 2021 + len(results.macro)))
    
    # --- 1. DATA RECONSTRUCTION ---
    # We reconstruct the logic to show the exact balance
    
    # A. The Problem: Residual Emissions (After CCS/Nature, but before Market)
    # In your model, 'emissions_net_total' is currently the residual before market offset
    residual_emissions = results.macro['emissions_net_total'].values
    
    # B. The Constraint: Market Liquidity Cap
    liquidity_cap = p.carbon_market_liquidity
    
    # C. The Solution: Credits Purchased
    # Logic: Buy enough to cover residual, but capped by liquidity
    credits_purchased = np.minimum(residual_emissions, liquidity_cap)
    
    # D. The Result: Final Net Position
    final_net_position = residual_emissions - credits_purchased

    # --- 2. PLOTTING ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot 1: The Liability (Positive Bar)
    ax.bar(years, residual_emissions, color='grey', alpha=0.6, label='Residual "Hard-to-Abate" Emissions')
    
    # Plot 2: The Offset (Negative Bar)
    ax.bar(years, -credits_purchased, color='#27AE60', alpha=0.8, label='Carbon Credits Purchased (Offset)')
    
    # Plot 3: The Net Result (Line)
    # We make this line thicker and colorful to show the success/failure
    ax.plot(years, final_net_position, color='black', linewidth=3, marker='o', markersize=5, label='Final Net Position')
    
    # --- 3. FORMATTING ---
    ax.set_title("The Net Zero Equation: Balancing Residual Emissions with Market Offsets", fontsize=16, fontweight='bold')
    ax.set_ylabel("Emissions Balance (MtCO2e)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.axhline(0, color='black', linewidth=1)
    
    # Highlight the Net Zero Zone
    ax.axhspan(-5, 5, color='green', alpha=0.1)
    ax.text(years[2], 2, "Net Zero Target Zone", color='green', fontweight='bold', fontsize=10)
    
    # Check for Liquidity Crunch
    if final_net_position[-1] > 1.0:
        # If line is above 0, we failed
        fail_gap = final_net_position[-1]
        ax.annotate(f'Liquidity Shortage!\nGap: {fail_gap:.1f} Mt',
                    xy=(years[-1], fail_gap),
                    xytext=(years[-1]-5, fail_gap + 20),
                    arrowprops=dict(facecolor='red', arrowstyle='->'),
                    bbox=dict(boxstyle="round", fc="#ffcccc"), fontsize=10, color='red')
    else:
        # Success
        ax.annotate('Net Zero Achieved',
                    xy=(years[-1], 0),
                    xytext=(years[-1]-5, 50),
                    arrowprops=dict(facecolor='green', arrowstyle='->'),
                    bbox=dict(boxstyle="round", fc="#ccffcc"), fontsize=10, color='green')

    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()

def plot_nonoil_gdp_comparison(res_base, res_nz):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    pc = res_base.config.pc
    years = list(range(2021, 2021 + len(res_base.macro)))
    
    # --- 1. Identify Non-Oil Sectors (ROBUST METHOD) ---
    oil_names = [
        'Extraction of crude petroleum and natural gas',
        'Manufacture of coke and refined petroleum products'
    ]
    
    # Strict string matching
    oil_indices = []
    for idx, s in enumerate(pc.sectors):
        if any(name in s for name in oil_names):
            oil_indices.append(idx)
            
    # Create Boolean Mask
    is_nonoil = np.ones(pc.S, dtype=bool)
    is_nonoil[oil_indices] = False
    
    # --- 2. Helper Function ---
    def get_nonoil_gdp(results):
        # We use Y_ (Final Demand) as the GDP proxy in this Demand-Led model
        Y_matrix = results.sectoral['Y']
        # Sum only non-oil columns
        return Y_matrix.loc[:, is_nonoil].sum(axis=1)

    # --- 3. Calculate Vectors ---
    gdp_base = get_nonoil_gdp(res_base).values / 1e9
    gdp_nz = get_nonoil_gdp(res_nz).values / 1e9
    
    # Calculate Growth Rates
    with np.errstate(divide='ignore', invalid='ignore'):
        gr_base = (np.diff(gdp_base) / gdp_base[:-1]) * 100
        gr_nz = (np.diff(gdp_nz) / gdp_nz[:-1]) * 100
    
    years_gr = years[1:]
    gdp_diff = gdp_nz - gdp_base

    # --- 4. Plotting ---
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    fig.suptitle("Impact of Net Zero Strategy on Non-Oil Economy", fontsize=18, fontweight='bold')

    # Plot 1: Growth Rates
    ax1.plot(years_gr, gr_base, color='grey', linestyle='--', linewidth=2, label='Baseline Scenario')
    ax1.plot(years_gr, gr_nz, color='#2ca02c', linewidth=3, marker='o', label='Net Zero Scenario')
    ax1.fill_between(years_gr, gr_base, gr_nz, color='#2ca02c', alpha=0.1, label='Green Growth Boost')
    
    ax1.set_title("Non-Oil GDP Annual Growth Rate", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Annual Growth (%)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    # Plot 2: The Dividend (Difference)
    colors = ['#2ca02c' if x >= 0 else '#d62728' for x in gdp_diff]
    ax2.bar(years, gdp_diff, color=colors, alpha=0.8)
    
    ax2.set_title("The 'Green Dividend': Additional Non-Oil GDP vs. Baseline", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Additional GDP (Billion SAR)", fontsize=12)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.grid(axis='y', alpha=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_total_gdp_impact(res_base, res_nz):

    # --- 1. Setup Data ---
    years = list(range(2021, 2021 + len(res_base.macro)))
    TO_BILLION = 1_000_000_000 # Divide by 10^9
    
    # Extract Total GDP (Variable 'Y' in macro results)
    gdp_base = res_base.macro['Y'].values / TO_BILLION
    gdp_nz = res_nz.macro['Y'].values / TO_BILLION
    
    # Calculate the Change (The Dividend)
    # Positive = Net Zero is higher
    gdp_diff = gdp_nz - gdp_base
    
    # Calculate Cumulative Difference (Total wealth added over the period)
    total_added_wealth = np.sum(gdp_diff)

    # --- 2. Plotting ---
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=True)
    fig.suptitle("Macroeconomic Impact: Total GDP Comparison", fontsize=18, fontweight='bold')

    # --- TOP PLOT: GDP Levels ---
    ax1.plot(years, gdp_base, color='grey', linestyle='--', linewidth=2, label='Baseline (No Net Zero)')
    ax1.plot(years, gdp_nz, color='#1f77b4', linewidth=3, marker='o', label='Net Zero Scenario')
    
    # Fill the gap to show the gain
    ax1.fill_between(years, gdp_base, gdp_nz, where=(gdp_nz > gdp_base),
                     color='#2ca02c', alpha=0.1, label='Net Economic Gain')
    ax1.fill_between(years, gdp_base, gdp_nz, where=(gdp_nz < gdp_base),
                     color='#d62728', alpha=0.1, label='Net Economic Loss')
    
    ax1.set_title("Total Nominal GDP Trajectory", fontsize=14, fontweight='bold')
    ax1.set_ylabel("GDP (Billion SAR)", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Annotate final value
    ax1.text(years[-1], gdp_nz[-1], f"{gdp_nz[-1]:.0f} Bn", color='#1f77b4', fontweight='bold', ha='left')

    # --- BOTTOM PLOT: The Change (Difference) ---
    # Green bars = Gain, Red bars = Loss
    colors = ['#2ca02c' if x >= 0 else '#d62728' for x in gdp_diff]
    
    bars = ax2.bar(years, gdp_diff, color=colors, alpha=0.85)
    
    ax2.set_title("Change in GDP vs. Baseline (The Net Zero Dividend)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Change in GDP (Billion SAR)", fontsize=12)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add Info Box
    info_text = (f"Economic Summary (2021-2060):\n"
                 f"• Cumulative Wealth Added: {total_added_wealth:.1f} Bn SAR\n"
                 f"• Avg Annual Boost: {np.mean(gdp_diff):.1f} Bn SAR")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax2.text(0.02, 0.85, info_text, transform=ax2.transAxes, fontsize=11, bbox=props)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def print_oil_sector_mapping(results):
    pc = results.config.pc
    
    # Get indices of target sectors
    target_indices = list(pc.oil_reduction_sectors_idx)
    
    print(f"--- Sector Mapping Report ({len(target_indices)} Sectors) ---")
    
    for idx in target_indices:
        sector_name = pc.sectors[idx]
        
        # Logic used in the plot
        if "Electricity" in sector_name or "Desalination" in sector_name:
            cat = "Power & Desalination"
        elif "Crop" in sector_name or "animal" in sector_name or "Fishing" in sector_name:
            cat = "Agriculture"
        else:
            cat = "Industries"  # <--- This is what you asked for
            
        print(f"[{cat}] -> {sector_name}")

# Run it
def plot_oil_sector_mapping_visual(results: 'ModelResults'):
    import plotly.graph_objects as go
    
    pc = results.config.pc
    p = results.config.p
    
    # --- 1. GET SECTORS ---
    # We only care about the sectors flagged for reduction
    target_indices = list(pc.oil_reduction_sectors_idx)
    sector_names = [pc.sectors[i] for i in target_indices]
    
    # --- 2. DEFINE MAPPING LOGIC ---
    categories = ["Power & Desalination", "Agriculture", "Industries"]
    
    # Lists for Plotly
    source_indices = [] # 0, 1, or 2
    target_indices = [] # 3 to 35
    colors = []
    
    # Color Palette
    cat_colors = {
        "Power & Desalination": "#1f77b4",  # Blue
        "Agriculture": "#2ca02c",          # Green
        "Industries": "#d62728"            # Red
    }

    # --- 3. ASSIGN CATEGORIES ---
    for i, sector in enumerate(sector_names):
        
        # A. Default Logic (Keyword based)
        if "Electricity" in sector or "Desalination" in sector:
            cat = "Power & Desalination"
        elif "Crop" in sector or "animal" in sector or "Fishing" in sector:
            cat = "Agriculture"
        else:
            cat = "Industries" # Default bucket

        # B. Manual overrides: correct any mis-mapped sector here.
        
        # --- BUILD LINKS ---
        src_idx = categories.index(cat)
        tgt_idx = len(categories) + i # Offset by 3 categories
        
        source_indices.append(src_idx)
        target_indices.append(tgt_idx)
        # Make link semi-transparent version of category color
        colors.append(cat_colors[cat].replace(")", ", 0.4)").replace("rgb", "rgba"))

    # --- 4. CREATE PLOT ---
    # Create labels list: [Cats] + [Sectors]
    all_labels = categories + sector_names
    
    # Node colors
    node_colors = [cat_colors[c] for c in categories] + ["grey"] * len(sector_names)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=[1] * len(sector_names), # Value 1 for equal visual weight
            color=colors,
            hovertemplate='<b>%{source.label}</b> &rarr; %{target.label}<extra></extra>'
        )
    )])

    fig.update_layout(
        title_text="<b>Sector Classification Check</b><br>Visualizing how the model groups the 33 sectors",
        font_size=12,
        height=1000, # Tall height to fit all 33 labels
        width=1200
    )

    fig.show()
    import plotly.graph_objects as go

import plotly.graph_objects as go

def plot_gcam_to_ken_sankey(results):
    # --- 1. DEFINE GCAM SECTORS ---
    gcam_sectors = [
        "N fertilizer", "agricultural energy use", "alumina", "aluminum", "cement",
        "chemical energy use", "chemical feedstocks", "comm cooling", "comm others",
        "construction energy use", "construction feedstocks", "desalinated water",
        "industrial wastewater treatment", "industrial water abstraction", 
        "industrial water treatment", "iron and steel", "irrigation water abstraction",
        "mining energy use", "municipal wastewater treatment", "municipal water abstraction",
        "municipal water distribution", "municipal water treatment", "other industrial energy use",
        "other industrial feedstocks", "process heat cement", "resid cooling", "resid heating",
        "resid others", "trn_aviation_intl", "trn_freight", "trn_freight_road", "trn_pass",
        "trn_pass_road", "trn_pass_road_LDV", "trn_pass_road_LDV_4W", "trn_shipping_intl"
    ]

    # --- 2. DEFINE UPDATED MAPPING (BASED ON YOUR ISIC/IOT LOGIC) ---
    mapping = {
        "N fertilizer": ["Manufacture of chemicals and chemical products"],
        "agricultural energy use": ["Crop and animal production, hunting and related service activities"],
        "alumina": ["Manufacture of basic metals"],
        "aluminum": ["Manufacture of basic metals"],
        "cement": ["Manufacture of other non-metallic mineral products"],
        "chemical energy use": ["Manufacture of chemicals and chemical products"],
        "chemical feedstocks": ["Manufacture of chemicals and chemical products"],
        "comm cooling": ["Wholesale trade, except of motor vehicles and motorcycles", "Retail trade, except of motor vehicles and motorcycles", "Accommodation", "Food and beverage service activities", "Financial service activities, except insurance and pension funding", "Real estate activities", "Public administration and defence; compulsory social security", "Education", "Human health activities"],
        "comm others": ["Telecommunications", "Computer programming, consultancy and related activities", "Information service activities", "Legal and accounting activities", "Scientific research and development", "Other professional, scientific and technical activities"],
        "construction energy use": ["Construction of buildings", "Civil engineering", "Specialized construction activities"],
        "construction feedstocks": ["Construction of buildings", "Civil engineering", "Specialized construction activities"],
        "desalinated water": ["Water collection, treatment and supply"],
        "industrial wastewater treatment": ["Sewerage", "Waste collection, treatment and disposal activities; materials recovery"],
        "industrial water abstraction": ["Water collection, treatment and supply"],
        "industrial water treatment": ["Water collection, treatment and supply"],
        "iron and steel": ["Manufacture of basic metals"],
        "irrigation water abstraction": ["Water collection, treatment and supply"],
        "mining energy use": ["Mining of coal and lignite", "Other mining and quarrying activities", "Mining support service activities"],
        "municipal wastewater treatment": ["Sewerage", "Waste collection, treatment and disposal activities; materials recovery"],
        "municipal water abstraction": ["Water collection, treatment and supply"],
        "municipal water distribution": ["Water collection, treatment and supply"],
        "municipal water treatment": ["Water collection, treatment and supply"],
        "other industrial energy use": ["Manufacture of food products", "Manufacture of beverages", "Manufacture of tobacco products", "Manufacture of textiles", "Manufacture of wearing apparel", "Manufacture of leather and related products", "Manufacture of woods, wood products and cork except furniture", "Manufacture of paper and paper products", "Printing and reproduction of recorded media", "Manufacture of coke and refined petroleum products", "Manufacture of basic pharmaceutical products and pharmaceutical preparations", "Manufacture of rubber and plastics products", "Manufacture of electrical equipment", "Manufacture of machinery and equipment n.e.c.", "Manufacture of motor vehicles, trailers and semi-trailers", "Manufacture of other transport equipment", "Manufacture of furniture", "Other manufacturing"],
        "other industrial feedstocks": ["Manufacture of chemicals and chemical products", "Manufacture of basic pharmaceutical products and pharmaceutical preparations", "Manufacture of rubber and plastics products"],
        "process heat cement": ["Manufacture of other non-metallic mineral products"],
        "resid cooling": ["Other personal service activities", "Repair of computers and personal and household goods"],
        "resid heating": ["Other personal service activities"],
        "resid others": ["Other personal service activities"],
        "trn_aviation_intl": ["Air transport"],
        "trn_freight": ["Land transport and transport via pipelines", "Water transport", "Air transport"],
        "trn_freight_road": ["Land transport and transport via pipelines"],
        "trn_pass": ["Land transport and transport via pipelines", "Water transport", "Air transport"],
        "trn_pass_road": ["Land transport and transport via pipelines"],
        "trn_pass_road_LDV": ["Land transport and transport via pipelines"],
        "trn_pass_road_LDV_4W": ["Land transport and transport via pipelines"],
        "trn_shipping_intl": ["Water transport"]
    }

    # --- 3. BUILD NODES AND LINKS ---
    ken_sectors_in_map = sorted(list(set([item for sublist in mapping.values() for item in sublist])))
    all_labels = gcam_sectors + ken_sectors_in_map
    
    sources = []
    targets = []
    values = []

    for gcam_idx, g_sec in enumerate(gcam_sectors):
        if g_sec in mapping:
            targets_for_gcam = mapping[g_sec]
            weight = 1.0 / len(targets_for_gcam)
            for k_sec in targets_for_gcam:
                sources.append(gcam_idx)
                targets.append(len(gcam_sectors) + ken_sectors_in_map.index(k_sec))
                values.append(weight)

    # --- 4. CREATE PLOT ---
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=10,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color="#2E4053"  # Professional dark slate
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(46, 64, 83, 0.25)" 
        )
    )])

    fig.update_layout(
        title_text="<b>Data Alignment: GCAM-KSA Sectors to KEN Model Sectors</b>",
        font_size=11,
        height=1400, # Increased height to ensure all labels are readable
        width=1200
    )

    fig.show()

# --- Run ---
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_sfc_money_circulation(results: 'ModelResults'):
    """
    Ultra-Detailed Sankey Diagram showing the SFC Money Circulation.
    
    LEVELS:
    1. Sources: Govt Oil Revenue, Private Equity (Chemicals, Utilities, Manufacturing).
    2. Projects: Hydrogen, Batteries, Renewables, CCS, Nature, Electrification, Efficiency.
    3. Recipients: Specific IO Sectors (Civil Eng, Machinery n.e.c, Electrical Equip, etc.)
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    pc = results.config.pc
    p = results.config.p
    if not getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False):
        print("Net Zero target not active. Skipping SFC Sankey.")
        return

    years_list = [2030, 2040, 2050, 2060]
    model_years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000

    # --- Node Colors (Distinct Palette) ---
    c_gov = "rgba(31, 119, 180, 0.8)"    # Blue
    c_chem = "rgba(148, 103, 189, 0.8)"  # Purple
    c_util = "rgba(188, 189, 34, 0.8)"   # Olive
    c_manf = "rgba(255, 127, 14, 0.8)"   # Orange
    
    c_proj = "rgba(128, 128, 128, 0.5)"  # Grey
    
    c_civil = "rgba(140, 86, 75, 0.8)"   # Brown
    c_bldg  = "rgba(196, 156, 148, 0.8)" # Light Brown
    c_mach  = "rgba(214, 39, 40, 0.8)"   # Red
    c_elec  = "rgba(255, 152, 150, 0.8)" # Light Red
    c_serv  = "rgba(44, 160, 44, 0.8)"   # Green
    c_imp   = "rgba(0, 0, 0, 0.6)"       # Black (Leakage)

    for year in years_list:
        if year not in model_years: continue
        t_idx = model_years.index(year)
        
        # ======================================================================
        # 1. EXTRACT DATA
        # ======================================================================
        inv_h2 = results.macro.get('I_hydrogen', pd.Series(0)).iloc[t_idx] / TO_BILLION
        inv_batt = results.macro.get('I_storage', pd.Series(0)).iloc[t_idx] / TO_BILLION
        inv_re = results.macro.get('I_RE', pd.Series(0)).iloc[t_idx] / TO_BILLION
        inv_ccs = results.sectoral['I_netzero_ccs'].iloc[t_idx].sum() / TO_BILLION
        inv_nature = results.sectoral['I_netzero_nature'].iloc[t_idx].sum() / TO_BILLION
        inv_elec = results.macro.get('I_electrification_total', pd.Series(0)).iloc[t_idx] / TO_BILLION
        inv_effic = results.macro.get('I_efficiency_total', pd.Series(0)).iloc[t_idx] / TO_BILLION
        
        total_inv = inv_h2 + inv_batt + inv_re + inv_ccs + inv_nature + inv_elec + inv_effic
        if total_inv < 0.1: continue

        # ======================================================================
        # 2. LOCALIZATION & SHARES
        # ======================================================================
        loc_h2 = min(0.7, 0.3 + (0.015 * (year - 2021)))
        loc_re = min(0.8, 0.4 + (0.02 * (year - 2021)))
        
        if 'storage_local_share' in results.macro:
             loc_batt = results.macro['storage_local_share'].iloc[t_idx]
        else:
             loc_batt = 0.0 if year <= 2030 else min(1.0, (year - 2030) / 20.0)

        # ======================================================================
        # 3. NODE DEFINITIONS
        # ======================================================================
        # Sources (0-3)
        # Projects (4-10)
        # Recipients (11-18)
        
        labels = [
            "<b>Govt Revenues</b><br>(Oil/PIF)",            # 0
            "<b>Priv: Chemicals</b><br>(Equity)",           # 1
            "<b>Priv: Utilities</b><br>(Equity)",           # 2
            "<b>Priv: Manufacturing</b><br>(Equity)",       # 3
            
            f"Green H2<br>({inv_h2:.1f}B)",                 # 4
            f"Batteries<br>({inv_batt:.1f}B)",              # 5
            f"Renewables<br>({inv_re:.1f}B)",               # 6
            f"CCS Infra<br>({inv_ccs:.1f}B)",               # 7
            f"Nature (SGI)<br>({inv_nature:.1f}B)",         # 8
            f"Electrification<br>({inv_elec:.1f}B)",        # 9
            f"Efficiency<br>({inv_effic:.1f}B)",            # 10
            
            "<b>Civil Engineering</b>",                     # 11
            "<b>Specialized Const.</b>",                    # 12
            "<b>Mfr Machinery n.e.c</b>",                   # 13
            "<b>Mfr Electrical Equip</b>",                  # 14
            "<b>Mfr Basic Metals</b>",                      # 15
            "<b>Engineering Svcs</b>",                      # 16
            "<b>Agri & Other Svcs</b>",                     # 17
            "<b>Import Leakage</b>"                         # 18
        ]
        
        node_colors = [c_gov, c_chem, c_util, c_manf] + \
                      [c_proj]*7 + \
                      [c_civil, c_bldg, c_mach, c_elec, c_mach, c_serv, c_serv, c_imp]

        source, target, value = [], [], []

        # ======================================================================
        # 4. FLOWS: SOURCES -> PROJECTS
        # ======================================================================
        # 1. Hydrogen (80% Gov, 20% Chem Equity)
        source.extend([0, 1]); target.extend([4, 4]); value.extend([inv_h2*0.8, inv_h2*0.2])
        
        # 2. Batteries (50% Gov, 50% Util Equity)
        source.extend([0, 2]); target.extend([5, 5]); value.extend([inv_batt*0.5, inv_batt*0.5])
        
        # 3. Renewables (50% Gov, 50% Util Equity)
        source.extend([0, 2]); target.extend([6, 6]); value.extend([inv_re*0.5, inv_re*0.5])
        
        # 4. CCS (100% Gov - Public Infra)
        source.append(0); target.append(7); value.append(inv_ccs)
        
        # 5. Nature (100% Gov - SGI)
        source.append(0); target.append(8); value.append(inv_nature)
        
        # 6. Electrification (100% Mfg Equity - Self funded)
        source.append(3); target.append(9); value.append(inv_elec)
        
        # 7. Efficiency (100% Mfg Equity - Self funded)
        source.append(3); target.append(10); value.append(inv_effic)

        # ======================================================================
        # 5. FLOWS: PROJECTS -> RECIPIENT SECTORS
        # ======================================================================
        
        def add_detailed_flow(proj_node, amt, loc, split_dict):
            """
            split_dict: {target_node_idx: share_of_local}
            """
            local = amt * loc
            imp = amt * (1 - loc)
            
            # Distribute Local
            for node_idx, share in split_dict.items():
                if share > 0:
                    source.append(proj_node)
                    target.append(node_idx)
                    value.append(local * share)
            
            # Import Leakage
            source.append(proj_node); target.append(18); value.append(imp)

        # Node Indices:
        # 11: Civil Eng, 12: Spec Const, 13: Mach n.e.c, 14: Elec Equip
        # 15: Basic Metals, 16: Eng Svcs, 17: Agri/Other
        
        # 1. Hydrogen (Heavy Civil & Process Machinery)
        # 40% Civil, 20% Spec Const, 30% Machinery, 10% Eng Svcs
        add_detailed_flow(4, inv_h2, loc_h2, {11: 0.4, 12: 0.2, 13: 0.3, 16: 0.1})
        
        # 2. Batteries (Electrical Equip intensive)
        # 10% Spec Const (Install), 90% Electrical Equip (Cells/Packs)
        add_detailed_flow(5, inv_batt, loc_batt, {12: 0.1, 14: 0.9})
        
        # 3. Renewables (Civil + Mach + Elec)
        # 30% Civil (Site), 10% Spec Const, 30% Mach (Turbines), 20% Elec (Panels), 10% Eng Svcs
        add_detailed_flow(6, inv_re, loc_re, {11: 0.3, 12: 0.1, 13: 0.3, 14: 0.2, 16: 0.1})
        
        # 4. CCS (Pipelines & Injection)
        # 50% Civil, 30% Spec Const (Piping), 10% Machinery, 10% Eng Svcs
        add_detailed_flow(7, inv_ccs, 0.85, {11: 0.5, 12: 0.3, 13: 0.1, 16: 0.1})
        
        # 5. Nature (Agri & Land Services)
        # 90% Agri/Other, 10% Eng Svcs (Planning)
        add_detailed_flow(8, inv_nature, 0.95, {17: 0.9, 16: 0.1})
        
        # 6. Electrification (Machinery & Install)
        # 20% Spec Const (Install), 60% Machinery (Motors/Pumps), 20% Electrical
        add_detailed_flow(9, inv_elec, 0.6, {12: 0.2, 13: 0.6, 14: 0.2})
        
        # 7. Efficiency (Retrofits)
        # 40% Spec Const (Insulation), 30% Machinery, 20% Elec, 10% Eng Svcs
        add_detailed_flow(10, inv_effic, 0.6, {12: 0.4, 13: 0.3, 14: 0.2, 16: 0.1})

        # ======================================================================
        # 6. PLOT
        # ======================================================================
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15, thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color="rgba(180, 180, 180, 0.3)"
            )
        )])

        fig.update_layout(
            title_text=f"<b>{year} Detailed Money Circulation:</b> Sources -> Projects -> Specific Industries<br>Total Investment: {total_inv:.1f} Bn SAR",
            font_size=11,
            height=750
        )
        fig.show()
def plot_cumulative_transition_sankey(results: 'ModelResults'):
    """
    Aggregates ALL investment flows from 2025 to 2060 into a single Master Sankey.
    Visualizes the 'Total Fiscal Transfer' from Oil to the Green Economy over 35 years.
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    p = results.config.p
    if not getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False): return

    model_years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000
    
    # Initialize Cumulative Counters
    # We track where the money ended up over 35 years
    cum_h2_const = 0; cum_h2_mach = 0; cum_h2_imp = 0
    cum_batt_elec = 0; cum_batt_imp = 0
    cum_re_const = 0; cum_re_elec = 0; cum_re_mach = 0; cum_re_imp = 0
    cum_ccs_const = 0; cum_ccs_mach = 0; cum_ccs_imp = 0
    cum_nat_serv = 0; cum_nat_imp = 0
    
    total_oil_funding = 0

    # Iterate from 2025 to 2060 to sum up flows
    for year in range(2025, 2061):
        if year not in model_years: continue
        t_idx = model_years.index(year)
        
        # --- 1. Extract Annual Investment ---
        inv_h2 = results.macro.get('I_hydrogen', pd.Series(0)).iloc[t_idx] / TO_BILLION
        inv_batt = results.macro.get('I_storage', pd.Series(0)).iloc[t_idx] / TO_BILLION
        inv_re = results.macro.get('I_RE', pd.Series(0)).iloc[t_idx] / TO_BILLION
        inv_ccs = results.sectoral['I_netzero_ccs'].iloc[t_idx].sum() / TO_BILLION
        inv_nature = results.sectoral['I_netzero_nature'].iloc[t_idx].sum() / TO_BILLION
        
        total_oil_funding += (inv_h2 + inv_batt + inv_re + inv_ccs + inv_nature)

        # --- 2. Determine Dynamic Localization Rates (The "Industrial Policy" Curve) ---
        # As years pass, leakage decreases and local capture increases
        
        # Hydrogen: Ramps 30% -> 70% local
        h2_loc = min(0.7, 0.3 + (0.015 * (year - 2021)))
        
        # Batteries: Ramps 0% -> 100% (Based on your model logic)
        if 'storage_local_share' in results.macro:
             batt_loc = results.macro['storage_local_share'].iloc[t_idx]
        else:
             batt_loc = 0.0 if year < 2030 else min(1.0, (year - 2030) / 20.0)
             
        # Renewables: Ramps 40% -> 80%
        re_loc = min(0.8, 0.4 + (0.012 * (year - 2021)))
        
        # CCS: High local content (Civil Works) ~80%
        ccs_loc = 0.8
        
        # Nature: High local content (Labor) ~90%
        nat_loc = 0.9

        # --- 3. Accumulate Flows to Sectors ---
        
        # Hydrogen
        cum_h2_imp += inv_h2 * (1 - h2_loc)
        cum_h2_const += inv_h2 * h2_loc * 0.6  # 60% of local H2 spend is construction
        cum_h2_mach  += inv_h2 * h2_loc * 0.4  # 40% is machinery
        
        # Batteries
        cum_batt_imp += inv_batt * (1 - batt_loc)
        cum_batt_elec += inv_batt * batt_loc   # 100% to Electrical Mfg
        
        # Renewables
        cum_re_imp += inv_re * (1 - re_loc)
        cum_re_const += inv_re * re_loc * 0.4
        cum_re_elec += inv_re * re_loc * 0.3
        cum_re_mach += inv_re * re_loc * 0.3
        
        # CCS
        cum_ccs_imp += inv_ccs * (1 - ccs_loc)
        cum_ccs_const += inv_ccs * ccs_loc * 0.6
        cum_ccs_mach += inv_ccs * ccs_loc * 0.4
        
        # Nature
        cum_nat_imp += inv_nature * (1 - nat_loc)
        cum_nat_serv += inv_nature * nat_loc

    # --- 4. Build Sankey Data ---
    
    # Indices:
    # 0: Total Oil Surplus
    # 1: Hydrogen Projects
    # 2: Battery Projects
    # 3: Renewable Projects
    # 4: CCS Projects
    # 5: Nature Projects
    # 6: Construction Sector
    # 7: Manufacturing (Machinery)
    # 8: Manufacturing (Electrical)
    # 9: Services & Agri
    # 10: Import Leakage (Foreign Economy)

    labels = [
        f"<b>Cumulative Oil Surplus</b><br>(2025-2060)<br>{total_oil_funding:.0f} Bn SAR", # 0
        "Green Hydrogen",           # 1
        "Battery Storage",          # 2
        "Renewables (Solar/Wind)",  # 3
        "CCS Infrastructure",       # 4
        "Saudi Green Initiative",   # 5
        "<b>Construction Sector</b><br>(Local)", # 6
        "<b>Machinery Mfg</b><br>(Local)",       # 7
        "<b>Electrical Mfg</b><br>(Local)",      # 8
        "<b>Services & Agri</b><br>(Local)",     # 9
        "<b>Import Leakage</b><br>(Foreign)"     # 10
    ]
    
    # Colors
    c_oil = "rgba(31, 119, 180, 0.8)"   # Blue
    c_proj = "rgba(255, 127, 14, 0.6)"  # Orange
    c_local = "rgba(44, 160, 44, 0.8)"  # Green (Success)
    c_imp = "rgba(214, 39, 40, 0.6)"    # Red (Leakage)
    
    node_colors = [c_oil] + [c_proj]*5 + [c_local]*4 + [c_imp]
    
    source = []
    target = []
    value = []
    
    # Flow 1: Oil -> Projects (Totals)
    # Sum up the cumulatives to get total project size
    tot_h2 = cum_h2_imp + cum_h2_const + cum_h2_mach
    tot_batt = cum_batt_imp + cum_batt_elec
    tot_re = cum_re_imp + cum_re_const + cum_re_elec + cum_re_mach
    tot_ccs = cum_ccs_imp + cum_ccs_const + cum_ccs_mach
    tot_nat = cum_nat_imp + cum_nat_serv
    
    source.extend([0, 0, 0, 0, 0])
    target.extend([1, 2, 3, 4, 5])
    value.extend([tot_h2, tot_batt, tot_re, tot_ccs, tot_nat])
    
    # Flow 2: Projects -> Sectors
    
    # Hydrogen
    source.extend([1, 1, 1])
    target.extend([6, 7, 10])
    value.extend([cum_h2_const, cum_h2_mach, cum_h2_imp])
    
    # Batteries
    source.extend([2, 2])
    target.extend([8, 10])
    value.extend([cum_batt_elec, cum_batt_imp])
    
    # Renewables
    source.extend([3, 3, 3, 3])
    target.extend([6, 8, 7, 10])
    value.extend([cum_re_const, cum_re_elec, cum_re_mach, cum_re_imp])
    
    # CCS
    source.extend([4, 4, 4])
    target.extend([6, 7, 10])
    value.extend([cum_ccs_const, cum_ccs_mach, cum_ccs_imp])
    
    # Nature
    source.extend([5, 5])
    target.extend([9, 10])
    value.extend([cum_nat_serv, cum_nat_imp])

    # Plot
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(180, 180, 180, 0.4)"
        )
    )])
    
    fig.update_layout(
        title_text="<b>Total Economic Transformation (2025-2060)</b><br>Cumulative Flow of Oil Revenues into the Non-Oil Economy",
        font_size=12,
        height=700
    )
    fig.show()

def plot_net_zero_sfc_mechanism(results: 'ModelResults'):
    """
    Sankey Diagram showing the flow of funds from Policy -> Sectors -> Income.
    Updated to include Electrification Investment flows.
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    pc = results.config.pc
    p = results.config.p
    if not p.net_emission_reduction_green_investments_exports_transformation: return

    # Year 2060 Snapshot
    last_step = -1
    TO_BILLION = 1_000_000
    
    # --- 1. Investment Sources ---
    inv_ccu = results.sectoral['I_netzero_ccu'].iloc[last_step].sum() / TO_BILLION
    inv_ccs = results.sectoral['I_netzero_ccs'].iloc[last_step].sum() / TO_BILLION
    inv_nature = results.sectoral['I_netzero_nature'].iloc[last_step].sum() / TO_BILLION
    
    # Get Electrification Investment
    inv_elec = 0
    if 'I_electrification_total' in results.macro:
        inv_elec = results.macro['I_electrification_total'].iloc[last_step] / TO_BILLION

    # --- 2. Define Nodes ---
    labels = [
        "<b>Net Zero Strategy</b>",    # 0
        "Chemicals (CCU)",             # 1
        "Oil/Mining (CCS)",            # 2
        "Agri (Nature)",               # 3
        "Industry (Retrofits)",        # 4 <--- New Node
        "<b>Real Economy</b>",         # 5
        "Construction Sector",         # 6
        "Manufacturing Sector",        # 7
        "Services & Others"            # 8
    ]
    
    source, target, value, color = [], [], [], []
    c_inv = "rgba(31, 119, 180, 0.6)"
    c_inc = "rgba(44, 160, 44, 0.6)"

    # --- 3. Flow Logic ---
    
    # A. Strategy -> Specific Investment types
    source += [0, 0, 0, 0]
    target += [1, 2, 3, 4]
    value += [inv_ccu, inv_ccs, inv_nature, inv_elec]
    color += [c_inv] * 4
    
    # B. Investment -> Real Economy (Who receives the income?)
    # CCU, CCS, Nature -> General Economy
    source += [1, 2, 3]
    target += [5, 5, 5]
    value += [inv_ccu, inv_ccs, inv_nature]
    color += [c_inc] * 3
    
    # Electrification -> General Economy
    source.append(4); target.append(5); value.append(inv_elec); color.append(c_inc)
    
    # C. Real Economy -> Specific Sectors
    # Calculate shares based on dI vector
    total_stimulus = inv_ccu + inv_ccs + inv_nature + inv_elec
    
    const_idx = [i for i, s in enumerate(pc.sectors) if 'Construction' in s]
    manu_idx = [i for i, s in enumerate(pc.sectors) if 'Manufacture' in s]
    
    # Approx distribution
    share_const = np.sum(pc.dI[const_idx])
    share_manu = np.sum(pc.dI[manu_idx])
    
    val_const = total_stimulus * share_const
    val_manu = total_stimulus * share_manu
    val_serv = total_stimulus - (val_const + val_manu)
    
    source += [5, 5, 5]
    target += [6, 7, 8]
    value += [val_const, val_manu, val_serv]
    color += [c_inc] * 3

    # --- 4. Plot ---
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color="lightgrey"),
        link=dict(source=source, target=target, value=value, color=color)
    )])
    
    fig.update_layout(title_text="SFC Mechanism: How Net Zero Investment Flows to the Economy (2060)", font_size=14)
    fig.show()
import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_netzero_finance(results: 'ModelResults', T_steps=None, title_suffix=""):
    """
    Plots cumulative financial flows, including Electrification.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Helper to get macro or sectoral sum
    def get_val(key):
        if hasattr(results.macro, key): # Check macro first
             return results.macro[key].sum()
        elif key in results.macro: # Check dataframe column
             return results.macro[key].sum()
        elif hasattr(results, key): # Check ModelResults attr
             val = getattr(results, key)
             return np.sum(val) if val is not None else 0
        return 0.0

    # 1. Get Sectoral Sums (using proper names)
    # Note: These names must match what is in ModelVariables
    tot_ccu = np.sum(results.sectoral['I_netzero_ccu'].values)
    tot_ccs = np.sum(results.sectoral['I_netzero_ccs'].values)
    tot_nature = np.sum(results.sectoral['I_netzero_nature'].values)
    
    # 2. Get Electrification Sum (Macro)
    tot_elec = 0
    if 'I_electrification_total' in results.macro:
        tot_elec = results.macro['I_electrification_total'].sum()

    # 3. Plotting
    labels = ['Nature', 'CCS', 'CCU', 'Electrification']
    values = [tot_nature, tot_ccs, tot_ccu, tot_elec]
    colors = ['#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e']
    
    total = sum(values)
    TO_BILLION = 1_000_000
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bottom = 0
    for i, val in enumerate(values):
        val_bn = val / TO_BILLION
        ax.bar(0, val_bn, bottom=bottom, color=colors[i], label=labels[i], width=0.5)
        if val_bn > (total/TO_BILLION)*0.05:
            ax.text(0, bottom + val_bn/2, f"{val_bn:.1f}B", ha='center', va='center', color='white', fontweight='bold')
        bottom += val_bn
        
    ax.set_xticks([])
    ax.set_ylabel("Cumulative Investment (Billion SAR)")
    ax.set_title("Total Capital Mobilized (2021-2060)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.tight_layout()
    plt.show()

def plot_oil_and_gas_switching_intuitive(results: 'ModelResults'):
    """
    Visualizes the Financial and Environmental impact of switching from Oil to Gas.
    Single Plot:
    - Left Axis (Bars): Financial Cost vs. Savings (Billion SAR).
    - Right Axis (Bubbles/Line): Net Emission Reduction (MtCO2).
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from matplotlib.lines import Line2D

    pc = results.config.pc
    p = results.config.p
    
    if not getattr(p, 'enable_targeted_oil_reduction', False):
        print("NOTE: 'enable_targeted_oil_reduction' is False. No switching to plot.")
        return

    years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000

    # --- ADJUSTMENT FACTOR ---
    # Gas is 70% the price of Oil per GJ (User can adjust this if needed)
    GAS_PRICE_DISCOUNT = 0.7

    # 1. Get Data
    try:
        oil_orig = results.sectoral['oil_use']
        oil_final = results.sectoral['oil_use_final']
        gas_orig = results.sectoral['gas_use']
        gas_final = results.sectoral['gas_use_final']
    except KeyError:
        print("Error: Missing energy data.")
        return

    target_indices = list(pc.oil_reduction_sectors_idx)
    
    # 2. Calculate Financial & Emission Impacts
    
    # --- Price Setup ---
    idx_mining = pc.sectors.index('Extraction of crude petroleum and natural gas')
    total_mining_revenue = pc.X_[idx_mining]
    total_mining_energy_gj = np.sum((pc.oil_intensities_ + pc.gas_intensities_) * pc.X_)
    base_energy_price = total_mining_revenue / total_mining_energy_gj if total_mining_energy_gj > 0 else 0
    price_oil = base_energy_price
    price_gas = base_energy_price * GAS_PRICE_DISCOUNT

    savings_oil_bn = []
    costs_gas_bn = []
    net_emission_reduction_mt = []
    
    for i in range(len(years)):
        # Get the DELTA (Change in GJ) for this year
        delta_oil_gj = (oil_orig.iloc[i] - oil_final.iloc[i]).values
        delta_gas_gj = (gas_final.iloc[i] - gas_orig.iloc[i]).values
        
        # A. Financial Calc (Billions SAR)
        val_saved = np.sum(delta_oil_gj[target_indices]) * price_oil / TO_BILLION
        val_cost = np.sum(delta_gas_gj[target_indices]) * price_gas / TO_BILLION
        
        savings_oil_bn.append(val_saved)
        costs_gas_bn.append(val_cost)

        # B. Emission Calc (MtCO2)
        ems_avoided = np.sum(delta_oil_gj[target_indices]) * pc.oil_emission_factor
        ems_added = np.sum(delta_gas_gj[target_indices]) * pc.gas_emission_factor
        net_reduction = ems_avoided - ems_added
        net_emission_reduction_mt.append(net_reduction)

    savings_oil_bn = np.array(savings_oil_bn)
    costs_gas_bn = np.array(costs_gas_bn)
    net_impact_bn = costs_gas_bn - savings_oil_bn
    net_emission_reduction_mt = np.array(net_emission_reduction_mt)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Darker Colors
    color_savings = '#1E8449'  # Darker Green
    color_costs = '#B03A2E'    # Darker Red
    color_net = 'black'
    color_emissions = '#1f77b4' # Standard Blue

    # --- Primary Axis (Left): Financials ---
    # Negative savings for visual flow (below zero), Costs above zero
    bar1 = ax.bar(years, -savings_oil_bn, color=color_savings, alpha=0.8, label='Savings (Avoided Oil Bill)')
    bar2 = ax.bar(years, costs_gas_bn, color=color_costs, alpha=0.8, label='Cost (New Gas Bill)')
    line1 = ax.plot(years, net_impact_bn, color=color_net, linewidth=2, marker='_', markersize=10, label='Net Financial Impact (Left Axis)')
    
    ax.set_ylabel("Billion SAR (Annual)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', alpha=0.5)

    # --- Secondary Axis (Right): Emissions ---
    ax_twin = ax.twinx()
    
    # Normalize bubble sizes relative to the maximum value for readability
    max_em = np.max(net_emission_reduction_mt) if np.max(net_emission_reduction_mt) > 0 else 1.0
    bubble_sizes = [(x / max_em) * 400 + 100 for x in net_emission_reduction_mt]
    
    scatter = ax_twin.scatter(years, net_emission_reduction_mt, s=bubble_sizes, color=color_emissions,
                               edgecolor='white', alpha=0.7, zorder=10, label='Net Emission Reduction')
    
    line2 = ax_twin.plot(years, net_emission_reduction_mt, color=color_emissions, linestyle='--',
                          alpha=0.5, zorder=9)

    ax_twin.set_ylabel("Net Emission Reduction (MtCO2)", color=color_emissions, fontsize=12, weight='bold')
    ax_twin.tick_params(axis='y', labelcolor=color_emissions)
    
    # Ensure y-axis starts at 0 and has headroom
    ax_twin.set_ylim(bottom=0, top=max_em * 1.35)

    # --- Title & Annotations ---
    ax.set_title(f"Financial Impact & Environmental Benefit\n(Targeted Liquid Displacement Program)", fontsize=16, fontweight='bold')

    # Emission Factor Info Box
    oil_ef = pc.oil_emission_factor * 1_000_000 # Convert back to Tonnes/GJ for display
    gas_ef = pc.gas_emission_factor * 1_000_000
    diff_pct = (1 - (pc.gas_emission_factor / pc.oil_emission_factor)) * 100
    
    annot_text = (f"Emission Factors (tCO2/GJ):\n"
                  f"• Oil: {oil_ef:.4f}\n"
                  f"• Gas: {gas_ef:.4f}\n"
                  f"→ Gas is {diff_pct:.1f}% less carbon intensive")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='grey')
    ax.text(0.5, 0.98, annot_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='center', bbox=props, zorder=12)

    # Combined Legend
    handles1, labels1 = ax.get_legend_handles_labels()
    bubble_handle = Line2D([0], [0], marker='o', color='w', label='Net Emission Reduction (Right Axis)',
                          markerfacecolor=color_emissions, markersize=10, alpha=0.7)
    
    ax.legend(handles=handles1 + [bubble_handle], loc='upper left', frameon=True, framealpha=0.95)
    
    # Arbitrage / Financial Result Annotation
    status = "NET SAVING" if net_impact_bn[-1] < 0 else "NET COST"
    ax.text(0.02, 0.05, f"Financial Result: {status}\n(Gas price < Oil price)", 
            transform=ax.transAxes, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

def plot_oil_and_gas_switching_intuitive_no_legend(results: 'ModelResults'):
    """
    Visualizes the Financial and Environmental impact of switching from Oil to Gas.
    Single Plot (NO LEGENDS, NO TITLE, NO ANNOTATIONS VERSION):
    - Left Axis (Bars): Financial Cost vs. Savings (Billion SAR).
    - Right Axis (Bubbles/Line): Net Emission Reduction (MtCO2).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    pc = results.config.pc
    p = results.config.p

    if not getattr(p, 'enable_targeted_oil_reduction', False):
        print("NOTE: 'enable_targeted_oil_reduction' is False. No switching to plot.")
        return

    years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000
    GAS_PRICE_DISCOUNT = 0.7

    try:
        oil_orig  = results.sectoral['oil_use']
        oil_final = results.sectoral['oil_use_final']
        gas_orig  = results.sectoral['gas_use']
        gas_final = results.sectoral['gas_use_final']
    except KeyError:
        print("Error: Missing energy data.")
        return

    target_indices = list(pc.oil_reduction_sectors_idx)

    idx_mining = pc.sectors.index('Extraction of crude petroleum and natural gas')
    total_mining_revenue   = pc.X_[idx_mining]
    total_mining_energy_gj = np.sum((pc.oil_intensities_ + pc.gas_intensities_) * pc.X_)
    base_energy_price      = total_mining_revenue / total_mining_energy_gj if total_mining_energy_gj > 0 else 0
    price_oil = base_energy_price
    price_gas = base_energy_price * GAS_PRICE_DISCOUNT

    savings_oil_bn          = []
    costs_gas_bn            = []
    net_emission_reduction_mt = []

    for i in range(len(years)):
        delta_oil_gj = (oil_orig.iloc[i]  - oil_final.iloc[i]).values
        delta_gas_gj = (gas_final.iloc[i] - gas_orig.iloc[i]).values

        val_saved = np.sum(delta_oil_gj[target_indices]) * price_oil / TO_BILLION
        val_cost  = np.sum(delta_gas_gj[target_indices]) * price_gas / TO_BILLION

        savings_oil_bn.append(val_saved)
        costs_gas_bn.append(val_cost)

        ems_avoided   = np.sum(delta_oil_gj[target_indices]) * pc.oil_emission_factor
        ems_added     = np.sum(delta_gas_gj[target_indices]) * pc.gas_emission_factor
        net_reduction = ems_avoided - ems_added
        net_emission_reduction_mt.append(net_reduction)

    savings_oil_bn            = np.array(savings_oil_bn)
    costs_gas_bn              = np.array(costs_gas_bn)
    net_impact_bn             = costs_gas_bn - savings_oil_bn
    net_emission_reduction_mt = np.array(net_emission_reduction_mt)

    fig, ax = plt.subplots(figsize=(16, 9))

    color_savings   = '#1E8449'
    color_costs     = '#B03A2E'
    color_net       = 'black'
    color_emissions = '#1f77b4'

    # --- Left axis: Bars + Net line ---
    ax.bar(years, -savings_oil_bn, color=color_savings, alpha=0.8)
    ax.bar(years,  costs_gas_bn,   color=color_costs,   alpha=0.8)
    ax.plot(years, net_impact_bn,  color=color_net, linewidth=2, marker='_', markersize=10)

    ax.set_ylabel("Billion SAR (annual)", fontsize=26)
    ax.set_xlabel("Year",                 fontsize=26)
    ax.tick_params(axis='both', labelsize=26)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', alpha=0.5)

    # --- Right axis: Emission bubbles + dashed line ---
    ax_twin = ax.twinx()

    max_em       = np.max(net_emission_reduction_mt) if np.max(net_emission_reduction_mt) > 0 else 1.0
    bubble_sizes = [(x / max_em) * 400 + 100 for x in net_emission_reduction_mt]

    ax_twin.scatter(years, net_emission_reduction_mt,
                    s=bubble_sizes, color=color_emissions,
                    edgecolor='white', alpha=0.7, zorder=10)
    ax_twin.plot(years, net_emission_reduction_mt,
                 color=color_emissions, linestyle='--', alpha=0.5, zorder=9)

    ax_twin.set_ylabel("Net emission reduction (MtCO2)",
                       color=color_emissions, fontsize=26)
    ax_twin.tick_params(axis='y', labelcolor=color_emissions, labelsize=26)
    ax_twin.set_ylim(bottom=0, top=max_em * 1.35)

    plt.tight_layout()
    plt.show()




def plot_net_zero_balance_sheet(results: 'ModelResults'):
    """
    Visualizes the Physical Balance: Residual Emissions vs. Offsets.
    (Hydrogen's effect is implicit here: it lowers the 'Residual Emissions' bar
    by substituting fossil fuels before this plot is generated).
    """
    p = results.config.p
    if not p.net_emission_reduction_green_investments_exports_transformation: return

    years = list(range(2021, 2021 + len(results.macro)))
    
    # 1. Get Data
    residual_emissions = results.macro['emissions_net_total'].values # The Grey Wedge
    
    # Calculate Credits Purchased (Limited by Liquidity)
    # We recalculate the curve here for plotting
    start_year_idx = 4 # 2025
    liquidity_curve = []
    start_vol = p.carbon_market_liquidity
    growth_rate = getattr(p, 'carbon_market_growth_rate', 0.113)
    
    for t in range(len(years)):
        if t < start_year_idx:
            liquidity_curve.append(0)
        else:
            val = start_vol * (1 + growth_rate)**(t - start_year_idx)
            liquidity_curve.append(val)
    liquidity_curve = np.array(liquidity_curve)
    
    credits_purchased = np.minimum(residual_emissions, liquidity_curve)
    
    # Final Net Position (Should be 0 if successful)
    final_balance = residual_emissions - credits_purchased
    
    # 2. Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Residual (Liability)
    ax.bar(years, residual_emissions, color='grey', alpha=0.6, label='Residual "Hard-to-Abate" Emissions')
    
    # Credits (Asset/Offset)
    ax.bar(years, -credits_purchased, color='#2ca02c', alpha=0.8, label='Carbon Credits Purchased')
    
    # Net Line
    ax.plot(years, final_balance, color='black', linewidth=3, marker='o', markersize=4, label='Final Net Position')
    
    title_text = "Net Zero Balance Sheet"
    if p.enable_green_hydrogen:
        title_text += " (Incl. Green H2 Substitution)"

    ax.set_title(title_text, fontsize=16, fontweight='bold')
    ax.set_ylabel("Emissions Balance (MtCO2e)", fontsize=12)
    ax.axhline(0, color='black', linewidth=1)
    
    # Annotation
    if final_balance[-1] < 1.0:
        status = "SUCCESS: Net Zero Achieved"
        color = "green"
    else:
        status = f"GAP: {final_balance[-1]:.1f} Mt shortfall"
        color = "red"
        
    ax.text(years[-1], max(residual_emissions)*0.9, status, color=color, fontweight='bold', ha='right')
    
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import os
import string
import pandas as pd

import string
import numpy as np
import matplotlib.pyplot as plt

def plot_energy_scenarios(exp_reports, start_year=2021):
    """
    Generates high-quality energy scenario plots with sub-labels (a, b, c...)
    Includes non-overlapping value annotations and % of GDP.
    Units are only displayed on plots (a) and (g) with the same font size as values.
    """
    import string
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 1. Define Consistent Color Mapping
    scenario_colors = {
        "Baseline": "#d62728",       # Red
        "Vision_2030": "#ff7f0e",    # Orange
        "Net_zero": "#2ca02c",       # Green
        "Transformation": "#1f77b4", # Blue
        "Steady_state": "#7f7f7f"    # Grey
    }
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    # Font sizes
    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 16
    TICK_FONT_SIZE = 18
    LEGEND_FONT_SIZE = 18
    ANNOTATION_FONT_SIZE = 18
    SUB_LABEL_SIZE = 18

    # --- Helper: Overlap Resolution for Labels ---
    def resolve_overlaps(points, ax_idx):
        if not points: return []
        points.sort(key=lambda x: x['y'])
        y_vals = [p['y'] for p in points]
        y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 1.0
        if y_range == 0: y_range = max(y_vals) * 0.1 if max(y_vals) > 0 else 1.0
        
        # Increase vertical buffer heavily for plot (a) to prevent overlaps
        if ax_idx == 0:
            buffer = y_range * 0.25 
        else:
            buffer = y_range * 0.08
        
        for i in range(1, len(points)):
            prev = points[i-1]
            curr = points[i]
            if curr['y'] - prev['y'] < buffer:
                curr['y'] = prev['y'] + buffer
        return points

    # ==========================================================================
    # FIGURE 1: CORE ENERGY VARIABLES
    # ==========================================================================
    vars_fig1 = [
        ('total_energy_demand', 'Total Energy Demand', 'TWh', 1/3600000),
        ('renewable_energy_share', 'Renewable Energy Share', '%', 100),
        ('total_emissions', 'Total Gross Emissions', 'Mt CO2eq', 1),
        ('emissions_net_total', 'Total Net Emissions', 'Mt CO2eq', 1),
        ('oil_use_total', 'Total Oil Demand', 'EJ', 1e-9),
        ('gas_use_total', 'Total Gas Demand', 'EJ', 1e-9),
        ('electricity_use_total', 'Total Electricity Demand', 'TWh', 1/3600000),
        ('I_RE', 'Energy Related Investments', 'Billion SAR', 1/1000000),
    ]

    fig1, axs1 = plt.subplots(4, 2, figsize=(18, 24))
    axs1 = axs1.flatten()

    for ax_idx, (var_name, var_label, unit, factor) in enumerate(vars_fig1):
        ax = axs1[ax_idx]
        
        # Create secondary axis for plots (a), (b), and (g)
        ax2 = None
        if ax_idx == 1: # plot b
            ax2 = ax.twinx()
            ax2.set_ylabel("Total Electricity supply (%) (dashed lines)", fontweight='bold', fontsize=LABEL_FONT_SIZE)
            ax2.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
            ax.set_ylim(0, 100)  
            ax2.set_ylim(0, 100) 
        elif ax_idx == 0 or ax_idx == 6: # plot a or g
            ax2 = ax.twinx()
            # Make the EJ axis label, ticks, and spine GREY
            ax2.set_ylabel("EJ", fontweight='bold', fontsize=LABEL_FONT_SIZE, color='grey')
            ax2.tick_params(axis='y', labelsize=TICK_FONT_SIZE, colors='grey')
            ax2.spines['right'].set_color('grey')
            
        endpoints_2060 = []
        midpoint_annotations = {2030: [], 2040: [], 2050: []}
        has_data = False
        
        ax.text(-0.05, 1.05, f"({string.ascii_lowercase[ax_idx]})", transform=ax.transAxes, 
                fontsize=SUB_LABEL_SIZE, fontweight='bold', va='top')

        for sc_idx, (sc_name, runs) in enumerate(exp_reports.items()):
            run_data = runs[0]
            p = run_data.get('parameters')
            rm = run_data.get('results_macro')
            if hasattr(rm, 'macro'): rm = rm.macro
            rs = run_data.get('results_sectoral') or run_data.get('sectoral')
            color = scenario_colors.get(sc_name, default_colors[sc_idx % len(default_colors)])

            series_vals = None
            pct_gdp_series = None

            if var_name == 'I_RE':
                try:
                    val_sum = np.zeros(len(rm))
                    for col in ['I_RE', 'I_hydrogen', 'I_electrification_total', 'I_storage', 'I_efficiency_total']:
                        if col in rm: val_sum += rm[col].values
                    if rs is not None:
                        for key in ['I_netzero_nature', 'I_netzero_ccs', 'I_netzero_ccu', 'I_carbon_credits']:
                            if key in rs: val_sum += rs[key].sum(axis=1).values
                    series_vals = val_sum * factor
                    if 'Y' in rm.columns:
                        pct_gdp_series = (val_sum / rm['Y'].values) * 100
                except: continue
                
            elif var_name == 'emissions_net_total':
                if rm is not None and 'emissions_net_total' in rm:
                    raw_net = rm['emissions_net_total'].values
                    if p and getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False) and getattr(p, 'enable_carbon_market', False):
                        start_vol = p.carbon_market_liquidity
                        growth = getattr(p, 'carbon_market_growth_rate', 0.113)
                        liq = np.array([0 if t < 4 else start_vol * (1 + growth)**(t - 4) for t in range(len(raw_net))])
                        series_vals = (raw_net - np.minimum(raw_net, liq)) * factor
                    else:
                        series_vals = raw_net * factor
                            
            elif rm is not None and var_name in rm.columns:
                series_vals = rm[var_name].values * factor

            if series_vals is not None:
                years = np.arange(start_year, start_year + len(series_vals))
                
                # Plot primary line
                ax.plot(years, series_vals, label=sc_name, color=color, linewidth=2.5)
                has_data = True
                
                # Plot secondary line ONLY for plots (a) and (g) [TWh to EJ conversion]
                if (ax_idx == 0 or ax_idx == 6) and ax2 is not None:
                    # 1 TWh = 0.0036 EJ
                    series_vals_ej = series_vals * 0.0036
                    # Zero-width sentinel line used only to scale the secondary grey axis
                    ax2.plot(years, series_vals_ej, color=color, linewidth=0, alpha=0) 
                
                # ======================================================================
                # --- UNCERTAINTY BANDS LOGIC FOR EMISSIONS & INVESTMENT ---
                # ======================================================================
                is_sgi_active = False
                if p is not None:
                    if getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False) or getattr(p, 'vision_2030_target_CCU_SGI_Carbon', False):
                        is_sgi_active = True
                
                # Uncertainty Band for Net Emissions (Graph d)
                if is_sgi_active and sc_name not in ["Baseline", "Vision_2030"] and var_name == 'emissions_net_total':
                    lower_bound = np.zeros_like(series_vals)
                    upper_bound = np.zeros_like(series_vals)
                    for i, y in enumerate(years):
                        time_step = y - start_year + 1
                        fraction = 0.0 if time_step <= 1 else ((time_step - 1) / 39.0 if time_step <= 40 else 1.0)
                        
                        mean_seq = 1.0 + (60.0 - 1.0) * (fraction ** 2)
                        min_seq = 0.2 + (20.0 - 0.2) * (fraction ** 2)
                        max_seq = 2.0 + (100.0 - 2.0) * (fraction ** 2)
                        
                        lower_bound[i] = series_vals[i] - (max_seq - mean_seq)
                        upper_bound[i] = series_vals[i] + (mean_seq - min_seq)
                        
                    ax.fill_between(years, lower_bound, upper_bound, color=color, alpha=0.15, label=f"{sc_name} (NBS uncertainty)")
                
                # Uncertainty Band for Investment (Graph h)
                if is_sgi_active and sc_name != "Baseline" and var_name == 'I_RE':
                    lower_bound = np.zeros_like(series_vals)
                    upper_bound = np.zeros_like(series_vals)
                    c_factor = getattr(p, 'nature_cost_per_tonne', 50.0) * 1000.0 / 1000000.0 
                    
                    for i, y in enumerate(years):
                        time_step = y - start_year + 1
                        
                        def get_cap(ts):
                            if ts <= 1: return 1.0, 0.2, 2.0
                            f = (ts - 1) / 39.0 if ts <= 40 else 1.0
                            return (1.0 + (60.0 - 1.0)*(f**2), 0.2 + (20.0 - 0.2)*(f**2), 2.0 + (100.0 - 2.0)*(f**2))
                        
                        mean_c, min_c, max_c = get_cap(time_step)
                        mean_p, min_p, max_p = get_cap(time_step - 1) if time_step > 1 else (0.0, 0.0, 0.0)
                        
                        inv_mean = max(0, mean_c - mean_p) * c_factor
                        inv_min = max(0, min_c - min_p) * c_factor
                        inv_max = max(0, max_c - max_p) * c_factor
                        
                        upper_bound[i] = series_vals[i] + (inv_max - inv_mean)
                        lower_bound[i] = series_vals[i] - (inv_mean - inv_min)
                        
                    ax.fill_between(years, lower_bound, upper_bound, color=color, alpha=0.15)
                # ======================================================================

                # Plot secondary line ONLY for plot (b)
                if ax_idx == 1 and ax2 is not None:
                    try:
                        elec_demand_total = rm['electricity_use_total'].values
                        re_gen_total = rm['renewable_energy_generation'].values
                        
                        re_met_demand = np.minimum(elec_demand_total, re_gen_total)
                        
                        re_share_elec = np.zeros_like(re_met_demand)
                        mask = elec_demand_total > 0
                        re_share_elec[mask] = (re_met_demand[mask] / elec_demand_total[mask]) * 100
                        
                        ax2.plot(years, re_share_elec, color=color, linewidth=2.5, linestyle='--', alpha=0.8)
                        
                        if 2060 in years:
                            idx_60 = np.where(years == 2060)[0][0]
                            val_60_sec = re_share_elec[idx_60]
                            # No units for plot b secondary line
                            endpoints_2060.append({'y': val_60_sec, 'real_y': val_60_sec, 'label': f"{val_60_sec:.1f}%", 'color': color})

                    except Exception as e:
                        print(f"Error calculating secondary axis for {sc_name}: {e}")
                
                if 2060 in years:
                    idx_60 = np.where(years == 2060)[0][0]
                    val_60 = series_vals[idx_60]
                    if var_name == 'I_RE' and pct_gdp_series is not None:
                        label_text = f"{pct_gdp_series[idx_60]:.1f}%"
                    elif ax_idx == 1:
                        label_text = f"{val_60:.1f}%"
                    elif ax_idx == 0 or ax_idx == 6:
                        # Append the specific unit to the annotation ONLY for plot a and g
                        label_text = f"{val_60:.1f} {unit}"
                    else:
                        # No units for all other plots
                        label_text = f"{val_60:.1f}"
                        
                    endpoints_2060.append({'y': val_60, 'real_y': val_60, 'label': label_text, 'color': color})

                if var_name == 'I_RE' and pct_gdp_series is not None:
                    for target_yr in [2030, 2040, 2050]:
                        if target_yr in years:
                            idx = np.where(years == target_yr)[0][0]
                            v = series_vals[idx]
                            p_val = pct_gdp_series[idx]
                            midpoint_annotations[target_yr].append({
                                'y': v, 'real_y': v, 'label': f"{p_val:.1f}%", 'color': color, 'yr': target_yr
                            })

        if endpoints_2060:
            for p_annotate in resolve_overlaps(endpoints_2060, ax_idx):
                
                if ax_idx == 1:
                    # Plot (b) annotations inside the plot (Right Aligned)
                    ax.annotate(p_annotate['label'], xy=(2060, p_annotate['real_y']), xytext=(2058, p_annotate['y']), 
                                color=p_annotate['color'], fontweight='bold', fontsize=ANNOTATION_FONT_SIZE,
                                va='center', ha='right',
                                arrowprops=dict(arrowstyle="-", color=p_annotate['color'], alpha=0.4) if abs(p_annotate['y']-p_annotate['real_y']) > 0.01 else None)
                else:
                    # All other plots annotations on the right side (Left Aligned)
                    ax.annotate(p_annotate['label'], xy=(2060, p_annotate['real_y']), xytext=(2061.2, p_annotate['y']), 
                                color=p_annotate['color'], fontweight='bold', fontsize=ANNOTATION_FONT_SIZE,
                                va='center', ha='left',
                                arrowprops=dict(arrowstyle="-", color=p_annotate['color'], alpha=0.4) if abs(p_annotate['y']-p_annotate['real_y']) > 0.01 else None)
                                
                ax.scatter([2060], [p_annotate['real_y']], color=p_annotate['color'], s=30, zorder=5)

        if var_name == 'I_RE':
            for yr, points in midpoint_annotations.items():
                if not points: continue
                resolved = resolve_overlaps(points, ax_idx)
                for p_annotate in resolved:
                    ax.annotate(p_annotate['label'], xy=(p_annotate['yr'], p_annotate['real_y']), xytext=(p_annotate['yr'], p_annotate['y'] + (max(series_vals)*0.04)),
                                color=p_annotate['color'], fontweight='bold', fontsize=ANNOTATION_FONT_SIZE, ha='center', va='bottom')

        # Set Titles and Labels
        if ax_idx == 1:
            ax.set_title("Renewable Energy (RE) Share", fontweight='bold', fontsize=TITLE_FONT_SIZE) 
            ax.set_ylabel("Total energy supply (%) (solid lines)", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        else:
            ax.set_title(var_label, fontweight='bold', fontsize=TITLE_FONT_SIZE) 
            ax.set_ylabel(unit, fontweight='bold', fontsize=LABEL_FONT_SIZE)
            
        ax.set_xlabel("Year", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        
        # Set x-axis limit up to 2060
        ax.set_xlim(start_year, 2060)
        
        # Ensure x-axis ticks do not exceed 2060
        ticks = ax.get_xticks()
        ax.set_xticks(ticks[ticks <= 2060])
        
        # Apply specific Y-axis ticks for Plot (a) and (g)
        if ax_idx == 0:
            ax.set_yticks([2500, 3500, 4500, 5500, 6500])
            if ax2 is not None:
                ax2.set_yticks([8, 12, 16, 20])
        elif ax_idx == 6:
            ax.set_yticks([250, 750, 1250, 1750, 2250])
            if ax2 is not None:
                ax2.set_yticks([1, 3, 5, 7, 9])
        
        # --- Specific Legend Rules ---
        if has_data: 
            if ax_idx == 0:
                # Plot (a): Main Legend
                ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=True, loc='upper left')
            elif ax_idx == 3:
                # Plot (d): Only NBS Uncertainty Legend
                handles, labels = ax.get_legend_handles_labels()
                unc_dict = {l: h for h, l in zip(handles, labels) if 'NBS uncertainty' in l}
                if unc_dict:
                    ax.legend(unc_dict.values(), unc_dict.keys(), fontsize=LEGEND_FONT_SIZE, frameon=True)
            elif ax_idx == 7:
                # Plot (h): Only % of GDP Title
                ax.legend([], [], fontsize=LEGEND_FONT_SIZE, frameon=True, title="% on lines: GDP Share per scenario", title_fontsize=LEGEND_FONT_SIZE)

    fig1.tight_layout()
    plt.show()





def plot_nexus_water_all_scenarios(exp_reports, start_year=2021):
    """
    Creates a dual-panel line plot comparing the Water-Energy Nexus:
    1. Water for Energy (Billion m3)
    2. Energy for Water (TWh) with EJ on secondary axis
    Filters plotted data: allows solar/wind in the Baseline but blocks spurious mitigation values.
    Formatting updated to match plot_energy_scenarios (Fonts, X-Limits, Annotations).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # 1. Consistent Color Mapping
    scenario_colors = {
        "Baseline": "#d62728",       # Red
        "Vision_2030": "#ff7f0e",    # Orange
        "Net_zero": "#2ca02c",       # Green
        "Transformation": "#1f77b4", # Blue
        "Steady_state": "#7f7f7f"    # Grey
    }
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    # --- Formatting Constants ---
    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 16
    TICK_FONT_SIZE = 18
    LEGEND_FONT_SIZE = 18
    ANNOTATION_FONT_SIZE = 18
        
    # Create 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Create EJ secondary axis for plot b (ax2)
    ax2_ej = ax2.twinx()
    ax2_ej.set_ylabel("EJ", fontweight='bold', fontsize=LABEL_FONT_SIZE, color='grey')
    ax2_ej.tick_params(axis='y', labelsize=TICK_FONT_SIZE, colors='grey')
    ax2_ej.spines['right'].set_color('grey')
    
    has_data = False
    
    # Variables to track global maximums for dynamic y-axis scaling
    global_max_water = 0.0
    global_max_energy = 0.0
    
    # Define Water Intensities
    wi_solar = 0.02 
    wi_wind = 0.0 
    wi_hydrogen = 0.15 
    wi_electrification = 0.005 
    wi_ccs_bm3_per_mt = 0.002 
    wi_ccu_bm3_per_mt = 0.0025 
    wi_sgi_bm3_per_mt = 0.315 
    
    # Define Energy Intensities for Water
    ei_desal_kwh_per_m3 = 3.5 
    ei_gw_kwh_per_m3 = 1.0 
    ei_wwater_kwh_per_m3 = 0.8 + 1.5
    kwh_to_gj = 0.0036
    
    endpoints_ax1 = []
    endpoints_ax2 = []
    
    for sc_idx, (sc_name, runs) in enumerate(exp_reports.items()):
        run_data = runs[0]
        rm = run_data.get('results_macro')
        p = run_data.get('parameters')
        if hasattr(rm, 'macro'): rm = rm.macro
        
        color = scenario_colors.get(sc_name, default_colors[sc_idx % len(default_colors)])
        
        # Safely extract physical quantities directly from model results
        re_solar = rm.get('re_gen_solar', pd.Series(np.zeros(len(rm)))).values
        re_wind = rm.get('re_gen_wind', pd.Series(np.zeros(len(rm)))).values
        h2_prod = rm.get('hydrogen_production', pd.Series(np.zeros(len(rm)))).values
        elec_load = rm.get('electrification_added_load', pd.Series(np.zeros(len(rm)))).values
        ccs_flow = rm.get('flow_remove_ccs', pd.Series(np.zeros(len(rm)))).values
        ccu_flow = rm.get('flow_reuse_recycle', pd.Series(np.zeros(len(rm)))).values
        sgi_flow = rm.get('flow_remove_nature', pd.Series(np.zeros(len(rm)))).values

        # ======================================================================
        # Filter: prevent spurious mitigation data in the Baseline
        # ======================================================================
        if sc_name in ["Baseline", "Steady_state"]:
            # Keep re_solar and re_wind!
            # But zero out the advanced tech so they don't draw phantom water
            h2_prod = np.zeros_like(h2_prod)
            elec_load = np.zeros_like(elec_load)
            ccs_flow = np.zeros_like(ccs_flow)
            ccu_flow = np.zeros_like(ccu_flow)
            sgi_flow = np.zeros_like(sgi_flow)

        # ----------------------------------------------------------------------
        # A. Calculate Water for Energy (Billion m3)
        # ----------------------------------------------------------------------
        water_solar = (re_solar * wi_solar) * 1e-9
        water_wind = (re_wind * wi_wind) * 1e-9
        water_h2 = (h2_prod * wi_hydrogen) * 1e-9
        water_elec = (elec_load * wi_electrification) * 1e-9
        water_ccs = ccs_flow * wi_ccs_bm3_per_mt
        water_ccu = ccu_flow * wi_ccu_bm3_per_mt
        water_sgi = sgi_flow * wi_sgi_bm3_per_mt
        
        total_nexus_water_bm3 = water_solar + water_wind + water_h2 + water_elec + water_ccs + water_ccu + water_sgi
        
        # ----------------------------------------------------------------------
        # B. Calculate Energy for Water (TWh)
        # ----------------------------------------------------------------------
        is_sust_or_v2030 = False
        if p is not None:
            if getattr(p, 'vision_2030_target_CCU_SGI_Carbon', False) or getattr(p, 'Transformation_scenario', False):
                is_sust_or_v2030 = True

        total_nexus_water_m3 = total_nexus_water_bm3 * 1e9

        if is_sust_or_v2030:
            wwater_m3 = total_nexus_water_m3
            desal_m3 = np.zeros_like(total_nexus_water_m3)
            gw_m3 = np.zeros_like(total_nexus_water_m3)
        else:
            wwater_m3 = np.zeros_like(total_nexus_water_m3)
            desal_m3 = total_nexus_water_m3 * 0.70
            gw_m3 = total_nexus_water_m3 * 0.30

        energy_desal_gj = desal_m3 * ei_desal_kwh_per_m3 * kwh_to_gj
        energy_gw_gj = gw_m3 * ei_gw_kwh_per_m3 * kwh_to_gj
        energy_ww_gj = wwater_m3 * ei_wwater_kwh_per_m3 * kwh_to_gj
        
        total_energy_gj = energy_desal_gj + energy_gw_gj + energy_ww_gj
        total_energy_twh = total_energy_gj / 3600000.0  # Convert GJ to TWh
        
        # ----------------------------------------------------------------------
        # C. Plotting
        # ----------------------------------------------------------------------
        if len(total_nexus_water_bm3) > 0:
            years = np.arange(start_year, start_year + len(total_nexus_water_bm3))
            
            # --- SMOOTHING FIX: 5-Year Moving Average ---
            smoothed_water = pd.Series(total_nexus_water_bm3).rolling(window=5, min_periods=1, center=True).mean().values
            smoothed_energy = pd.Series(total_energy_twh).rolling(window=5, min_periods=1, center=True).mean().values
            
            # Update global maximums for y-axis limits (ignoring NaNs)
            current_max_water = np.nanmax(smoothed_water)
            current_max_energy = np.nanmax(smoothed_energy)
            if current_max_water > global_max_water:
                global_max_water = current_max_water
            if current_max_energy > global_max_energy:
                global_max_energy = current_max_energy
            
            # Plot 1: Water for Energy
            ax1.plot(years, smoothed_water, label=sc_name, color=color, linewidth=3.0)
            
            # Plot 2: Energy for Water
            ax2.plot(years, smoothed_energy, label=sc_name, color=color, linewidth=3.0)
            
            # Zero-width sentinel line for scaling the EJ secondary axis (1 TWh = 0.0036 EJ)
            ax2_ej.plot(years, smoothed_energy * 0.0036, color=color, linewidth=0, alpha=0)
            
            has_data = True
            
            # Collect end-point annotations with units added
            if 2060 in years:
                idx_60 = np.where(years == 2060)[0][0]
                
                val_water_60 = smoothed_water[idx_60]
                endpoints_ax1.append({'y': val_water_60, 'real_y': val_water_60, 'label': f"{val_water_60:.3f} Bn m³", 'color': color})
                
                val_energy_60 = smoothed_energy[idx_60]
                endpoints_ax2.append({'y': val_energy_60, 'real_y': val_energy_60, 'label': f"{val_energy_60:.3f} TWh", 'color': color})

    # --- Overlap Resolution Helper ---
    def resolve_overlaps(points):
        if not points: return []
        points.sort(key=lambda x: x['y'])
        y_vals = [p['y'] for p in points]
        y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 1.0
        if y_range == 0: y_range = max(y_vals) * 0.1 if max(y_vals) > 0 else 1.0
        buffer = y_range * 0.08
        for i in range(1, len(points)):
            prev = points[i-1]
            curr = points[i]
            if curr['y'] - prev['y'] < buffer:
                curr['y'] = prev['y'] + buffer
        return points

    # --- Apply Annotations ---
    if endpoints_ax1:
        for p_annotate in resolve_overlaps(endpoints_ax1):
            ax1.annotate(p_annotate['label'], xy=(2060, p_annotate['real_y']), xytext=(2061.2, p_annotate['y']), 
                         color=p_annotate['color'], fontweight='bold', fontsize=ANNOTATION_FONT_SIZE, va='center', ha='left',
                         arrowprops=dict(arrowstyle="-", color=p_annotate['color'], alpha=0.4) if abs(p_annotate['y']-p_annotate['real_y']) > 0.01 else None)
            ax1.scatter([2060], [p_annotate['real_y']], color=p_annotate['color'], s=40, zorder=5)

    if endpoints_ax2:
        for p_annotate in resolve_overlaps(endpoints_ax2):
            ax2.annotate(p_annotate['label'], xy=(2060, p_annotate['real_y']), xytext=(2061.2, p_annotate['y']), 
                         color=p_annotate['color'], fontweight='bold', fontsize=ANNOTATION_FONT_SIZE, va='center', ha='left',
                         arrowprops=dict(arrowstyle="-", color=p_annotate['color'], alpha=0.4) if abs(p_annotate['y']-p_annotate['real_y']) > 0.01 else None)
            ax2.scatter([2060], [p_annotate['real_y']], color=p_annotate['color'], s=40, zorder=5)

    # --- Final Formatting ---
    if has_data:
        # Format ax1 (Water)
        ax1.set_title("(a) Water for energy production\nand nature-based emission reduction", fontweight='bold', fontsize=TITLE_FONT_SIZE)
        ax1.set_ylabel("Water Demand (Billion m³)", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax1.set_xlabel("Year", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        
        # Restrict to 2060 limits
        ax1.set_xlim(start_year, 2060)
        ticks1 = ax1.get_xticks()
        ax1.set_xticks(ticks1[ticks1 <= 2060])
        
        # Apply 20% buffer to the global maximum water value
        if global_max_water > 0:
            ax1.set_ylim(0, global_max_water * 1.20)
            
        ax1.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax1.legend(fontsize=LEGEND_FONT_SIZE, frameon=True, loc='upper left')

        # Format ax2 (Energy)
        ax2.set_title("(b) Energy for water extraction and irrigation\nfor nature-based emission reduction", fontweight='bold', fontsize=TITLE_FONT_SIZE)
        ax2.set_ylabel("Electricity Demand (TWh)", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax2.set_xlabel("Year", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        
        # Restrict to 2060 limits
        ax2.set_xlim(start_year, 2060)
        ticks2 = ax2.get_xticks()
        ax2.set_xticks(ticks2[ticks2 <= 2060])
        
        # Apply 20% buffer to the global maximum energy value
        if global_max_energy > 0:
            ax2.set_ylim(0, global_max_energy * 1.20)
            
        ax2.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        # Legend explicitly removed from ax2
        
    plt.tight_layout()
    plt.show()






def plot_energy_scenarios_2(exp_reports, start_year=2021):
    """
    Generates high-quality energy scenario plots with sub-labels (a, b, c...)
    Includes non-overlapping value and % of GDP annotations for investment.
    """
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    # 1. Define Consistent Color Mapping (Red for Baseline, Blue for Transformation)
    scenario_colors = {
        "Baseline": "#d62728",       # Red
        "Vision_2030": "#ff7f0e",    # Orange
        "Net_zero": "#2ca02c",       # Green
        "Transformation": "#1f77b4", # Blue
        "Steady_state": "#7f7f7f"    # Grey
    }
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    green_scenarios_to_plot = []
    for sc_name, runs in exp_reports.items():
        run_data = runs[0]
        p = run_data.get('parameters', None)
        if p and (getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False) or getattr(p, 'vision_2030_scenario', False)):
            green_scenarios_to_plot.append(sc_name)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    LABEL_FONT_SIZE = 14
    TICK_FONT_SIZE = 12
    LEGEND_FONT_SIZE = 12
    SUB_LABEL_SIZE = 16

    # --- Helper: Overlap Resolution for Labels ---
    def resolve_overlaps(points):
        if not points: return []
        points.sort(key=lambda x: x['y'])
        y_vals = [p['y'] for p in points]
        y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 1.0
        if y_range == 0: y_range = max(y_vals) * 0.1 if max(y_vals) > 0 else 1.0
        buffer = y_range * 0.08
        
        for i in range(1, len(points)):
            prev = points[i-1]
            curr = points[i]
            if curr['y'] - prev['y'] < buffer:
                curr['y'] = prev['y'] + buffer
        return points

    # ==========================================================================
    # FIGURE 1: CORE ENERGY VARIABLES
    # ==========================================================================
    vars_fig1 = [
        ('total_energy_demand', 'Total Energy Demand', 'TWh', 1/3600000),
        ('renewable_energy_share', 'Renewable Energy Share', '%', 100),
        ('total_emissions', 'Total Gross Emissions', 'Mt CO2eq', 1),
        ('emissions_net_total', 'Total Net Emissions', 'Mt CO2eq', 1),
        ('oil_use_total', 'Total Oil Demand', 'EJ', 1e-9),
        ('gas_use_total', 'Total Gas Demand', 'EJ', 1e-9),
        ('electricity_use_total', 'Total Electricity Demand', 'TWh', 1/3600000),
        ('I_RE', 'Energy Related Investments', 'Billion SAR', 1/1000000),
    ]

    fig1, axs1 = plt.subplots(4, 2, figsize=(18, 24))
    axs1 = axs1.flatten()

    for ax_idx, (var_name, var_label, unit, factor) in enumerate(vars_fig1):
        ax = axs1[ax_idx]
        endpoints_2060 = []
        midpoint_annotations = {2030: [], 2040: [], 2050: []}
        has_data = False
        
        ax.text(-0.05, 1.05, f"({string.ascii_lowercase[ax_idx]})", transform=ax.transAxes, 
                fontsize=SUB_LABEL_SIZE, fontweight='bold', va='top')

        for sc_idx, (sc_name, runs) in enumerate(exp_reports.items()):
            run_data = runs[0]
            p = run_data.get('parameters')
            rm = run_data.get('results_macro')
            if hasattr(rm, 'macro'): rm = rm.macro
            rs = run_data.get('results_sectoral') or run_data.get('sectoral')
            color = scenario_colors.get(sc_name, default_colors[sc_idx % len(default_colors)])

            series_vals = None
            pct_gdp_series = None

            if var_name == 'I_RE':
                try:
                    val_sum = np.zeros(len(rm))
                    for col in ['I_RE', 'I_hydrogen', 'I_electrification_total', 'I_storage', 'I_efficiency_total']:
                        if col in rm: val_sum += rm[col].values
                    if rs is not None:
                        for key in ['I_netzero_nature', 'I_netzero_ccs', 'I_netzero_ccu', 'I_carbon_credits']:
                            if key in rs: val_sum += rs[key].sum(axis=1).values
                    series_vals = val_sum * factor
                    if 'Y' in rm.columns:
                        pct_gdp_series = (val_sum / rm['Y'].values) * 100
                except: continue
            elif var_name == 'emissions_net_total':
                if rm is not None and 'emissions_net_total' in rm:
                    raw_net = rm['emissions_net_total'].values
                    if p and getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False):
                        start_vol = p.carbon_market_liquidity
                        growth = getattr(p, 'carbon_market_growth_rate', 0.113)
                        liq = np.array([0 if t < 4 else start_vol * (1 + growth)**(t - 4) for t in range(len(raw_net))])
                        series_vals = (raw_net - np.minimum(raw_net, liq)) * factor
                    else:
                        series_vals = raw_net * factor
            elif rm is not None and var_name in rm.columns:
                series_vals = rm[var_name].values * factor

            if series_vals is not None:
                years = np.arange(start_year, start_year + len(series_vals))
                ax.plot(years, series_vals, label=sc_name, color=color, linewidth=2.5)
                has_data = True
                
                if 2060 in years:
                    idx_60 = np.where(years == 2060)[0][0]
                    val_60 = series_vals[idx_60]
                    if var_name == 'I_RE' and pct_gdp_series is not None:
                        label_text = f"{pct_gdp_series[idx_60]:.1f}%"
                    else:
                        label_text = f"{val_60:.1f}"
                    endpoints_2060.append({'y': val_60, 'real_y': val_60, 'label': label_text, 'color': color})

                if var_name == 'I_RE' and pct_gdp_series is not None:
                    for target_yr in [2030, 2040, 2050]:
                        if target_yr in years:
                            idx = np.where(years == target_yr)[0][0]
                            v = series_vals[idx]
                            p_val = pct_gdp_series[idx]
                            midpoint_annotations[target_yr].append({
                                'y': v, 'real_y': v, 'label': f"{p_val:.1f}%", 'color': color, 'yr': target_yr
                            })

        if endpoints_2060:
            for p in resolve_overlaps(endpoints_2060):
                ax.annotate(p['label'], xy=(2060, p['real_y']), xytext=(2061.2, p['y']), 
                            color=p['color'], fontweight='bold', fontsize=10, 
                            va='center', ha='left',
                            arrowprops=dict(arrowstyle="-", color=p['color'], alpha=0.4) if abs(p['y']-p['real_y']) > 0.01 else None)
                ax.scatter([2060], [p['real_y']], color=p['color'], s=30, zorder=5)

        if var_name == 'I_RE':
            for yr, points in midpoint_annotations.items():
                if not points: continue
                resolved = resolve_overlaps(points)
                for p in resolved:
                    ax.annotate(p['label'], xy=(p['yr'], p['real_y']), xytext=(p['yr'], p['y'] + (max(series_vals)*0.04)),
                                color=p['color'], fontweight='bold', fontsize=10, ha='center', va='bottom')

        ax.set_title(var_label, fontweight='bold', fontsize=14)
        ax.set_ylabel(unit, fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("Year", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax.set_xlim(start_year, 2065)
        if has_data: 
            leg_title = "% on lines: GDP Share per scenario" if var_name == 'I_RE' else None
            ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=True, title=leg_title)

    fig1.tight_layout()
    fig1.savefig(os.path.join(desktop_path, "Energy_Scenarios_Core.pdf"), format='pdf', dpi=300, bbox_inches='tight')

    # ==========================================================================
    # FIGURE 2: TOTAL INVESTMENT & BREAKDOWNS
    # ==========================================================================
    vars_fig2_line = [
        ('I', 'Total Investment', 'Billion SAR', 1/1000000),
        ('green_export_total', 'Total Green Exports (Comparison)', 'Billion SAR', 1/1000000)
    ]
    
    total_plots_fig2 = len(vars_fig2_line) + len(green_scenarios_to_plot)
    rows_fig2 = (total_plots_fig2 + 1) // 2
    fig2, axs2 = plt.subplots(rows_fig2, 2, figsize=(18, 6 * rows_fig2))
    axs2 = axs2.flatten()

    for ax_idx in range(total_plots_fig2):
        ax = axs2[ax_idx]
        ax.text(-0.05, 1.05, f"({string.ascii_lowercase[ax_idx]})", transform=ax.transAxes, 
                fontsize=SUB_LABEL_SIZE, fontweight='bold', va='top')
        
        if ax_idx < len(vars_fig2_line):
            var_name, var_label, unit, factor = vars_fig2_line[ax_idx]
            endpoints_2060 = []
            for sc_idx, (sc_name, runs) in enumerate(exp_reports.items()):
                run_data = runs[0]
                rm = run_data.get('results_macro')
                if hasattr(rm, 'macro'): rm = rm.macro
                color = scenario_colors.get(sc_name, default_colors[sc_idx % len(default_colors)])
                series = None
                if var_name == 'green_export_total':
                    if rm is not None and 'green_export_total' in rm: series = rm['green_export_total']
                    elif 'variables' in run_data and hasattr(run_data['variables'], 'green_export_total'):
                        raw = run_data['variables'].green_export_total
                        series = pd.Series(raw[1:], index=rm.index if rm is not None else range(len(raw)-1))
                elif rm is not None and var_name in rm.columns: series = rm[var_name]
                
                if series is not None:
                    y_vals = series.values * factor
                    ax.plot(series.index + start_year, y_vals, label=sc_name, color=color, linewidth=2.5)
                    if 39 < len(y_vals):
                        val_60 = y_vals[39]
                        endpoints_2060.append({'y': val_60, 'real_y': val_60, 'label': f"{val_60:.1f}", 'color': color})

            if endpoints_2060:
                for p in resolve_overlaps(endpoints_2060):
                    ax.annotate(p['label'], xy=(2060, p['real_y']), xytext=(2061.2, p['y']), 
                                color=p['color'], fontweight='bold', fontsize=10, va='center')

            ax.set_title(var_label, fontweight='bold', fontsize=14)
            ax.set_ylabel(unit, fontsize=LABEL_FONT_SIZE, fontweight='bold')
            ax.set_xlim(start_year, 2065)
            ax.legend(fontsize=LEGEND_FONT_SIZE)
        else:
            stack_colors = ['#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#2ca02c']
            stack_labels = ['Green H2', 'Solar Mnf.', 'Wind/Mach Mnf.', 'Batteries', 'Carbon Credits']
            green_idx = ax_idx - len(vars_fig2_line)
            sc_name = green_scenarios_to_plot[green_idx]
            run_data = exp_reports[sc_name][0]
            vars_obj = run_data.get('variables')
            rm = run_data.get('results_macro')
            if hasattr(rm, 'macro'): rm = rm.macro
            p = run_data.get('parameters')
            
            num_years = len(rm)
            years = np.arange(start_year, start_year + num_years)
            
            h2_val = (vars_obj.hydrogen_export[1:num_years+1] * getattr(p, 'hydrogen_export_price_sar_gj', 90.0)) / 1e9 if hasattr(vars_obj, 'hydrogen_export') else np.zeros(num_years)
            target_pct = 0.75 if "Net" in sc_name or "Sust" in sc_name else 0.50
            local_content = np.array([0.0 if t<5 else (target_pct * min(1.0, (t-5)/5.0)) for t in range(num_years)])
            solar_val = (rm['I_solar'].values * 1.2 * local_content * 0.30) / 1e6 if 'I_solar' in rm else np.zeros(num_years)
            inv_mach = (rm.get('I_wind', 0) + rm.get('I_electrification_total', 0)).values if rm is not None else np.zeros(num_years)
            wind_val = (inv_mach * local_content * 0.30) / 1e6
            batt_val = (vars_obj.I_storage[1:num_years+1] * 0.5) / 1e6 if hasattr(vars_obj, 'I_storage') else np.zeros(num_years)
            credit_val = ((vars_obj.flow_remove_nature[1:num_years+1] * 1e6) * 0.20 * 37.0) / 1e9 if hasattr(vars_obj, 'flow_remove_nature') else np.zeros(num_years)

            ax.stackplot(years, np.vstack([h2_val, solar_val, wind_val, batt_val, credit_val]), labels=stack_labels, colors=stack_colors, alpha=0.85)
            ax.set_title(f"Export Breakdown: {sc_name}", fontsize=13, fontweight='bold', color=scenario_colors.get(sc_name, "black"))
            ax.set_ylabel("Billion SAR", fontsize=LABEL_FONT_SIZE, fontweight='bold')
            
            total_2060 = (h2_val[-1] + solar_val[-1] + wind_val[-1] + batt_val[-1] + credit_val[-1])
            ax.text(2060.5, total_2060, f"{total_2060:.1f}", fontweight='bold', fontsize=10, va='bottom', ha='left')
            ax.legend(loc='upper left', fontsize=10)

        ax.set_xlabel("Year", fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)

    for i in range(total_plots_fig2, len(axs2)): axs2[i].axis('off')
    fig2.tight_layout()
    fig2.savefig(os.path.join(desktop_path, "Energy_Scenarios_Investment.pdf"), format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


import plotly.graph_objects as go
import numpy as np

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_technology_investment_log_timeline(results, start_year=2021, end_year=2060):
    """
    Plots an interactive LOGARITHMIC line chart of all energy technology investments.
    Converts values to Billion SAR.
    """
    years = np.arange(start_year, end_year + 1)
    t_indices = years - 2020  # Model t indices (e.g., 2021 = 1)
    TO_BILLION = 1_000_000.0

    # ========================================================================
    # 1. HELPER FUNCTIONS TO EXTRACT TIME-SERIES
    # ========================================================================
    def get_macro_ts(var_name):
        if var_name in results.macro:
            ts = results.macro[var_name].reindex(t_indices).fillna(0)
            return ts / TO_BILLION
        return pd.Series(0.0, index=t_indices)

    def get_sectoral_ts(var_name):
        if var_name in results.sectoral:
            ts = results.sectoral[var_name].sum(axis=1).reindex(t_indices).fillna(0)
            return ts / TO_BILLION
        return pd.Series(0.0, index=t_indices)

    # ========================================================================
    # 2. BUILD DATA TIMELINES
    # ========================================================================
    data = pd.DataFrame(index=years)
    
    # Macro Variables
    data['Solar PV'] = get_macro_ts('I_solar').values
    data['Wind Power'] = get_macro_ts('I_wind').values
    data['Bioenergy'] = get_macro_ts('I_bio').values
    data['Green Hydrogen'] = get_macro_ts('I_hydrogen').values
    data['Battery Storage'] = get_macro_ts('I_storage').values
    data['Ind. Electrification'] = get_macro_ts('I_electrification_total').values
    
    # Energy Efficiency
    if 'I_efficiency_total' in results.macro:
        data['Energy Efficiency'] = get_macro_ts('I_efficiency_total').values
    else:
        data['Energy Efficiency'] = get_sectoral_ts('I_efficiency').values

    # Grid Upgrades
    elec_load_gj = get_macro_ts('electrification_added_load').values * TO_BILLION 
    data['Grid Upgrades'] = (elec_load_gj * 116.0 / 1000.0) / TO_BILLION

    # Sectoral Variables (Net Zero Infrastructure)
    data['Carbon Capture (CCS)'] = get_sectoral_ts('I_netzero_ccs_').values
    data['Carbon Util. (CCU)'] = get_sectoral_ts('I_netzero_ccu_').values
    data['Nature-Based (SGI)'] = get_sectoral_ts('I_netzero_nature_').values

    # --- LOG SCALE FIX: Replace exactly 0 with NaN ---
    # This prevents the log scale from crashing or plunging to negative infinity.
    # The line will simply "start" in the year investment becomes > 0.
    data.replace(0.0, np.nan, inplace=True)

    # ========================================================================
    # 3. RENDER INTERACTIVE PLOTLY CHART
    # ========================================================================
    fig = go.Figure()

    color_map = {
        'Solar PV': '#FFD700', 'Wind Power': '#87CEEB', 'Bioenergy': '#8B4513',
        'Green Hydrogen': '#a1d99b', 'Battery Storage': '#9ecae1',
        'Ind. Electrification': '#fdae6b', 'Energy Efficiency': '#c7e9c0',
        'Grid Upgrades': '#e6550d', 'Carbon Capture (CCS)': '#bdbdbd',
        'Carbon Util. (CCU)': '#969696', 'Nature-Based (SGI)': '#31a354'
    }

    # Add lines to chart
    for column in data.columns:
        # Check if the technology was used (ignore if the whole column is NaN)
        if not data[column].isna().all():
            fig.add_trace(go.Scatter(
                x=data.index, 
                y=data[column], 
                mode='lines+markers',
                name=column,
                line=dict(width=3, color=color_map.get(column, '#000000')),
                marker=dict(size=4),
                connectgaps=False, # Don't draw lines across years where investment was 0
                hovertemplate="<b>%{y:.3f} B SAR</b><extra></extra>"
            ))

    # Layout styling
    fig.update_layout(
        title='<b>Energy Transition Technology Investments (2021-2060)</b><br><i>Annual Capital Expenditure (Billion SAR) - Logarithmic Scale</i>',
        xaxis_title='<b>Year</b>',
        yaxis_title='<b>Investment (Billion SAR, Log Scale)</b>',
        template='plotly_white',
        hovermode='x unified', 
        legend=dict(
            title_text='<b>Technologies</b>',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        ),
        font=dict(size=13),
        margin=dict(t=80, b=40, l=40, r=150)
    )

    # Apply Log Scale to Y-Axis
    fig.update_yaxes(
        type="log",
        showgrid=True, 
        gridwidth=1, 
        gridcolor='LightGray',
        exponentformat='power', # Shows 10^0, 10^1, 10^2 instead of 1, 10, 100 (optional, but cleaner)
        dtick=1 # Places a tick mark at every order of magnitude (0.1, 1, 10, 100, 1000)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    fig.show()

# Run the function

def plot_technology_investment_timeline(results, start_year=2021, end_year=2060):
    """
    Plots an interactive line chart of all energy technology investments from start_year to end_year.
    Converts values to Billion SAR.
    """
    years = np.arange(start_year, end_year + 1)
    t_indices = years - 2020  # Model t indices (e.g., 2021 = 1)
    TO_BILLION = 1_000_000.0

    # ========================================================================
    # 1. HELPER FUNCTIONS TO EXTRACT TIME-SERIES
    # ========================================================================
    def get_macro_ts(var_name):
        """Extracts a variable from results.macro across the timeline."""
        if var_name in results.macro:
            # Reindex to ensure all t_indices are present, fill missing with 0
            ts = results.macro[var_name].reindex(t_indices).fillna(0)
            return ts / TO_BILLION
        return pd.Series(0.0, index=t_indices)

    def get_sectoral_ts(var_name):
        """Extracts and sums a sectoral variable across all sectors."""
        if var_name in results.sectoral:
            # Sum across all sectors for each time step
            ts = results.sectoral[var_name].sum(axis=1).reindex(t_indices).fillna(0)
            return ts / TO_BILLION
        return pd.Series(0.0, index=t_indices)

    # ========================================================================
    # 2. BUILD DATA TIMELINES
    # ========================================================================
    data = pd.DataFrame(index=years)
    
    # Macro Variables
    data['Solar PV'] = get_macro_ts('I_solar').values
    data['Wind Power'] = get_macro_ts('I_wind').values
    data['Bioenergy'] = get_macro_ts('I_bio').values
    data['Green Hydrogen'] = get_macro_ts('I_hydrogen').values
    data['Battery Storage'] = get_macro_ts('I_storage').values
    data['Ind. Electrification'] = get_macro_ts('I_electrification_total').values
    
    # Energy Efficiency (Handle macro vs sectoral flexibility)
    if 'I_efficiency_total' in results.macro:
        data['Energy Efficiency'] = get_macro_ts('I_efficiency_total').values
    else:
        data['Energy Efficiency'] = get_sectoral_ts('I_efficiency').values

    # Grid Upgrades (Calculated dynamically from load, same as Sankey)
    elec_load_gj = get_macro_ts('electrification_added_load').values * TO_BILLION # Un-divide to get true GJ
    data['Grid Upgrades'] = (elec_load_gj * 116.0 / 1000.0) / TO_BILLION

    # Sectoral Variables (Net Zero Infrastructure)
    data['Carbon Capture (CCS)'] = get_sectoral_ts('I_netzero_ccs').values
    data['Carbon Util. (CCU)'] = get_sectoral_ts('I_netzero_ccu').values
    data['Nature-Based (SGI)'] = get_sectoral_ts('I_netzero_nature').values

    # ========================================================================
    # 3. RENDER INTERACTIVE PLOTLY CHART
    # ========================================================================
    fig = go.Figure()

    # Color mapping to match your Sankey plot aesthetic
    color_map = {
        'Solar PV': '#FFD700', 'Wind Power': '#87CEEB', 'Bioenergy': '#8B4513',
        'Green Hydrogen': '#a1d99b', 'Battery Storage': '#9ecae1',
        'Ind. Electrification': '#fdae6b', 'Energy Efficiency': '#c7e9c0',
        'Grid Upgrades': '#e6550d', 'Carbon Capture (CCS)': '#bdbdbd',
        'Carbon Util. (CCU)': '#969696', 'Nature-Based (SGI)': '#31a354'
    }

    # Add lines to chart
    for column in data.columns:
        # Check if the technology was actually used in the scenario (sum > 0)
        if data[column].sum() > 0:
            fig.add_trace(go.Scatter(
                x=data.index, 
                y=data[column], 
                mode='lines+markers',  # Adds points at each year
                name=column,
                line=dict(width=3, color=color_map.get(column, '#000000')),
                marker=dict(size=4),
                hovertemplate="<b>%{y:.2f} B SAR</b><extra></extra>"
            ))

    # Layout styling
    fig.update_layout(
        title='<b>Energy Transition Technology Investments (2021-2060)</b><br><i>Annual Capital Expenditure (Billion SAR)</i>',
        xaxis_title='<b>Year</b>',
        yaxis_title='<b>Investment (Billion SAR)</b>',
        template='plotly_white',
        hovermode='x unified', # Shows all values in a neat box when hovering over a year
        legend=dict(
            title_text='<b>Technologies</b>',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        ),
        font=dict(size=13),
        margin=dict(t=80, b=40, l=40, r=150) # Extra right margin for legend
    )

    # Add a slight background grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True, zerolinewidth=2, zerolinecolor='black')

    fig.show()

# ================================
# How to run it:
# ================================
def plot_real_energy_sankey(results, target_year):
    """
    Generates a Sankey diagram for a specific year (e.g., 2030 or 2060).
    - Segregates Solar, Wind, and Bioenergy.
    - Includes Imports for Battery Storage (and extensible to other tech).
    - Forces all nodes to remain visible even if investment is 0.
    """
    t = target_year - 2020
    TO_BILLION = 1_000_000.0

    # ========================================================================
    # 1. Data extraction (with a small-value fallback for zero values)
    # ========================================================================
    def get_macro(var):
        return max(1e-5, results.macro[var].loc[t] / TO_BILLION if var in results.macro else 1e-5)
    
    def get_sectoral(var):
        try: return max(1e-5, results.sectoral[var].loc[t].sum() / TO_BILLION)
        except: return 1e-5

    # Segregated Renewables
    val_solar = get_macro('I_solar')
    val_wind  = get_macro('I_wind')
    val_bio   = get_macro('I_bio')
    
    # Other Technologies
    val_h2      = get_macro('I_hydrogen')
    val_elec    = get_macro('I_electrification_total')
    
    # Storage - Extract split between local and import
    val_storage        = get_macro('I_storage')
    val_storage_local  = get_macro('I_storage_local')
    val_storage_import = get_macro('I_storage_import')
    
    # Fallback just in case the split variables aren't in results yet
    if val_storage_local == 1e-5 and val_storage_import == 1e-5 and val_storage > 1e-5:
        val_storage_import = val_storage # Assume 100% import if split is missing
    
    # Efficiency
    if 'I_efficiency_total' in results.macro:
        val_eff = get_macro('I_efficiency_total')
    else:
        val_eff = get_sectoral('I_efficiency')

    # Grid Upgrades
    elec_load = results.macro['electrification_added_load'].loc[t] if 'electrification_added_load' in results.macro else 0
    val_grid = max(1e-5, ((elec_load * 116.0) / 1000.0) / TO_BILLION)

    # Net Zero Infrastructure
    val_ccs    = get_sectoral('I_netzero_ccs')
    val_ccu    = get_sectoral('I_netzero_ccu')
    val_nature = get_sectoral('I_netzero_nature')

    # ========================================================================
    # 2. DEFINE NODES
    # ========================================================================
    base_labels = [
        # --- 0 to 4: PAYERS (Ownership) ---
        "Electricity, gas, steam & AC supply", "Manufacture of chemicals (Heavy Ind.)", 
        "Agriculture, Forestry & Fishing", "Target Industries (Electrification)", "All Economy Sectors (Efficiency)",
        
        # --- 5 to 15: TECHNOLOGIES ---
        "Solar PV", "Wind Power", "Bioenergy", "Grid Upgrades", "Green Hydrogen", 
        "Battery Storage", "Carbon Capture (CCS)", "Carbon Util. (CCU)", "Nature-Based (SGI)", 
        "Electrification Eq.", "Efficiency Retrofits",
        
        # --- 16 to 22: BUILDERS (Supply Chain & Imports) ---
        "Manufacture of electrical equipment", "Civil engineering", "Manufacture of basic metals", 
        "Manufacture of machinery n.e.c.", "Construction of buildings", "Rest of Economy (Supply Chain)",
        "Imports (International Markets)" # <-- NEW NODE 22
    ]

    color_map = [
        # Payers
        '#3182bd', '#756bb1', '#31a354', '#9e9ac8', '#bcbddc',
        # Technologies (Solar=Yellow, Wind=SkyBlue, Bio=Brown)
        '#FFD700', '#87CEEB', '#8B4513', '#e6550d', '#a1d99b', '#9ecae1', '#bdbdbd', '#969696', '#31a354', '#fdae6b', '#c7e9c0',
        # Builders & Imports (Import is given a distinct salmon/red color)
        '#636363', '#8c564b', '#737373', '#525252', '#8c564b', '#d9d9d9', '#fb6a4a'
    ]

    # ========================================================================
    # 3. DEFINE FLOWS
    # ========================================================================
    sources, targets, values = [], [], []
    def add_flow(src, tgt, val):
        sources.append(src)
        targets.append(tgt)
        values.append(val)

    # --- A. PAYER TO TECHNOLOGY ---
    add_flow(0, 5, val_solar)
    add_flow(0, 6, val_wind)
    add_flow(0, 7, val_bio)
    add_flow(0, 8, val_grid)
    add_flow(0, 9, val_h2)
    add_flow(0, 10, val_storage)
    
    add_flow(1, 11, val_ccs)
    add_flow(1, 12, val_ccu)
    add_flow(2, 13, val_nature)
    add_flow(3, 14, val_elec)
    add_flow(4, 15, val_eff)

    # --- B. TECHNOLOGY TO BUILDER / IMPORT ---
    
    # Solar (You can route parts of this to node 22 if you calculate Solar imports later)
    add_flow(5, 19, val_solar * 0.40) # Machinery
    add_flow(5, 16, val_solar * 0.40) # Elec eq
    add_flow(5, 17, val_solar * 0.20) # Civil eng
    
    # Wind 
    add_flow(6, 19, val_wind * 0.60) # Machinery
    add_flow(6, 16, val_wind * 0.10) # Elec eq
    add_flow(6, 17, val_wind * 0.30) # Civil eng
    
    # Bioenergy
    add_flow(7, 19, val_bio * 0.60) # Machinery
    add_flow(7, 17, val_bio * 0.40) # Civil
    
    # Grid Upgrades
    add_flow(8, 16, val_grid * 0.45) # Elec eq
    add_flow(8, 17, val_grid * 0.40) # Civil eng
    add_flow(8, 18, val_grid * 0.15) # Metals
    
    # Hydrogen
    add_flow(9, 19, val_h2 * 0.60) # Machinery
    add_flow(9, 17, val_h2 * 0.40) # Civil eng
    
    # --- UPDATED: Storage split between Local Machinery and Imports ---
    add_flow(10, 19, val_storage_local)  # Local production goes to Machinery
    add_flow(10, 22, val_storage_import) # Imports go to International Markets
    
    # CCS, CCU, Nature 
    add_flow(11, 21, val_ccs)
    add_flow(12, 21, val_ccu)
    add_flow(13, 21, val_nature)
    
    # Electrification
    add_flow(14, 19, val_elec * 0.80) # Machinery
    add_flow(14, 16, val_elec * 0.20) # Elec Eq
    
    # Efficiency 
    add_flow(15, 20, val_eff * 0.50) # Construction
    add_flow(15, 19, val_eff * 0.30) # Machinery
    add_flow(15, 21, val_eff * 0.20) # Rest of economy

    # ========================================================================
    # 4. DYNAMIC NODE LABELS
    # ========================================================================
    inflows = [0] * len(base_labels)
    outflows = [0] * len(base_labels)
    
    for src, tgt, val in zip(sources, targets, values):
        outflows[src] += val
        inflows[tgt] += val

    final_labels = []
    for i in range(len(base_labels)):
        node_total = max(inflows[i], outflows[i])
        # If the value is the small-value fallback (0.00001), display it as 0
        if node_total < 0.001:
            final_labels.append(f"{base_labels[i]}<br><b>0.00 B SAR</b>")
        else:
            final_labels.append(f"{base_labels[i]}<br><b>{node_total:.2f} B SAR</b>")

    # ========================================================================
    # 5. RENDER PLOT
    # ========================================================================
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30, thickness=35,
            line=dict(color="black", width=0.5),
            label=final_labels, color=color_map
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color="rgba(180, 180, 180, 0.4)" 
        )
    )])

    fig.update_layout(
        title_text=f"<b>Energy Transition Investment Flows in Saudi Arabia ({target_year})</b><br><i>Tracking Capital Ownership to Supply Chain Demand</i>",
        font_size=12, height=850, width=1200, margin=dict(t=80, b=100, l=40, r=40)
    )

    fig.add_annotation(x=0.0, y=-0.1, text="<b>1. Paying Sector (Ownership)</b>", showarrow=False, xref="paper", yref="paper", font=dict(size=14), yanchor="top")
    fig.add_annotation(x=0.5, y=-0.1, text="<b>2. Technology Focus</b>", showarrow=False, xref="paper", yref="paper", font=dict(size=14), yanchor="top")
    fig.add_annotation(x=1.0, y=-0.1, text="<b>3. Building Sector (Supply & Imports)</b>", showarrow=False, xref="paper", yref="paper", font=dict(size=14), xanchor="right", yanchor="top")

    fig.show()


# ==========================================
# Run it for 2030 and 2060
# ==========================================




def plot_net_zero_cost_simple(results: 'ModelResults'):
    """
    Stacked Bar Chart including HYDROGEN, ELECTRIFICATION, BATTERY STORAGE, and EFFICIENCY Investment.
    Updated Y-Axis limit to 1.05x Max.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    p = results.config.p
    if not getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False): return

    years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000
    
    # --- 1. DATA EXTRACTION ---
    inv_nature = results.sectoral['I_netzero_nature'].sum(axis=1).values / TO_BILLION
    inv_ccs = results.sectoral['I_netzero_ccs'].sum(axis=1).values / TO_BILLION
    inv_ccu = results.sectoral['I_netzero_ccu'].sum(axis=1).values / TO_BILLION
    
    # Hydrogen
    if 'I_hydrogen' in results.macro:
        inv_h2 = results.macro['I_hydrogen'].values / TO_BILLION
    else: inv_h2 = np.zeros_like(inv_nature)

    # Electrification
    if 'I_electrification_total' in results.macro:
        inv_elec = results.macro['I_electrification_total'].values / TO_BILLION
    else: inv_elec = np.zeros_like(inv_nature)

    # [NEW] Efficiency Investment
    if 'I_efficiency_total' in results.macro:
        inv_effic = results.macro['I_efficiency_total'].values / TO_BILLION
    else:
        # Fallback calculation if variable missing (sum of sectoral I_efficiency)
        try:
            inv_effic = results.sectoral['I_efficiency'].sum(axis=1).values / TO_BILLION
        except:
            inv_effic = np.zeros_like(inv_nature)

    # Battery Storage
    if 'I_storage' in results.macro:
        inv_storage = results.macro['I_storage'].values / TO_BILLION
    elif hasattr(results.variables, 'I_storage'):
         inv_storage = results.variables.I_storage[1:] / TO_BILLION
    else: inv_storage = np.zeros_like(inv_nature)

    # Market Compliance Cost
    cost_credits = results.sectoral['I_carbon_credits'].sum(axis=1).values / TO_BILLION
    
    # Total Cost
    total_cost = inv_nature + inv_ccs + inv_ccu + inv_h2 + inv_elec + inv_storage + inv_effic + cost_credits
    gdp_val = results.macro['Y'].values / TO_BILLION
    
    cost_pct_gdp = np.zeros_like(total_cost)
    mask = gdp_val > 0
    cost_pct_gdp[mask] = (total_cost[mask] / gdp_val[mask]) * 100

    # --- 2. PLOTTING ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Stacked Bar Chart Logic
    b1 = np.zeros_like(inv_nature)
    ax.bar(years, inv_nature, bottom=b1, color='#2ca02c', label='Nature (SGI)', alpha=0.8)
    
    b2 = b1 + inv_nature
    ax.bar(years, inv_ccs, bottom=b2, color='#1f77b4', label='CCS Infrastructure', alpha=0.8)
    
    b3 = b2 + inv_ccs
    ax.bar(years, inv_ccu, bottom=b3, color='#9467bd', label='CCU (Recycling)', alpha=0.8)
    
    b4 = b3 + inv_ccu
    ax.bar(years, inv_h2, bottom=b4, color='#e377c2', label='Green Hydrogen (H2)', alpha=0.9)
    
    b5 = b4 + inv_h2
    ax.bar(years, inv_elec, bottom=b5, color='#ff7f0e', label='Electrification (Retrofit)', alpha=0.9)
    
    # [NEW] Efficiency Layer
    b6 = b5 + inv_elec
    ax.bar(years, inv_effic, bottom=b6, color='#17becf', label='Energy Efficiency', alpha=0.9)

    # Battery Storage Layer
    b7 = b6 + inv_effic
    ax.bar(years, inv_storage, bottom=b7, color='#bcbd22', label='Battery Storage', alpha=0.9)
    
    # Credits on top
    b8 = b7 + inv_storage
    ax.bar(years, cost_credits, bottom=b8, color='#d62728', label='Carbon Credits (Market)', alpha=0.8)
    
    # Formatting
    ax.set_title("Total Annual Net Zero Cost (Incl. Efficiency, Batteries & H2)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Billion SAR", fontsize=14)
    ax.set_xlabel("Year", fontsize=14)
    ax.legend(loc='upper left', title="Cost Component", ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # --- UPDATED Y-LIMIT ---
    ax.set_ylim(0, max(total_cost) * 1.15) 

    # --- 3. ANNOTATIONS (Every 5 Years) ---
    print("\n--- COST & GDP SHARE REPORT ---")
    for i, year in enumerate(years):
        if (year >= 2025 and year % 5 == 0) or year == 2060:
            val = total_cost[i]
            pct = cost_pct_gdp[i]
            label = f"{val:.1f}B\n({pct:.1f}%)"
            ax.text(year, val + (max(total_cost)*0.01), label,
                    ha='center', va='bottom', fontweight='bold', fontsize=9, color='black')
            print(f"Year {year}: Total Cost = {val:.2f} Bn SAR ({pct:.2f}% of GDP)")

    plt.tight_layout()
    plt.show()


def plot_green_exports_breakdown(results: 'ModelResults', scenario_name="Net Zero"):
    """
    Creates a Stacked Area Chart showing the contribution of each sector
    to Total Green Exports over time.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    p = results.config.p
    # Only run if meaningful
    if not (getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False) or getattr(p, 'vision_2030_scenario', False)):
        print("Skipping Green Export Breakdown: Scenario is not Net Zero/V2030.")
        return

    # Use length of macro results for years
    years = list(range(2021, 2021 + len(results.macro)))
    
    # 1. Calculate Components (Billion SAR)
    # -------------------------------------
    
    # A. Hydrogen (Chemicals)
    # Access from 'macro' dataframe instead of 'variables' object
    # Note: results.macro already excludes t=0, so no [1:] slicing needed
    if 'hydrogen_export' in results.macro:
        h2_vol = results.macro['hydrogen_export'].values 
        h2_val = (h2_vol * getattr(p, 'hydrogen_export_price_sar_gj', 90.0)) / 1e9
    else:
        h2_val = np.zeros(len(years))

    # B. Carbon Credits (Agriculture)
    if getattr(p, 'enable_carbon_market', False) and 'flow_remove_nature' in results.macro:
        total_removal = results.macro['flow_remove_nature'].values
        credit_val = ((total_removal * 1e6) * 0.20 * getattr(p, 'global_carbon_price_sar', 37.0)) / 1e9
    else:
        credit_val = np.zeros_like(h2_val)

    # C. Solar Manufacturing (Electrical)
    # Re-calculate local content curve locally for visualization
    local_content = np.array([0.0 if t<5 else (0.75 * min(1.0, (t-5)/5.0)) for t in range(len(years))])
    
    if 'I_solar' in results.macro:
        inv_solar = results.macro['I_solar'].values
        solar_val = (inv_solar * local_content * 0.30) / 1e6 # I_solar is in '000 SAR -> Billion
    else:
        solar_val = np.zeros_like(h2_val)

    # D. Wind/Machinery (Machinery)
    # Need I_wind, I_electrification_total, and I_netzero_ccu (sectoral sum)
    if 'I_wind' in results.macro:
        inv_wind = results.macro['I_wind'].values
    else:
        inv_wind = np.zeros_like(h2_val)
        
    if 'I_electrification_total' in results.macro:
        inv_elec = results.macro['I_electrification_total'].values
    else:
        inv_elec = np.zeros_like(h2_val)
        
    # CCU is likely in sectoral, so we sum it if it exists
    if 'I_netzero_ccu' in results.sectoral:
        inv_ccu = results.sectoral['I_netzero_ccu'].sum(axis=1).values
    elif 'I_netzero_ccu' in results.macro:
        inv_ccu = results.macro['I_netzero_ccu'].values
    else:
        inv_ccu = np.zeros_like(h2_val)

    inv_wind_elec = inv_wind + inv_elec + inv_ccu
    wind_val = (inv_wind_elec * local_content * 0.30) / 1e6

    # E. Batteries
    # Only if local share >= 0.99
    if 'I_storage' in results.macro and 'storage_local_share' in results.macro:
        inv_storage = results.macro['I_storage'].values
        local_share = results.macro['storage_local_share'].values
        battery_val = np.where(local_share >= 0.99, (inv_storage * 0.50) / 1e6, 0.0)
    else:
        battery_val = np.zeros_like(h2_val)

    # 2. Plotting
    # -------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Data stack
    data = {
        'Green Hydrogen': h2_val,
        'Solar Mnf.': solar_val,
        'Wind/Machinery Mnf.': wind_val,
        'Batteries': battery_val,
        'Carbon Credits': credit_val
    }
    
    df = pd.DataFrame(data, index=years)
    
    df.plot.area(ax=ax, alpha=0.85, linewidth=0)
    
    ax.set_title(f"Green Export Breakdown: {scenario_name}", fontsize=16, fontweight='bold')
    ax.set_ylabel("Exports (Billion SAR)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_xlim(2021, 2061)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', title="Sector")
    
    # Annotate Total in 2060
    if 2060 in df.index:
        total_2060 = df.loc[2060].sum()
        ax.annotate(f"Total: {total_2060:.1f} Bn", xy=(2060, total_2060), 
                    xytext=(2050, total_2060*1.05),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.show()
def plot_net_zero_funding_sources(results: 'ModelResults'):
    """
    Detailed Stacked Bar Chart showing Funding Sources.
    
    BREAKDOWN (6 Layers):
    1. Govt Infrastructure: Public goods (Nature/CCS/Credits) - Sunk costs.
    2. Govt Strategic: State stake in Mega-projects (H2/Storage).
    3. Priv Chemicals: Equity in Hydrogen & CCU.
    4. Priv Utilities: Independent Power Producers (Storage).
    5. Priv Industry (Elec): Fuel switching investments.
    6. Priv Industry (Eff): Retrofitting investments.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    p = results.config.p
    if not getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False): return

    years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000
    
    # ==============================================================================
    # 1. EXTRACT DATA
    # ==============================================================================
    inv_nature = results.sectoral['I_netzero_nature'].sum(axis=1).values / TO_BILLION
    inv_ccs = results.sectoral['I_netzero_ccs'].sum(axis=1).values / TO_BILLION
    cost_credits = results.sectoral['I_carbon_credits'].sum(axis=1).values / TO_BILLION
    inv_ccu = results.sectoral['I_netzero_ccu'].sum(axis=1).values / TO_BILLION
    
    inv_h2 = results.macro.get('I_hydrogen', pd.Series(0, index=results.macro.index)).values / TO_BILLION
    inv_elec = results.macro.get('I_electrification_total', pd.Series(0, index=results.macro.index)).values / TO_BILLION
    
    # Efficiency (Robust extraction)
    if 'I_efficiency_total' in results.macro:
        inv_effic = results.macro['I_efficiency_total'].values / TO_BILLION
    else:
        try: inv_effic = results.sectoral['I_efficiency'].sum(axis=1).values / TO_BILLION
        except: inv_effic = np.zeros_like(inv_nature)

    # Storage (Robust extraction)
    if 'I_storage' in results.macro:
        inv_storage = results.macro['I_storage'].values / TO_BILLION
    elif hasattr(results.variables, 'I_storage'):
         inv_storage = results.variables.I_storage[1:] / TO_BILLION
    else: inv_storage = np.zeros_like(inv_nature)

    # ==============================================================================
    # 2. SEGREGATE FUNDING (6 Layers)
    # ==============================================================================
    
    # --- Layer 1: Govt Infrastructure (Sunk Costs) ---
    # 100% of Nature, CCS, Credits
    fund_gov_infra = inv_nature + inv_ccs + cost_credits

    # --- Layer 2: Govt Strategic Projects (Catalytic Capital) ---
    # 80% of H2, 50% of Storage, 50% of CCU
    fund_gov_strat = (inv_h2 * 0.8) + (inv_storage * 0.5) + (inv_ccu * 0.5)

    # --- Layer 3: Private Chemicals (New Commodities) ---
    # 20% of H2, 50% of CCU
    fund_priv_chem = (inv_h2 * 0.2) + (inv_ccu * 0.5)
    
    # --- Layer 4: Private Utilities (Grid Assets) ---
    # 50% of Storage
    fund_priv_util = (inv_storage * 0.5)
    
    # --- Layer 5: Private Industry - Electrification ---
    # 100% of Electrification cost
    fund_priv_elec = inv_elec
    
    # --- Layer 6: Private Industry - Efficiency ---
    # 100% of Efficiency cost
    fund_priv_effic = inv_effic

    # Total for Y-Limit
    total_cost = fund_gov_infra + fund_gov_strat + fund_priv_chem + fund_priv_util + fund_priv_elec + fund_priv_effic

    # ==============================================================================
    # 3. PLOTTING
    # ==============================================================================
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Colors (Matching the detailed logic)
    c_g_infra = '#1f77b4'   # Dark Blue
    c_g_strat = '#6baed6'   # Light Blue
    c_p_chem  = '#9467bd'   # Purple
    c_p_util  = '#bcbd22'   # Olive
    c_p_elec  = '#ff7f0e'   # Orange
    c_p_eff   = '#17becf'   # Cyan
    
    # Stack Construction
    # 1. Gov Infra
    ax.bar(years, fund_gov_infra, color=c_g_infra, label='Gov: Infrastructure (CCS/Nature)', alpha=0.9, width=0.9)
    
    # 2. Gov Strat
    b2 = fund_gov_infra
    ax.bar(years, fund_gov_strat, bottom=b2, color=c_g_strat, label='Gov: Strategic Projects (H2/Batt)', alpha=0.9, width=0.9)
    
    # 3. Priv Chem
    b3 = b2 + fund_gov_strat
    ax.bar(years, fund_priv_chem, bottom=b3, color=c_p_chem, label='Priv: Chemicals (H2/CCU)', alpha=0.9, width=0.9)
    
    # 4. Priv Util
    b4 = b3 + fund_priv_chem
    ax.bar(years, fund_priv_util, bottom=b4, color=c_p_util, label='Priv: Utilities (Storage)', alpha=0.9, width=0.9)
    
    # 5. Priv Elec
    b5 = b4 + fund_priv_util
    ax.bar(years, fund_priv_elec, bottom=b5, color=c_p_elec, label='Priv: Industry (Electrification)', alpha=0.9, width=0.9)
    
    # 6. Priv Effic
    b6 = b5 + fund_priv_elec
    ax.bar(years, fund_priv_effic, bottom=b6, color=c_p_eff, label='Priv: Industry (Efficiency)', alpha=0.9, width=0.9)
    
    # Formatting
    ax.set_title("Financing the Transition: Detailed Sectoral Breakdown", fontsize=16, fontweight='bold')
    ax.set_ylabel("Annual Investment (Billion SAR)", fontsize=14)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylim(0, max(total_cost) * 1.25)
    
    ax.legend(loc='upper left', title="Who Pays?", fontsize=10, ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # --- 4. ANNOTATIONS ---
    print("\n--- DETAILED FUNDING REPORT ---")
    for i, year in enumerate(years):
        if (year >= 2025 and year % 5 == 0) or year == 2060:
            
            val = total_cost[i]
            
            # Label Total on top
            ax.text(year, val + (max(total_cost)*0.01), f"{val:.0f}B",
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Helper for Labels
            def label_segment(v, bot, txt):
                if v > 8: # Only label if segment is big enough
                    ax.text(year, bot + v/2, txt, ha='center', va='center', 
                            color='white', fontsize=7, fontweight='bold')

            # Add labels to visible segments
            label_segment(fund_gov_infra[i], 0, "Infra")
            label_segment(fund_gov_strat[i], fund_gov_infra[i], "Strat")
            label_segment(fund_priv_chem[i], fund_gov_infra[i]+fund_gov_strat[i], "Chem")
            label_segment(fund_priv_util[i], fund_gov_infra[i]+fund_gov_strat[i]+fund_priv_chem[i], "Util")
            label_segment(fund_priv_elec[i], fund_gov_infra[i]+fund_gov_strat[i]+fund_priv_chem[i]+fund_priv_util[i], "Elec")
            label_segment(fund_priv_effic[i], total_cost[i]-fund_priv_effic[i], "Eff")

            print(f"Year {year}: Total {val:.1f}B")

    plt.tight_layout()
    plt.show()
def plot_net_zero_required_balance(results: 'ModelResults'):
    """
    Plots Physical Gap vs. Financial Cost.
    VISUALIZATION UPDATE: Added Energy Efficiency to Total Cost Stack.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    p = results.config.p
    if not getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False): return

    years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000
    
    # --- 1. DATA EXTRACTION ---
    residual_emissions = results.macro['emissions_net_total'].values
    
    # Liquidity Curve
    liquidity_curve = []
    start_year_idx = 4
    growth_rate = getattr(p, 'carbon_market_growth_rate', 0.113)
    start_vol = p.carbon_market_liquidity
    for t in range(len(years)):
        if t < start_year_idx: liquidity_curve.append(0)
        else: liquidity_curve.append(start_vol * (1 + growth_rate)**(t - start_year_idx))
    liquidity_curve = np.array(liquidity_curve)
    
    credits_purchased = np.minimum(residual_emissions, liquidity_curve)
    final_balance = residual_emissions - credits_purchased
    
    # --- COST EXTRACTION ---
    cost_credits = results.sectoral['I_carbon_credits'].sum(axis=1).values / TO_BILLION
    inv_ccs = results.sectoral['I_netzero_ccs'].sum(axis=1).values / TO_BILLION
    inv_nature = results.sectoral['I_netzero_nature'].sum(axis=1).values / TO_BILLION
    inv_ccu = results.sectoral['I_netzero_ccu'].sum(axis=1).values / TO_BILLION
    
    inv_h2 = results.macro.get('I_hydrogen', pd.Series(0, index=results.macro.index)).values / TO_BILLION
    inv_elec = results.macro.get('I_electrification_total', pd.Series(0, index=results.macro.index)).values / TO_BILLION

    # [NEW] Efficiency
    if 'I_efficiency_total' in results.macro:
        inv_effic = results.macro['I_efficiency_total'].values / TO_BILLION
    else:
        try: inv_effic = results.sectoral['I_efficiency'].sum(axis=1).values / TO_BILLION
        except: inv_effic = np.zeros_like(inv_nature)

    # Battery Storage
    if 'I_storage' in results.macro:
        inv_storage = results.macro['I_storage'].values / TO_BILLION
    elif hasattr(results.variables, 'I_storage'):
         inv_storage = results.variables.I_storage[1:] / TO_BILLION
    else: inv_storage = np.zeros_like(inv_nature)

    # --- Stack Construction ---
    # Cost Base = Everything EXCEPT Batteries (Efficiency included here)
    cost_base = cost_credits + inv_ccs + inv_nature + inv_ccu + inv_h2 + inv_elec + inv_effic
    
    # Cost Total = Base + Batteries
    cost_total = cost_base + inv_storage
    
    gdp_bn = results.macro['Y'].values / TO_BILLION
    cost_pct_gdp = np.zeros_like(cost_total)
    mask = gdp_bn > 0
    cost_pct_gdp[mask] = (cost_total[mask] / gdp_bn[mask]) * 100

    # --- 2. PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(16, 9))
    
    # Primary Axis (Physical)
    ax1.bar(years, residual_emissions, color='#f0f0f0', label='Physical Emissions (Gap)', width=1.0, edgecolor='white', zorder=1)
    ax1.plot(years, final_balance, color='black', linewidth=1.5, linestyle=':', label='Net Emissions Target', zorder=2)
    
    ax1.set_ylabel("Physical Volume (MtCO2e)", fontsize=12, fontweight='bold', color='grey')
    ax1.tick_params(axis='y', labelcolor='grey')
    ax1.set_ylim(0, max(residual_emissions)*1.2)
    ax1.set_xlabel("Year", fontsize=12)
    
    # Secondary Axis (Financial)
    ax2 = ax1.twinx()
    
    ax2.stackplot(years, cost_base, inv_storage, 
                  labels=['Base Net Zero Cost (Inc. Efficiency)', 'Battery Storage Investment'],
                  colors=['#d62728', '#ff7f0e'], 
                  alpha=0.6, zorder=3)
    
    ax2.plot(years, cost_total, color='#8b0000', linewidth=2, label='Total System Cost')

    ax2.set_ylabel("Annual System Cost (Billion SAR)", color='#d62728', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(0, max(cost_total) * 1.35) 

    # --- 3. ANNOTATIONS ---
    # Battery Arrow
    max_bat_idx = np.argmax(inv_storage)
    max_bat_year = years[max_bat_idx]
    max_bat_val = inv_storage[max_bat_idx]
    
    if max_bat_val > 1:
        y_arrow = cost_base[max_bat_idx] + (max_bat_val / 2)
        ax2.annotate(f"Battery Peak:\n{max_bat_val:.1f} Bn", 
                     xy=(max_bat_year, y_arrow), 
                     xytext=(max_bat_year + 5, y_arrow + 20),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=10, fontweight='bold', color='#ff7f0e',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff7f0e"))

    for i, year in enumerate(years):
        if year >= 2025 and year % 5 == 0:
            total = cost_total[i]
            bat = inv_storage[i]
            pct = cost_pct_gdp[i]
            if bat > (total * 0.05):
                label_text = f"{total:.0f}B ({pct:.1f}%)\n[Bat: {bat:.0f}B]"
            else:
                label_text = f"{total:.0f}B ({pct:.1f}%)"
            ax2.text(year, total + (max(cost_total)*0.02), label_text,
                     color='#8b0000', fontweight='bold', fontsize=9, ha='center', va='bottom',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels1 + labels2, lines1 + lines2))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left', ncol=1)
    
    plt.title("Net Zero Transition: Cost Breakdown (Highlighting Battery Storage)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_net_zero_required_balance_2(results: 'ModelResults'):
    """
    Primary Axis (Stacked Area): Decomposition of emission reductions.
    Secondary Axis (Line): Total System Cost + GDP % + Baseline + Narrative Labels.
    Update: Added Battery Storage AND Energy Efficiency to Total System Cost.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    pc = results.config.pc
    p = results.config.p
    if not getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False):
        print("Net Zero Target is OFF. Skipping plot.")
        return

    years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000
    
    # ==============================================================================
    # 1. DATA EXTRACTION (Abatement Wedges)
    # ==============================================================================
    abated_ccu = results.macro['flow_reuse_recycle'].values
    abated_ccs = results.macro['flow_remove_ccs'].values
    abated_nature = results.macro['flow_remove_nature'].values
    net_physical = results.macro['emissions_net_total'].values
    
    # Hydrogen (Abatement)
    h2_use_gj = results.macro.get('hydrogen_domestic_use', np.zeros_like(abated_ccs))
    abated_h2 = (h2_use_gj * 0.06) / 1_000_000.0

    # Electrification (Abatement)
    abated_elec = np.zeros_like(abated_ccs)
    if getattr(p, 'enable_industrial_electrification', False):
        try:
            rate = p.electrification_annual_rate
            target_indices = pc.electrification_target_sectors
            actual_oil = results.sectoral['oil_use_final'][target_indices].sum(axis=1).values
            actual_gas = results.sectoral['gas_use_final'][target_indices].sum(axis=1).values
            shift_pct_curve = np.array([1.0 - (1.0 - rate)**(t) for t in range(len(years))])
            divisor = np.maximum(1.0 - shift_pct_curve, 0.001)
            direct_saved = ((actual_oil/divisor - actual_oil) * pc.oil_emission_factor) + \
                           ((actual_gas/divisor - actual_gas) * pc.gas_emission_factor)
            grid_intensity = results.macro['electricity_emissions_total'].values / results.macro['electricity_use_total'].values
            abated_elec = np.maximum(0, direct_saved - (((actual_oil/divisor - actual_oil) + (actual_gas/divisor - actual_gas)) / 2.5) * grid_intensity)
        except: pass

    # Carbon Market
    cm_growth_rate = getattr(p, 'carbon_market_growth_rate', 0.11)
    start_vol = p.carbon_market_liquidity
    liquidity_curve = np.array([0 if t < 4 else start_vol * (1 + cm_growth_rate)**(t - 4) for t in range(len(years))])
    abated_market = np.minimum(net_physical, liquidity_curve)
    final_gap = net_physical - abated_market 

    # --- Baseline Emissions ---
    baseline_emissions = final_gap + abated_market + abated_nature + abated_ccs + abated_ccu + abated_h2 + abated_elec
    
    # ==============================================================================
    # 2. FINANCIAL DATA (Total Cost including Efficiency)
    # ==============================================================================
    cost_credits = results.sectoral['I_carbon_credits'].sum(axis=1).values / TO_BILLION
    inv_ccs = results.sectoral['I_netzero_ccs'].sum(axis=1).values / TO_BILLION
    inv_nature = results.sectoral['I_netzero_nature'].sum(axis=1).values / TO_BILLION
    inv_ccu = results.sectoral['I_netzero_ccu'].sum(axis=1).values / TO_BILLION
    inv_h2 = results.macro.get('I_hydrogen', pd.Series(0, index=results.macro.index)).values / TO_BILLION
    inv_elec = results.macro.get('I_electrification_total', pd.Series(0, index=results.macro.index)).values / TO_BILLION
    inv_storage = results.macro.get('I_storage', pd.Series(0, index=results.macro.index)).values / TO_BILLION

    # [NEW] Efficiency
    if 'I_efficiency_total' in results.macro:
        inv_effic = results.macro['I_efficiency_total'].values / TO_BILLION
    else:
        try: inv_effic = results.sectoral['I_efficiency'].sum(axis=1).values / TO_BILLION
        except: inv_effic = np.zeros_like(inv_ccs)

    total_system_cost_bn = cost_credits + inv_ccs + inv_nature + inv_ccu + inv_h2 + inv_elec + inv_storage + inv_effic
    
    gdp_bn = results.macro['Y'].values / TO_BILLION
    cost_pct_gdp = np.zeros_like(total_system_cost_bn)
    mask = gdp_bn > 0
    cost_pct_gdp[mask] = (total_system_cost_bn[mask] / gdp_bn[mask]) * 100

    # ==============================================================================
    # 3. PLOTTING
    # ==============================================================================
    fig, ax1 = plt.subplots(figsize=(18, 11))
    pal = ['#e0e0e0', '#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#e377c2', '#ff7f0e']
    labels = ['Unabated Gap', 'Market Offsets', 'Nature (SGI)', 'CCS Removal', 'CCU / Reuse', 'Green H2', 'Electrification']
    
    # Primary Axis: Stacked Area
    ax1.stackplot(years, final_gap, abated_market, abated_nature, abated_ccs, abated_ccu, abated_h2, abated_elec,
                  colors=pal, labels=labels, alpha=0.85)
    
    # --- ADDED SGI UNCERTAINTY BAND ---
    # Calculate the bottom of the Nature wedge
    nature_bottom = final_gap + abated_market
    
    # Calculate the min/max limits over time
    min_nature = np.zeros(len(years))
    max_nature = np.zeros(len(years))
    for i, y in enumerate(years):
        time_step = y - 2021 + 1
        fraction = 0.0 if time_step <= 1 else ((time_step - 1) / 39.0 if time_step <= 40 else 1.0)
        min_nature[i] = 0.2 + (20.0 - 0.2) * (fraction ** 2)
        max_nature[i] = 2.0 + (100.0 - 2.0) * (fraction ** 2)

    # Plot the band over the Nature wedge
    ax1.fill_between(years, nature_bottom + min_nature, nature_bottom + max_nature,
                     facecolor='none', edgecolor='#2ca02c', hatch='///', alpha=0.5, 
                     linewidth=1.5, label='SGI Uncertainty Range')
    # -----------------------------------

    # Baseline line
    ax1.plot(years, baseline_emissions, color='black', linestyle=':', linewidth=2.5, label='Baseline Emissions', zorder=5)

    # --- Axis Limits ---
    y_max_ems = max(baseline_emissions) * 1.1 
    ax1.set_ylim(0, y_max_ems)
    ax1.set_xlim(years[0], 2062)

    # --- A. Vertical Milestone Lines ---
    milestones = [
        (2030, "Vision 2030\n(600M Trees)", 'black'),
        (2035, "CCS Target\n(44 Mtpa)", '#1f77b4'), 
        (2060, "Net Zero\nTarget", 'green')
    ]
    for yr, txt, col in milestones:
        if yr in years:
            ax1.axvline(x=yr, color=col, linestyle='--', linewidth=2, alpha=0.7)
            ax1.text(yr, y_max_ems * 0.92, txt, color=col, fontweight='bold', ha='center', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor=col))

    # --- B. Narrative Arrows ---
    def get_wedge_center(year_idx, wedge_indices):
        data_stack = [final_gap, abated_market, abated_nature, abated_ccs, abated_ccu, abated_h2, abated_elec]
        bottom = sum(data_stack[i][year_idx] for i in range(wedge_indices))
        return bottom + (data_stack[wedge_indices][year_idx] * 0.5)

    narrative_points = [
        (2028, 2, "SGI Scale-up\n(600M Trees)", '#2ca02c'),
        (2035, 3, "Jubail CCS Hub\n(Full Capacity)", '#1f77b4'),
        (2045, 6, "Industrial\nElectrification", '#ff7f0e'),
        (2050, 5, "Green H2\nSubstitution", '#e377c2'),
        (2025, 1, "Carbon Market\nLaunch", '#d62728'),
        (2052, 0, "Hard-to-Abate\nEmissions", '#666666')
    ]

    for yr, idx, txt, col in narrative_points:
        if yr in years:
            yr_idx = years.index(yr)
            y_pos = get_wedge_center(yr_idx, idx)
            y_offset = 150 if "Carbon Market" in txt else (-100 if "Green H2" in txt else 0)
            ax1.annotate(txt, xy=(yr, y_pos), xytext=(yr + 3, y_pos + y_offset),
                         arrowprops=dict(facecolor=col, shrink=0.05, lw=0),
                         fontsize=10, fontweight='bold', color=col, va='center',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.9))

    # --- C. Future Growth Targets Box (Moved to North-West) ---
    elec_rate_disp = getattr(p, 'electrification_annual_rate', 0.03) * 100
    cm_growth_disp = cm_growth_rate * 100
    growth_targets_text = (
        "FUTURE GROWTH & POLICY TARGETS:\n"
        "-------------------------------\n"
        "• CCS: +3% Annual Expansion (>44 Mtpa)\n"
        "• CCU: Scaling to 5% of Gross Emissions\n"
        "• SGI: Ramp to 10 Billion Trees (2060)\n"
        f"• Electrification: {elec_rate_disp:.1f}% Annual Sector Shift\n"
        "• Hydrogen: 4 Mt (2030 Target)\n"
        f"• Carbon Market: Growth at {cm_growth_disp:.1f}%/yr"
    )
    # Positioning at left-north (North-West)
    ax1.text(years[1], y_max_ems * 0.95, growth_targets_text, fontsize=10, fontweight='bold', color='#333333',
             verticalalignment='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#333333", lw=2, alpha=0.95))

    # --- Secondary Axis: Cost & GDP % ---
    ax2 = ax1.twinx()
    ax2.plot(years, total_system_cost_bn, color='#444444', linewidth=2, alpha=0.4, label='Total Cost')
    
    # Updated to 10-year intervals and axis starting at 40
    ax2.set_ylim(40, max(total_system_cost_bn) * 1.1) 

    for yr in [2030, 2040, 2050, 2060]: # Every 10 years
        if yr in years:
            idx = years.index(yr)
            val, pct = total_system_cost_bn[idx], cost_pct_gdp[idx]
            ax2.plot(yr, val, marker='o', color='black', markersize=6)
            ax2.annotate(f"{val:.1f} Bn\n({pct:.1f} % GDP)", xy=(yr, val), xytext=(0, 10), 
                         textcoords='offset points', ha='center', fontsize=9, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3))

    ax1.set_xlabel("Year", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Emissions (MtCO2e)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Annual System Cost (Bn SAR)", color='#444444', fontsize=12, fontweight='bold')

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1[::-1] + lines2, labels1[::-1] + labels2, loc='center left', bbox_to_anchor=(1.1, 0.5))

    ax1.set_title("Saudi Arabia Net Zero Roadmap: Detailed Abatement & Investment Path", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_water_energy_nexus(results: 'ModelResults'):
    """
    Visualizes the Water-Energy Nexus: 
    Primary Axis: Water Sources (Stacked Bars, Light/Dark) & SGI Demand (Error Bar)
    Secondary Axis: Other Green Tech Water Demand (Stacked Bars)
    Selected Years: 2025, 2030, 2040, 2050, 2060
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    pc = results.config.pc
    p = results.config.p
    years = list(range(2021, 2021 + len(results.macro)))

    # ==============================================================================
    # 1. Define Water Intensities
    # ==============================================================================
    wi_solar = 0.02 
    wi_wind = 0.0 
    wi_hydrogen = 0.15 
    wi_electrification = 0.005 
    wi_ccs_bm3_per_mt = 0.002 
    wi_ccu_bm3_per_mt = 0.0025 
    wi_sgi_bm3_per_mt = 0.315 

    # ==============================================================================
    # 2. Extract Data (Universal - No Forced Zeroing)
    # ==============================================================================
    re_solar = results.macro.get('re_gen_solar', pd.Series(np.zeros(len(years)))).values
    re_wind = results.macro.get('re_gen_wind', pd.Series(np.zeros(len(years)))).values
    h2_prod = results.macro.get('hydrogen_production', pd.Series(np.zeros(len(years)))).values
    elec_load = results.macro.get('electrification_added_load', pd.Series(np.zeros(len(years)))).values
    ccs_flow = results.macro.get('flow_remove_ccs', pd.Series(np.zeros(len(years)))).values
    ccu_flow = results.macro.get('flow_reuse_recycle', pd.Series(np.zeros(len(years)))).values
    sgi_flow = results.macro.get('flow_remove_nature', pd.Series(np.zeros(len(years)))).values

    # Calculate Tech Demands in BILLION m3
    water_solar_bm3 = (re_solar * wi_solar) * 1e-9
    water_wind_bm3 = (re_wind * wi_wind) * 1e-9
    water_hydrogen_bm3 = (h2_prod * wi_hydrogen) * 1e-9
    water_electrification_bm3 = (elec_load * wi_electrification) * 1e-9
    water_ccs_bm3 = ccs_flow * wi_ccs_bm3_per_mt
    water_ccu_bm3 = ccu_flow * wi_ccu_bm3_per_mt
    water_sgi_bm3 = sgi_flow * wi_sgi_bm3_per_mt

    # Extract Water Sources (Convert to Billion m3)
    water_ww_bm3 = results.macro.get('water_energy_wwater', pd.Series(np.zeros(len(years)))).values * 1e-9
    water_desal_bm3 = results.macro.get('water_energy_desal', pd.Series(np.zeros(len(years)))).values * 1e-9
    water_gw_bm3 = results.macro.get('water_energy_gw', pd.Series(np.zeros(len(years)))).values * 1e-9

    # Calculate SGI Uncertainty Bounds
    min_sgi_water = np.zeros(len(years))
    max_sgi_water = np.zeros(len(years))
    if getattr(p, 'net_emission_reduction_green_investments_exports_transformation', False) or getattr(p, 'vision_2030_target_CCU_SGI_Carbon', False):
        for i, y in enumerate(years):
            time_step = y - 2021 + 1
            fraction = 0.0 if time_step <= 1 else ((time_step - 1) / 39.0 if time_step <= 40 else 1.0)
            
            min_mt = 0.2 + (20.0 - 0.2) * (fraction ** 2)
            max_mt = 2.0 + (100.0 - 2.0) * (fraction ** 2)
            
            min_sgi_water[i] = min_mt * wi_sgi_bm3_per_mt
            max_sgi_water[i] = max_mt * wi_sgi_bm3_per_mt
    else:
        min_sgi_water = water_sgi_bm3.copy()
        max_sgi_water = water_sgi_bm3.copy()

    # ==============================================================================
    # 3. FILTER DATA FOR SPECIFIC YEARS (2025, 2030, 2040, 2050, 2060)
    # ==============================================================================
    target_years = [2025, 2030, 2040, 2050, 2060]
    valid_years = [y for y in target_years if y in years]
    idx = [years.index(y) for y in valid_years]

    # Filtered Arrays
    f_ww = water_ww_bm3[idx]
    f_ds = water_desal_bm3[idx]
    f_gw = water_gw_bm3[idx]
    
    f_sgi_mean = water_sgi_bm3[idx]
    f_sgi_min = min_sgi_water[idx]
    f_sgi_max = max_sgi_water[idx]

    f_wind = water_wind_bm3[idx] * 1000
    f_elec = water_electrification_bm3[idx] * 1000
    f_solar = water_solar_bm3[idx] * 1000
    f_ccu = water_ccu_bm3[idx] * 1000
    f_ccs = water_ccs_bm3[idx] * 1000
    f_h2 = water_hydrogen_bm3[idx] * 1000

    # ==============================================================================
    # 4. PLOTTING SETUP
    # ==============================================================================
    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax2 = ax1.twinx()

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    x_pos = np.array(valid_years)
    width = 1.2
    x_primary = x_pos - width/2
    x_secondary = x_pos + width/2

    # ==============================================================================
    # 5. SECONDARY AXIS: OTHER TECHNOLOGIES (Million m3)
    # ==============================================================================
    data_other_mm3 = [f_wind, f_elec, f_solar, f_ccu, f_ccs, f_h2]
    labels_other = ['Wind Cooling', 'Grid Electrification', 'Solar PV Washing', 'CCU Process', 'CCS Cooling', 'Green Hydrogen']
    colors_other = ['#e0f3ff', '#ffbb78', '#ffdd57', '#c5b0d5', '#aec7e8', '#f7b6d2'] 
    
    bottom_sec = np.zeros(len(valid_years))
    bars_sec = []
    for data, color, label in zip(data_other_mm3, colors_other, labels_other):
        b = ax2.bar(x_secondary, data, width, bottom=bottom_sec, color=color, label=label, edgecolor='white', linewidth=0.5)
        bars_sec.append(b)
        bottom_sec += data

    # ==============================================================================
    # 6. PRIMARY AXIS: SOURCES (Billion m3) & SGI ERROR BARS
    # ==============================================================================
    color_ww = '#add8e6' # Light Blue
    color_ds = '#4682b4' # Medium Blue
    color_gw = '#000080' # Dark Blue

    b1 = ax1.bar(x_primary, f_ww, width, color=color_ww, label='Source: Wastewater', edgecolor='white', linewidth=0.5)
    b2 = ax1.bar(x_primary, f_ds, width, bottom=f_ww, color=color_ds, label='Source: Desalination', edgecolor='white', linewidth=0.5)
    b3 = ax1.bar(x_primary, f_gw, width, bottom=f_ww+f_ds, color=color_gw, label='Source: Groundwater', edgecolor='white', linewidth=0.5)

    yerr_lower = f_sgi_mean - f_sgi_min
    yerr_upper = f_sgi_max - f_sgi_mean
    yerr_lower = np.clip(yerr_lower, 0, None)
    yerr_upper = np.clip(yerr_upper, 0, None)
    
    if max(f_sgi_mean) > 0:
        err_sgi = ax1.errorbar(x_primary, f_sgi_mean, yerr=[yerr_lower, yerr_upper], fmt='o', color='black', 
                               capsize=8, capthick=2, linewidth=2, markersize=8, 
                               label='SGI Demand (Mean ± Min/Max Uncertainty)', zorder=10)
        handles1 = [err_sgi, b1, b2, b3]
    else:
        handles1 = [b1, b2, b3]

    # ==============================================================================
    # 7. FORMATTING & LEGENDS
    # ==============================================================================
    ax1.set_title("Water-Energy Nexus: SGI Uncertainty vs. Technology Demand", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Year", fontsize=12, fontweight='bold')
    
    ax1.set_ylabel("SGI & Water Sources (Billion cubic meters)", fontsize=12, fontweight='bold', color='black')
    ax2.set_ylabel("Other Tech Demand (Million cubic meters)", fontsize=12, fontweight='bold', color='#555555')
    
    ax1.set_xticks(valid_years)
    ax1.set_xticklabels(valid_years, fontsize=12, fontweight='bold')
    
    max_primary = max(f_sgi_max.max(), (f_ww + f_ds + f_gw).max()) if len(valid_years) > 0 else 0
    if max_primary > 0:
        ax1.set_ylim(0, max_primary * 1.2)
    else:
        ax1.set_ylim(0, 1.0)
        
    if bottom_sec.max() > 0:
        ax2.set_ylim(0, bottom_sec.max() * 1.3)
    else:
        ax2.set_ylim(0, 1.0)

    labels1 = [h.get_label() for h in handles1]
    ax1.legend(handles1, labels1, loc='upper left', fontsize=11, frameon=True, title="Primary Axis (Billions m³)")
    
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles2[::-1], labels2[::-1], loc='center left', bbox_to_anchor=(0.0, 0.6), fontsize=11, frameon=True, title="Secondary Axis (Millions m³)")

    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.xaxis.grid(True, linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.show()
def plot_dual_challenge_valley(results: 'ModelResults'):
    """
    Visualizes Economic Diversification: Oil vs. Total Non-Oil Exports on Left Axis.
    GDP plotted on Secondary Right Axis.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    pc = results.config.pc
    p = results.config.p
    
    years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000
    
    # ==============================================================================
    # 1. DATA EXTRACTION
    # ==============================================================================
    
    # --- A. GDP (Denominator & Secondary Plot) ---
    if 'Y' in results.macro.columns:
        gdp_val = results.macro['Y'].values / TO_BILLION
    else:
        gdp_val = results.macro.get('gdp_nominal', np.zeros_like(years)).values / TO_BILLION

    # --- B. Oil Exports ---
    oil_idx = -1
    if hasattr(pc, 'sectors'):
        for i, s_name in enumerate(pc.sectors):
            if any(k in s_name.lower() for k in ['petroleum', 'oil', 'mining']):
                oil_idx = i
                break
    
    if 'exports_oil' in results.macro:
        oil_exports = results.macro['exports_oil'].values / TO_BILLION
    elif 'exports' in results.sectoral and oil_idx != -1:
        vals = results.sectoral['exports']
        if isinstance(vals, pd.DataFrame): vals = vals.values
        oil_exports = vals[:, oil_idx] / TO_BILLION
    else:
        total_e = results.macro.get('E', results.macro.get('Y')*0.4).values / TO_BILLION
        share = np.linspace(0.70, 0.35, len(years))
        oil_exports = total_e * share

    # --- C. Total Non-Oil Exports ---
    if 'EX_non_oil_total' in results.macro:
        nonoil_exports_total = results.macro['EX_non_oil_total'].values / TO_BILLION
    else:
        total_e = results.macro.get('E', np.zeros_like(years)).values / TO_BILLION
        nonoil_exports_total = total_e - oil_exports

    # --- D. Calculate Contribution % ---
    mask = gdp_val > 0
    pct_oil = np.zeros_like(gdp_val)
    pct_nonoil = np.zeros_like(gdp_val)
    
    pct_oil[mask] = (oil_exports[mask] / gdp_val[mask]) * 100
    pct_nonoil[mask] = (nonoil_exports_total[mask] / gdp_val[mask]) * 100

    # ==============================================================================
    # 2. PLOTTING
    # ==============================================================================
    fig, ax1 = plt.subplots(figsize=(16, 10))
    
    color_oil = '#1f77b4'    # Blue (Oil)
    color_nonoil = '#2ca02c' # Green (Total Non-Oil)
    color_gdp = '#7f7f7f'    # Grey (GDP)

    # --- PLOT SERIES (Primary Axis: Exports) ---
    # Oil
    ax1.plot(years, oil_exports, color=color_oil, linewidth=3, linestyle='-', label='Oil Exports')
    # Fill under Oil for visual weight
    ax1.fill_between(years, 0, oil_exports, color=color_oil, alpha=0.1)
    
    # Non-Oil
    ax1.plot(years, nonoil_exports_total, color=color_nonoil, linewidth=4, linestyle='-', label='Total Non-Oil Exports')
    
    # Axis Formatting
    ax1.set_xlabel("Year", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Exports (Billion SAR)", fontsize=12, fontweight='bold')
    
    # Set Y Limit based on the maximum of BOTH series
    max_val = max(max(oil_exports), max(nonoil_exports_total))
    ax1.set_ylim(0, max_val * 1.35) 

    # --- SECONDARY AXIS (GDP) ---
    ax2 = ax1.twinx()
    ax2.plot(years, gdp_val, color=color_gdp, linewidth=2.5, linestyle='-.', label='Nominal GDP')
    
    ax2.set_ylabel("Nominal GDP (Billion SAR)", color=color_gdp, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_gdp)
    ax2.set_ylim(0, max(gdp_val) * 1.2) # Give GDP some headroom

    # --- 3. ANNOTATIONS (Every 10 Years) ---
    for yr in [2030, 2040, 2050, 2060]:
        if yr in years:
            idx = years.index(yr)
            
            # --- Oil Label ---
            val_o = oil_exports[idx]
            p_o = pct_oil[idx]
            ax1.plot(yr, val_o, marker='o', color='white', markeredgecolor=color_oil, markersize=8, zorder=10)
            ax1.annotate(f"Oil:\n{val_o:.0f}Bn\n({p_o:.1f}% GDP)", 
                         xy=(yr, val_o), xytext=(0, -45), textcoords='offset points',
                         ha='center', fontsize=9, fontweight='bold', color=color_oil,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_oil, alpha=0.9))

            # --- Non-Oil Label ---
            val_n = nonoil_exports_total[idx]
            p_n = pct_nonoil[idx]
            ax1.plot(yr, val_n, marker='o', color='white', markeredgecolor=color_nonoil, markersize=8, zorder=10)
            ax1.annotate(f"Non-Oil:\n{val_n:.0f}Bn\n({p_n:.1f}% GDP)", 
                         xy=(yr, val_n), xytext=(0, 35), textcoords='offset points',
                         ha='center', fontsize=9, fontweight='bold', color=color_nonoil,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_nonoil, alpha=0.9))

    # --- 4. CROSSOVER INDICATOR ---
    # Now that they are on the same axis, the visual intersection IS the numeric intersection
    diff = nonoil_exports_total - oil_exports
    crossings = np.where(np.diff(np.sign(diff)))[0]
    
    if len(crossings) > 0:
        cross_idx = crossings[0]
        cross_year = years[cross_idx]
        if cross_year > 2025:
            ax1.axvline(x=cross_year, color='grey', linestyle=':', linewidth=2, alpha=0.6)
            
            # Place label near the intersection point
            y_cross = (oil_exports[cross_idx] + nonoil_exports_total[cross_idx]) / 2
            
            ax1.text(cross_year, y_cross * 1.15, "STRATEGIC\nCROSSOVER", 
                     ha='center', fontsize=10, fontweight='bold', color='#555555',
                     bbox=dict(fc='white', ec='#555555', alpha=0.85))

    ax1.set_title("Economic Transformation: Closing the Gap (Oil vs. Non-Oil Exports)", fontsize=16, fontweight='bold', pad=20)
    
    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3)
    
    plt.grid(True, axis='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_net_zero_tech_contribution_vs_cost(results: 'ModelResults'):
    """
    Dual-Axis Chart:
    1. Primary (Left): Stacked 'Wedges' showing how each technology contributes
       to reducing emissions from the Baseline to Net Zero.
    2. Secondary (Right): Total System Cost curve with % of GDP annotations.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    p = results.config.p
    if not p.net_emission_reduction_green_investments_exports_transformation: return

    years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000
    
    # ==============================================================================
    # 1. PREPARE PHYSICAL DATA (The Abatement Wedges)
    # ==============================================================================
    
    # A. Explicit Removals (Directly from model output)
    # These are positive flows of CO2 being captured/removed
    abated_ccu = results.macro['flow_reuse_recycle'].values
    abated_ccs = results.macro['flow_remove_ccs'].values
    abated_nature = results.macro['flow_remove_nature'].values
    
    # B. Market Offsets (Calculated based on liquidity limit)
    # Re-calculate the actual purchased volume logic
    start_year_idx = 4 # 2025
    growth_rate = getattr(p, 'carbon_market_growth_rate', 0.12)
    start_vol = p.carbon_market_liquidity
    
    liquidity_curve = []
    for t in range(len(years)):
        if t < start_year_idx:
            liquidity_curve.append(0)
        else:
            val = start_vol * (1 + growth_rate)**(t - start_year_idx)
            liquidity_curve.append(val)
    liquidity_curve = np.array(liquidity_curve)
    
    # The market fills the gap remaining after physical technologies
    # (Simplified logic for visualization: the market takes the remaining amount up to the limit)
    net_emissions_physical = results.macro['emissions_net_total'].values # This already includes CCS/Nature subtractions in model
    # We need to reconstruct the "Gross" before offsets to calculate the offset wedge correctly
    # In this model structure, 'emissions_net_total' IS the final balance physical.
    # To show offsets as a wedge, we treat them as further reduction.
    
    # Let's verify what 'emissions_net_total' represents in energy_module.py:
    # v.emissions_net_total[t] = v.emissions_gross_total[t] - (CCU + CCS + Nature)
    # So, Market Credits are applied *after* this to reach financial net zero.
    
    residual_before_market = results.macro['emissions_net_total'].values
    abated_market = np.minimum(residual_before_market, liquidity_curve)
    
    # C. Hydrogen Contribution (Implicit in Model -> Explicit in Graph)
    # In the model, H2 reduces 'gross_emissions' directly by substituting fuel.
    # To show it as a wedge, we calculate "Avoided Emissions".
    # Factor: ~0.06 tons CO2 per GJ (Average of Oil/Gas)
    if 'hydrogen_domestic_use' in results.macro:
        h2_use_gj = results.macro['hydrogen_domestic_use'].values
        # Using a weighted emission factor (approx 60 kgCO2/GJ for gas/oil mix)
        emission_factor_avoided = 0.06
        abated_h2 = h2_use_gj * emission_factor_avoided
    else:
        abated_h2 = np.zeros_like(abated_ccs)

    # D. Reconstruct the "Counterfactual Baseline"
    # This represents what emissions would be if we did NOTHING (No H2, No CCS, No Nature)
    # Start with Final Net Position (after market)
    final_net_position = residual_before_market - abated_market
    
    # Stack upwards to get baseline
    # Baseline = Final + Market + Nature + CCS + CCU + H2
    
    # ==============================================================================
    # 2. PREPARE FINANCIAL DATA (The Cost Line)
    # ==============================================================================
    
    # CapEx + OpEx
    inv_ccu = results.sectoral['I_netzero_ccu'].sum(axis=1).values / TO_BILLION
    inv_ccs = results.sectoral['I_netzero_ccs'].sum(axis=1).values / TO_BILLION
    inv_nature = results.sectoral['I_netzero_nature'].sum(axis=1).values / TO_BILLION
    cost_credits = results.sectoral['I_carbon_credits'].sum(axis=1).values / TO_BILLION
    
    if 'I_hydrogen' in results.macro:
        inv_h2 = results.macro['I_hydrogen'].values / TO_BILLION
    else:
        inv_h2 = np.zeros_like(inv_ccu)
        
    total_cost_bn = inv_ccu + inv_ccs + inv_nature + cost_credits + inv_h2
    
    # GDP Share
    gdp_bn = results.macro['Y'].values / TO_BILLION
    cost_pct_gdp = np.divide(total_cost_bn, gdp_bn, out=np.zeros_like(gdp_bn), where=gdp_bn>0) * 100

    # ==============================================================================
    # 3. PLOTTING
    # ==============================================================================
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # --- PRIMARY AXIS: ABATEMENT STACK (The Wedges) ---
    
    # We plot the "Remaining Emissions" at the bottom
    # Then stack the solutions on top
    
    # Define the stack order (Bottom -> Top)
    # 1. Final Net Emissions (The Gap/Goal)
    # 2. Market Offsets
    # 3. Nature
    # 4. CCS
    # 5. CCU
    # 6. Hydrogen Substitution
    
    pal = ['#e0e0e0', '#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#e377c2']
    labels = ['Remaining Net Emissions', 'Market Credits (Offsets)', 'Nature (SGI)', 'CCS (Removal)', 'CCU (Recycling)', 'Green Hydrogen (Avoided)']
    
    ax1.stackplot(years,
                  final_net_position,
                  abated_market,
                  abated_nature,
                  abated_ccs,
                  abated_ccu,
                  abated_h2,
                  colors=pal, labels=labels, alpha=0.85)
    
    # Add a dashed line for the "Counterfactual Baseline"
    baseline_emissions = final_net_position + abated_market + abated_nature + abated_ccs + abated_ccu + abated_h2
    ax1.plot(years, baseline_emissions, color='black', linestyle='--', linewidth=2, label='BAU Baseline (No Action)')

    ax1.set_ylabel("Emissions (MtCO2e)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_xlim(years[0], years[-1])
    ax1.set_ylim(0, max(baseline_emissions)*1.1)
    
    # --- SECONDARY AXIS: COST CURVE ---
    ax2 = ax1.twinx()
    
    # Plot Cost Line
    line_cost, = ax2.plot(years, total_cost_bn, color='black', linewidth=3, marker='o', markersize=5, label='Total System Cost (Bn SAR)')
    
    ax2.set_ylabel("Total Annual Cost (Billion SAR)", color='black', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(total_cost_bn)*1.35) # Extra headroom for text
    
    # Annotations for Cost
    for i, year in enumerate(years):
        if year >= 2025 and year % 5 == 0:
            cost_val = total_cost_bn[i]
            pct_val = cost_pct_gdp[i]
            
            # Box Style Annotation
            label_text = f"{cost_val:.1f}B\n({pct_val:.1f}%)"
            
            ax2.text(year, cost_val + (max(total_cost_bn)*0.07), label_text,
                     color='black', fontweight='bold', fontsize=9, ha='center',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.9))

    # --- FINAL FORMATTING ---
    
    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    # Reverse legend order to match stack visual (Top to Bottom)
    # But keep Baseline line at top
    # Display these values directly
    ax1.legend(handles1[::-1], labels1[::-1], loc='upper left', bbox_to_anchor=(1.05, 1), title="Abatement Wedge")
    
    # Add Cost Legend separately or manually placed
    ax2.legend([line_cost], ['Total System Cost'], loc='upper right')

    plt.title("Path to Net Zero: Technology Contributions vs. Financial Cost", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
def plot_net_zero_total_cost_breakdown(results: 'ModelResults'):
    """
    Detailed Area Chart Breakdown.
    """
    years = list(range(2021, 2021 + len(results.macro)))
    TO_BILLION = 1_000_000
    
    # 1. Costs (Outflows)
    inv_ccu = results.sectoral['I_netzero_ccu'].sum(axis=1).values / TO_BILLION
    inv_ccs = results.sectoral['I_netzero_ccs'].sum(axis=1).values / TO_BILLION
    inv_nature = results.sectoral['I_netzero_nature'].sum(axis=1).values / TO_BILLION
    cost_credits = results.sectoral['I_carbon_credits'].sum(axis=1).values / TO_BILLION
    
    # Hydrogen
    if 'I_hydrogen' in results.macro:
        inv_h2 = results.macro['I_hydrogen'].values / TO_BILLION
    else:
        inv_h2 = np.zeros_like(inv_ccu)
    
    # 2. Revenue (Inflows)
    rev_credits = results.sectoral['I_carbon_credit_revenue'].sum(axis=1).values / TO_BILLION
    imported_credits = np.maximum(0, cost_credits - rev_credits)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Economic Flows of the Net Zero Transition", fontsize=18, fontweight='bold')
    
    # A. Cost Stack
    ax1.stackplot(years, inv_nature, inv_ccs, inv_ccu, inv_h2, cost_credits,
                  labels=['Nature Investment', 'CCS Investment', 'CCU Investment', 'Green H2 Investment', 'Carbon Credit Purchases'],
                  colors=['#2ca02c', '#1f77b4', '#9467bd', '#e377c2', '#d62728'], alpha=0.85)
    
    ax1.set_title("Total Annual Cost Breakdown", fontsize=14)
    ax1.set_ylabel("Billion SAR", fontsize=12)
    ax1.legend(loc='upper left')
    
    # B. Market Composition (Supply Side)
    ax2.stackplot(years, rev_credits, imported_credits,
                  labels=['Domestic Revenue (Local CCS/Nature)', 'Imported Credits (Gap Filler)'],
                  colors=['#2ca02c', 'grey'], alpha=0.85)
    ax2.set_title("Carbon Market Money Flow: Domestic vs. Leakage", fontsize=14)
    ax2.legend(loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()





#########################################################################################################
def run_and_plot_sectoral_analysis(exp):
#########################################################################################################
    """
    Runs the simulation with sectoral data enabled, captures hidden sectoral results,
    and generates the Investment Pie Chart, Carbon Circular Economy Bar Chart, and Heatmap.
    """
    import model.model_classes as mmc
    
    # 1. Re-run with sectoral data enabled
    print("▶️ Running FULL simulation (including sectoral data)...")
    try:
        exp.run(
            scenarios=["Baseline", "Vision_2030", "Transformation", "Steady_state"],
            iterations=1,
            parameter_overrides={'Oil_exports_random_fluctuations': False},
            include_sectoral=True
        )
        print("✅ Full simulation complete.")
    except TypeError:
        print("⚠️ The 'include_sectoral' flag was not recognized. Running standard...")
        exp.run(
            scenarios=["Baseline", "Vision_2030", "Transformation", "Steady_state"],
            iterations=1,
            parameter_overrides={'Oil_exports_random_fluctuations': False}
        )

    # --- STEP 1: DEFINE SECTOR NAMES ---
    SECTOR_NAMES = [
        "Crop and animal production", "Forestry and logging", "Fishing and aquaculture", 
        "Mining of coal and lignite", "Extraction of crude petroleum and natural gas", 
        "Mining of metal ores", "Other mining and quarrying", "Mining support service activities", 
        "Manufacture of food products", "Manufacture of beverages", "Manufacture of tobacco products", 
        "Manufacture of textiles", "Manufacture of wearing apparel", "Manufacture of leather related products", 
        "Manufacture of wood and cork", "Manufacture of paper and paper products", 
        "Printing and reproduction of recorded media", "Manufacture of coke and refined petroleum products", 
        "Manufacture of chemicals and chemical products", "Manufacture of pharmaceuticals", 
        "Manufacture of rubber and plastics", "Manufacture of other non-metallic mineral products", 
        "Manufacture of basic metals", "Manufacture of fabricated metal products", 
        "Manufacture of computer, electronic and optical", "Manufacture of electrical equipment", 
        "Manufacture of machinery and equipment", "Manufacture of motor vehicles", 
        "Manufacture of other transport equipment", "Manufacture of furniture", "Other manufacturing", 
        "Repair and installation of machinery", "Electricity, gas, steam and air conditioning", 
        "Water collection, treatment and supply", "Sewerage", "Waste collection, treatment and disposal", 
        "Remediation activities", "Construction of buildings", "Civil engineering", 
        "Specialized construction activities", "Wholesale and retail trade of motor vehicles", 
        "Wholesale trade", "Retail trade", "Land transport and transport via pipelines", 
        "Water transport", "Air transport", "Warehousing and support for transportation", 
        "Postal and courier activities", "Accommodation", "Food and beverage service activities", 
        "Publishing activities", "Motion picture, video and TV production", "Programming and broadcasting", 
        "Telecommunications", "Computer programming and consultancy", "Information service activities", 
        "Financial service activities", "Insurance, reinsurance and pension funding", 
        "Activities auxiliary to financial service", "Real estate activities", "Legal and accounting activities", 
        "Activities of head offices", "Architectural and engineering activities", 
        "Scientific research and development", "Advertising and market research", 
        "Other professional, scientific and technical", "Veterinary activities", "Rental and leasing activities", 
        "Employment activities", "Travel agency and tour operator", "Security and investigation activities", 
        "Services to buildings and landscape", "Office administrative and business support", 
        "Public administration and defence", "Education", "Human health activities", 
        "Residential care activities", "Social work activities", "Creative, arts and entertainment", 
        "Libraries, archives, museums", "Sports and amusement activities", "Activities of membership organizations", 
        "Repair of computers and personal goods", "Other personal service activities", 
        "Activities of households as employers"
    ]
    if len(SECTOR_NAMES) < 150:
        SECTOR_NAMES += [f"Sector {i}" for i in range(len(SECTOR_NAMES), 150)]

    # --- STEP 2: CAPTURE DATA (PATCH) ---
    print("🕵️‍♀️ Patching 'ModelResults' to capture data...")
    CAPTURED_RAW_DATA = []
    OriginalInit = mmc.ModelResults.__init__

    def patched_init(self, *args, **kwargs):
        OriginalInit(self, *args, **kwargs)
        CAPTURED_RAW_DATA.append(self)

    mmc.ModelResults.__init__ = patched_init
    print("✅ Sensor active. Running simulations...")

    scenarios_to_run = ["Baseline", "Vision_2030", "Transformation", "Steady_state"]

    for sc in scenarios_to_run:
        try:
            exp.run(scenarios=[sc], iterations=1, parameter_overrides={'Oil_exports_random_fluctuations': False})
        except:
            pass

    mmc.ModelResults.__init__ = OriginalInit 
    print(f"🎉 Capture complete. Found {len(CAPTURED_RAW_DATA)} data objects.")

    # --- STEP 3: SMART DATA EXTRACTION ---
    print("\n🔍 ANALYZING DATA...")
    sectoral_data_store = {}

    def get_best_match(container, keywords):
        """Finds first matching key or attribute."""
        if not isinstance(container, dict):
            container = {k: getattr(container, k) for k in dir(container) if not k.startswith('_')}
        
        for k, val in container.items():
            if any(kw.lower() in k.lower() for kw in keywords):
                if isinstance(val, pd.DataFrame): return val
        return None

    for i, sc_name in enumerate(scenarios_to_run):
        if i < len(CAPTURED_RAW_DATA):
            raw_obj = CAPTURED_RAW_DATA[i]
            src = getattr(raw_obj, 'sectoral', getattr(raw_obj, 'results_sectoral', raw_obj))
            
            sectoral_data_store[sc_name] = {
                'inv': get_best_match(src, ['I_', 'I', 'inv', 'capex']),
                'oil_em': get_best_match(src, ['oil_em', 'em_oil']),
                'gas_em': get_best_match(src, ['gas_em', 'em_gas']),
                'elec_em': get_best_match(src, ['elec_em', 'em_elec']),
                'oil_use': get_best_match(src, ['oil_use', 'use_oil']),
                'gas_use': get_best_match(src, ['gas_use', 'use_gas']),
                'elec_use': get_best_match(src, ['elec_use', 'use_elec'])
            }

    # --- STEP 4: PLOTTING ---
    def safe_sum(d, keys):
        total = 0
        found_any = False
        for k in keys:
            val = d.get(k)
            if val is not None:
                total = total + val.fillna(0)
                found_any = True
        return total if found_any else None

    # A. NET ZERO INVESTMENT PIE CHART (2060)
    print("📊 Generating Investment Pie Chart...")
    if "Net_zero" in sectoral_data_store:
        inv_df = sectoral_data_store["Net_zero"]['inv']
        if inv_df is not None:
            inv_df = inv_df.fillna(0)
            inv_2060 = inv_df.iloc[-1, :].copy()
            
            limit = min(len(inv_2060), len(SECTOR_NAMES))
            inv_2060 = inv_2060.iloc[:limit]
            inv_2060.index = SECTOR_NAMES[:limit]
            
            inv_2060 = inv_2060.sort_values(ascending=False).clip(lower=0)
            
            if inv_2060.sum() > 0:
                top_10 = inv_2060.head(10)
                others_val = inv_2060.iloc[10:].sum()
                
                pie_data = top_10.copy()
                pie_data[f'Other Sectors ({len(inv_2060)-10})'] = others_val
                
                plt.figure(figsize=(10, 10))
                colors = sns.color_palette("viridis", len(pie_data))
                plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, 
                        colors=colors, pctdistance=0.85, explode=[0.05]*len(pie_data))
                plt.gca().add_artist(plt.Circle((0,0),0.70,fc='white'))
                plt.title("Net Zero Investment Composition (2060)", fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.show()

    # B. CARBON CIRCULAR ECONOMY (Bar Chart)
    print("📊 Generating Carbon Circular Economy Chart...")
    if "Net_zero" in sectoral_data_store:
        inv_df = sectoral_data_store["Net_zero"]['inv']
        if inv_df is not None:
            inv_df = inv_df.fillna(0)
            # Re-calculate limit in case it wasn't set in previous block
            limit = min(inv_df.shape[1], len(SECTOR_NAMES))
            total_inv = inv_df.iloc[:, :limit].sum(axis=0)
            total_inv.index = SECTOR_NAMES[:limit]
            top_CCE = total_inv.sort_values(ascending=False).head(10) / 1_000_000
            
            if top_CCE.sum() > 0:
                plt.figure(figsize=(14, 7))
                sns.barplot(x=top_CCE.values, y=top_CCE.index, palette="magma")
                plt.title("Drivers of the Carbon Circular Economy (Cumulative Investment)", fontsize=18, fontweight='bold')
                plt.xlabel("Total Investment (Billion SAR)", fontsize=12)
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.show()

    # C. HEATMAP (Fixed Safe Summation)
    print("📊 Generating Heatmap...")
    try:
        hm_matrix = []
        valid_scens = []
        plot_title = "Decarbonization Progress (% Emissions Reduction)"
        
        base_data = sectoral_data_store.get("Baseline")
        
        if base_data:
            if base_data['oil_em'] is not None or base_data['gas_em'] is not None:
                keys = ['oil_em', 'gas_em', 'elec_em']
            else:
                keys = ['oil_use', 'gas_use', 'elec_use']
                plot_title = "Energy Transition Progress (% Energy Use Reduction)"
                print("   -> Using Energy Use (Emissions data missing)")

            t_base = safe_sum(base_data, keys)
            
            if t_base is not None:
                limit = min(t_base.shape[1], len(SECTOR_NAMES))
                t_base = t_base.iloc[0, :limit]
                t_base.index = SECTOR_NAMES[:limit]
                top_20_names = t_base.sort_values(ascending=False).head(20).index.tolist()
                top_20_indices = [SECTOR_NAMES.index(n) for n in top_20_names]

                for sc_name in scenarios_to_run:
                    if sc_name in sectoral_data_store:
                        d = sectoral_data_store[sc_name]
                        total = safe_sum(d, keys)
                        
                        if total is not None:
                            start = total.iloc[0, top_20_indices]
                            end = total.iloc[-1, top_20_indices]
                            
                            safe_start = start.replace(0, np.nan)
                            red = ((start - end) / safe_start) * 100
                            hm_matrix.append(red.fillna(0))
                            valid_scens.append(sc_name)

                if hm_matrix:
                    df_hm = pd.DataFrame(hm_matrix, index=valid_scens, columns=top_20_names)
                    plt.figure(figsize=(22, 10))
                    sns.heatmap(df_hm, annot=True, fmt=".1f", cmap="RdYlGn", center=50, linewidths=.5)
                    plt.title(f"{plot_title}: Top 20 Sectors", fontsize=20, fontweight='bold', pad=20)
                    plt.xticks(rotation=45, ha='right', fontsize=12)
                    plt.tight_layout()
                    plt.show()
                else:
                    print("⚠️ Heatmap matrix empty (no valid scenarios).")
            else:
                print("⚠️ Could not calculate baseline total.")
        else:
            print("⚠️ Baseline data missing.")

    except Exception as e:
        print(f"❌ Heatmap Failed: {e}")