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
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import numpy as np
import textwrap
from IPython.display import display as IPdisplay

from .model_classes import ModelConfig, ModelResults
from .calibration import ParametersCalibrated
from .utils import string_to_value


# Note: endyear must match the value set in the run configuration.
startyear = 1
endyear = 41

def get_srange(config: ModelConfig):
    return [str(i) for i in range(1, len(config.pc.sectors)+1)]

##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
# AGGREGATE ECONOMIC MACRO PLOTS
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
def create_macro_graphs(results: 'ModelResults', displayPlots: bool = True):
    """Create various plots based on model results."""
    import matplotlib.pyplot as plt

    # Create copies of data and convert from Thousand SAR to Billion SAR
    df = results.macro.copy() / 1_000_000
    # For index/rates/inflation that shouldn't be divided, we revert them below
    
    df_ = results.sectoral.copy() # Keeping a copy just in case, though not directly plotted here
    pc = results.config.pc
    
    # Revert variables that are NOT monetary (e.g., inflation rates, indices, growth rates)
    # Since they were divided by 1,000,000 above, we multiply them back to their original state.
    non_monetary_vars = ['inflation', 'inflation_consumers', 'deflator_gdp', 'price_index_cons']
    for var in non_monetary_vars:
        if var in df.columns:
            df[var] = df[var] * 1_000_000

    # Set years for x-axis labels
    endyear = len(df)
    years = list(range(2021, 2021 + endyear))  # 2021 to endyear inclusive
    df.index = years # Assign years directly to the index for easier plotting

    # --- CUSTOM X-AXIS TICKS (2021, 2025, 2030, 2035...) ---
    custom_ticks = [2021] + list(range(2025, years[-1] + 1, 5))

    # Plot Main macro variables in one big plot - nominal AND real
    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    (ax1) = axs
    df.plot(y=['Y','Y_non_oil','C','I_total','Gov_exp','P','EX', 'IM', 'YD_wage'],
            title='Key macro variables in nominal terms', xlabel='Year', ylabel='Billion SAR', ax=ax1)
    ax1.set_xticks(custom_ticks)               
    ax1.set_xticklabels(custom_ticks, rotation=0)  
    ax1.legend(labels=[ "Final demand GDP","Non-oil final demand GDP",  "Consumption C", "Investment total",\
         "Government Spending Gov Exp","Profits P", "Exports EX", "Imports IM", "Disposable Wage income YD_wage"])

    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    (ax1) = axs
    df.plot(y=['y','y_non_oil','c','i_total','gov_exp','pi_real','ex', 'im', 'w'],
            title='Key macro variables in real terms (base year 2021=1 price level of 1)', xlabel='Year', ylabel='Billion SAR', ax=ax1)
    ax1.set_xticks(custom_ticks)               
    ax1.set_xticklabels(custom_ticks, rotation=0)  
    ax1.legend(labels=[ "Final demand GDP real","Non-oil final demand GDP real",  "Consumption C real", "Investment total real",\
         "Government Spending Gov Exp real","Profits P real", "Exports EX real", "Imports IM real", "real wages w"])

    ###########################################################################
    # TIMESERIES PLOT: plot the main time series that we feed into the model
    ###########################################################################
    # TIMESERIES PLOT — FIGURE 1 (original with annotations + % of GDP)
    fig1, ax1 = plt.subplots(1, 1, figsize=(13, 7))
    ax1.set_title("Key Time Series for KSA Economy Fed into Model", fontsize=13)

    scale = 1_000_000_000  # Thousand SAR → Trillion SAR

    # --- Pull raw series ---
    s_gdp     = pc.ts_national['Gross Domestic Product (4)']        / scale
    s_inv     = pc.ts_national['Gross Fixed Capital Formation (4)'] / scale
    s_gov     = pc.ts_government['Total Expenditures (5)']          / scale
    s_exp     = pc.ts_national['Exports of Goods & Services (4)']   / scale
    s_oil     = pc.ts_national['Oil Sectors (4)']                   / scale
    s_nonoil  = pc.ts_national['Non-Oil Sector (4)']                / scale
    s_imp     = pc.ts_national['Imports of Goods & Services (4)']   / scale

    # --- KEY FIX: align all series to a common index ---
    all_series_raw = [s_gdp, s_inv, s_gov, s_exp, s_oil, s_nonoil, s_imp]
    common_index   = s_gdp.index

    s_gdp, s_inv, s_gov, s_exp, s_oil, s_nonoil, s_imp = [
        s.reindex(common_index) for s in all_series_raw
    ]

    # --- Series config: (data, label, color) ---
    ts_series = [
        (s_gdp,    "GDP",                    '#1f77b4'),
        (s_inv,    "Investments",            '#d62728'),
        (s_gov,    "Gov. Total Expenditures",'#2ca02c'),
        (s_exp,    "Exports",                '#ff7f0e'),
        (s_oil,    "Oil Sectors",            '#9467bd'),
        (s_nonoil, "Non-Oil Sectors",        '#8c564b'),
        (s_imp,    "Imports",                '#17becf'),
    ]

    # --- Get last valid GDP value for % of GDP calculation ---
    gdp_last_idx = s_gdp.last_valid_index()
    gdp_last_val = s_gdp[gdp_last_idx]

    legend_handles = []

    for series, label, color in ts_series:
        line, = ax1.plot(series.index, series.values, color=color, linewidth=1.8)

        # --- Build legend label with % of GDP at last year ---
        last_idx = series.last_valid_index()
        last_val = series[last_idx]

        if label == "GDP":
            legend_label = f"GDP  ({last_idx})"
        else:
            if not pd.isna(last_val) and gdp_last_val > 0:
                pct = last_val / gdp_last_val * 100
                legend_label = f"{label}  —  {pct:.1f}% of GDP ({last_idx})"
            else:
                legend_label = f"{label}  ({last_idx})"

        legend_handles.append(
            mlines.Line2D([], [], color=color, linewidth=1.8, label=legend_label)
        )

    # --- Legend note explaining % of GDP ---
    ax1.legend(
        handles=legend_handles,
        title="Series  |  % of GDP is share of last available year's GDP",
        title_fontsize=7.5,
        fontsize=8,
        loc='upper left',
        framealpha=0.85,
        edgecolor='gray'
    )

    ax1.set_xlabel("Year", fontweight='normal', fontsize=14)
    ax1.set_ylabel("Trillion SAR", fontweight='normal', fontsize=14)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.tick_params(axis='both', labelsize=12)

    fig1.tight_layout()
    plt.show()



    

    ###############################
    # Plot Main macro variables with some more details
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    (ax1) = axs
    df.plot(y=['X', 'Y','Y_non_oil','Y_production','Y_distribution','Q_s','Q_d','unmet_demand_total', 'C', 'I_private', 'I_public','IM', 'Gov_exp', 'P', 'EX_oil', 'EX_non_oil','YD_wage','YD_profit'],
            title='Demand Details Nominal', xlabel='Year', ylabel='Billion SAR', ax=ax1, style=line_styles)
    ax1.set_xticks(custom_ticks)               
    ax1.set_xticklabels(custom_ticks, rotation=0)  
    ax1.legend(labels=["Total Output X", "Final demand GDP Y (expenditure)","Non-oil final demand GDP Y","Final demand Y (production)","Final demand Y (distribution)", "Supply choice Q_s","Desired demand Q_d","Unmet demand total", "Consumption C", "Investment private", "Investment public","Imports",
               "Government Spending", "Profits P", "Oil Exports", " Non-oil Exports","YD_wage wage disp. inc.","YD_profit disp. profit inc."])

# Multiple Plots for comparison
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axs
    
    # Define the exact ticks requested
    specific_ticks = [2021, 2030, 2040, 2050, 2060]
    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xticks(specific_ticks)               
        if ax in [ax1, ax2, ax3]:
            ax.set_xticklabels([]) # Remove years from first row plots
        else:
            ax.set_xticklabels(specific_ticks, rotation=45) # Slight rotation for smaller subplots

    df.plot(y=['X', 'Y', 'C', 'I_private', 'I_public', 'Gov_exp', 'P', 'EX', 'IM', 'W'],
            title='Demand Nominal', xlabel='', ylabel='Billion SAR', ax=ax1)
    ax1.legend(labels=["Total Output X", "GDP demand Y", "Consumption", "Investment private",
               "Investment public", "Government Spending", "Profits", "Exports", "Imports", "Wages"], fontsize=8)

    df.plot(y=['YD_wage', 'YD_profit'],
            title='Households Nominal', xlabel='', ylabel='', ax=ax2)
    ax2.legend(labels=["Income wages", "Income profits"], fontsize=8)

    df['Net_trade'] = df['EX'] - df['IM']
    df.plot(y=['Net_trade', 'EX', 'IM', 'EX_oil', 'EX_non_oil','Y'],
            title='Trade Nominal', xlabel='', ylabel='', ax=ax3)
    ax3.legend(labels=["Net trade Exports - Imports", "Exports",
               "Imports", "Oil Exports", " Non oil Exports","GDP demand Y"], fontsize=8)

    df['Expenditure'] = df['Gov_exp']
    df['Revenue'] = df['Gov_rev']
    # Re-assigning to itself handles the warning, though mathematically redundant
    df['GP'] = df['GP'] 
    df['TH'] = df['TH']

    df.plot(y=['Expenditure', 'Revenue', 'GP', 'TH'], title='Government Nominal',
            xlabel='Year', ylabel='Billion SAR', ax=ax4)
    ax4.legend(fontsize=8)

    df.plot(y=['K','k', 'Gov_net_wealth', 'Gov_ext_assets', 'V', 'L', ], title='Stocks Nominal',
            xlabel='Year', ylabel='', ax=ax5)
    ax5.legend(labels=["Capital Nominal","Capital real", "Gov. net wealth (NFA-Bonds)",
               "Gov. gross ext. assets (SAMA + PIF)", "HH Savings", "Firm Loans"], fontsize=8)

    df.plot(y=['X', 'Y', 'P', 'W', 'IntP'], title='Firms Nominal',
            xlabel='Year', ylabel='', ax=ax6)
    ax6.legend(labels=["Total output X",
               "Total Income GDP Y", "Profits", "Wage Costs", "Intermediate Inputs"], fontsize=8)

    for _i, _ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=16, fontweight='bold', va='top')
 
 
    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    # Stocks large view
    df['Gov_deficit'] = df['Gov_rev']-df['Gov_exp']
    (ax1) = axs
    df.plot(y=['Y','Gov_rev','Gov_exp','Gov_deficit'], title='Government revenues, expenditures, and deficit compared to GDP (nominal)',
            xlabel='Year', ylabel='Billion SAR', ax=ax1)
    ax1.set_xticks(custom_ticks)               
    ax1.set_xticklabels(custom_ticks, rotation=0) 
    ax1.legend(labels=['GDP','Gov revenue','Gov expenditure','Gov deficit'])


    # ── Balance of Payments ──
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ((ax_bop1, ax_bop2), (ax_bop3, ax_bop4)) = axs
    for ax in [ax_bop1, ax_bop2, ax_bop3, ax_bop4]:
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(custom_ticks, rotation=0)

    # Panel 1: Current account decomposition
    df.plot(y=['trade_balance', 'secondary_income_balance', 'primary_income_balance', 'current_account'],
            title='Current Account (BPM6)', xlabel='Year', ylabel='Billion SAR', ax=ax_bop1)
    ax_bop1.legend(labels=['Trade balance', 'Secondary income (remittances)', 'Primary income', 'Current account'], fontsize=8)
    ax_bop1.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

    # Panel 2: Financial account — FDI and Gov_ext_assets change
    df.plot(y=['FDI_net_total', 'Gov_ext_assets_change', 'financial_account'],
            title='Financial Account', xlabel='Year', ylabel='Billion SAR', ax=ax_bop2)
    ax_bop2.legend(labels=['FDI net inflows', 'ΔGov. ext. assets (SAMA + PIF)', 'Financial account'], fontsize=8)
    ax_bop2.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

    # Panel 3: Gov_ext_assets stock and income
    df.plot(y=['Gov_ext_assets', 'Gov_net_wealth', 'Gov_ext_assets_income'],
            title='Government External & Net Wealth + Asset Income', xlabel='Year', ylabel='Billion SAR', ax=ax_bop3)
    ax_bop3.legend(labels=['Gov. gross ext. assets (SAMA + PIF)', 'Gov. net wealth (NFA-Bonds)', 'Gov. ext. asset income (→ Gov_rev & BoP)'], fontsize=8)

    # Panel 4: Remittances and BoP check
    df.plot(y=['remittances', 'BoP_check'],
            title='Remittances & BoP Identity Check', xlabel='Year', ylabel='Billion SAR', ax=ax_bop4)
    ax_bop4.legend(labels=['Remittances (outflow)', 'BoP check (should ≈ 0)'], fontsize=8)
    ax_bop4.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

    for _i, _ax in enumerate([ax_bop1, ax_bop2, ax_bop3, ax_bop4]):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=16, fontweight='bold', va='top')

    plt.suptitle('Balance of Payments', fontsize=14, y=1.02)
    plt.tight_layout()








##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
# SECTORAL ECONOMIC MACRO PLOTS
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
def create_sectoral_graphs(results: 'ModelResults', displayPlots: bool = True):
    """Create various plots based on model results."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import textwrap

    df = results.macro.copy()
    df_ = results.sectoral.copy()
    pc = results.config.pc
    endyear = len(df) # Extract endyear dynamically based on the dataframe length
    # Set years for x-axis labels
    years = list(range(2021, 2021 + endyear))  # 2021 to 2060 inclusive

    ####################################################################
    # Create one-letter NACE codes for each sector
    ###################################################################
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

    def aggregate_to_nace_sections(df):
        section_labels = pd.Series(sector_number_to_nace_section_full)
        return df.T.groupby(section_labels.values).sum().T
 
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
    # Plot NOMINAL Sectoral Output Y_ 
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['Y'].columns
    axs.set_title("NOMINAL Sectoral OUTPUT Y_s (GDP) - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['Y'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['Y'].index[-1]
    top_sectors_start = results.sectoral['Y'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['Y'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    # Plot REAL Sectoral Output Y_ 
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['y'].columns
    axs.set_title("REAL Sectoral OUTPUT y_s (GDP) - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['y'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['y'].index[-1]
    top_sectors_start = results.sectoral['y'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['y'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)



    #############################################################################################################################
    # Plot Sectoral TOTAL OUTPUT X_ NOMINAL 
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['X'].columns
    axs.set_title("NOMINAL Sectoral TOTAL OUTPUT X_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['X'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['X'].index[-1]
    top_sectors_start = results.sectoral['X'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['X'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    # Plot Sectoral TOTAL OUTPUT X_ REAL
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['x'].columns
    axs.set_title("REAL Sectoral TOTAL OUTPUT x_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['x'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['x'].index[-1]
    top_sectors_start = results.sectoral['x'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['x'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    #############################################################################################################################
    # Plot NOMINAL SUPPLY CHOICE Q_s 
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['Q_s'].columns
    axs.set_title("NOMINAL Sectoral SUPPLY CHOICE Q_s (GDP) - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['Q_s'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['Q_s'].index[-1]
    top_sectors_start = results.sectoral['Q_s'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['Q_s'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)
    
    # Plot REAL SUPPLY CHOICE Q_s 
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['q_s'].columns
    axs.set_title("REAL Sectoral SUPPLY CHOICE q_s (GDP) - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['q_s'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['q_s'].index[-1]
    top_sectors_start = results.sectoral['q_s'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['q_s'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    #############################################################################################################################
    # Plot Sectoral INVESTMENT I_
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['I'].columns
    axs.set_title("Sectoral INVESTMENT I_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['I'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['I'].index[-1]
    top_sectors_start = results.sectoral['I'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['I'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    #############################################################################################################################
    # Plot Sectoral WAGES W_ NOMINAL
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['W'].columns
    axs.set_title("NOMINAL Sectoral WAGES W_ - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['W'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['W'].index[-1]
    top_sectors_start = results.sectoral['W'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['W'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)


    # Plot Sectoral WAGES W_ REAL
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['w'].columns
    axs.set_title("REAL Sectoral WAGES w_ - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['w'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['w'].index[-1]
    top_sectors_start = results.sectoral['w'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['w'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    #############################################################################################################################
    # Plot AGGREGATE Sectoral Output Y_onedigit_
    Y_onedigit_ = aggregate_to_nace_sections(results.sectoral['Y'])
    fig, axs = plt.subplots(1, 1, figsize=(16, 10))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  

    # Use aggregated Y DataFrame
    sector = Y_onedigit_.columns
    axs.set_title("Sectoral OUTPUT Y_s (GDP) by NACE 1-digit section")
    axs.plot(Y_onedigit_ / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    wrapped_labels = [textwrap.fill(str(s), width=28) for s in sector]
    axs.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10, title="NACE Section", labels=wrapped_labels)
    plt.subplots_adjust(right=0.78, bottom=0.14)  

    end_period = Y_onedigit_.index[-1]
    top_sectors_start = Y_onedigit_.loc[1].nlargest(5).index  
    top_sectors_end = Y_onedigit_.loc[end_period].nlargest(5).index  

    sector_notes_start = [f"Start - Section {section}" for section in top_sectors_start]
    sector_notes_end = [f"End - Section {section}" for section in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, 0, notes, wrap=True, horizontalalignment='center', fontsize=9)

    sectors = results.sectoral['X'].columns
    pc.IM_ = pd.Series(pc.IM_, index=sectors) 

    #############################################################################################################################
    # Plot Sectoral IMPORTS IM_ 
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['IM'].columns
    axs.set_title("Sectoral IMPORTS model CALCULATED IM_s (GDP) - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['IM'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['IM'].index[-1]
    top_sectors_start = results.sectoral['IM'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['IM'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    #############################################################################################################################
    # Plot Sectoral Capital Stock K_
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['K'].columns
    axs.set_title("Nominal sectoral Capital Stock K_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['K'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['K'].index[-1]
    top_sectors_start = results.sectoral['K'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['K'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['k'].columns
    axs.set_title("REAL sectoral Capital Stock k_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['k'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['k'].index[-1]
    top_sectors_start = results.sectoral['k'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['k'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    #############################################################################################################################
    # Plot Sectoral INTERMEDIATE INPUT PURCHASES IntP_ 
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['IntP'].columns
    axs.set_title("NOMINAL sectoral INTERMEDIATE INPUT IntP_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['IntP'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['IntP'].index[-1]
    top_sectors_start = results.sectoral['IntP'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['IntP'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    # Plot Sectoral INTERMEDIATE INPUT PURCHASES IntP_ 
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['intP'].columns
    axs.set_title("REAL sectoral INTERMEDIATE INPUT intP_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['intP'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['intP'].index[-1]
    top_sectors_start = results.sectoral['intP'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['intP'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    #############################################################################################################################
    # Plot Sectoral EXPORTS EX_ 
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    sector = results.sectoral['EX'].columns
    axs.set_title("Sectoral EXPORTS EX_s - with top sectors in first and last modeling period labeled")
    axs.plot(results.sectoral['EX'] / 1_000_000, label=sector)
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=13)
    end_period = results.sectoral['EX'].index[-1]
    top_sectors_start = results.sectoral['EX'].loc[1].nlargest(5).index  
    top_sectors_end = results.sectoral['EX'].loc[end_period].nlargest(5).index  
    sector_notes_start = [f"Start - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_start]
    sector_notes_end = [f"End - Sector {sector}: {pc.sectors[sector]}" for sector in top_sectors_end]
    sector_notes = sector_notes_start + sector_notes_end
    notes = "\n".join(["; ".join(sector_notes[i:i+3]) for i in range(0, len(sector_notes), 3)])
    plt.figtext(0.5, -0.25, notes, wrap=True, horizontalalignment='center', fontsize=9)

    #############################################################################################################################
    # Plot Sectoral Output Y_ for the top 10 sectors at the end of the modeling period
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.set_xticks(df.index[::5])               
    axs.set_xticklabels(years[::5], rotation=0)  
    end_period = results.sectoral['Y'].index[-1]  
    top_10_sectors = results.sectoral['Y'].loc[end_period].nlargest(10).index  
    Y_top_10 = results.sectoral['Y'][top_10_sectors]
    for sector in top_10_sectors:
        axs.plot(Y_top_10.index, Y_top_10[sector] / 1_000_000, label=f"Sector {sector}: {pc.sectors[sector]}")
    axs.set_title("Sectoral Output Y_s (GDP) - Top 10 Sectors in last modeling period")
    axs.set_xlabel("Time")
    axs.set_ylabel("Billion SAR")
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    #############################################################################################################################
    # Plot sectoral decomposition of components of X total output, in relation Y value added GDP and to Imports!
    # Extract data for period 1 START CALIBRATION
    sectors = results.sectoral['X'].columns
    X_ = results.sectoral['X'].loc[1] / 1_000_000
    IntP_ = results.sectoral['IntP'].loc[1] / 1_000_000
    W_ = results.sectoral['W'].loc[1] / 1_000_000
    P_ = results.sectoral['P'].loc[1] / 1_000_000
    Y_ = results.sectoral['Y'].loc[1] / 1_000_000
    IM_ = results.sectoral['IM'].loc[1] / 1_000_000
    Q_d_ = results.sectoral['Q_d'].loc[1] / 1_000_000
    EX_ = results.sectoral['EX'].loc[1] / 1_000_000
    GY_ = results.sectoral['GY'].loc[1] / 1_000_000
    C_ = results.sectoral['C'].loc[1] / 1_000_000
    Other_Y_ = Y_ - W_ - P_
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.25
    indices = np.arange(len(sectors))
    
    ax.bar(indices - bar_width/2, IntP_, bar_width, label='Intermediate Inputs (X)')
    ax.bar(indices - bar_width/2, W_, bar_width, bottom=IntP_, label='Wages (X)')
    ax.bar(indices - bar_width/2, P_, bar_width, bottom=IntP_ + W_, label='Profits (X)')
    ax.bar(indices + bar_width/2, Y_, bar_width,color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width, IM_, bar_width, label='Imports model calculated')
    ax.bar(indices + bar_width*1.5, pc.IM_ / 1_000_000, bar_width, label='Imports calibrated')
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Billion SAR')
    ax.set_title('START Sectoral Decomposition of Total Output X in comparison to Value Added Y and Imports model calculated and calibrated, Period 1')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    #############################################################################################################################
    # AGREGATE Plot sectoral decomposition of components of X total output, in relation Y value added GDP and to Imports!
    # START IN PERIOD 1
    IntP_agg = aggregate_to_nace_sections(results.sectoral['IntP']).loc[1] / 1_000_000
    W_agg    = aggregate_to_nace_sections(results.sectoral['W']).loc[1] / 1_000_000
    P_agg    = aggregate_to_nace_sections(results.sectoral['P']).loc[1] / 1_000_000
    Y_agg    = aggregate_to_nace_sections(results.sectoral['Y']).loc[1] / 1_000_000
    IM_agg   = aggregate_to_nace_sections(results.sectoral['IM']).loc[1] / 1_000_000
    Qd_agg   = aggregate_to_nace_sections(results.sectoral['Q_d']).loc[1] / 1_000_000
    EX_agg   = aggregate_to_nace_sections(results.sectoral['EX']).loc[1] / 1_000_000
    GY_agg   = aggregate_to_nace_sections(results.sectoral['GY']).loc[1] / 1_000_000
    C_agg    = aggregate_to_nace_sections(results.sectoral['C']).loc[1] / 1_000_000
    IM_calib_agg = aggregate_to_nace_sections(pd.DataFrame(pc.IM_).T).iloc[0] / 1_000_000
    Other_Y_agg = Y_agg - W_agg - P_agg
    long_to_short = {v: k for k, v in nace_section_full_names.items()}
    sections_long = Y_agg.index
    sections_short = [long_to_short.get(name, name) for name in sections_long]
    bar_width = 0.25
    indices = np.arange(len(sections_short))
    columns_long = Y_onedigit_.columns
    legend_labels = [long_to_short.get(col, col) for col in columns_long]
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.bar(indices - bar_width/2, IntP_agg, bar_width, label='Intermediate Inputs (X)')
    ax.bar(indices - bar_width/2, W_agg, bar_width, bottom=IntP_agg, label='Wages (X)')
    ax.bar(indices - bar_width/2, P_agg, bar_width, bottom=IntP_agg + W_agg, label='Profits (X)')
    ax.bar(indices + bar_width/2, Y_agg, bar_width, color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width, IM_agg, bar_width, label='Imports model calculated')
    ax.bar(indices + bar_width*1.5, IM_calib_agg, bar_width, label='Imports calibrated')
    ax.set_xlabel('NACE 1-digit Section')
    ax.set_ylabel('Billion SAR')
    ax.set_title('START: Decomposition of Total Output X (NACE 1-digit) vs. Value Added Y and Imports')
    ax.set_xticks(indices)
    ax.set_xticklabels(sections_short, rotation=0)
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.23),  
        ncol=3,
        frameon=True,
        fontsize=11
    )
    wrapped_section_names = [textwrap.fill(f"{short}: {long}", width=60)
                            for short, long in zip(sections_short, sections_long)]
    lines = ["; ".join(wrapped_section_names[i:i+4]) for i in range(0, len(wrapped_section_names), 4)]
    notes = "\n".join(lines)
    plt.subplots_adjust(bottom=0.22)  
    plt.figtext(0.5, 0.04, notes, wrap=True, ha='center', va='top', fontsize=9)
        
    #############################################################################################################################
    # Plot sectoral decomposition of components of X total output, in relation Y value added GDP and to Imports!
    # Extract data for LAST period END of model
    sectors = results.sectoral['X'].columns
    end_period = results.sectoral['Y'].index[-1]
    X_ = results.sectoral['X'].loc[end_period] / 1_000_000
    IntP_ = results.sectoral['IntP'].loc[end_period] / 1_000_000
    W_ = results.sectoral['W'].loc[end_period] / 1_000_000
    P_ = results.sectoral['P'].loc[end_period] / 1_000_000
    Y_ = results.sectoral['Y'].loc[end_period] / 1_000_000
    IM_ = results.sectoral['IM'].loc[end_period] / 1_000_000
    Q_d_ = results.sectoral['Q_d'].loc[end_period] / 1_000_000
    EX_ = results.sectoral['EX'].loc[end_period] / 1_000_000
    GY_ = results.sectoral['GY'].loc[end_period] / 1_000_000
    C_ = results.sectoral['C'].loc[end_period] / 1_000_000
    Other_Y_ = Y_ - W_ - P_
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.25
    indices = np.arange(len(sectors))
    
    ax.bar(indices - bar_width/2, IntP_, bar_width, label='Intermediate Inputs (X)')
    ax.bar(indices - bar_width/2, W_, bar_width, bottom=IntP_, label='Wages (X)')
    ax.bar(indices - bar_width/2, P_, bar_width, bottom=IntP_ + W_, label='Profits (X)')
    ax.bar(indices + bar_width/2, Y_, bar_width,color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width, IM_, bar_width, label='Imports model calculated')
    ax.bar(indices + bar_width*1.5, pc.IM_ / 1_000_000, bar_width, label='Imports calibrated')
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Billion SAR')
    ax.set_title('END Sectoral Decomposition of Total Output X in comparison to Value Added Y and Imports model calculated and calibrated, LAST PERIOD')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    #############################################################################################################################
    # LAST PERIOD AGREGATE Plot sectoral decomposition of components of X total output, in relation Y value added GDP and to Imports!
    # LAST PERIOD
    IntP_agg = aggregate_to_nace_sections(results.sectoral['IntP']).loc[end_period] / 1_000_000
    W_agg    = aggregate_to_nace_sections(results.sectoral['W']).loc[end_period] / 1_000_000
    P_agg    = aggregate_to_nace_sections(results.sectoral['P']).loc[end_period] / 1_000_000
    Y_agg    = aggregate_to_nace_sections(results.sectoral['Y']).loc[end_period] / 1_000_000
    IM_agg   = aggregate_to_nace_sections(results.sectoral['IM']).loc[end_period] / 1_000_000
    Qd_agg   = aggregate_to_nace_sections(results.sectoral['Q_d']).loc[end_period] / 1_000_000
    EX_agg   = aggregate_to_nace_sections(results.sectoral['EX']).loc[end_period] / 1_000_000
    GY_agg   = aggregate_to_nace_sections(results.sectoral['GY']).loc[end_period] / 1_000_000
    C_agg    = aggregate_to_nace_sections(results.sectoral['C']).loc[end_period] / 1_000_000
    IM_calib_agg = aggregate_to_nace_sections(pd.DataFrame(pc.IM_).T).iloc[0] / 1_000_000

    Other_Y_agg = Y_agg - W_agg - P_agg
    long_to_short = {v: k for k, v in nace_section_full_names.items()}
    sections_long = Y_agg.index
    sections_short = [long_to_short.get(name, name) for name in sections_long]
    bar_width = 0.25
    indices = np.arange(len(sections_short))
    columns_long = Y_onedigit_.columns
    legend_labels = [long_to_short.get(col, col) for col in columns_long]

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.bar(indices - bar_width/2, IntP_agg, bar_width, label='Intermediate Inputs (X)')
    ax.bar(indices - bar_width/2, W_agg, bar_width, bottom=IntP_agg, label='Wages (X)')
    ax.bar(indices - bar_width/2, P_agg, bar_width, bottom=IntP_agg + W_agg, label='Profits (X)')
    ax.bar(indices + bar_width/2, Y_agg, bar_width, color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width, IM_agg, bar_width, label='Imports model calculated')
    ax.bar(indices + bar_width*1.5, IM_calib_agg, bar_width, label='Imports calibrated')

    ax.set_xlabel('NACE 1-digit Section')
    ax.set_ylabel('Billion SAR')
    ax.set_title('LAST PERIOD: Decomposition of Total Output X (NACE 1-digit) vs. Value Added Y and Imports')
    ax.set_xticks(indices)
    ax.set_xticklabels(sections_short, rotation=0)
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.23),  
        ncol=3,
        frameon=True,
        fontsize=11
    )
    wrapped_section_names = [textwrap.fill(f"{short}: {long}", width=60)
                            for short, long in zip(sections_short, sections_long)]
    lines = ["; ".join(wrapped_section_names[i:i+4]) for i in range(0, len(wrapped_section_names), 4)]
    notes = "\n".join(lines)
    plt.subplots_adjust(bottom=0.22)  
    plt.figtext(0.5, 0.04, notes, wrap=True, ha='center', va='top', fontsize=8)

    # Reset data for period 1 so that other plots are still for calibration period
    sectors = results.sectoral['X'].columns
    X_ = results.sectoral['X'].loc[1] / 1_000_000
    IntP_ = results.sectoral['IntP'].loc[1] / 1_000_000
    W_ = results.sectoral['W'].loc[1] / 1_000_000
    P_ = results.sectoral['P'].loc[1] / 1_000_000
    Y_ = results.sectoral['Y'].loc[1] / 1_000_000
    IM_ = results.sectoral['IM'].loc[1] / 1_000_000
    Q_d_ = results.sectoral['Q_d'].loc[1] / 1_000_000
    EX_ = results.sectoral['EX'].loc[1] / 1_000_000
    GY_ = results.sectoral['GY'].loc[1] / 1_000_000
    C_ = results.sectoral['C'].loc[1] / 1_000_000

    #############################################################################################################################
    # Plot Y model calculated vs. Y calibrated sectoral
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.25
    indices = np.arange(len(sectors))
    ax.bar(indices - bar_width/2, Y_, bar_width,color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width/2, pc.Y_ / 1_000_000, bar_width, label='Y_ sectoral CALIBRATED ')
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Billion SAR')
    ax.set_title('Sectoral Value Added Y_ MODEL CALCULATED vs. Y_ sectoral CALIBRATED Period 1')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    #############################################################################################################################
    # Plot Y model calculated vs. Y calibrated sectoral
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.25
    indices = np.arange(len(sectors))
    ax.bar(indices - bar_width/2, results.sectoral['Y'].loc[2] / 1_000_000, bar_width,color='skyblue', label='Model calculated GDP Value added Y')
    ax.bar(indices + bar_width/2, pc.Y_ / 1_000_000, bar_width, label='Y_ sectoral CALIBRATED ')
    ax.set_xlabel('Sectors')
    ax.set_ylabel('Billion SAR')
    ax.set_title('Period 2 Sectoral Value Added Y_ MODEL CALCULATED vs. Y_ sectoral CALIBRATED Period 2')
    ax.set_xticks(indices)
    ax.set_xticklabels(sectors, rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    #############################################################################################################################
    if displayPlots:
        plt.tight_layout()
        plt.show()

    return fig, ax



##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
# DETAILED SECTORAL ECONOMIC MACRO PLOTS
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
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

    # # Plot Y and Imports side by side, where Imports CALIBRATED are larger than output model calculated period 1
    # fig, ax = plt.subplots(figsize=(15, 8))
    # # Filter sectors where imports are larger than output
    # # Add sector notes as a legend or text

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

##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
# BALANCE SHEET
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
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


##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
# CORRELATION HEATMAP
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
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



##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
# CREATE TRANSITION MATRIX
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
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



##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
# WATER GRAPHS
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
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

    water_user_emp = results.macro['water_use_total_national']
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

    for _i, _ax in enumerate(axs):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=16, fontweight='bold', va='top')

    prices = results.sectoral["p"].copy()

    plt.tight_layout()
    plt.show()




def create_combined_water_graphs(results: ModelResults):
    """Create three consolidated subplots for water use, source shares, and investments."""
    pc = results.config.pc

     
    pc = results.config.pc
    x_wwater = results.sectoral["X"][pc.sector_wwater]
    x_wwater = results.sectoral["X"][pc.sector_wwater]
    x_wwater_real = results.sectoral["x"][pc.sector_wwater]
    x_wwater_real = results.sectoral["x"][pc.sector_wwater]
    water_use_wwater_biophysical = results.macro["water_use_wwater_tot"]

    # --- Define all required variables ---
    water_use_desal = results.macro["water_use_desal_tot"]

    # Ensure wastewater total is defined
    if "water_use_wwater_tot" not in results.macro:
        results.macro["water_use_wwater_tot"] = results.sectoral["X"][pc.sector_wwater]
    
    water_use_wwater = results.macro["water_use_wwater_tot"]
    x_wwater = results.sectoral["X"][pc.sector_wwater]
    water_user_emp = results.macro['water_use_total_national']  # Including household use

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

    for _i, _ax in enumerate(axs):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=16, fontweight='bold', va='top')

    plt.tight_layout()
    plt.show()

