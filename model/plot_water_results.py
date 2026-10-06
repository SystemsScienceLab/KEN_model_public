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
import numpy as np
import pandas as pd
import os
import sys
import warnings

# Ensure we can import from the model directory
sys.path.append(os.getcwd())

import model.model as m
from model.experiment_runner import ExperimentRunner
from model.water_module import get_supply_mix_config, get_agr_data
from model.experiment_plots import SCENARIO_LABELS

# ==========================================
# 1. ENHANCED EXPERIMENT RUNNER
# ==========================================

class WaterResultRunner(ExperimentRunner):
    """
    A specialized runner to ensure all water-related data (Agriculture, 
    Policy Targets, etc.) are correctly injected into each scenario run.
    """
    def single_run(self, scenario: str, seed: int, parameter_overrides: dict = None):
        # Map scenario names to internal configuration keys
        scenario_mapping = {
            "Baseline": "Baseline_scenario",
            "Vision_2030": "Vision_2030_scenario",
            "Transformation": "Transformation_scenario"
        }

        # Map to water module specific scenario names
        water_module_scenarios = {
            "Baseline": "BAU",
            "Vision_2030": "2030 Vision",
            "Transformation": "Transformation"
        }
        
        current_sc_key = scenario_mapping.get(scenario)
        scenario_config = {k: (k == current_sc_key) for k in scenario_mapping.values()}

        np.random.seed(seed)
        parameters = self.scenario_params[scenario].default_values()
        
        # 1. Initialize calibrated parameters with scenario config
        from model.calibration import ParametersCalibrated
        pc = ParametersCalibrated(parameters, verbose=False, scenario_config=scenario_config)
        
        # 2. Agriculture data (crop module, value-chain coupling, water) are set in
        # ParametersCalibrated (calibration.py section 8.2) for the scenario's water-module
        # scenario and population growth, so they are not re-injected here. Irrigation cost
        # series for the plots ({Year: SAR}) are taken from there, for 2021-2060.
        agr_invest = {y: v for y, v in pc.agr_irrigation_invest_dict.items() if y <= 2060}
        agr_om = {y: v for y, v in pc.agr_irrigation_om_dict.items() if y <= 2060}
        agr_subsidy = {y: v for y, v in pc.agr_irrigation_subsidy_dict.items() if y <= 2060}

        # 3. Inject Water Allocation Policy Parameters
        base_s, target_s, target_year = get_supply_mix_config(scenario_config)
        pc.water_allocation_target_year = target_year
        for k, val in base_s.items():
            source, sector = k.split('_')
            setattr(pc, f"{source}_allocation_{sector}_share_base", val)
        for k, val in target_s.items():
            source, sector = k.split('_')
            setattr(pc, f"{source}_allocation_{sector}_share_target", val)

        # 4. Run the model
        from model.model import ModelConfig, run_model
        config = ModelConfig(parameters, pc, self.endyear)
        parameters.Warnings_ON = False
        results = run_model(config, verbose=False)

        return {
            "results_macro": results.macro, 
            "results_sectoral": results.sectoral, 
            "scenario": scenario,
            # Add new cost data
            "irrigation_invest": agr_invest,
            "irrigation_om": agr_om,
            "irrigation_subsidy": agr_subsidy
        }

# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================

# Formatting Constants for Line Plots
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 18
ANNOTATION_FONT_SIZE = 18
SUB_LABEL_SIZE = 18

def resolve_overlaps(points):
    """Helper to prevent overlapping end-point annotations"""
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

def plot_water_use_comparison(all_results):
    """
    Plot 1: water_use_scenarios_comparison.png
    Line plot: Formatted with annotations, single legend, no main title.
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    scenarios = list(all_results.keys())
    years = all_results[scenarios[0]].index + 2020
    
    scenario_colors = {
        "Baseline":       "#d62728",  # Red
        "Vision_2030":    "#ff7f0e",  # Orange
        "Net_zero":       "#2ca02c",  # Green
        "Transformation": "#1f77b4",  # Blue
    }
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    plot_configs = [
        ('water_use_agr', 'Agricultural Water Use'),
        ('water_use_ind', 'Total Industrial Water Use'),
        ('water_use_ser', 'Total Service Water Use'),
        ('water_use_hh', 'Total Household Water Use'),
        ('water_use_total_national', 'Total Water Use (Demand)')
    ]
    
    for i, (var_name, title) in enumerate(plot_configs):
        ax = axs[i]
        endpoints_2060 = []
        
        for s_idx, sc in enumerate(scenarios):
            df = all_results[sc]
            data = df[var_name] / 1e9
            color = scenario_colors.get(sc, default_colors[s_idx % len(default_colors)])
            ax.plot(years, data, label=SCENARIO_LABELS.get(sc, sc), color=color, linewidth=2.5)
            
            # Extract 2060 value for annotation
            if 2060 in years:
                idx_60 = np.where(years == 2060)[0][0]
                val_60 = data.iloc[idx_60]
                endpoints_2060.append({'y': val_60, 'real_y': val_60, 'label': f"{val_60:.1f}", 'color': color})
        
        # Apply non-overlapping annotations
        if endpoints_2060:
            for p_annotate in resolve_overlaps(endpoints_2060):
                ax.annotate(p_annotate['label'], xy=(2060, p_annotate['real_y']), xytext=(2061.2, p_annotate['y']), 
                            color=p_annotate['color'], fontweight='bold', fontsize=ANNOTATION_FONT_SIZE, va='center', ha='left',
                            arrowprops=dict(arrowstyle="-", color=p_annotate['color'], alpha=0.4) if abs(p_annotate['y']-p_annotate['real_y']) > 0.01 else None)
                ax.scatter([2060], [p_annotate['real_y']], color=p_annotate['color'], s=30, zorder=5)

        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
        ax.set_ylabel("Billion m3", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("Year", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        
        # Lock x-axis to 2060 limits
        ax.set_xlim(years.min(), 2060)
        ticks = ax.get_xticks()
        ax.set_xticks(ticks[ticks <= 2060])
        
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Only show legend on the very first plot
        if i == 0:
            ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=True, loc='upper left')

    # Hide unused subplot(s)
    if len(plot_configs) < len(axs):
        for j in range(len(plot_configs), len(axs)):
            axs[j].axis('off')

    for _i, _ax in enumerate(axs[:len(plot_configs)]):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=SUB_LABEL_SIZE, fontweight='bold', va='top')

    plt.tight_layout()
    plt.show()

def plot_water_balance_verification(all_results):
    """
    Plot 2: water_balance_verification.png
    Area Stackplot: Maintained original formatting. Main title removed.
    Legend strictly on the first plot only.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    axes = axes.flatten()
    
    scenarios = list(all_results.keys())
    years = all_results[scenarios[0]].index + 2020
    
    for i, sc in enumerate(scenarios):
        df = all_results[sc]
        ax = axes[i]
        
        # Data in Billion m3
        desal = df["water_use_desal_tot"] / 1e9
        wwater = df["water_use_wwater_tot"] / 1e9
        gw = df["water_use_gw_tot"] / 1e9
        total = df['water_use_total_national'] / 1e9
        
        ax.stackplot(years, desal, wwater, gw, 
                    labels=['Desalination', 'Wastewater Reuse', 'Groundwater'],
                    colors=['#3498db', '#2ecc71', '#e67e22'], alpha=0.8)
        ax.plot(years, total, color='black', linestyle='--', linewidth=2, label='Total Demand')
        
        ax.set_title(f"Supply Mix: {sc}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Billion m3", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.4)
        
        # Only show legend on the very first plot
        if i == 0:
            ax.legend(loc='upper left', fontsize=12)
        
    for _i, _ax in enumerate(axes):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=16, fontweight='bold', va='top')

    plt.tight_layout()
    plt.show()

def plot_scenario_trends_comparison(all_results):
    """
    Plot 3: water_scenario_comparison_trends.png
    Line plot: Formatted with annotations, single legend, no main title.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    scenarios = list(all_results.keys())
    years = all_results[scenarios[0]].index + 2020
    
    metrics = {
        "Total Water Supply": 'water_use_total_national',
        "Desalination Supply": "water_use_desal_tot",
        "Wastewater Reuse Supply": "water_use_wwater_tot",
        "Groundwater Supply": "water_use_gw_tot"
    }
    
    colors = {
        "Baseline":       "#d62728",  # Red
        "Vision_2030":    "#ff7f0e",  # Orange
        "Net_zero":       "#2ca02c",  # Green
        "Transformation": "#1f77b4",  # Blue
    }
    
    for i, (title, var_name) in enumerate(metrics.items()):
        ax = axes[i]
        endpoints_2060 = []
        
        for sc in scenarios:
            data = all_results[sc][var_name] / 1e9
            ax.plot(years, data, label=SCENARIO_LABELS.get(sc, sc), color=colors.get(sc), linewidth=3)
            
            # Extract 2060 value for annotation
            if 2060 in years:
                idx_60 = np.where(years == 2060)[0][0]
                val_60 = data.iloc[idx_60]
                endpoints_2060.append({'y': val_60, 'real_y': val_60, 'label': f"{val_60:.1f}", 'color': colors.get(sc)})
                
        # Apply non-overlapping annotations
        if endpoints_2060:
            for p_annotate in resolve_overlaps(endpoints_2060):
                ax.annotate(p_annotate['label'], xy=(2060, p_annotate['real_y']), xytext=(2061.2, p_annotate['y']), 
                            color=p_annotate['color'], fontweight='bold', fontsize=ANNOTATION_FONT_SIZE, va='center', ha='left',
                            arrowprops=dict(arrowstyle="-", color=p_annotate['color'], alpha=0.4) if abs(p_annotate['y']-p_annotate['real_y']) > 0.01 else None)
                ax.scatter([2060], [p_annotate['real_y']], color=p_annotate['color'], s=30, zorder=5)
        
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
        ax.set_ylabel("Billion m3", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("Year", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        
        # Lock x-axis to 2060 limits
        ax.set_xlim(years.min(), 2060)
        ticks = ax.get_xticks()
        ax.set_xticks(ticks[ticks <= 2060])
        
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Only show legend on the very first plot
        if i == 0:
            ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=True, loc='upper left')
        
    for _i, _ax in enumerate(axes):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=SUB_LABEL_SIZE, fontweight='bold', va='top')

    plt.tight_layout()
    plt.show()

def plot_sectoral_supply_mix(all_results, all_sectoral_results):
    """
    Plot 4: water_sectoral_supply_mix.png
    Bar Plot: Maintained original formatting. Main title removed.
    Legend strictly on the first plot only.
    """
    scenarios = list(all_results.keys())
    sectors = ['Agriculture', 'Industry', 'Service', 'Household']
    years = all_results[scenarios[0]].index + 2020
    
    # Filter years: 2021, 2026, 2031, ..., up to end
    selected_indices = [i for i, y in enumerate(years) if (y >= 2021) and ((y - 2021) % 5 == 0)]
    selected_years = years[selected_indices]
    
    fig, axes = plt.subplots(4, len(scenarios), figsize=(6 * len(scenarios), 18), sharex=True)
    
    bar_width = 3.5 
    
    for i, sc in enumerate(scenarios):
        df_macro = all_results[sc]
        sectoral = all_sectoral_results[sc]
        
        # 1. Agriculture (Indices 0:3)
        desal_agr = (sectoral['water_use_desal'].iloc[:, 0:3].sum(axis=1) / 1e9).iloc[selected_indices]
        ww_agr = (sectoral['water_use_wwater'].iloc[:, 0:3].sum(axis=1) / 1e9).iloc[selected_indices]
        gw_agr = (sectoral['water_use_gw'].iloc[:, 0:3].sum(axis=1) / 1e9).iloc[selected_indices]
        
        # 2. Industry (Indices 3:40)
        desal_ind = (sectoral['water_use_desal'].iloc[:, 3:40].sum(axis=1) / 1e9).iloc[selected_indices]
        ww_ind = (sectoral['water_use_wwater'].iloc[:, 3:40].sum(axis=1) / 1e9).iloc[selected_indices]
        gw_ind = (sectoral['water_use_gw'].iloc[:, 3:40].sum(axis=1) / 1e9).iloc[selected_indices]
        
        # 3. Service (Indices 40:)
        desal_ser = (sectoral['water_use_desal'].iloc[:, 40:].sum(axis=1) / 1e9).iloc[selected_indices]
        ww_ser = (sectoral['water_use_wwater'].iloc[:, 40:].sum(axis=1) / 1e9).iloc[selected_indices]
        gw_ser = (sectoral['water_use_gw'].iloc[:, 40:].sum(axis=1) / 1e9).iloc[selected_indices]
        
        # 4. Household (from Macro)
        desal_hh = (df_macro['water_use_desal_hh'] / 1e9).iloc[selected_indices]
        ww_hh = (df_macro['water_use_wwater_hh'] / 1e9).iloc[selected_indices]
        gw_hh = (df_macro['water_use_gw_hh'] / 1e9).iloc[selected_indices]
        
        data_to_plot = [
            (desal_agr, ww_agr, gw_agr),
            (desal_ind, ww_ind, gw_ind),
            (desal_ser, ww_ser, gw_ser),
            (desal_hh, ww_hh, gw_hh)
        ]
        
        for j, (desal, ww, gw) in enumerate(data_to_plot):
            ax = axes[j, i]
            
            p1 = ax.bar(selected_years, desal, width=bar_width, color='#3498db', label='Desal', alpha=0.9, edgecolor='white', linewidth=0.5)
            p2 = ax.bar(selected_years, ww, bottom=desal, width=bar_width, color='#2ecc71', label='WW Reuse', alpha=0.9, edgecolor='white', linewidth=0.5)
            p3 = ax.bar(selected_years, gw, bottom=desal+ww, width=bar_width, color='#e67e22', label='Groundwater', alpha=0.9, edgecolor='white', linewidth=0.5)
            
            total_heights = desal + ww + gw
            for x, val in zip(selected_years, total_heights):
                if val > 0:
                    ax.text(x, val, f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            if not total_heights.empty and total_heights.max() > 0:
                ax.set_ylim(0, total_heights.max() * 1.15)
            
            if j == 0:
                ax.set_title(f"{sc}", fontsize=16, fontweight='bold')
            if i == 0:
                ax.set_ylabel(f"{sectors[j]}\n(Bln m3)", fontsize=14, fontweight='bold')
            
            ax.grid(True, axis='y', linestyle=':', alpha=0.4)
            ax.set_xticks(selected_years)
            ax.set_xticklabels(selected_years, fontsize=10)
            
            # Only show legend on the very first plot
            if j == 0 and i == 0:
                ax.legend(loc='upper left', fontsize=12)

    for _i, _ax in enumerate(axes.flatten()):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=16, fontweight='bold', va='top')

    plt.tight_layout()
    plt.show()

def plot_irrigation_costs(all_results):
    """
    Plot 5: irrigation_costs_comparison.png
    Line plot: Formatted with annotations, single legend, no main title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()
    
    scenarios = list(all_results.keys())
    years = list(all_results[scenarios[0]]["irrigation_invest"].keys())
    years_arr = np.array(years)
    
    metrics = [
        ("irrigation_invest", "Irrigation Investment Cost"),
        ("irrigation_om", "Irrigation O&M Cost"),
        ("irrigation_subsidy", "Irrigation Subsidy")
    ]
    
    colors = {
        "Baseline":       "#d62728",  # Red
        "Vision_2030":    "#ff7f0e",  # Orange
        "Net_zero":       "#2ca02c",  # Green
        "Transformation": "#1f77b4",  # Blue
    }
    
    for i, (key, title) in enumerate(metrics):
        ax = axes[i]
        endpoints_2060 = []
        
        for sc in scenarios:
            data_dict = all_results[sc][key]
            values = [data_dict.get(y, 0) for y in years]
            values_million = [v / 1e6 for v in values]
            
            ax.plot(years, values_million, label=SCENARIO_LABELS.get(sc, sc), color=colors.get(sc), linewidth=2.5)
            
            # Extract 2060 value for annotation
            if 2060 in years_arr:
                idx_60 = np.where(years_arr == 2060)[0][0]
                val_60 = values_million[idx_60]
                endpoints_2060.append({'y': val_60, 'real_y': val_60, 'label': f"{val_60:.1f}", 'color': colors.get(sc)})
                
        # Apply non-overlapping annotations
        if endpoints_2060:
            for p_annotate in resolve_overlaps(endpoints_2060):
                ax.annotate(p_annotate['label'], xy=(2060, p_annotate['real_y']), xytext=(2061.2, p_annotate['y']), 
                            color=p_annotate['color'], fontweight='bold', fontsize=ANNOTATION_FONT_SIZE, va='center', ha='left',
                            arrowprops=dict(arrowstyle="-", color=p_annotate['color'], alpha=0.4) if abs(p_annotate['y']-p_annotate['real_y']) > 0.01 else None)
                ax.scatter([2060], [p_annotate['real_y']], color=p_annotate['color'], s=30, zorder=5)

        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
        ax.set_ylabel("Million SAR", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("Year", fontweight='bold', fontsize=LABEL_FONT_SIZE)
        
        # Lock x-axis to 2060 limits
        ax.set_xlim(min(years), 2060)
        ticks = ax.get_xticks()
        ax.set_xticks(ticks[ticks <= 2060])
        
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Only show legend on the very first plot
        if i == 0:
            ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=True, loc='upper left')
        
    for _i, _ax in enumerate(axes):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=SUB_LABEL_SIZE, fontweight='bold', va='top')

    plt.tight_layout()
    plt.show()

def plot_agri_trilemma(all_results, all_sectoral_results, agr_sector=0, save_path=None):
    """
    Plot 6: agri_trilemma_water_food_trade.pdf
    Line plot: food self-sufficiency, agricultural imports and agricultural water use.
    (a) Domestic agricultural output and (b) agricultural imports, both real (2021 prices),
    (c) agricultural water use with the 2021 level (cap of the transformation scenario).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()

    scenarios = list(all_results.keys())
    years = all_results[scenarios[0]].index + 2020

    colors = {
        "Baseline":       "#d62728",  # Red
        "Vision_2030":    "#ff7f0e",  # Orange
        "Net_zero":       "#2ca02c",  # Green
        "Transformation": "#1f77b4",  # Blue
    }

    # (title, y-axis label, data per scenario)
    panels = [
        ("Domestic Agricultural Output", "Billion SAR (2021 prices)",
         lambda sc: all_sectoral_results[sc]['x'].iloc[:, agr_sector] / 1e6),
        ("Agricultural Imports", "Billion SAR (2021 prices)",
         lambda sc: all_sectoral_results[sc]['im'].iloc[:, agr_sector] / 1e6),
        ("Agricultural Water Use", "Billion m3",
         lambda sc: all_results[sc]['water_use_agr'] / 1e9),
    ]

    for i, (title, ylabel, get_data) in enumerate(panels):
        ax = axes[i]
        endpoints_2060 = []

        for sc in scenarios:
            data = get_data(sc)
            ax.plot(years, data, label=SCENARIO_LABELS.get(sc, sc), color=colors.get(sc), linewidth=2.5)

            # Extract 2060 value for annotation
            if 2060 in years:
                idx_60 = np.where(years == 2060)[0][0]
                val_60 = data.iloc[idx_60]
                endpoints_2060.append({'y': val_60, 'real_y': val_60, 'label': f"{val_60:.1f}", 'color': colors.get(sc)})

        # Agricultural water use: mark the 2021 level, the cap of the transformation scenario
        if i == 2:
            level_2021 = get_data(scenarios[0]).iloc[0]
            ax.axhline(level_2021, color='gray', linestyle='--', linewidth=1.5)
            ax.text(2047, level_2021, "2021 level (cap)", color='gray', fontsize=LABEL_FONT_SIZE - 2,
                    ha='center', va='bottom')

        # Apply non-overlapping annotations
        if endpoints_2060:
            for p_annotate in resolve_overlaps(endpoints_2060):
                ax.annotate(p_annotate['label'], xy=(2060, p_annotate['real_y']), xytext=(2061.2, p_annotate['y']),
                            color=p_annotate['color'], fontweight='bold', fontsize=ANNOTATION_FONT_SIZE, va='center', ha='left',
                            arrowprops=dict(arrowstyle="-", color=p_annotate['color'], alpha=0.4) if abs(p_annotate['y']-p_annotate['real_y']) > 0.01 else None)
                ax.scatter([2060], [p_annotate['real_y']], color=p_annotate['color'], s=30, zorder=5)

        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("Year", fontweight='bold', fontsize=LABEL_FONT_SIZE)

        # Lock x-axis to 2060 limits
        ax.set_xlim(years.min(), 2060)
        ticks = ax.get_xticks()
        ax.set_xticks(ticks[ticks <= 2060])

        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax.grid(True, linestyle=':', alpha=0.6)

        # Only show legend on the very first plot
        if i == 0:
            ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=True, loc='upper left')

    for _i, _ax in enumerate(axes):
        _ax.text(-0.05, 1.05, f"({chr(ord('a') + _i)})", transform=_ax.transAxes,
                 fontsize=SUB_LABEL_SIZE, fontweight='bold', va='top')

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def plot_water_results():
    print("Starting Water Results Generation...")
    
    # 1. Run Scenarios
    runner = WaterResultRunner(startyear=1, endyear=40)
    scenarios_to_run = ["Baseline", "Vision_2030", "Transformation"]
    raw_results = runner.run(scenarios_to_run, iterations=1, verbose=True)
    
    # Process macro results for existing plots
    processed_results = {sc: raw_results[sc][0]["results_macro"] for sc in scenarios_to_run}
    
    # Process sectoral results for the new plot
    processed_sectoral = {sc: raw_results[sc][0]["results_sectoral"] for sc in scenarios_to_run}

    # Process cost data
    processed_costs = {
        sc: {
            "irrigation_invest": raw_results[sc][0]["irrigation_invest"],
            "irrigation_om": raw_results[sc][0]["irrigation_om"],
            "irrigation_subsidy": raw_results[sc][0]["irrigation_subsidy"]
        } for sc in scenarios_to_run
    }
    
    # 2. Generate all plots
    print("\nGenerating Plots...")
    plot_water_use_comparison(processed_results)
    plot_water_balance_verification(processed_results)
    plot_scenario_trends_comparison(processed_results)
    plot_sectoral_supply_mix(processed_results, processed_sectoral)
    plot_irrigation_costs(processed_costs)
    plot_agri_trilemma(processed_results, processed_sectoral)
    
    print("\n" + "="*50)
    print("All water resource results have been generated successfully.")
    print("="*50)

if __name__ == "__main__":
    plot_water_results()