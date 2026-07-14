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

"""
Module for exporting model data to LaTeX format.

This module provides functionality to export various model components
(such as the list of input-output sectors) as formatted LaTeX tables.
"""

import pandas as pd
import os
from typing import Optional
from .calibration import ParametersCalibrated


def _ensure_tables_dir():
    """Ensure the Tables directory exists."""
    if not os.path.exists('Tables'):
        os.makedirs('Tables')


def export_sectors_to_latex(
    pc: ParametersCalibrated,
    output_path: str = "sectors_table.tex",
    caption: str = "List of Input-Output Sectors in the Model",
    label: str = "tab:sectors",
    include_indices: bool = True,
    longtable: bool = True
) -> str:
    """
    Export the list of input-output sectors as a LaTeX table.
    
    Parameters
    ----------
    pc : ParametersCalibrated
        The calibrated parameters object containing the sectors list.
    output_path : str, optional
        Filename for the LaTeX file (will be saved in Tables/). Default: "sectors_table.tex"
    caption : str, optional
        Caption for the LaTeX table.
    label : str, optional
        Label for referencing the table in LaTeX.
    include_indices : bool, optional
        Whether to include sector indices in the table. Default: True
    longtable : bool, optional
        Whether to use longtable environment (for multi-page tables). Default: True
        
    Returns
    -------
    str
        The generated LaTeX code.
    """
    
    _ensure_tables_dir()
    full_path = os.path.join('Tables', output_path)
    
    # Create DataFrame with sectors
    if include_indices:
        df = pd.DataFrame({
            'Index': range(len(pc.sectors)),
            'Sector Name': pc.sectors
        })
        column_format = 'cl'
    else:
        df = pd.DataFrame({
            'Sector Name': pc.sectors
        })
        column_format = 'l'
    
    # Generate LaTeX code
    if longtable:
        latex_code = df.to_latex(
            index=False,
            escape=False,
            longtable=True,
            column_format=column_format,
            caption=caption,
            label=label
        )
    else:
        latex_code = df.to_latex(
            index=False,
            escape=False,
            column_format=column_format,
            caption=caption,
            label=label
        )
    
    # Save to file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"LaTeX table exported to: {full_path}")
    print(f"Total sectors: {len(pc.sectors)}")
    
    return latex_code


def export_sectors_with_details_to_latex(
    pc: ParametersCalibrated,
    output_path: str = "sectors_detailed_table.tex",
    caption: str = "Input-Output Sectors with Economic Details",
    label: str = "tab:sectors_detailed",
    longtable: bool = True
) -> str:
    """
    Export the list of input-output sectors with additional economic details as a LaTeX table.
    
    Parameters
    ----------
    pc : ParametersCalibrated
        The calibrated parameters object containing the sectors list and data.
    output_path : str, optional
        Filename for the LaTeX file (will be saved in Tables/).
    caption : str, optional
        Caption for the LaTeX table.
    label : str, optional
        Label for referencing the table in LaTeX.
    longtable : bool, optional
        Whether to use longtable environment. Default: True
        
    Returns
    -------
    str
        The generated LaTeX code.
    """
    
    _ensure_tables_dir()
    full_path = os.path.join('Tables', output_path)
    
    # Create DataFrame with sectors and economic indicators
    df = pd.DataFrame({
        'Index': range(len(pc.sectors)),
        'Sector Name': pc.sectors,
        'Output Share (\\%)': (pc.dX * 100).round(2),
        'GDP Share (\\%)': ((pc.Y_ / pc.Y_.sum()) * 100).round(2),
        'Investment Share (\\%)': (pc.dI * 100).round(2)
    })
    
    # Generate LaTeX code
    if longtable:
        latex_code = df.to_latex(
            index=False,
            escape=False,
            longtable=True,
            column_format='clrrr',
            caption=caption,
            label=label,
            float_format="%.2f"
        )
    else:
        latex_code = df.to_latex(
            index=False,
            escape=False,
            column_format='clrrr',
            caption=caption,
            label=label,
            float_format="%.2f"
        )
    
    # Save to file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"LaTeX table with details exported to: {full_path}")
    print(f"Total sectors: {len(pc.sectors)}")
    
    return latex_code


def export_vision2030_sectors_to_latex(
    pc: ParametersCalibrated,
    output_path: str = "vision2030_sectors_table.tex",
    caption: str = "Vision 2030 Target Sectors",
    label: str = "tab:vision2030_sectors"
) -> str:
    """
    Export Vision 2030 target sectors with their growth targets as a LaTeX table.
    
    Parameters
    ----------
    pc : ParametersCalibrated
        The calibrated parameters object.
    output_path : str, optional
        Filename for the LaTeX file (will be saved in Tables/).
    caption : str, optional
        Caption for the LaTeX table.
    label : str, optional
        Label for referencing the table in LaTeX.
        
    Returns
    -------
    str
        The generated LaTeX code.
    """
    
    _ensure_tables_dir()
    full_path = os.path.join('Tables', output_path)
    
    # Filter Vision 2030 sectors
    vision_indices = pc.vision2030_sectors
    
    df = pd.DataFrame({
        'Index': vision_indices,
        'Sector Name': [pc.sectors[i] for i in vision_indices],
        'Yearly Growth Target (\\%)': (pc.sectoral_yearly_target_growth_rates_V2030_[vision_indices] * 100).round(2),
        'Total Target (\\%)': (pc.sectoral_total_targets_V2030_[vision_indices] * 100).round(2)
    })
    
    # Generate LaTeX code
    latex_code = df.to_latex(
        index=False,
        escape=False,
        column_format='clrr',
        caption=caption,
        label=label,
        float_format="%.2f"
    )
    
    # Save to file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"Vision 2030 sectors LaTeX table exported to: {full_path}")
    print(f"Total Vision 2030 sectors: {len(vision_indices)}")
    
    return latex_code


def export_growth_rates_to_latex(
    summary_df: pd.DataFrame,
    output_path: str = "growth_rates_comparison.tex",
    caption: str = "GDP Growth Rate Comparison Across Scenarios",
    label: str = "tab:growth_rates",
    longtable: bool = False
) -> str:
    """
    Export growth rates analysis summary to a LaTeX table.
    
    This function takes the output from analyze_growth_rates() and converts it
    to a well-formatted LaTeX table.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame output from analyze_growth_rates() function.
    output_path : str, optional
        Filename for the LaTeX file (will be saved in Tables/).
    caption : str, optional
        Caption for the LaTeX table.
    label : str, optional
        Label for referencing the table in LaTeX.
    longtable : bool, optional
        Whether to use longtable environment. Default: False (table fits on one page)
        
    Returns
    -------
    str
        The generated LaTeX code.
        
    Example
    -------
    >>> from model.analyze_growth_rates import analyze_growth_rates
    >>> summary_df = analyze_growth_rates(exp_reports)
    >>> export_growth_rates_to_latex(summary_df)
    """
    
    _ensure_tables_dir()
    full_path = os.path.join('Tables', output_path)
    
    # Make a copy to avoid modifying the original
    df = summary_df.copy()
    
    # Rename columns for better LaTeX formatting
    column_mapping = {
        'Scenario': 'Scenario',
        'Empirical GDP Growth 2013-2023 (%)': 'Empirical\\newline Growth\\newline 2013-2023\\newline (\\%)',
        'Model GDP Growth (Geometric, %)': 'Model Growth\\newline (Geometric)\\newline (\\%)',
        'Model GDP Growth (Arithmetic, %)': 'Model Growth\\newline (Arithmetic)\\newline (\\%)',
        'Y Start': 'GDP Start\\newline (Million SAR)',
        'Y End': 'GDP End\\newline (Million SAR)',
        'Total Growth (%)': 'Total Growth\\newline (\\%)'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Format numeric columns
    numeric_cols = [col for col in df.columns if col != 'Scenario']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    
    # Generate LaTeX code
    if longtable:
        latex_code = df.to_latex(
            index=False,
            escape=False,
            longtable=True,
            column_format='lrrrrrr',
            caption=caption,
            label=label
        )
    else:
        latex_code = df.to_latex(
            index=False,
            escape=False,
            column_format='lrrrrrr',
            caption=caption,
            label=label
        )
    
    # Save to file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"Growth rates LaTeX table exported to: {full_path}")
    print(f"Number of scenarios: {len(df)}")
    
    return latex_code


def export_detailed_growth_rates_to_latex(
    exp_reports: dict,
    variables: list = None,
    output_path: str = "detailed_growth_rates.tex",
    caption: str = r"Detailed Growth Rates by Variable and Scenario, in \%",
    label: str = "tab:detailed_growth",
    longtable: bool = True
) -> str:
    """
    Export detailed growth rates for multiple variables across scenarios to LaTeX.
    
    Parameters
    ----------
    exp_reports : dict
        Dictionary of experiment reports from ExperimentRunner.
    variables : list, optional
        List of variable names to analyze. Default: ['Y', 'C', 'I_total', 'EX', 'IM', 'P']
    output_path : str, optional
        Filename for the LaTeX file (will be saved in Tables/).
    caption : str, optional
        Caption for the LaTeX table.
    label : str, optional
        Label for referencing the table in LaTeX.
    longtable : bool, optional
        Whether to use longtable environment. Default: True
        
    Returns
    -------
    str
        The generated LaTeX code.
    """

    # Human-readable labels for variables
    VARIABLE_LABELS = {
        'Y':        'Output (GDP, Y)',
        'C':        'Consumption (C)',
        'I_total':  'Investment (I) total',
        'EX':       'Exports (EX)',
        'IM':       'Imports (IM)',
        'P':        'Profits (P)',
        'Gov_exp':  'Gov. expenditures',
        'YD_wage':  'Disposable wage income (YD wage)',
    }

    _ensure_tables_dir()
    full_path = os.path.join('Tables', output_path)
    
    if variables is None:
        variables = ['Y', 'C', 'I_total', 'EX', 'IM', 'P']
    
    # Build detailed growth rates table
    growth_data = []
    
    for var in variables:
        label_text = VARIABLE_LABELS.get(var, var)
        row = {'Variable': label_text}
        for scenario_name, scenario_runs in exp_reports.items():
            report = scenario_runs[0]
            df = report['results_macro'].copy()
            
            if var in df.columns:
                start_val = df.loc[1, var]
                end_val = df[var].iloc[-1]
                years = len(df) - 1
                
                if start_val > 0:
                    growth_rate = (end_val / start_val) ** (1 / years) - 1
                    row[scenario_name] = growth_rate * 100
                else:
                    row[scenario_name] = None
            else:
                row[scenario_name] = None
        
        growth_data.append(row)
    
    df = pd.DataFrame(growth_data)
    
    # Rename scenario columns: replace underscores with spaces for display
    scenario_col_mapping = {col: col.replace('_', ' ') for col in df.columns if col != 'Variable'}
    df = df.rename(columns=scenario_col_mapping)

    # Format numeric columns
    for col in df.columns:
        if col != 'Variable':
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Generate column format string
    num_scenarios = len(exp_reports)
    column_format = 'l' + 'r' * num_scenarios
    
    # Generate LaTeX code
    if longtable:
        latex_code = df.to_latex(
            index=False,
            escape=False,
            longtable=True,
            column_format=column_format,
            caption=caption,
            label=label
        )
    else:
        latex_code = df.to_latex(
            index=False,
            escape=False,
            column_format=column_format,
            caption=caption,
            label=label
        )
    
    # Save to file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"Detailed growth rates LaTeX table exported to: {full_path}")
    print(f"Variables analyzed: {len(variables)}")
    print(f"Scenarios compared: {len(exp_reports)}")
    
    return latex_code