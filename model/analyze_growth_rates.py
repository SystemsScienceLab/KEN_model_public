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

def analyze_growth_rates(exp_reports, variables_to_analyze=None, show_detailed=True):
    """
    Analyze and compare growth rates across multiple scenarios.
    
    Parameters
    ----------
    exp_reports : dict
        Dictionary of experiment reports from ExperimentRunner, where keys are 
        scenario names and values are lists of run reports.
    variables_to_analyze : list, optional
        List of variable names to analyze. Default: ['Y', 'C', 'I_total', 'EX', 'IM', 'P']
    show_detailed : bool, optional
        Whether to show detailed growth rates by variable. Default: True
    
    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame containing summary statistics for all scenarios.
    """
    import pandas as pd
    
    if variables_to_analyze is None:
        variables_to_analyze = ['Y', 'C', 'I_total', 'EX', 'IM', 'P']
    
    print("=" * 80)
    print("GROWTH RATE COMPARISON ACROSS SCENARIOS")
    print("=" * 80)

    # Create a summary dataframe
    growth_summary = []

    for scenario_name, scenario_runs in exp_reports.items():
        # Take the first run (or average if multiple runs)
        report = scenario_runs[0]
        
        # Get macro results and parameters
        df = report['results_macro'].copy()
        pc = report['parameters_calibrated']
        
        # Calculate growth rate of Y
        df['gY'] = df['Y'].pct_change() * 100
        
        # Get start and end values
        Y_startyear = df.loc[1, 'Y']
        Y_endyear = df['Y'].iloc[-1]
        years = len(df) - 1
        
        # Calculate average growth rate (geometric mean)
        Y_avg_growth_rate_geometric = (Y_endyear / Y_startyear) ** (1 / years) - 1
        
        # Calculate average growth rate (arithmetic mean of period growth rates)
        Y_avg_growth_rate_arithmetic = df['gY'].mean()
        
        # Store results
        growth_summary.append({
            'Scenario': scenario_name,
            'Empirical GDP Growth 2013-2023 (%)': pc.gGY_avg * 100,
            'Model GDP Growth (Geometric, %)': Y_avg_growth_rate_geometric * 100,
            'Model GDP Growth (Arithmetic, %)': Y_avg_growth_rate_arithmetic,
            'Y Start': Y_startyear / 1e6,  # in millions
            'Y End': Y_endyear / 1e6,  # in millions
            'Total Growth (%)': ((Y_endyear / Y_startyear) - 1) * 100
        })



 
        
        # Print individual scenario details
        print(f"\n{scenario_name.upper()} SCENARIO:")
        print("-" * 80)
        print(f"  Empirical GDP growth rate 2013-2023: {pc.gGY_avg*100:.2f}%")
        print(f"  Model GDP growth (geometric mean): {Y_avg_growth_rate_geometric * 100:.2f}%")
        print(f"  Model GDP growth (arithmetic mean): {Y_avg_growth_rate_arithmetic:.2f}%")
        print(f"  GDP start (billion SAR): {Y_startyear / 1e9:.2f}")
        print(f"  GDP end (billion SAR): {Y_endyear / 1e9:.2f}")
        print(f"  Total growth over period: {((Y_endyear / Y_startyear) - 1) * 100:.2f}%")

    # Create summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    summary_df = pd.DataFrame(growth_summary)
    summary_df = summary_df.round(2)
    print(summary_df.to_string(index=False))

    # Calculate additional growth rates for key variables
    if show_detailed:
        print("\n" + "=" * 80)
        print("DETAILED GROWTH RATES BY VARIABLE")
        print("=" * 80)

        for var in variables_to_analyze:
            print(f"\n{var}:")
            for scenario_name, scenario_runs in exp_reports.items():
                report = scenario_runs[0]
                df = report['results_macro'].copy()
                
                if var in df.columns:
                    start_val = df.loc[1, var]
                    end_val = df[var].iloc[-1]
                    years = len(df) - 1
                    
                    if start_val > 0:
                        growth_rate = (end_val / start_val) ** (1 / years) - 1
                        print(f"  {scenario_name:20s}: {growth_rate * 100:6.2f}%")

        # FDI share of total investment overview
        print(f"\nFDI Share of Total Investment:")
        for scenario_name, scenario_runs in exp_reports.items():
            report = scenario_runs[0]
            df = report['results_macro'].copy()
            pc = report['parameters_calibrated']
            
            if 'FDI_share_of_investment' in df.columns:
                t_first = df.index[0]
                t_last = df.index[-1]
                t_mid = df.index[len(df) // 2]
                
                fdi_first = df.loc[t_first, 'FDI_share_of_investment'] * 100
                fdi_mid = df.loc[t_mid, 'FDI_share_of_investment'] * 100
                fdi_last = df.loc[t_last, 'FDI_share_of_investment'] * 100
                
                year_first = 2020 + t_first
                year_mid = 2020 + t_mid
                year_last = 2020 + t_last
                
                print(f"  {scenario_name:20s}:  {year_first}={fdi_first:5.2f}%  |  "
                      f"{year_mid}={fdi_mid:5.2f}%  |  {year_last}={fdi_last:5.2f}%")
            else:
                enable_fdi = getattr(report['parameters'], 'enable_FDI', False)
                print(f"  {scenario_name:20s}:  FDI disabled (enable_FDI={enable_fdi})")
    
    return summary_df