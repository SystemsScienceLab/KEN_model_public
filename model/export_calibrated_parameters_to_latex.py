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
Module for exporting all calibrated parameters to LaTeX tables with numerical values.

This module provides a comprehensive function to export all calibrated parameter 
values, matrices, and vectors to LaTeX longtables and CSV files.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional


def _ensure_tables_dir():
    """Ensure the Tables directory exists."""
    if not os.path.exists('Tables'):
        os.makedirs('Tables')


def export_calibrated_parameters_to_latex(
    pc,
    parameters,
    output_path: str = "calibrated_parameters_complete.tex",
    include_csv_exports: bool = True
) -> str:
    """
    Export all calibrated parameter values to a comprehensive LaTeX document.
    
    This function generates:
    1. A complete longtable with all parameter values organized by category
    2. CSV exports of matrices and vectors for supplementary material
    3. Summary statistics and verification checks
    
    Parameters
    ----------
    pc : ParametersCalibrated
        The calibrated parameters object containing all calibration data.
    parameters : ModelParameters
        The model parameters object containing manual parameter values.
    output_path : str, optional
        Filename for the LaTeX file (will be saved in Tables/).
        Default: "calibrated_parameters_complete.tex"
    include_csv_exports : bool, optional
        Whether to export CSV files for matrices/vectors. Default: True
        
    Returns
    -------
    str
        The generated LaTeX code.
    """
    
    _ensure_tables_dir()
    full_path = os.path.join('Tables', output_path)
    
    def fmt_scalar(val):
        """Format scalar values for LaTeX"""
        if isinstance(val, (int, np.integer)):
            return f"{val:,}"
        elif isinstance(val, (float, np.floating)):
            if abs(val) < 0.0001 and val != 0:
                return f"{val:.2e}"
            else:
                return f"{val:.6f}"
        else:
            return str(val)
    
    def fmt_vector_sample(vec, name=""):
        """Format vector with sample values"""
        if len(vec) <= 5:
            vals = ', '.join([f'{x:.4f}' for x in vec])
            return f"[{vals}]"
        else:
            first = ', '.join([f'{x:.4f}' for x in vec[:3]])
            last = ', '.join([f'{x:.4f}' for x in vec[-2:]])
            return f"[{first}, ..., {last}]"
    
    def fmt_pct(val):
        """Format as percentage"""
        return f"{val*100:.2f}\\%"
    
    # Start building LaTeX document
    latex_lines = []
    
    # Document header
    latex_lines.extend([
        "% Calibrated Model Parameters - Complete Reference",
        "% Generated automatically from calibration data",
        "% ",
        "",
        "\\begin{longtable}{p{0.30\\textwidth} p{0.15\\textwidth} p{0.20\\textwidth} p{0.25\\textwidth}}",
        "\\caption{Calibrated Model Parameters (Saudi Arabia, Base Year 2021)} \\label{tab:calibrated_parameters_complete} \\\\",
        "\\toprule",
        "\\textbf{Parameter} & \\textbf{Symbol} & \\textbf{Value} & \\textbf{Source/Notes} \\\\",
        "\\midrule",
        "\\endfirsthead",
        "",
        "\\multicolumn{4}{c}%",
        "{{\\tablename\\ \\thetable{} -- continued from previous page}} \\\\",
        "\\toprule",
        "\\textbf{Parameter} & \\textbf{Symbol} & \\textbf{Value} & \\textbf{Source/Notes} \\\\",
        "\\midrule",
        "\\endhead",
        "",
        "\\midrule",
        "\\multicolumn{4}{r}{{Continued on next page}} \\\\",
        "\\endfoot",
        "",
        "\\bottomrule",
        "\\endlastfoot",
        ""
    ])
    
    # 1. PRODUCTION TECHNOLOGY PARAMETERS
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{1. Production Technology Parameters}} \\\\",
        "\\midrule",
        f"Capital productivity (baseline) & $e_K$ & {fmt_scalar(parameters.eK)} & Uniform across sectors \\\\",
        f"Capital-output ratio & $KX$ & {fmt_scalar(parameters.KX)} & From structural CGE literature \\\\",
        f"Desalination capital productivity & $e_{{K,desal}}$ & {fmt_scalar(pc.eK_desal)} & Calibrated from operational data \\\\",
        f"Wastewater capital productivity & $e_{{K,wwater}}$ & {fmt_scalar(pc.eK_wwater)} & Calibrated from operational data \\\\",
        f"Labor productivity (sectoral) & $\\alpha_s$ & {fmt_vector_sample(pc.alpha_)} & $\\max(Y_s/W_s, 0)$ from IOT \\\\",
        f"Intermediate input productivity & $\\beta_s$ & {fmt_vector_sample(pc.beta_)} & $\\max(Y_s/IntP_s, 0)$ from IOT \\\\",
        f"Capital productivity coefficient & $\\kappa_s$ & {fmt_vector_sample(pc.kappa_)} & $\\max(Y_s/K_s, 0)$ from IOT \\\\",
        f"Number of sectors & $S$ & {pc.S} & 86 sectors total \\\\",
        "\\midrule",
        ""
    ])
    
    # 2. DEMAND COMPOSITION SHARES
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{2. Demand Composition Shares (from 2021 IOTs)}} \\\\",
        "\\midrule",
        f"Sectoral consumption shares & $dC_s$ & {fmt_vector_sample(pc.dC)} & Sum: {fmt_scalar(np.sum(pc.dC))} \\\\",
        f"Sectoral government shares & $dG_s$ & {fmt_vector_sample(pc.dG)} & Sum: {fmt_scalar(np.sum(pc.dG))} \\\\",
        f"Sectoral investment shares & $dI_s$ & {fmt_vector_sample(pc.dI)} & Sum: {fmt_scalar(np.sum(pc.dI))} \\\\",
        f"Sectoral output shares & $dX_s$ & {fmt_vector_sample(pc.dX)} & Sum: {fmt_scalar(np.sum(pc.dX))} \\\\",
        f"Sectoral capital shares & $dK_s$ & {fmt_vector_sample(pc.dK)} & Sum: {fmt_scalar(np.sum(pc.dK))} \\\\",
        "\\midrule",
        ""
    ])
    
    # 3. BEHAVIORAL PARAMETERS
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{3. Behavioral Parameters}} \\\\",
        "\\midrule",
        f"MPC out of wage income & $\\alpha_1$ & {fmt_scalar(parameters.α1)} & Set to 1.0 (calibrated) \\\\",
        f"MPC out of profit income & $\\alpha_2$ & {fmt_scalar(pc.α2)} & Calibrated: $C^{{profit}}/YD^{{profit}}$ \\\\",
        f"MPC out of wealth & $\\alpha_3$ & {fmt_scalar(parameters.α3)} & Set to zero (baseline) \\\\",
        f"Fixed consumption & $\\alpha_0$ & {fmt_scalar(parameters.α0)} & Baseline assumption \\\\",
        f"Investment adjustment rate & $\\gamma_i$ & {fmt_scalar(parameters.gamma_i)} & Naqvi \\& Stockhammer (2018) \\\\",
        f"Profit-financed investment & invest\\_profit & {fmt_scalar(pc.invest_profit)} & Calibrated from bank credit data \\\\",
        "\\midrule",
        ""
    ])
    
    # 4. FINANCIAL PARAMETERS
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{4. Financial Parameters}} \\\\",
        "\\midrule",
        f"Loan repayment rate & $\\rho$ & {fmt_scalar(parameters.ρ)} & 5\\% (20-year standard credit) \\\\",
        f"Interest rate markup & $\\mu$ & {fmt_scalar(pc.mu)} & Calibrated ({fmt_pct(pc.mu)} spread) \\\\",
        f"Deposit interest rate & $r^M$ & {fmt_scalar(pc.rm)} & Calibrated ({fmt_pct(pc.rm)}) \\\\",
        f"Loan interest rate & $r^L$ & {fmt_scalar(pc.rl)} & $r^M + \\mu$ ({fmt_pct(pc.rl)}) \\\\",
        f"Government bond interest rate & $r^{{Bond}}$ & {fmt_scalar(pc.interest_gov_bonds)} & SAMA bills assumption ({fmt_pct(pc.interest_gov_bonds)}) \\\\",
        f"Initial household wealth & $V_0$ & {fmt_scalar(pc.V)} & Thousand SAR (monetary data 2021) \\\\",
        f"Initial total loans & $L_0$ & {fmt_scalar(pc.L)} & Thousand SAR (bank credit 2021) \\\\",
        f"Initial government wealth & $GV_0$ & {fmt_scalar(pc.GV)} & Thousand SAR (PIF assets est.) \\\\",
        f"Initial government bonds & $Bond_0$ & {fmt_scalar(pc.Bond)} & Thousand SAR (public debt 2021) \\\\",
        f"Initial government net wealth & $NW^{{gov}}_0$ & {fmt_scalar(pc.Gov_net_wealth)} & Thousand SAR ($GV_0 - Bond_0$) \\\\",
        "\\midrule",
        ""
    ])
    
    # 5. FISCAL PARAMETERS
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{5. Fiscal Parameters}} \\\\",
        "\\midrule",
        f"VAT rate & $\\tau^{{VAT}}$ & {fmt_scalar(pc.tau_vat)} & ({fmt_pct(pc.tau_vat)}) \\\\",
        f"Household tax rate & $\\tau^H$ & {fmt_scalar(pc.τ)} & Non-VAT taxes \\\\",
        f"Production subsidy rates & $\\tau^{{sub}}_s$ & {fmt_vector_sample(pc.tau_sub_production_)} & 86-sector vector \\\\",
        f"Production tax rates & $\\tau^{{prod}}_s$ & {fmt_vector_sample(pc.tau_tax_production_)} & 86-sector vector \\\\",
        f"Product tax rates & $\\tau^{{net}}_s$ & {fmt_vector_sample(pc.tau_tax_products_net_)} & 86-sector vector \\\\",
        f"Investment tax rate & $\\tau^{{inv}}$ & {fmt_scalar(pc.tau_tax_investments)} & Homogeneous \\\\",
        "\\midrule",
        ""
    ])
    
    # 6. PUBLIC-PRIVATE OWNERSHIP
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{6. Public-Private Ownership Shares}} \\\\",
        "\\midrule",
        f"Public sector shares & $s^{{public}}_s$ & {fmt_vector_sample(pc.s_public_)} & Oil, desalination, energy \\\\",
        f"Private sector shares & $s^{{private}}_s$ & {fmt_vector_sample(pc.s_private_)} & Complement ($1 - s^{{public}}$) \\\\",
        f"Oil sector public share & - & {fmt_pct(pc.s_public_[pc.crude_oil_sector])} & From oil revenues/profits \\\\",
        f"Desalination public share & - & {fmt_pct(pc.s_public_[pc.sector_desal])} & Fully public (100\\%) \\\\",
        "\\midrule",
        ""
    ])
    
    # 7. AVERAGE GROWTH RATES
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{7. Average Growth Rates (2013-2023 CAGR)}} \\\\",
        "\\midrule",
        f"GDP growth & $\\overline{{g^Y}}$ & {fmt_scalar(pc.gY_avg)} & {fmt_pct(pc.gY_avg)} per year \\\\",
        f"Investment growth & $\\overline{{g^I}}$ & {fmt_scalar(pc.gI_avg)} & {fmt_pct(pc.gI_avg)} per year \\\\",
        f"Total exports growth & $\\overline{{g^{{EX}}}}$ & {fmt_scalar(pc.gEXP_avg)} & {fmt_pct(pc.gEXP_avg)} per year \\\\",
        f"Oil exports growth & $\\overline{{g^{{EX,oil}}}}$ & {fmt_scalar(pc.gEXP_oil_avg)} & {fmt_pct(pc.gEXP_oil_avg)} per year \\\\",
        f"Non-oil exports growth & $\\overline{{g^{{EX,non-oil}}}}$ & {fmt_scalar(pc.gEXP_non_oil_avg)} & {fmt_pct(pc.gEXP_non_oil_avg)} per year \\\\",
        f"Imports growth & $\\overline{{g^{{IM}}}}$ & {fmt_scalar(pc.gIM_avg)} & {fmt_pct(pc.gIM_avg)} per year \\\\",
        f"Government exp. growth & $\\overline{{g^{{GY}}}}$ & {fmt_scalar(pc.gGY_avg)} & {fmt_pct(pc.gGY_avg)} per year \\\\",
        f"Wage growth & $\\overline{{g^W}}$ & {fmt_scalar(pc.gW)} & {fmt_pct(pc.gW)} per year \\\\",
        f"Inflation (non-oil deflator) & $\\overline{{\\pi}}$ & {fmt_scalar(pc.inflation_avg)} & {fmt_pct(pc.inflation_avg)} per year \\\\",
        f"Desalination capacity growth & $\\overline{{g^{{desal}}}}$ & {fmt_scalar(pc.gDesalCap_avg)} & {fmt_pct(pc.gDesalCap_avg)} per year \\\\",
        f"Wastewater capacity growth & $\\overline{{g^{{wwater}}}}$ & {fmt_scalar(pc.gWWaterCap_avg)} & {fmt_pct(pc.gWWaterCap_avg)} per year \\\\",
        "\\midrule",
        ""
    ])
    
    # 8. WATER SECTOR PARAMETERS
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{8. Water Sector Parameters}} \\\\",
        "\\midrule",
        # wwater_share_of_water_use is not exported: it is only used when Add_new_wastewater_sector=True
        # (inactive in standard runs). The existing Sewerage IO sector is used directly instead.
        f"Desalination share of water use & $s^{{desal}}$ & {fmt_scalar(pc.desal_share_of_water_use)} & {fmt_pct(pc.desal_share_of_water_use)} \\\\",
        f"Total water use (initial) & $W^{{total}}_0$ & {fmt_scalar(pc.total_water_use_initial)} & m$^3$ (2021) \\\\",
        f"Household water consumption & $W^{{HH}}$ & {fmt_scalar(pc.water_use_hh_2021)} & m$^3$ (2021) \\\\",
        f"Water price (initial) & $p^{{water}}$ & {fmt_scalar(pc.water_price)} & Thousand SAR/m$^3$ \\\\",
        f"Desalination capital & $K^{{desal}}$ & {fmt_scalar(pc.K_desal)} & Thousand SAR \\\\",
        f"Wastewater capital & $K^{{wwater}}$ & {fmt_scalar(pc.K_wwater)} & Thousand SAR \\\\",
        f"Desalination depreciation rate & $\\delta^{{desal}}$ & {fmt_scalar(pc.δ_desal)} & 25-year lifetime \\\\",
        f"Wastewater depreciation rate & $\\delta^{{wwater}}$ & {fmt_scalar(pc.δ_wwater)} & 25-year lifetime \\\\",
        f"Desalination utilization & $u^{{desal}}$ & {fmt_scalar(pc.u_desal)} & {fmt_pct(pc.u_desal)} \\\\",
        f"Wastewater utilization & $u^{{wwater}}$ & {fmt_scalar(pc.u_wwater)} & {fmt_pct(pc.u_wwater)} \\\\",
        f"Water use intensities & $w_s$ & {fmt_vector_sample(pc.water_use_intensities_)} & m$^3$/thousand SAR (86 sectors) \\\\",
        "\\midrule",
        ""
    ])
    
    # 9. ENERGY SECTOR PARAMETERS
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{9. Energy Sector Parameters}} \\\\",
        "\\midrule",
        f"Gas intensities & $e^{{gas}}_s$ & {fmt_vector_sample(pc.gas_intensities_)} & GJ/thousand SAR (86 sectors) \\\\",
        f"Electricity intensities & $e^{{elec}}_s$ & {fmt_vector_sample(pc.electricity_intensities_)} & GJ/thousand SAR (86 sectors) \\\\",
        f"Oil intensities & $e^{{oil}}_s$ & {fmt_vector_sample(pc.oil_intensities_)} & GJ/thousand SAR (86 sectors) \\\\",
        f"Gas emission factor & $\\epsilon^{{gas}}$ & {fmt_scalar(pc.gas_emission_factor)} & MtCO$_2$/GJ (IPCC 2006) \\\\",
        f"Oil emission factor & $\\epsilon^{{oil}}$ & {fmt_scalar(pc.oil_emission_factor)} & MtCO$_2$/GJ (IPCC 2006) \\\\",
        f"RE emission factor & $\\epsilon^{{RE}}$ & {fmt_scalar(pc.re_emission_factor)} & Zero operational emissions \\\\",
        f"Initial RE generation (2021) & $RE_0$ & {fmt_scalar(pc.initial_RE_generation)} & GJ \\\\",
        f"RE solar share (2023) & $s^{{solar}}_{{2023}}$ & {fmt_scalar(pc.re_solar_share)} & {fmt_pct(pc.re_solar_share)} \\\\",
        f"RE wind share (2023) & $s^{{wind}}_{{2023}}$ & {fmt_scalar(pc.re_wind_share)} & {fmt_pct(pc.re_wind_share)} \\\\",
        f"RE bioenergy share (2023) & $s^{{bio}}_{{2023}}$ & {fmt_scalar(pc.re_bio_share)} & {fmt_pct(pc.re_bio_share)} \\\\",
        "\\midrule",
        ""
    ])
    
    # 10. RENEWABLE ENERGY INVESTMENT
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{10. Renewable Energy Investment Parameters}} \\\\",
        "\\midrule",
        f"Solar PV CapEx & $C^{{solar}}$ & {fmt_scalar(pc.capex_solar_per_gj)} & kSAR/GJ-year (691k USD/MW, CF=0.28) \\\\",
        f"Onshore Wind CapEx & $C^{{wind}}$ & {fmt_scalar(pc.capex_wind_per_gj)} & kSAR/GJ-year (1,041k USD/MW, CF=0.40) \\\\",
        f"Bioenergy CapEx & $C^{{bio}}$ & {fmt_scalar(pc.capex_bio_per_gj)} & kSAR/GJ-year (4,900k USD/MW, CF=0.85) \\\\",
        f"Weighted avg RE CapEx & $\\overline{{C^{{RE}}}}$ & {fmt_scalar(pc.re_capex_per_gj_year)} & kSAR/GJ-year (2023 mix) \\\\",
        "\\midrule",
        ""
    ])
    
    # 11. INITIAL CONDITIONS - SECTORAL
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{11. Initial Conditions - Sectoral Stocks (2021)}} \\\\",
        "\\midrule",
        f"Sectoral output & $X_{{s,0}}$ & {fmt_vector_sample(pc.X_)} & Thousand SAR (Sum: {fmt_scalar(np.sum(pc.X_))}) \\\\",
        f"Sectoral final demand & $Y_{{s,0}}$ & {fmt_vector_sample(pc.Y_)} & Thousand SAR (Sum: {fmt_scalar(np.sum(pc.Y_))}) \\\\",
        f"Sectoral capital stock & $K_{{s,0}}$ & {fmt_vector_sample(pc.K_)} & Thousand SAR (Sum: {fmt_scalar(np.sum(pc.K_))}) \\\\",
        f"Sectoral consumption & $C_{{s,0}}$ & {fmt_vector_sample(pc.C_)} & Thousand SAR (Sum: {fmt_scalar(np.sum(pc.C_))}) \\\\",
        f"Sectoral government & $G_{{s,0}}$ & {fmt_vector_sample(pc.G_)} & Thousand SAR (Sum: {fmt_scalar(np.sum(pc.G_))}) \\\\",
        f"Sectoral exports & $EX_{{s,0}}$ & {fmt_vector_sample(pc.EX_)} & Thousand SAR (Sum: {fmt_scalar(np.sum(pc.EX_))}) \\\\",
        f"Sectoral imports & $IM_{{s,0}}$ & {fmt_vector_sample(pc.IM_)} & Thousand SAR (Sum: {fmt_scalar(np.sum(pc.IM_))}) \\\\",
        f"Sectoral wages & $W_{{s,0}}$ & {fmt_vector_sample(pc.W_)} & Thousand SAR (Sum: {fmt_scalar(np.sum(pc.W_))}) \\\\",
        f"Sectoral profits & $P_{{s,0}}$ & {fmt_vector_sample(pc.P_)} & Thousand SAR (Sum: {fmt_scalar(np.sum(pc.P_))}) \\\\",
        f"Markup & $\\theta_s$ & {fmt_vector_sample(pc.θ_)} & Calibrated from $P/(X-P)$ \\\\",
        "\\midrule",
        ""
    ])
    
    # 12. INITIAL CONDITIONS - AGGREGATES
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{12. Initial Conditions - Aggregates (2021)}} \\\\",
        "\\midrule",
        f"Total consumption & $C_0$ & {fmt_scalar(np.sum(pc.C_))} & Thousand SAR \\\\",
        f"Total government spending & $GY_0$ & {fmt_scalar(pc.GY)} & Thousand SAR \\\\",
        f"Total investment & $I_0$ & {fmt_scalar(pc.I_total)} & Thousand SAR \\\\",
        f"Desalination investment & $I^{{desal}}_0$ & {fmt_scalar(pc.I_desal)} & Thousand SAR (avg. 2013-2023) \\\\",
        f"Wastewater investment & $I^{{wwater}}_0$ & {fmt_scalar(pc.I_wwater)} & Thousand SAR (avg. 2021-2023) \\\\",
        f"Total exports & $EX_0$ & {fmt_scalar(np.sum(pc.EX_))} & Thousand SAR \\\\",
        f"Total imports & $IM_0$ & {fmt_scalar(np.sum(pc.IM_))} & Thousand SAR \\\\",
        f"GDP (production approach) & $GDP_0$ & {fmt_scalar(pc.GDP)} & Thousand SAR \\\\",
        f"Household wage income & $YD^{{wage}}_0$ & {fmt_scalar(pc.YD_wage)} & Thousand SAR \\\\",
        f"Household profit income & $YD^{{profit}}_0$ & {fmt_scalar(pc.YD_profit)} & Thousand SAR \\\\",
        f"Bank profits & $PB_0$ & {fmt_scalar(pc.PB)} & Thousand SAR \\\\",
        f"Aggregate depreciation rate & $\\delta$ & {fmt_scalar(pc.δ)} & Weighted average \\\\",
        "\\midrule",
        ""
    ])
    
    # 13. IO MATRICES
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{13. Input-Output Matrices and Coefficients}} \\\\",
        "\\midrule",
        f"A matrix (tech coefficients) & $\\mathbf{{A}}_0$ & $86 \\times 86$ & See supplementary file \\\\",
        f"Z matrix (intermediate flows) & $\\mathbf{{Z}}_0$ & $86 \\times 86$ & See supplementary file \\\\",
        f"Identity matrix & $\\mathbf{{I}}$ & $86 \\times 86$ & Standard \\\\",
        "\\midrule",
        ""
    ])
    
    # 14. AR MODEL PARAMETERS
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{14. Autoregressive (AR) Model Parameters}} \\\\",
        "\\midrule",
        f"GDP optimal lag (AIC) & $L^Y_{{AIC}}$ & {pc.Y_optimal_lag_aic} & Data-driven selection \\\\",
        f"GDP optimal lag (BIC) & $L^Y_{{BIC}}$ & {pc.Y_optimal_lag_bic} & Data-driven selection \\\\",
        f"Deflator optimal lag (AIC) & $L^\\pi_{{AIC}}$ & {pc.Deflator_optimal_lag_aic} & For inflation forecasting \\\\",
        f"Deflator optimal lag (BIC) & $L^\\pi_{{BIC}}$ & {pc.Deflator_optimal_lag_bic} & For inflation forecasting \\\\",
        f"Gov. exp. optimal lag (AIC) & $L^{{GY}}_{{AIC}}$ & {pc.GY_optimal_lag_aic} & Fiscal forecasting \\\\",
        f"Gov. exp. optimal lag (BIC) & $L^{{GY}}_{{BIC}}$ & {pc.GY_optimal_lag_bic} & Fiscal forecasting \\\\",
        f"Oil exports optimal lag (AIC) & $L^{{oil}}_{{AIC}}$ & {pc.Oil_optimal_lag_aic} & Oil activity forecasting \\\\",
        f"Oil exports optimal lag (BIC) & $L^{{oil}}_{{BIC}}$ & {pc.Oil_optimal_lag_bic} & Oil activity forecasting \\\\",
        "\\midrule",
        ""
    ])
    
    # 15. KEY SECTOR INDICES
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{15. Key Sector Indices}} \\\\",
        "\\midrule",
        f"Desalination sector index & - & {pc.sector_desal} & {pc.sectors[pc.sector_desal]} \\\\",
        f"Wastewater sector index & - & {pc.sector_wwater} & {pc.sectors[pc.sector_wwater]} \\\\",
        f"Water sector index & - & {pc.sector_water} & {pc.sectors[pc.sector_water]} \\\\",
        f"Crude oil sector index & - & {pc.crude_oil_sector} & {pc.sectors[pc.crude_oil_sector]} \\\\",
        f"Refined oil sector index & - & {pc.refined_oil_sector} & {pc.sectors[pc.refined_oil_sector]} \\\\",
        f"Electricity/gas/AC sector & - & {pc.electricity_gas_AC_sector[0]} & {pc.sectors[pc.electricity_gas_AC_sector[0]]} \\\\",
        f"Agriculture sector index & - & {pc.agr_sector} & {pc.sectors[pc.agr_sector]} \\\\",
        f"Construction sectors & - & {len(pc.sectors_construction)} sectors & Indices: {pc.sectors_construction} \\\\",
        f"Tourism sectors & - & {len(pc.sectors_tourism)} sectors & Indices: {pc.sectors_tourism} \\\\",
        f"AI/High-tech sectors & - & {len(pc.sectors_AI_hightech)} sectors & Indices: {pc.sectors_AI_hightech} \\\\",
        f"Vision 2030 sectors (total) & - & {len(pc.vision2030_sectors)} sectors & See Vision 2030 table \\\\",
        "\\midrule",
        ""
    ])
    
    # 16. DATA SOURCES SUMMARY
    latex_lines.extend([
        "\\multicolumn{4}{l}{\\textbf{16. Data Sources Summary}} \\\\",
        "\\midrule",
        "\\multicolumn{4}{p{0.90\\textwidth}}{",
        "\\textbf{Input-Output Tables:} GASTAT, 86-sector tables 2013-2023 with desalination/wastewater disaggregation. \\\\",
        "\\textbf{National Accounts:} GASTAT time series 1969-2023, GDP by expenditure and institutional sectors. \\\\",
        "\\textbf{Monetary Data:} SAMA (Saudi Arabian Monetary Authority), bank credits, government deposits, financial liabilities. \\\\",
        "\\textbf{Energy Data:} Historical electricity generation mix (2000-2023), renewable capacity, IRENA cost data. \\\\",
        "\\textbf{Water Data:} Ministry of Environment, desalination database, water use intensities by sector. \\\\",
        "\\textbf{Fiscal Data:} Ministry of Finance, government expenditures, oil revenues, public debt, SAMA bills. \\\\",
        "\\textbf{Trade Statistics:} GASTAT exports/imports by commodity and year. \\\\",
        "\\textbf{Investment Data:} Gross fixed capital formation, perpetual inventory method for capital stock estimation. \\\\",
        "} \\\\",
        ""
    ])
    
    # Close the table
    latex_lines.append("\\end{longtable}")
    
    # Join all lines
    latex_content = '\n'.join(latex_lines)
    
    # Save to file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"\n{'='*80}")
    print(f"Calibrated parameters LaTeX table exported to: {full_path}")
    print(f"{'='*80}")
    # Count parameter rows (those with & and \\)
    row_end = '\\\\'
    print(f"Total parameters documented: {len([l for l in latex_lines if '&' in l and row_end in l])}")
    
    # Export CSV files if requested
    if include_csv_exports:
        print(f"\nExporting supplementary CSV and LaTeX files...")
        
        # A matrix
        csv_path = os.path.join('Tables', 'A_matrix_86x86.csv')
        tex_path = os.path.join('Tables', 'A_matrix_86x86.tex')
        df_A = pd.DataFrame(pc.A, index=pc.sectors, columns=pc.sectors)
        df_A.to_csv(csv_path)
        
        # Create custom LaTeX for A matrix with numbered indices and A4 landscape fit
        df_A_numbered = pd.DataFrame(pc.A, 
                                     index=range(1, len(pc.sectors)+1), 
                                     columns=range(1, len(pc.sectors)+1))
        
        with open(tex_path, 'w') as f:
            f.write("% Technical Coefficients Matrix A (86x86)\n")
            f.write("% Use with: \\input{Tables/A_matrix_86x86.tex}\n")
            f.write("% Note: Requires landscape, adjustbox, and tabular packages\n\n")
            f.write("\\begin{landscape}\n")
            f.write("\\begin{adjustbox}{width=1.0\\textheight, height=0.95\\textwidth, keepaspectratio}\n")
            f.write("\\tiny\n")
            f.write(df_A_numbered.to_latex(float_format="%.2f", longtable=False, 
                                          caption="Technical Coefficients Matrix A (86x86) - Sectors numbered 1-86", 
                                          label="tab:A_matrix", escape=False))
            f.write("\\end{adjustbox}\n")
            f.write("\\end{landscape}\n")
        
        print(f"  - A_matrix_86x86.csv")
        print(f"  - A_matrix_86x86.tex (numbered indices, A4 landscape, tiny font)")
        
        # Z matrix (io_table)
        csv_path = os.path.join('Tables', 'Z_matrix_86x86.csv')
        tex_path = os.path.join('Tables', 'Z_matrix_86x86.tex')
        pc.io_table.to_csv(csv_path)
        pc.io_table.to_latex(tex_path, float_format="%.2f", longtable=True,
                            caption="Intermediate Input Flows Matrix Z (86x86)", 
                            label="tab:Z_matrix")
        print(f"  - Z_matrix_86x86.csv")
        print(f"  - Z_matrix_86x86.tex")
        
        # Sectoral values
        csv_path = os.path.join('Tables', 'sectoral_initial_values.csv')
        tex_path = os.path.join('Tables', 'sectoral_initial_values.tex')
        df_sectoral = pd.DataFrame({
            'Sector': pc.sectors,
            'X_output': pc.X_,
            'Y_final_demand': pc.Y_,
            'K_capital': pc.K_,
            'C_consumption': pc.C_,
            'G_government': pc.G_,
            'I_investment': pc.gross_capital_formation,
            'EX_exports': pc.EX_,
            'IM_imports': pc.IM_,
            'W_wages': pc.W_,
            'P_profits': pc.P_,
            'alpha_labor_prod': pc.alpha_,
            'beta_intermediate_prod': pc.beta_,
            'kappa_capital_prod': pc.kappa_,
            'theta_markup': pc.θ_,
            'dC_cons_share': pc.dC,
            'dG_gov_share': pc.dG,
            'dI_inv_share': pc.dI,
            'dX_output_share': pc.dX,
            'dK_capital_share': pc.dK,
            's_public': pc.s_public_,
            's_private': pc.s_private_
        })
        df_sectoral.to_csv(csv_path, index=False)
        df_sectoral.to_latex(tex_path, index=False, float_format="%.4f", longtable=True,
                            caption="Sectoral Initial Values (2021)", 
                            label="tab:sectoral_values")
        print(f"  - sectoral_initial_values.csv")
        print(f"  - sectoral_initial_values.tex")
        
        # Intensities
        csv_path = os.path.join('Tables', 'sectoral_intensities.csv')
        tex_path = os.path.join('Tables', 'sectoral_intensities.tex')
        df_intensities = pd.DataFrame({
            'Sector': pc.sectors,
            'water_intensity_m3_per_kSAR': pc.water_use_intensities_,
            'gas_intensity_GJ_per_kSAR': pc.gas_intensities_,
            'electricity_intensity_GJ_per_kSAR': pc.electricity_intensities_,
            'oil_intensity_GJ_per_kSAR': pc.oil_intensities_
        })
        df_intensities.to_csv(csv_path, index=False)
        df_intensities.to_latex(tex_path, index=False, float_format="%.6f", longtable=True,
                               caption="Sectoral Resource Use Intensities (Water, Gas, Electricity, Oil)", 
                               label="tab:sectoral_intensities")
        print(f"  - sectoral_intensities.csv")
        print(f"  - sectoral_intensities.tex")
        
        # Growth rates and other aggregates
        csv_path = os.path.join('Tables', 'aggregate_parameters.csv')
        tex_path = os.path.join('Tables', 'aggregate_parameters.tex')
        df_aggregates = pd.DataFrame({
            'Parameter': [
                'gY_avg', 'gI_avg', 'gEXP_avg', 'gEXP_oil_avg', 'gEXP_non_oil_avg',
                'gIM_avg', 'gGY_avg', 'gW', 'inflation_avg', 'gDesalCap_avg', 'gWWaterCap_avg',
                'eK', 'KX', 'alpha_1', 'alpha_2', 'alpha_3', 'gamma_i', 'rho', 'mu',
                'r_M', 'r_L', 'tau_VAT', 'V_0', 'L_0', 'GV_0', 'Bond_0', 'GDP_0',
                'C_0', 'GY_0', 'I_0', 'EX_0', 'IM_0'
            ],
            'Value': [
                pc.gY_avg, pc.gI_avg, pc.gEXP_avg, pc.gEXP_oil_avg, pc.gEXP_non_oil_avg,
                pc.gIM_avg, pc.gGY_avg, pc.gW, pc.inflation_avg, pc.gDesalCap_avg, pc.gWWaterCap_avg,
                parameters.eK, parameters.KX, parameters.α1, pc.α2, parameters.α3, parameters.gamma_i, 
                parameters.ρ, pc.mu, pc.rm, pc.rl, pc.tau_vat, pc.V, pc.L, pc.GV, pc.Bond, pc.GDP,
                np.sum(pc.C_), pc.GY, pc.I_total, np.sum(pc.EX_), np.sum(pc.IM_)
            ]
        })
        df_aggregates.to_csv(csv_path, index=False)
        df_aggregates.to_latex(tex_path, index=False, float_format="%.6f",
                              caption="Aggregate Parameters and Initial Conditions", 
                              label="tab:aggregate_parameters")
        print(f"  - aggregate_parameters.csv")
        print(f"  - aggregate_parameters.tex")
    
    print(f"\n{'='*80}")
    print(f"Export complete!")
    print(f"{'='*80}\n")
    
    return latex_content
