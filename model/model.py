# =============================================================================
# =============================================================================
# AUTHORSHIP of model components and paper sections:
# =============================================================================
# Michael Miess (all model components and paper sections): Conceptualization (including methodology of the model), data curation, formal analysis, investigation, methodology, software (model architecture and code), supervision, validation, visualization, writing – original draft, writing – review and editing.
# Ansir Ilyas (focus on energy model components and paper sections): Conceptualization, data curation, formal analysis, investigation, software (model code), validation, visualization, writing – original draft, writing – review and editing. 
# Dan Wang (focus on water model components and paper sections): Conceptualization, data curation, formal analysis, investigation, software (model code), validation, visualization, writing – original draft, writing – review and editing. 
# Asjad Naqvi: Conceptualization, methodology, investigation, validation, writing – review and editing. 
# Yoshihide Wada: Conceptualization (including initial conception of the project and study), funding acquisition, project administration, resources, supervision, investigation, validation, writing – review and editing.
# =============================================================================
# Joel Foramitti (technical consultant): Initial software architecture development, first coding of preliminary model versions (base model from literature), technical support
# =============================================================================

import numpy as np
import pandas as pd
from dataclasses import fields

from .model_classes import ModelConfig, ModelVariables, ModelResults, ModelParameters
from .calibration import ParametersCalibrated
from .endogenize_input_output_matrix_A import endogenize_input_output_matrix_A
from .energy_module import energy_module
from .investment_module import investment_module
from .exports_module import exports_module
from .green_exports_module import calculate_green_exports_and_substitution
from .ar_model_estimation import estimate_ar_model, estimate_ar_model_log, estimate_ar_model_growthrates, forecast, determine_optimal_lag_order, determine_optimal_lag_order_log, determine_optimal_lag_order_growthrates, identify_structural_break
from .water_module import replace_agr_X, water_accounting, apply_irrigation_costs
from .remittances_module import remittances_module
import warnings

############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
# I. Run model: config file with calibrated and manual parameters & time steps, some tax-related preparations
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################



def run_model(config: ModelConfig, verbose=False):
    """Perform a simulation of the model for a given number of timesteps."""

    parameters = config.p
    parameters_cal = config.pc
    timesteps = config.T  # Number of simulation steps
    variables = ModelVariables(timesteps, parameters_cal)  # Dynamic variables

    # Run first step
    run_model_step(variables, parameters, parameters_cal, 1, verbose=verbose)

    # Run remaining steps
    for t in range(2, timesteps + 1):
        run_model_step(variables, parameters,
                       parameters_cal, t, verbose=verbose)

    return prepare_results(timesteps, variables, config)
############################################################################################################################################################################################################################



def _phased_tax_rate(t, rate_1, timing_1, rate_2, timing_2, phase_in_years):
    """Return the effective tax rate at period t with a piecewise-linear phase-in.

    When a new tax tier becomes active (timing_1 or timing_2 > 0 and t >= timing_x),
    the rate is ramped linearly from 0 (or the previous tier's rate) to the target
    rate over `phase_in_years` periods, avoiding a discontinuous jump in model output.

    Parameters
    ----------
    t               : current simulation period
    rate_1          : target rate for the first tax tier
    timing_1        : period at which tier-1 becomes active (0 = never)
    rate_2          : target rate for the second tax tier
    timing_2        : period at which tier-2 becomes active (0 = never)
    phase_in_years  : number of periods for the linear ramp (1 = instant step-change)
    """
    n = max(phase_in_years, 1)  # guard against zero to avoid division by zero
    if timing_2 > 0 and t >= timing_2:
        # Tier-2 active: ramp from rate_1 toward rate_2
        frac = min((t - timing_2 + 1) / n, 1.0)
        return rate_1 + (rate_2 - rate_1) * frac
    elif timing_1 > 0 and t >= timing_1:
        # Tier-1 active: ramp from 0 toward rate_1
        frac = min((t - timing_1 + 1) / n, 1.0)
        return rate_1 * frac
    return 0.0



############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
# II. IO table recalculation for desalination and wastewater shares
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
def calc_io_table(v: ModelVariables, p: ModelParameters, pc: ParametersCalibrated, t: int):
    """
    Calculate dynamic IO Table

    Adjust IO column (intermediary inputs) of water sector
    Add desalinated share as intermediary input for water
    Reduce all other intermediary inputs by desalinated share
    """
    desal_share = np.minimum(
        (v.k_[t-1][pc.sector_desal] * pc.u_desal * pc.eK_desal) /
        v.x_[t-1][pc.sector_water], 1
    )
    A = pc.A.copy()
    A[:, pc.sector_water] = pc.A[:, pc.sector_water] * (1 - desal_share)
    A[pc.sector_desal, pc.sector_water] = desal_share
    return A, desal_share



############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
# III. RUN MODEL STEP (main simulation loop)
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################


def run_model_step(v: ModelVariables, p: ModelParameters, pc: ParametersCalibrated, t: int, verbose=False):
    """Perform a single simulation step of the model."""

    ########################################################################################################################################################################################################################
    # 1. Update A matrix according to desalination share
    ########################################################################################################################################################################################################################
    v.A__[t], v.desal_share[t] = calc_io_table(v, p, pc, t)

    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    # 2. Growth rate 0 tests: steady state initialization, set all exogenous growth rates to zero
    # To check whether model remains in equilibrium solution
    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    if p.Growth_rates_0_test:
        pc.gY_avg = 0
        pc.gW = 0
        pc.I_desal = 0
        pc.I_wwater = 0
        pc.gDesalCap_avg = 0
        pc.gWWaterCap_avg = 0
        pc.gI_avg = 0
        pc.gGY_avg = 0
        pc.gEXP_oil_avg = 0
        pc.gEXP_non_oil_avg = 0
        pc.gIM_avg = 0
        p.non_oil_export_growth_adjustment = 0
        p.oil_growth_adjustment = 0
        p.profit_tax_rate_1 = 0
        p.wage_tax_rate_1 = 0
        p.profit_tax_rate_2 = 0
        p.wage_tax_rate_2 = 0
        p.wage_population_growth_adjustment = 0
        pc.inflation_avg = 0
        pc.inflation_2021 = 0
        pc.inflation_2022 = 0
        pc.sectoral_yearly_growth_rates_V2030_ = np.zeros(pc.S)
        pc.sectoral_yearly_growth_rates_non_oil_exports_ = np.zeros(pc.S)
        p.ρ = 0
        p.ρ_ext = 0
        pc.rl = 0
        pc.interest_gov_bonds = 0
        p.remittances_growth_adjustment = 0
        pc.gRemittances_avg = 0
        if t == 1:
            v.Sectoral_investment_distribution_[t]= v.X_[t-1] / np.sum(v.X_[t-1])
            v.I_[t] = v.I_total[t-1] * v.Sectoral_investment_distribution_[t]
        else:
            v.I_[t] = v.I_[t-1]
    ########################################################################################################################################################################################################################

    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    # 3. Spending decisions
    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################

    ########################################################################################################################################################################################################################
    # 3.1 Prices
    ########################################################################################################################################################################################################################
    # Prices based on last round's unit costs + markup
    # (no negative prices allowed),
    # set price floor and inflation ceiling by government control
    # Prices are set to 1 in the first period by calibration of sectoral mark-up rates
    # Total NOMINAL output X - NOMINAL profits P = total COSTS by definition is used for COST ACCOUNTING,
    # With past inflation rates as proxy for expectations, this yields cyclical dynamics; the AR-based forecast is used instead:
    if p.Inflation_timeseries_model:
        if t == 1:
            # PK MOTIVATION (Godley-Lavoie, 2012; Lavoie, 2014 §9):
            # The 2021-2022 period was dominated by transitory global supply-chain
            # disruptions and post-pandemic normalisation — shocks that are exogenous
            # to this structural long-run model and are not captured in its equations.
            # Saudi firms under bounded rationality, aware of the country's institutionally
            # anchored inflation (fixed exchange rate peg, administered energy prices,
            # price regulation), would NOT embed these transitory spikes into their
            # structural cost-markup pricing decisions.
            # Using inflation_avg also prevents the AR expectations series from being
            # seeded with a non-structural outlier that would otherwise trigger an
            # artifactual mean-reverting deflation signal at t=3.
            # (pc.inflation_2021 and pc.inflation_2022 remain stored for sensitivity analyses.)
            v.Inflation_forecast[t] = pc.inflation_avg
            if verbose:
                print("v.Inflation_forecast[t] t=1: structural anchor inflation_avg (not transitory 2021 data)",
                      v.Inflation_forecast[t], "time ", t)
        elif t == 2:
            # Same reasoning: structural anchor instead of the transitory 2022 rate.
            # The 2022 non-oil deflator spike was driven by global commodity/energy
            # prices — an exogenous shock absent from this model's equations.
            v.Inflation_forecast[t] = pc.inflation_avg
            if verbose:
                print("v.Inflation_forecast[t] t=2: structural anchor inflation_avg (not transitory 2022 data)",
                      v.Inflation_forecast[t], "time ", t)
        else:                           # Calculate forecast via growth rates
            Inflation_model_fit = estimate_ar_model_growthrates(
                pc.Deflator_non_oil_GDP_timeseries, pc.Deflator_optimal_lag_aic, pc.begin_time_series_estimation_deflator)
            # This creates a forecast for the next two periods
            v.Inflation_forecast[t] = forecast(
                Inflation_model_fit, steps=1).iloc[-1]
            if verbose:
                print(
                    "v.Inflation_forecast[t] using time series model", v.Inflation_forecast[t], "time ", t)

    # ─── Normal-cost pricing: EMA-smoothed unit cost share ────────────────────
    # PK motivation (Kalecki/Godley-Lavoie tradition): firms set prices on NORMAL
    # unit costs, not on actual period-to-period realised cost shares.  A simple
    # exponential moving average prevents volatile profits in early periods from
    # feeding directly into prices and causing the observed swing-in of the
    # cumulative deflator.  price_smooth_factor controls how quickly firms revise
    # their normal-cost estimate (0 = never revise, 1 = fully use current cost).
    _raw_cost_share = np.where(
        v.X_[t-1] != 0,
        (v.X_[t-1] - v.P_tot_[t-1]) / v.X_[t-1],
        v.unit_cost_smooth_[t-1]   # fallback: carry forward if X = 0
    )
    v.unit_cost_smooth_[t] = (
        p.price_smooth_factor * _raw_cost_share
        + (1.0 - p.price_smooth_factor) * v.unit_cost_smooth_[t-1]
    )

    # Either use time series model forecasted inflation (preferred choice, AR-based expectations mirroring bounded rationality),
    # or use past inflation as proxy for expected inflation (naive expectations), which yield cyclical dynamics (interesting dynamics, but NOT preferred option). 
    if p.Inflation_timeseries_model:
        v.p_[t] = np.minimum(np.maximum(
            v.unit_cost_smooth_[t] * (1 + pc.θ_) * (1 + v.Inflation_forecast[t]),
            np.maximum(p.price_floor, v.p_[t-1] * (1.0 - p.price_decline_limit))),
            v.p_[t-1]*(1 + p.price_ceiling)
        )
    else:
        v.p_[t] = np.minimum(np.maximum(
            v.unit_cost_smooth_[t] * (1 + pc.θ_) * (1 + v.inflation[t-1]),
            np.maximum(p.price_floor, v.p_[t-1] * (1.0 - p.price_decline_limit))),
            v.p_[t-1]*(1 + p.price_ceiling)
        )
    # Assumption: in desalination and wastewater, technological progress and inflation approximately cancel,
    # so the price index is pinned to 1 (currently, no better data available for this correction, this is subject to future work).
    v.p_[t][pc.sector_desal] = 1
    v.p_[t][pc.sector_wwater] = 1
    # Optional: pin financial-sector price to 1 to prevent the large downward drift
    # observed in unconstrained runs (due to high profit shares / calibration artefacts).
    # WARNING: setting this True affects bank profit calculations downstream; re-check
    # financial-sector consistency if results change unexpectedly.
    if p.Fix_finance_sector_price:
        v.p_[t][pc.sector_finance] = 1

    ########################################################################################################################################################################################################################
    # 3.2 Inflation and price indices
    ########################################################################################################################################################################################################################

    # Inflation as GDP deflator weighted with Total output X
    # Accounting with Laspeyres index
    v.inflation[t] = (
        pc.inflation_avg if t == 1 else
        np.sum(v.dX_[t-1] * v.p_[t]) / np.sum(v.dX_[t-1] * v.p_[t-1]) - 1
    )

    # Inflation consumers, weighted with consumption vector dC
    v.inflation_consumers[t] = (
        pc.inflation_avg if t == 1 else
        np.sum(pc.dC * v.p_[t]) / np.sum(pc.dC * v.p_[t-1]) - 1
    )

    # Price indices
    # Derive cumulative inflation rates, starting with the price level of one, adding yearly inflations
    v.deflator_gdp[t] = (
        1 + pc.inflation_avg if t == 1 else
        v.deflator_gdp[t-1] * (1 + v.inflation[t])
    )

    v.price_index_cons[t] = (
        1 + pc.inflation_avg if t == 1 else
        v.price_index_cons[t-1] * (1 + v.inflation_consumers[t])
    )
    # Append inflation of the current period to the time series model
    pc.Deflator_non_oil_GDP_timeseries = pd.concat(
        [pc.Deflator_non_oil_GDP_timeseries, pd.Series([v.deflator_gdp[t]])], ignore_index=True)

    ########################################################################################################################################################################################################################
    # 3.3 Wages (nominal and real, real wages necessary for potential coefficient calibration)
    ########################################################################################################################################################################################################################
    # Wage growth determination: Wages are adjusted with past inflation (standard choice)
    # Otherwise, they can be adjusted according to the maximum of inflation and GDP growth forecast (wage growth limit)
    # Last option, least preferred, because data is scarce (only 4-6 data points from IOTs so far), is to take an average growth rate from the past nominal wage growth trend in IOTs.
    if p.Inflation_based_wage_growth:
        v.gW[t] = v.inflation[t-1]
    elif p.Wage_growth_limit:
        v.gW[t] = np.maximum(0, np.minimum(
            v.inflation[t-1], v.Y_growth_rate_forecast[t]))
    else:
        v.gW[t] = pc.gW

    # WAGES change with the composition of sectoral production, i.e. they adapt while the economy is transforming
    # At the same time, wage growth is adjusted with exponential growth rates taken from population forecasts for Saudi Arabia according to SSPs (see calibration file).
    if p.Sectoral_wage_growth:
        if t == 1:
            v.W_[t] = v.W_[t-1] * (1 + v.gW[t]) * \
                (1 + p.wage_population_growth_adjustment)**t
        elif t == 2:
            # Blend t=1 and t=3+ regimes to prevent a structural jump at the wage formula switch.
            # PK motivation: wages are institutionally set and do not jump discontinuously.
            target_W2 = v.X_[t-1] * pc.wage_share_in_X_ * \
                (1 + v.gW[t]) * (1 + p.wage_population_growth_adjustment)**t
            v.W_[t] = 0.5 * v.W_[t-1] * (1 + v.gW[t]) * (1 + p.wage_population_growth_adjustment) \
                + 0.5 * target_W2
        elif t > 2:  # Attach wages to X_ with a fixed calibrated wage share. A formulation using sectoral Y_ growth rates does not work here, because many Y_ sectors are negative, which distorts the growth rates.
            v.W_[t] = v.X_[t-1] * pc.wage_share_in_X_ * \
                (1 + v.gW[t]) * (1 + p.wage_population_growth_adjustment)**t
    else:
        v.W_[t] = v.W_[t-1] * (1 + v.gW[t]) * \
            (1 + p.wage_population_growth_adjustment)**t
    v.W[t] = np.sum(v.W_[t])  # Total Nominal wages

    # Calculate real wages according to current price level
    v.w_[t] = v.W_[t] / v.p_[t]
    v.w[t] = np.sum(v.w_[t])  # Total Nominal wages

    ########################################################################################################################################################################################################################
    # 3.4 Household (HH) consumption
    ########################################################################################################################################################################################################################
    # Disposable income is spent on consumption, consumption out of wealth is set to 0 for now
    # Calibrated alpha2 - MPC out of PROFIT income (profit income dependent on model assumptions,
    # includes all additions such as resident, non-resident consumption, etc.)

    # The idea behind this switch is that households might want to keep REAL consumption stable, and adjust their consumption to inflation up front.
    # Stable REAL consumption seems like a reasonable assumption, especially in a country like Saudi Arabia with relatively stable inflation rates historically.

    # NOTE: (YD_wage - remittances) is used here because remittances are a leakage that leaves Saudi Arabia
    # and does not finance domestic consumption. Remittances are already computed earlier in the step.
    # α2 is calibrated consistently in calibration.py to match residual consumption after this leakage.
    # Standard choice for model documentation and first paper:
    # adapt_consumption_inflation = True
    adapt_consumption_inflation = True
    if adapt_consumption_inflation:
        v.C[t] = (
            (
                (p.α0 + (p.α1 * (v.YD_wage[t-1] - v.remittances[t-1]))) +
                (np.maximum(pc.α2 * v.YD_profit[t-1], 0)) +
                (p.α3 * v.V[t-1])
            ) * (1 + v.Inflation_forecast[t]) / (1 + pc.tau_vat)
        )       # Standard: include the inflation forecast to keep real consumption stable.
    else:
        v.C[t] = (
            (
                (p.α0 + (p.α1 * (v.YD_wage[t-1] - v.remittances[t-1]))) +
                (np.maximum(pc.α2 * v.YD_profit[t-1], 0)) +
                (p.α3 * v.V[t-1])
            ) / (1 + pc.tau_vat)
        )
    # Determine sectoral nominal consumption
    v.C_[t] = pc.dC * v.C[t]
    ########################################################################################################################################################################################################################

    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    # 4. Investment module: economic and energy investment for capital stock
    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    investment_module(v, p, pc, t, verbose=verbose)
    ########################################################################################################################################################################################################################

    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    # 5. Exports, imports, government spending (exogenous, preferably time-series based)
    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    # Exports (in the exogenous option) grow with average growth rates or times series estimations.
    # They can be adjusted for scenarios with exogenous parameters.
    exports_module(v, p, pc, t, verbose=verbose)
    ########################################################################################################################################################################################################################

    #####################################################################################################################################################################################################################
    # 5.1 Imports — usually not used, because imports are the residual between desired demand and domestic supply in standard model choice
    #####################################################################################################################################################################################################################
    if p.Import_timeseries_model:
        Import_model_fit = estimate_ar_model_growthrates(
            pc.IM_timeseries, pc.IM_optimal_lag_aic, pc.begin_time_series_estimation_im)
        v.IM_growth_rate_forecast[t] = forecast(
            Import_model_fit, steps=1).iloc[-1]
        v.IM_[t] = v.IM_[t-1] * \
            (1 + v.IM_growth_rate_forecast[t] + p.import_growth_adjustment)
        pc.IM_timeseries = pd.concat(
            [pc.IM_timeseries, pd.Series([v.IM_[t]])], ignore_index=True)
    else:
        v.IM_[t] = np.maximum(0, v.IM_[t-1] * (1 + p.import_growth_adjustment))
        # Do not allow negative imports (these are either meaningless or should be defined as exports)
    # 3. Apply net-zero effects (add exports and subtract imports)
    # We place this AFTER the baseline IM calculation so the substitution is subtracted
    # from the calculated baseline import volume.
    if p.net_emission_reduction_green_investments_exports_transformation:
        calculate_green_exports_and_substitution(v, p, pc, t)

    #####################################################################################################################################################################################################################
    # 5.2 Government spending
    #####################################################################################################################################################################################################################
    # We have to differentiate between gov consumption (IOTs) and total gov expenditure (ts_government) to be consistent with different data sources
    # Government expenditures in ts_government data we should match:  1,038,933

    # If government spending is countercyclical, then government spending is increased when output is negative by the amount of negative growth rate last period
    if p.Countercyclical_gov_policy:
        if p.GY_timeseries_model:
            GY_model_fit = estimate_ar_model_growthrates(
                pc.GY_timeseries, pc.GY_optimal_lag_aic, pc.begin_time_series_estimation_gov)
            # Pick the latest value.
            v.GY_growth_rate_forecast[t] = forecast(
                GY_model_fit, steps=1).iloc[-1]
            if p.Gov_expenditures_oil_exports_cyclicality and p.Oil_exports_random_fluctuations:
                if t > 1:
                    start = max(1, t - 5)
                    base_GY = np.maximum(
                        (np.mean([v.GY[τ] for τ in range(start, t)], axis=0)), 0)
                else:
                    base_GY = v.GY[t-1]
                v.GY[t] = base_GY * (1 + float(v.GY_growth_rate_forecast[t]) + v.oil_exports_shock[t-1]
                                     * p.government_oil_exports_cyclicality_factor + v.Gov_confidence_boost[t])
            else:
                v.GY[t] = v.GY[t-1] * (1 + float(v.GY_growth_rate_forecast[t]))
            pc.GY_timeseries = pd.concat(
                [pc.GY_timeseries, pd.Series([v.GY[t]])], ignore_index=True)
        else:
            if p.Gov_expenditures_oil_exports_cyclicality and p.Oil_exports_random_fluctuations:
                if t > 1:
                    start = max(1, t - 5)
                    base_GY = np.maximum(
                        (np.mean([v.GY[τ] for τ in range(start, t)], axis=0)), 0)
                else:
                    base_GY = v.GY[t-1]
                v.GY[t] = base_GY * (1 + pc.gGY_avg + v.oil_exports_shock[t-1] *
                                     p.government_oil_exports_cyclicality_factor + v.Gov_confidence_boost[t])
            else:
                v.GY[t] = v.GY[t-1] * \
                    (1 + pc.gGY_avg + v.Gov_confidence_boost[t])
    else:
        # Here estimate AR time series model for the growth rates of government expenditures, based on past growth rates applied to GY
        if p.GY_timeseries_model:
            GY_model_fit = estimate_ar_model_growthrates(
                pc.GY_timeseries, pc.GY_optimal_lag_aic, pc.begin_time_series_estimation_gov)
            # Pick the latest value.
            v.GY_growth_rate_forecast[t] = forecast(
                GY_model_fit, steps=1).iloc[-1]
            # Option to correlate government expenditure pro-cyclically (with lag) with the oil export shock from last period
            if p.Gov_expenditures_oil_exports_cyclicality and p.Oil_exports_random_fluctuations:
                if t > 1:
                    start = max(1, t - 5)
                    base_GY = np.maximum(
                        (np.mean([v.GY[τ] for τ in range(start, t)], axis=0)), 0)
                else:
                    base_GY = v.GY[t-1]
                v.GY[t] = base_GY * (1 + float(v.GY_growth_rate_forecast[t]) +
                                     v.oil_exports_shock[t-1] * p.government_oil_exports_cyclicality_factor)
            else:
                v.GY[t] = v.GY[t-1] * (1 + float(v.GY_growth_rate_forecast[t]))
            pc.GY_timeseries = pd.concat(
                [pc.GY_timeseries, pd.Series([v.GY[t]])], ignore_index=True)
        else:
            if p.Gov_expenditures_oil_exports_cyclicality and p.Oil_exports_random_fluctuations:
                if t > 1:
                    start = max(1, t - 5)
                    base_GY = np.maximum(
                        (np.mean([v.GY[τ] for τ in range(start, t)], axis=0)), 0)
                else:
                    base_GY = v.GY[t-1]
                v.GY[t] = base_GY * (1 + pc.gGY_avg + v.oil_exports_shock[t-1]
                                     * p.government_oil_exports_cyclicality_factor)
            else:
                v.GY[t] = v.GY[t-1] * (1 + pc.gGY_avg)

    # Create sectoral government consumption
    v.GY_[t] = pc.dG * v.GY[t]

    v.Sub_production_[t] = pc.tau_sub_production_ * v.X_[t]
    # Gov. expenditure includes consumption, subsidies, investment, interest payments on gov bonds, and residual expenditures
    # Bond interest/repayment term floored at zero: when Bond < 0 (gov is net creditor) this would
    # otherwise create negative Gov_exp. Sovereign assets are tracked separately via Gov_ext_assets.
    # Two-phase sovereign financing: interest and principal repayment on international sovereign bonds (Bond_external) are added when Phase 2 is active.
    # Interest cost: analogous to domestic bond interest in Gov_exp.
    # Repayment cost: analogous to domestic bond principal repayment (ρ × Bond[t-1]) — financed by widening the domestic fiscal deficit.
    v.Bond_external_interest_cost[t] = pc.interest_gov_bonds * max(v.Bond_external[t - 1], 0.0)
    v.Bond_external_repayment[t]     = p.ρ_ext             * max(v.Bond_external[t - 1], 0.0)
    v.Gov_exp[t] = v.GY[t] + np.sum(v.Sub_production_[t]) + v.GI[t] + np.maximum(v.Bond_domestic[t-1], 0) * np.sum(pc.interest_gov_bonds + p.ρ) + pc.gov_exp_resid + v.Bond_external_interest_cost[t] + v.Bond_external_repayment[t]
    # Real government expenditure
    v.gov_exp[t] = v.Gov_exp[t] / v.deflator_gdp[t]

    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    # 6. Firm supply decision & actual production: Supply choice Q_s_ by firms depends on expectations and production with limitational (preferably real) production function
    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################

    ########################################################################################################################################################################################################################
    # 6.1 Expected growth rate of final demand
    ########################################################################################################################################################################################################################
    # Initalize with average growth rate, and then let the model calculate new expected growth rate
    # by the growth of final demand. We assume that firms can observe the growth rates of final demand components
    # Lagged (t-2) formulation found numerically unstable; preferred AR formulation below.
    
    # Alternative aggregate formulation retained for reference; published scenarios use the AR time series model below.
    if t <= 1:
        v.gY_e[t] = pc.gY_avg
    else:
        # Average aggregate demand growth rate all sectors from last period for expectation formation (index uses t-1)
        v.gY_e[t] = (np.sum((pc.dC * v.C[t] + pc.dG * v.GY[t] + v.I_demand_[t] + v.EX_oil_[t] + v.EX_non_oil_[t] - v.IM_[t])) /
                     np.sum((pc.dC * v.C[t-1] + pc.dG * v.GY[t-1] + pc.dI * v.I_total[t-1] + v.EX_oil_[t-1] + v.EX_non_oil_[t-1] - v.IM_[t-1])))-1
        # Sectoral variant retained for reference; not used in published scenarios.
        v.gY_e_[t] = ((pc.dC * v.C[t] + pc.dG * v.GY[t] + v.I_demand_[t] + v.EX_oil_[t] + v.EX_non_oil_[t] - v.IM_[t]) /
                      (pc.dC * v.C[t-1] + pc.dG * v.GY[t-1] + pc.dI * v.I_total[t-1] + v.EX_oil_[t-1] + v.EX_non_oil_[t-1] - v.IM_[t-1]))-1

    
    ########################################################################################################################################################################################################################
    # Remark: This is the PREFERRED expectation formulation based on a time series model of past Y growth rates
    ########################################################################################################################################################################################################################
    # Expected growth rate of Y as a time series model of past Y timeseries (via growth rates in standard choice)
    # Remark: this parameter has a strong choice on model results, thus it is also shown in the scenario comparison run file run_model_COMPARE_scenarios.ipynb
    # This might mediate the effect of the high growth rates, and lead the AR model on a more realistic expansion path.
    if p.Y_timeseries_model:
        if t == 1:
            v.Y_growth_rate_forecast[t] = pc.gY_2021
            if verbose:
                print("v.Y_growth_rate_forecast[t] 2021 from data",
                      v.Y_growth_rate_forecast[t], "time ", t)
        elif t == 2:
            v.Y_growth_rate_forecast[t] = pc.gY_2022
            if verbose:
                print("v.Y_growth_rate_forecast[t] 2022 from data",
                      v.Y_growth_rate_forecast[t], "time ", t)
            # For 2023, gY_2023  = -0.037, which is a recession and would not represent this parameter as a proxy of the recent growth rates in Saudi Arabia before running the model
            # For this reason, growth rates are estimated on more normal growth rates that approximately represent past growth. 
        # Calculate forecast via logs like in Poledna, Miess et al. (2024) ABM paper
        elif p.Y_timeseries_model_log:
            Y_model_fit = estimate_ar_model_log(
                pc.Y_timeseries, pc.Y_optimal_lag_aic, pc.begin_time_series_estimation_Y)
            Y_level_forecast = np.exp(forecast(Y_model_fit, steps=1).iloc[-1])
            # Growth rate of final demand
            v.Y_growth_rate_forecast[t] = (
                Y_level_forecast / np.sum(v.Y_[t-1])) - 1
        else: # This is the STANDARD AR estimation choice used throughout all scenarios: Calculate forecast via growth rates
            Y_model_fit = estimate_ar_model_growthrates(
                pc.Y_timeseries, pc.Y_optimal_lag_aic, pc.begin_time_series_estimation_Y)
            # This creates a forecast for the next two periods, pick the next period's growth rate (t+1) for the current period's expectation formation
            v.Y_growth_rate_forecast[t] = forecast(
                Y_model_fit, steps=1).iloc[-1]
        print("v.Y_growth_rate_forecast[t]", v.Y_growth_rate_forecast[t], "time ", t)

    # Supply choice depends on past realized DEMAND v.Y_[t-1] and the expected GDP growth rate!
    if p.Y_timeseries_model:
        # Use maximum here to avoid that Q_s_ is negative (which makes no economic sense)
        # If countercyclicical gov policy is true: Forecasts are boosted by government confidence boost, as a function of a negative gY[t-1]
        v.Q_s_[t] = np.maximum(
            0, v.Y_[t-1] * (1 + v.Y_growth_rate_forecast[t] + v.Gov_confidence_boost[t]))
    else:
        v.Q_s_[t] = np.maximum(
            0, v.Y_[t-1] * (1 + v.gY_e[t] + v.Gov_confidence_boost[t]))
    if verbose:
        print("v.Y_growth_rate_forecast[t] incl. gov. confidence boost",
            v.Y_growth_rate_forecast[t] + v.Gov_confidence_boost[t], "time ", t)
    # Alternative: use the average Y growth rate (1+pc.gY_avg) to stabilize the model if it becomes unstable.
    # Sectoral expectations can cause these dynamics when imports exceed domestic production in some sectors.

    # Determine real supply choice
    # Convert nominal supply choice to real supply choice using prices
    v.q_s_[t] = v.Q_s_[t] / v.p_[t]

    # Update the calibrated productivity coefficients to currently available input factors before new production occurs:
    # beta and Kappy from t-1, labor is more flexible, all is scaled to last period's output because production happens after supply decision
    # Idea: there is some flexibility and excess capacity each year, if this is used by more production, production capacity is adapted every year. This seems like a reasonable assumption.


    # Test whether productivity parameters indeed scale back to output
    v.Y_test_alpha_[t-1] = pc.alpha_[t] * v.w_[t-1]
    v.Y_test_beta_[t-1] = pc.beta_[t] * v.intP_[t-1]
    v.Y_test_kappa_[t-1] = pc.kappa_[t] * v.k_[t-1]

    ########################################################################################################################################################################################################################
    # 6.2 Aggregate desired demand: Add all demand components
    ########################################################################################################################################################################################################################
    # HH consumption + govt + consumption + investment + EX - IM
    # Investment inkl. private and public investment for NACE + desalination
    v.Q_d_[t] = pc.dC * v.C[t] + pc.dG * v.GY[t] + \
        v.I_demand_[t] + v.EX_oil_[t] + v.EX_non_oil_[t] - v.IM_[t]
    # Convert nominal desired demand to real desired demand using prices
    v.q_d_[t] = v.Q_d_[t] / v.p_[t]
    # Oil extraction capacity assumed non-binding (Saudi Arabia's large proven reserves).
    #
    # SET Y_ EQUAL TO DEMAND BEFORE SUPPLY AND PRODUCTION CONSTRAINTS happen
    # Important: set Y_ = to demand to make demand components PULL GDP before supply decision and before SUPPLY constraints!
    v.Y_[t] = v.Q_d_[t]
    v.y_[t] = v.q_d_[t]


    # Productivity coefficients (alpha, beta, kappa) are either updated to be consistent
    # with the current A matrix (if Update_productivity_coefficients=True, the standard),
    # or held constant at their calibrated values. When updated, they are recalculated only
    # at t=1 using the initial A matrix and then held fixed for t>1 to ensure model stability.
    if p.Update_productivity_coefficients:
        if t == 1:
            # Recalculate intermediate inputs using the new A matrix, then derive coefficients
            v.IntP_[t-1] = np.sum(v.A__[t], axis=0) * v.X_[t-1]
            v.alpha_[t] = np.maximum(v.Y_[t] / v.W_[t-1], 0)
            v.beta_[t] = np.maximum(v.Y_[t] / v.IntP_[t-1], 0)
            v.kappa_[t] = np.maximum(v.Y_[t] / v.K_[t-1], 0)
        else:
            # Hold coefficients constant after t=1
            v.alpha_[t] = v.alpha_[t-1]
            v.beta_[t] = v.beta_[t-1]
            v.kappa_[t] = v.kappa_[t-1]
    else:
        # Use the calibrated constants throughout
        v.alpha_[t] = pc.alpha_
        v.beta_[t] = pc.beta_
        v.kappa_[t] = pc.kappa_




    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    # 7. Limitational production with Leontief production function
    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    # Limitational Leontief production function: Each factor of production (intermediate inputs, capital, labor) limits total output by sectors.
    # Assume away demand restrictions through intermediate inputs for steady state calibration: turn flexible according to exogenous parameter so that growth is not reduced through this.
    # Possibility to implement capacity constraints on intermediate inputs and on certain sectors for certain times (e.g. water sector).
    # All production takes place in REAL terms, so prices matter here: in real terms it matters what you can actually afford to produce, given prices.
    # But main behavioral equations hinge on variables in nominal terms (Keynesian thinking), and are then converted into real choices through prices.
    ########################################################################
    if p.Limitational_nominal_production:   # Assumption: all production takes place according to NOMINAL terms
        if t == p.constraint_period_agr:
            Min1 = np.minimum(
                v.beta_[t] * v.IntP_[t-1] * pc.non_agr_ * (1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]) +
                v.beta_[t] * v.IntP_[t-1] * pc.s_agr_ *
                (1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]),
                v.kappa_[t] * v.K_[t-1] * (1 + v.gk_[t-1]) * p.flex_capital)
            Min2 = np.minimum(
                v.alpha_[t] * v.W_[t], Min1)
        elif t == p.constraint_period_nonagr:
            Min1 = np.minimum(
                v.beta_[t] * v.IntP_[t-1] * pc.non_agr_ * (1 - p.capacity_constraint_nonagr) * (1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]) +
                v.beta_[t] * v.IntP_[t-1] * pc.s_agr_ *
                (1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]),
                v.kappa_[t] * v.K_[t-1] * (1 + v.gk_[t-1]) * p.flex_capital)
            Min2 = np.minimum(
                v.alpha_[t] * v.W_[t], Min1)
        else:
            Min1 = np.minimum(v.beta_[t] * v.IntP_[t-1] * (1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]),
                              v.kappa_[t] * v.K_[t-1] * (1 + v.gk_[t-1]) * p.flex_capital)
            Min2 = np.minimum(
                v.alpha_[t] * v.W_[t], Min1)
        v.Y_s_[t] = np.minimum(v.Q_s_[t], Min2)
    ########################################################################
    elif p.Limitational_real_production:   # Assumption: all production takes place according to REAL terms
        if t == p.constraint_period_agr:
            Min1 = np.minimum(
                v.beta_[t] * v.intP_[t-1] * pc.non_agr_ * (1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]) +
                v.beta_[t] * v.intP_[t-1] * pc.s_agr_ *
                (1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]),
                v.kappa_[t] * v.k_[t-1] * (1 + v.gk_[t-1]) * p.flex_capital)
            Min2 = np.minimum(
                v.alpha_[t] * v.w_[t], Min1)
        elif t == p.constraint_period_nonagr:
            Min1 = np.minimum(
                v.beta_[t] * v.intP_[t-1] * pc.non_agr_ * (1 - p.capacity_constraint_nonagr) * (1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]) +
                v.beta_[t] * v.intP_[t-1] * pc.s_agr_ *
                (1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]),
                v.kappa_[t] * v.k_[t-1] * (1 + v.gk_[t-1]) * p.flex_capital)
            Min2 = np.minimum(
                v.alpha_[t] * v.w_[t], Min1)
        else:
            Min1 = np.minimum(v.beta_[t] * v.intP_[t-1] * (1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]),
                              v.kappa_[t] * v.k_[t-1] * (1 + v.gk_[t-1]) * p.flex_capital)
            Min2 = np.minimum(
                v.alpha_[t] * v.w_[t], Min1)
        v.y_s_[t] = np.minimum(v.q_s_[t], Min2)
        # Convert back to nominal production
        v.Y_s_[t] = v.y_s_[t] * v.p_[t]
    ########################################################################
    else:
        if p.No_limitational_production_demand_led:
            # OR set supply equal to demand for a completely demand-led economy
            v.Y_s_[t] = v.Q_d_[t]
        if p.No_limitational_production_supply_led:
            # If no limitational production, set supply equal to supply choice for a completely supply-(choice)-led economy
            v.Y_s_[t] = v.Q_s_[t]

    #####################################################################################################################################################################################################################
    # 7.1 Evaluate binding constraints relative to supply choice
    #####################################################################################################################################################################################################################
    if p.Limitational_nominal_production:
        v.binding_intermediate_input_gap_[t] = np.maximum(0, v.Q_s_[t] - v.beta_[t] * v.IntP_[t-1] * (
            # Gap for intermediate inputs
            1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]))
        v.binding_capital_stock_gap_[t] = np.maximum(
            # Gap for capital stock
            0, v.Q_s_[t] - (v.kappa_[t] * v.K_[t-1] * (1 + v.gk_[t-1]) + p.flex_capital))
        v.binding_wage_gap_[t] = np.maximum(
            0, v.Q_s_[t] - v.alpha_[t] * v.W_[t])
    else:
        v.binding_intermediate_input_gap_[t] = np.maximum(0, v.q_s_[t] - v.beta_[t] * v.intP_[t-1] * (
            # Gap for intermediate inputs
            1 + p.flex_intermediate_inputs + v.gI_endog_[t-1]))
        v.binding_capital_stock_gap_[t] = np.maximum(
            # Gap for capital stock
            0, v.q_s_[t] - (v.kappa_[t] * v.k_[t-1] * (1 + v.gk_[t-1]) + p.flex_capital))
        v.binding_wage_gap_[t] = np.maximum(
            0, v.q_s_[t] - v.alpha_[t] * v.w_[t])

    # Create a metric to quantify overall supply constraint binding for all individual constraints
    # Set metric to zero if the supply choice is zero
    if p.Limitational_nominal_production:
        v.supply_constraint_IntP_[t] = np.where(
            v.Q_s_[t] != 0,
            v.binding_intermediate_input_gap_[t] / (v.Q_s_[t]),
            0
        )
        v.supply_constraint_K_[t] = np.where(
            v.Q_s_[t] != 0,
            v.binding_capital_stock_gap_[t] / (v.Q_s_[t]),
            0
        )

        v.supply_constraint_W_[t] = np.where(
            v.Q_s_[t] != 0,
            v.binding_wage_gap_[t] / (v.Q_s_[t]),
            0
        )
    else:
        v.supply_constraint_IntP_[t] = np.where(
            v.q_s_[t] != 0,
            v.binding_intermediate_input_gap_[t] / (v.q_s_[t]),
            0
        )
        v.supply_constraint_K_[t] = np.where(
            v.q_s_[t] != 0,
            v.binding_capital_stock_gap_[t] / (v.q_s_[t]),
            0
        )

        v.supply_constraint_W_[t] = np.where(
            v.q_s_[t] != 0,
            v.binding_wage_gap_[t] / (v.q_s_[t]),
            0
        )

    if t == p.constraint_period_nonagr:
        v.Y_s_[t] = v.Y_s_[t] * (1 - p.capacity_constraint_nonagr)

    # Adjusting total sectoral demand Q_d_ when it exceeds sectoral realized production Y_
    # Q_d_[t] will then have to adapt to production, either by increasing imports (option 1)
    # or (option 2) by restricting demand components. 
    # Order of restrictions currently assumed: non oil exports, government consumption, household consumption, investment
    ########################################################################################################################################################################################################################

    ########################################################################################################################################################################################################################
    # Remittances forward projection (called here after wages W[t] are known)
    # Full BoP closure is computed later in section 12.4b, after imports are finalized
    ########################################################################################################################################################################################################################
    remittances_module(v, p, pc, t)

    ########################################################################################################################################################################################################################
    # 8. Unmet demand & supply-demand reconciliation
    ########################################################################################################################################################################################################################
    # Calculate unmet demand - difference between desired demand and supply-constrained production
    v.Unmet_demand_[t] = v.Q_d_[t] - v.Y_s_[t]
    # Ensure no negative values for unmet demand
    v.Unmet_demand_[t][v.Unmet_demand_[t] < 0] = 0

    # Meet unmet demand by import increases (PREFERRED OPTION IN ALL CURRENT SCENARIOS)
    if p.Import_pure_adjustment:
        # Imports are increased if demand cannot be satisfied otherwise
        for sector in range(len(v.Unmet_demand_[t])):
            if v.Unmet_demand_[t][sector] != 0:
                if p.Warnings_ON:
                    if verbose:
                        print("Unmet demand", v.Unmet_demand_[
                              t][sector], " sector ", sector)
                # In case of capacity constraints in AGR, they are NOT allowed to simply import more!
                # Do not allow negative imports (these are either meaningless or should be defined as exports)
                v.IM_[t][sector] += v.Unmet_demand_[t][sector]
                v.IM_[t][sector] = np.maximum(0, v.IM_[t][sector])
                v.Q_d_[t][sector] -= v.Unmet_demand_[t][sector]
                v.Unmet_demand_[t][sector] = 0

    # Meet unment demand by reducing demand components until possible, then increase imports
    # (This switch/option is NOT ACTIVE in any of the scenarios)
    if p.Demand_reduction_adjustment:
        # Reduce non-oil exports to fill the gap
        for sector in range(len(v.EX_non_oil_[t])):
            if v.Unmet_demand_[t][sector] != 0:
                if v.Unmet_demand_[t][sector] > v.EX_non_oil_[t][sector]:
                    v.Unmet_demand_[t][sector] -= v.EX_non_oil_[t][sector]
                    v.EX_non_oil_[t][sector] = 0
                else:
                    v.EX_non_oil_[t][sector] -= v.Unmet_demand_[t][sector]
                    v.Unmet_demand_[t][sector] = 0
        if p.Countercyclical_gov_policy:
            v.GY_[t] = v.GY_[t]
            # If counteryclical policy is active, government consumption is not reduced!
        else:  # If non-oil exports turn negative, reduce government consumption
            for sector in range(len(v.GY_[t])):
                if v.Unmet_demand_[t][sector] != 0:
                    if v.Unmet_demand_[t][sector] > v.GY_[t][sector]:
                        v.Unmet_demand_[t][sector] -= v.GY_[t][sector]
                        # v.Q_d_[t][sector] -= v.GY_[t][sector]
                        v.GY_[t][sector] = 0
                    else:
                        v.GY_[t][sector] -= v.Unmet_demand_[t][sector]
                        v.Unmet_demand_[t][sector] = 0
        # If government consumption turns negative, reduce household consumption
        for sector in range(len(v.C_[t])):
            if v.Unmet_demand_[t][sector] != 0:
                if v.Unmet_demand_[t][sector] > v.C_[t][sector]:
                    v.Unmet_demand_[t][sector] -= v.C_[t][sector]
                    v.C_[t][sector] = 0
                else:
                    v.C_[t][sector] -= v.Unmet_demand_[t][sector]
                    v.Unmet_demand_[t][sector] = 0
        # If household consumption turns negative, reduce investment!
        for sector in range(len(v.I_demand_[t])):
            if v.Unmet_demand_[t][sector] > v.I_demand_[t][sector]:
                v.Unmet_demand_[t][sector] -= v.I_demand_[t][sector]
                v.I_demand_[t][sector] = 0
            else:
                v.I_demand_[t][sector] -= v.Unmet_demand_[t][sector]
                v.Unmet_demand_[t][sector] = 0
        # OPTION 1: If unmet demand is still not zero, increase imports by this amount!
        if p.Import_residual_adjustment:
            for sector in range(len(v.Unmet_demand_[t])):
                if v.Unmet_demand_[t][sector] != 0:
                    if verbose:
                        print(
                            f"Sector {sector}: Unmet demand is not zero before import correction. Unmet demand: {v.Unmet_demand_[t][sector]} at time {t} is added to imports")
                    # Do not allow negative imports (these are either meaningless or should be defined as exports)
                    v.IM_[t][sector] += v.Unmet_demand_[t][sector]
                    v.IM_[t][sector] = np.maximum(0, v.IM_[t][sector])
                    v.Unmet_demand_[t][sector] = 0
        # Reducing desired demand might affect expectations next period, so this has an effect and might matter substantially, depending on expectation mechanism
        if p.Demand_desired_adjustment:
            for sector in range(len(v.Unmet_demand_[t])):
                if v.Unmet_demand_[t][sector] != 0:
                    if verbose:
                        print(
                            f"Sector {sector}: Unmet demand is not zero before correction. Unmet demand: {v.Unmet_demand_[t][sector]} at time {t} is deducted from sectoral demand")
                    v.Q_d_[t][sector] -= v.Unmet_demand_[t][sector]
                    v.Unmet_demand_[t][sector] = 0

    # Throw a warning if unmet demand is not zero (sectoral and total)
    v.unmet_demand_total[t] = np.sum(v.Unmet_demand_[t])
    if p.Warnings_ON:
        if v.unmet_demand_total[t] != 0:
            warnings.warn(
                f"TOTAL unmet demand is not zero. Unmet demand TOTAL: {v.unmet_demand_total[t]} at time {t}. This is :{v.unmet_demand_total[t]/np.sum(v.Y_)*100:.0f} % of GDP Output Y.")
        for sector in range(len(v.Unmet_demand_[t])):
            if v.Unmet_demand_[t][sector] != 0:
                warnings.warn(
                    f"Sector {sector}: Unmet demand is not zero. Unmet demand: {v.Unmet_demand_[t][sector]} at time {t}")

    # Re-calculate the sums of the demand components after adjustment.
    v.C[t] = np.sum(v.C_[t])
    v.GY[t] = np.sum(v.GY_[t])

    # Check again whether investment desired and investment demand match in the aggregate
    # NOTE: I_demand_[t] may have been reduced by the unmet-demand loop above, so this
    # check captures any gap that survives after rationing. Storage imports (routing to IM
    # rather than domestic demand) are excluded from the threshold — same logic as CHECK 1.
    # Carbon credit investments (v.I_carbon_credits_[t]) are also excluded — they are computed
    # in energy_module and routed to foreign asset purchases, never into domestic I_demand_.
    v.I_total_desired[t] = np.sum(v.I_[t])
    v.I_total_demand[t] = np.sum(v.I_demand_[t])  # For diagnostics and calibration

    _storage_import_t    = v.I_storage_import[t] if hasattr(v, 'I_storage_import') else 0.0
    _carbon_credit_t     = v.I_carbon_credits_[t] if hasattr(v, 'I_carbon_credits_') else 0.0
    _carbon_credit_sum_t = float(np.sum(_carbon_credit_t)) if hasattr(_carbon_credit_t, '__len__') else float(_carbon_credit_t)

    # Adjusted demand: add back known leakage items that never hit I_demand_
    _adjusted_demand_t = v.I_total_demand[t] + _storage_import_t + _carbon_credit_sum_t

    _inv_gap_pct = (
        abs(_adjusted_demand_t - v.I_total_desired[t]) / v.I_total_desired[t] * 100
        if v.I_total_desired[t] > 0 else 0.0
    )

    # Use ownership total (sum of v.I_[t]) instead of demand total:
    # v.I_total[t-1] feeds into investment_module's I_ALL_except_energy_desal_wwater base;
    # using demand here instead of ownership propagates any demand-ownership gap forward.
    v.I_total[t] = np.sum(v.I_[t])


    ########################################################################################################################################################################################################################
    # 9. Re-calculate output (realized final demand and supply)
    ########################################################################################################################################################################################################################
    # Nominal Y
    v.Y_[t] = (pc.dC * v.C[t] + pc.dG * v.GY[t] + v.I_demand_[t] +
               v.EX_oil_[t] + v.EX_non_oil_[t] - v.IM_[t])

    # Negative Y for some sectors follows from import structure of Saudi economy.
    # Sectoral negative Y_ is not forbidden; the aggregate remains non-negative.
    if p.Y_GDP_sectoral_zero_max:
        v.Y_[t] = np.maximum(0, np.minimum(v.Q_d_[t], v.Y_[t]))

    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    # 10. Endogenous industry structures & industrial policy (Vision 2030)
    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    # Changes in production structures depende on excess imports and scarcity in intermediate inputs and capital, which affects investment, and Vision 2030 industrial policy
    if p.Endogenous_A_matrix:
        # Compute excess import measure: If IM>Y, then excess_import = how much larger IM is than Y (but limited to 1), if IM<=Y then 0
        v.excess_import_[t] = np.nan_to_num(np.maximum(
            np.minimum((v.IM_[t] - v.Y_[t]) / v.Y_[t], 1), 0))
        v.excess_scarcity_[t] = np.nan_to_num(np.maximum(np.minimum(
            v.supply_constraint_IntP_[t] + v.supply_constraint_K_[t], 1), 0))
        endogenize_input_output_matrix_A(v, p, pc, t)
    # Production structures remain exogenous
    else:
        v.excess_import_[t] = 0
        # Calculate total output X based on A matrix and all final demand components
        v.X_[t] = np.dot(np.linalg.inv(pc.I - v.A__[t]), v.Y_[t])
        # This is the input from the water module on the water consumption in the agriculture sector
        # Based on food demand, crop plantation schedules, water use per crop, irrigation technologies, etc.
        replace_agr_X(v, pc, t)
        # Re-create Z matrix from last period to have a direct representation of IO Shares
        v.Z__[t] = v.A__[t]  @ np.diag(v.X_[t])
        # This is the same as the Z matrix
        # Intermediate purchases nominal as matrix
        v.IntP__[t] = v.A__[t] * v.X_[t]
        # Intermediate sales
        v.IntS_[t] = np.dot(v.A__[t], v.X_[t])  # Intermediate sales nominal
        # Intermediate purchases
        v.IntP_[t] = np.sum(v.A__[t], axis=0) * v.X_[t]
        # Deriving a proxy for A from X, Y and Intermediate input matrix to test here, relevant for endogenizing A matrix
        # Shares of intermediate purchases
        v.IntP_shares__[t] = v.IntP__[t] / v.IntP_[t]
        v.A_proxy__[t] = v.IntP__[t] * (1/v.X_[t])

    ######################################################################################################################################################################################################################
    ######################################################################################################################################################################################################################
    # 11. Energy module: energy demand, supply, prices, emissions
    ######################################################################################################################################################################################################################
    ######################################################################################################################################################################################################################
    # Some investment-related equations may be moved to the investment module and lagged if required.

    # CAPTURE RETURN VALUES from energy_module
    (v.I_[t], v.I_demand_[t], v.gas_use_[t], v.oil_use_final_[t], v.gas_use_final_[t], 
     v.electricity_use_[t], v.oil_use_[t], v.energy_reduction_factor_[t],
     v.gas_use_total[t], v.oil_use_total[t], v.electricity_use_total[t], v.renewable_energy_generation[t],
     v.I_efficiency_[t], v.I_netzero_nature_[t], v.emissions_net_total[t], v.emissions_gross_total[t],
     v.I_carbon_credit_revenue_[t], v.I_carbon_credits_[t], v.flow_reuse_recycle[t], v.flow_remove_ccs[t],
     v.flow_remove_nature[t], v.marginal_cost_of_efficiency_[t]) = energy_module(v, p, pc, t)

    ######################################################################################################################################################################################################################

    ######################################################################################################################################################################################################################
    ######################################################################################################################################################################################################################
    # 12. Flow-flow and stock-flow accounting
    ######################################################################################################################################################################################################################
    ######################################################################################################################################################################################################################

    ######################################################################################################################################################################################################################
    # 12.1 Nominal-real conversions, GDP components, growth rates
    ######################################################################################################################################################################################################################
    # Calculate real GDP y after endogenous changes
    v.y_[t] = v.Y_[t] / v.p_[t]

    # Real total output / physical quantities
    v.x_[t] = v.X_[t] / v.p_[t]

    # Determine sectoral real consumption.
    v.c_[t] = v.C_[t] / v.p_[t]
    v.c[t] = np.sum(v.c_[t])

    # Differentiate oil and non-oil GDP (national accounts definition, scaled to GASTAT 2021 ratios)
    v.Y_non_oil_[t] = v.Y_[t] * pc.non_oil_private_sectors_ * pc.Y_non_oil_scale
    v.Y_oil_[t]     = v.Y_[t] * pc.sectors_oil_              * pc.Y_oil_scale
    # Real oil and non oil GDP y
    v.y_non_oil_[t] = v.Y_non_oil_[t] / v.p_[t]
    v.y_oil_[t]     = v.Y_oil_[t]     / v.p_[t]
    # Government activities GDP proxy and net taxes (national accounts definition)
    v.Y_gov_[t]      = v.Y_[t] * pc.gov_sectors_ * pc.Y_gov_scale
    v.y_gov_[t]      = v.Y_gov_[t] / v.p_[t]
    v.Y_net_taxes[t] = pc.ratio_net_taxes_gdp * np.sum(v.Y_[t])
    # Other GDP = Government activities + Net taxes on products
    v.Y_other_gdp[t] = np.sum(v.Y_gov_[t]) + v.Y_net_taxes[t]

    # Check whether the unmet demand mechanism has left differences between Y_s_[t] and Y_[t] after the adjustment. Should be 0!
    v.Y_diff_[t] = v.Y_s_[t] - v.Y_[t]
    if p.Warnings_ON:
        for sector in range(len(v.Y_diff_[t])):
            if v.Y_diff_[t][sector] != 0:
                warnings.warn(
                    f"Sector {sector} at time {t}: Unmet Demand gap: Y_diff_  = v.Y_s_[t] -  v.Y_[t]: {v.Y_diff_[t][sector]}")

    # Growth rate of Nominal GDP output Y (realized GDP)
    v.gY[t] = (np.sum(v.Y_[t]) / np.sum(v.Y_[t-1])) - \
        1  # Growth rate of REALIZED final demand
    # Growth rate of REALIZED final demand NON-OIL
    v.gY_non_oil[t] = (np.sum(v.Y_non_oil_[t]) / np.sum(v.Y_non_oil_[t-1])) - 1
    v.gY_oil[t] = (np.sum(v.Y_oil_[t]) / np.sum(v.Y_oil_[t-1])) - \
        1  # Growth rate of REALIZED final demand OIL
    # Growth rate of REAL GDP output Y (realized GDP)
    v.gy[t] = (np.sum(v.y_[t]) / np.sum(v.y_[t-1])) - \
        1  # Growth rate of REALIZED final demand
    # Growth rate of REALIZED final demand NON-OIL
    v.gy_non_oil[t] = (np.sum(v.y_non_oil_[t]) / np.sum(v.y_non_oil_[t-1])) - 1
    v.gy_oil[t] = (np.sum(v.y_oil_[t]) / np.sum(v.y_oil_[t-1])) - \
        1  # Growth rate of REALIZED final demand OIL
    # Growth rate of government activities GDP proxy
    v.gY_gov[t] = (np.sum(v.Y_gov_[t]) / np.sum(v.Y_gov_[t-1])) - 1
    v.gy_gov[t] = (np.sum(v.y_gov_[t]) / np.sum(v.y_gov_[t-1])) - 1
    # Sectoral growth rate of final demand
    v.gY_[t] = np.nan_to_num((v.Y_[t] / v.Y_[t-1]) - 1, nan=0.0)
    v.gX_[t] = np.nan_to_num((v.X_[t] / v.X_[t-1]) - 1, nan=0.0)
    v.gy_[t] = np.nan_to_num((v.y_[t] / v.y_[t-1]) - 1, nan=0.0)
    v.gx_[t] = np.nan_to_num((v.x_[t] / v.x_[t-1]) - 1, nan=0.0)
    # Update the time series model to calculate the growth rate of Y for next period
    pc.Y_timeseries = pd.concat(
        [pc.Y_timeseries, pd.Series([np.sum(v.Y_[t])])], ignore_index=True)

    # Update Sectoral share of total output
    v.dX_[t] = v.X_[t] / np.sum(v.X_[t])
    # Calculate this ratio, can be used for investment function
    v.X_Y_ratio_[t] = v.X_[t] / v.Y_[t]

    # Calculate real intermediate purchases/sales
    # Intermediate purchases real (corrected by sectoral inflation)
    v.intP_[t] = v.IntP_[t] / v.p_[t]
    # Intermediate sales real (corrected by sectoral inflation)
    v.intS_[t] = v.IntS_[t] / v.p_[t]

    # Real investment total
    # Investment total is recomputed here for consistency across all sectoral flows.
    # Use ownership total consistently; see also the first assignment above.
    v.I_total[t] = np.sum(v.I_[t])
    v.i_total[t] = v.I_total[t] / v.deflator_gdp[t]

    # Real imports calculation after supply-demand restrictions AND endogenous import reductions
    v.im_[t] = v.IM_[t] / v.p_[t]  # Import share of total output
    v.gIM_[t] = v.IM_[t] / v.IM_[t-1] - 1  # Growth rate of imports

    # Calculate a factor by how much changed GDP is larger in period 1 than the calibrated GDP, potentially to corrent consumption
    v.alpha_2_increase_factor[t] = np.sum(v.Y_[t])/np.sum(pc.Y_)

    #####################################################################################################################################################################################################################
    # 12.2 Capital dynamics
    #####################################################################################################################################################################################################################
    # Capital capacity utilization
    v.u_[t] = v.x_[t] / (v.k_[t-1] * pc.eK_)

    # Capital stock Evolution
    # Capital stock evolution with depreciation, with a fixed composition of the capital stock
    v.K_[t] = v.K_[t-1] * (1 - pc.δ_) + v.I_[t-1]

    v.gK[t] = (np.sum(v.K_[t]) / np.sum(v.K_[t-1])) - \
        1  # Growth rate of NOMINAL capital stock
    # Sectoral growth rate of NOMINAL capital stock
    v.gK_[t] = (v.K_[t] / v.K_[t-1]) - 1

    # Real capital stock (physical quantities)
    v.k_[t] = v.K_[t] / v.p_[t]
    v.gk[t] = (np.sum(v.k_[t]) / np.sum(v.k_[t-1])) - \
        1  # Growth rate of REAL capital stock
    # Sectoral growth rate of REAL capital stock
    v.gk_[t] = (v.k_[t] / v.k_[t-1]) - 1

    #####################################################################################################################################################################################################################
    # 12.3 Profits
    #####################################################################################################################################################################################################################
    # Taxes IOT for profit accounting
    v.Tax_products_net_[t] = pc.tau_tax_products_net_ * v.X_[t]
    v.Tax_production_[t] = pc.tau_tax_production_ * v.X_[t]
    v.Tax_investments_[t] = (
        v.I_private_[t] + v.I_public_[t]) * pc.tau_tax_investments



    # TOTAL realized profits (private + public)
    v.P_[t] = v.X_[t] - v.IntP_[t] - v.W_[t] \
        - v.Tax_production_[t] - v.Tax_products_net_[t]   \
        + v.Sub_production_[t] - v.Tax_investments_[t] 


    # Profits Update with the CARBON CREDIT mechanism
    # This must remain here to impact firm finances immediately
    v.P_[t] -= v.I_carbon_credits_[t]
    v.P_[t] += v.I_carbon_credit_revenue_[t]

    # Real profits
    v.pi_real_[t] = v.P_[t] / v.p_[t]

    #####################################################################################################################################################################################################################
    # 12.4 Government revenue accounting and bond issuance
    #####################################################################################################################################################################################################################
    # Value added tax (VAT) - VAT is paid on NET consumption
    # (as is calculated in consumption function)
    v.VAT[t] = v.C[t] * pc.tau_vat

    # Time-dependent profit tax — phased in linearly over p.tax_phase_in_years periods
    eff_profit_rate = _phased_tax_rate(
        t,
        p.profit_tax_rate_1, p.profit_tax_rate_timing_1,
        p.profit_tax_rate_2, p.profit_tax_rate_timing_2,
        p.tax_phase_in_years,
    )
    v.Tax_profits[t] = np.sum(pc.s_private_ * v.P_[t]) * eff_profit_rate

    # Time-dependent wage tax — phased in linearly over p.tax_phase_in_years periods
    eff_wage_rate = _phased_tax_rate(
        t,
        p.wage_tax_rate_1, p.wage_tax_rate_timing_1,
        p.wage_tax_rate_2, p.wage_tax_rate_timing_2,
        p.tax_phase_in_years,
    )
    v.Tax_wages[t] = np.sum(v.W_[t]) * eff_wage_rate

    # Government household taxes calibrated to match expenditure data,
    # are then levied on profit income
    v.TH[t] = np.sum(pc.s_private_ * v.P_[t]) * pc.τ

    # Private profits are reduced by oil profits taken by the government
    v.GP[t] = np.sum(pc.s_public_ * v.P_[t])
    # Government also receives its ownership share of SFC bank profits
    v.GP[t] += pc.gov_bank_share * v.PB[t]

    # Income on government gross external asset stock (SAMA reserves + PIF foreign holdings)
    # Positive when Gov_ext_assets > 0 → government earns investment income (PIF returns, reserve interest)
    # Negative when Gov_ext_assets < 0 → government pays interest to RoW on net foreign debt
    # Uses lagged asset stock (known) → no circularity with BoP computed below
    v.Gov_ext_assets_income[t] = p.r_gov_ext_assets * v.Gov_ext_assets[t - 1]

    # Government revenue accounting:
    # Gov_ext_assets_income included: when positive it is investment income (PIF/SAMA returns),
    # when negative it reduces Gov_rev (interest cost on net foreign debt to RoW).
    # PIF_income included: earned when government net wealth is positive.
    v.Gov_rev[t] = (
        v.TH[t] + v.VAT[t] + v.GP[t] +
        np.sum(v.Tax_products_net_[t]) + np.sum(v.Tax_production_[t]) +
        np.sum(v.Tax_investments_[t]) +
        v.Tax_profits[t] + v.Tax_wages[t] +
        p.Gov_ext_assets_extraction * v.Gov_ext_assets_income[t]   # Only extracted share enters Gov_rev; remainder reinvested into Gov_ext_assets stock
    )

    # Government fiscal surplus → PIF: compute Gov_pif_investment HERE (before Bond)
    # so the Bond equation correctly reflects that the PIF-invested fraction does NOT reduce bonds.
    # SFC budget constraint: fiscal surplus is split between (a) paying down bonds and (b) investing in PIF.
    v.gov_fiscal_balance[t] = v.Gov_rev[t] - v.Gov_exp[t]
    v.Gov_pif_investment[t] = p.gov_surplus_pif_rate * max(v.gov_fiscal_balance[t], 0.0)

    # Government bond issuance / retirement, split 60 % domestic / 40 % external.
    # pc.domestic_bond_share = 0.60 (fixed, rounded from SAMA 2021 outstanding: 59.6% domestic).
    # Two cases:
    #   Surplus (Gov_rev > Gov_exp): PIF absorbs gov_surplus_pif_rate share of the surplus;
    #     only the remaining (1 − gov_surplus_pif_rate) fraction retires bonds.
    #     gov_surplus_pif_rate=1 → entire surplus → PIF; bonds only fall by natural repayment (ρ·Bond[t-1]).
    #     gov_surplus_pif_rate=0 → entire surplus retires bonds (standard baseline case).
    #   Deficit (Gov_exp > Gov_rev): Gov_pif_investment=0; full deficit is bond-financed.
    # _bond_change > 0 → new issuance; _bond_change < 0 → net retirement.
    # Gov_ext_assets_income feeds through Gov_rev: positive asset income reduces the financing need.
    if v.gov_fiscal_balance[t] > 0:
        # Surplus: PIF absorbs the gov_surplus_pif_rate share; remainder retires bonds
        _bond_change = -(1.0 - p.gov_surplus_pif_rate) * v.gov_fiscal_balance[t]
    else:
        # Deficit: fully bond-financed
        _bond_change = -v.gov_fiscal_balance[t]  # positive value (= deficit size)
    v.Bond_domestic[t] = v.Bond_domestic[t-1] * (1 - p.ρ) + pc.domestic_bond_share * _bond_change
    v.Bond_external_routine_issuance[t] = (1 - pc.domestic_bond_share) * _bond_change

    # Gov_net_wealth is computed stock-based BELOW, after Gov_ext_assets[t] is fully finalized.
    # Definition: Gov_net_wealth[t] = Gov_ext_assets[t] − Bond[t]  (guarantees GNW ≤ Gov_ext_assets when Bond ≥ 0)

    ########################################################################################################################################################################################################################
    # 12.4b Balance of Payments (BPM6 structure)
    ########################################################################################################################################################################################################################
    # Computed here (after imports are finalized in section 8) rather than earlier.
    # Structure follows Naqvi (2025) "Balance of Payments" and BPM6:
    #   CAB + Capital Account + Financial Account = 0
    #
    # For Saudi Arabia:
    #   - Large trade surplus (oil exports), large remittance outflows (foreign workers)
    #   - FDI inflows represent foreign capital entering the country
    #   - Fixed exchange rate (3.75 SAR/USD peg) → SAMA intervenes → reserves adjust
    #   - Gov_ext_assets (SAMA reserves + PIF foreign holdings) is the natural clearing/residual variable

    # ── Current Account (CAB) ──
    # (1) Trade balance: exports minus imports of goods and services
    v.trade_balance[t] = np.sum(v.EX_oil_[t]) + np.sum(v.EX_non_oil_[t]) - np.sum(v.IM_[t])

    # (2) Primary income balance: investment income net (FDI returns, NFA income)
    #     Gov_ext_assets_income already computed in section 12.4 (enters both Gov_rev and BoP)
    # FDI income repatriation: return on NET FDI stock repatriated abroad (outflow).
    # Uses FDI_net_stock_total (cumulative net inflows only) instead of gross stock,
    # preventing amplification by the gross-to-net ratio from FDI_module.
    # Capped at p.max_FDI_income_share of private profits to prevent model instability.
    if t > 1 and v.FDI_net_stock_total[t - 1] > 0:
        _fdi_income_raw = p.r_FDI_repatriation * v.FDI_net_stock_total[t - 1]
        _private_profits = np.sum(pc.s_private_ * v.P_[t])
        FDI_income_payments = min(_fdi_income_raw, p.max_FDI_income_share * max(_private_profits, 0.0))
    else:
        FDI_income_payments = 0.0
    v.FDI_income_payments[t] = FDI_income_payments
    # Bond_external interest is an income outflow to foreign bondholders → enters primary income balance as a debit
    v.primary_income_balance[t] = v.Gov_ext_assets_income[t] - FDI_income_payments - v.Bond_external_interest_cost[t]

    # (3) Secondary income balance: personal remittances (outflow, already computed by remittances_module)
    v.secondary_income_balance[t] = -v.remittances[t]

    # (4) Current account = trade + primary income + secondary income
    v.current_account[t] = v.trade_balance[t] + v.primary_income_balance[t] + v.secondary_income_balance[t]

    # ── Capital Account (BPM6 definition) ──
    # Debt forgiveness, investment grants, non-produced asset transfers — negligible for Saudi
    v.capital_account[t] = 0.0

    # ── Financial Account & Gov_ext_assets (GEA) closure ──
    # BPM6 decomposition (Godley-Lavoie SFC convention):
    #   CA + FA = 0   where FA = FDI_net_inflow − ΔGov_ext_assets
    #   → ΔGov_ext_assets = CA + FDI_net_inflow  (in gross terms, before extraction)
    #
    # For Saudi Arabia's fixed-exchange-rate economy (3.75 SAR/USD peg):
    #   • Trade surplus     → SAMA receives USD or other currencies     → Gov_ext_assets ↑ (captured in CA)
    #   • Remittances out   → workers send USD or other currencies      → Gov_ext_assets ↓ (captured in CA)
    #   • FDI income repat. → profits sent abroad                       → Gov_ext_assets ↓ (captured in CA)
    #   • Asset income extraction (p.Gov_ext_assets_extraction fraction) leaves Gov_ext_assets
    #     and enters Gov_rev; only the REINVESTED share (1−e) stays in Gov_ext_assets.
    #   • Gov_pif_investment (FA outflow): government acquires foreign assets (PIF) using fiscal surplus.
    #     BPM6: acquisition of reserve assets = financial account debit (outflow from Saudi Arabia to RoW).
    #     Counterpart credit = CA surplus that generated the fiscal surplus (oil exports already in CA).
    v.financial_account[t] = -(v.current_account[t] + v.capital_account[t])
    v.Gov_ext_assets_change[t] = (v.current_account[t]
                                   - p.Gov_ext_assets_extraction * v.Gov_ext_assets_income[t]
                                   + v.FDI_net_total[t])
    # Routine external bond issuance: the 40 % external share of the fiscal deficit brings in foreign currency
    # (foreign investors send USD to SAMA) → Gov_ext_assets increases; Bond_external liability stock increases.
    v.Gov_ext_assets_change[t] += v.Bond_external_routine_issuance[t]
    # Gov_pif_investment: government fiscal surplus invested in PIF (foreign assets)
    # Added here (before _Gov_ext_assets_tentative) so the two-phase floor logic sees the full Gov_ext_assets change.
    v.Gov_ext_assets_change[t] += v.Gov_pif_investment[t]
    # Two-phase sovereign financing floor:
    #   Phase 1 (Gov_ext_assets > 50% of initial stock): deficit absorbed by drawing down SAMA/PIF reserves (normal)
    #   Phase 2 (Gov_ext_assets < 50% of initial stock): gap financed by issuing sovereign bonds to RoW at interest_gov_bonds
    #   Rationale: the government treats half the initial sovereign wealth fund as a strategic reserve floor
    #   and switches to bond issuance (at 4%) rather than depleting it further.
    #   The floor is inert whenever Gov_ext_assets stays above this threshold (zero impact on normal results).
    _Gov_ext_assets_floor = 0.5 * pc.Gov_ext_assets_initial
    _Gov_ext_assets_tentative = v.Gov_ext_assets[t - 1] + v.Gov_ext_assets_change[t]
    # Step 1: amortise existing external bonds at rate ρ_ext; add routine new issuance from the 60/40 split
    # (floored at zero; Bond_external_repayment[t] was computed in Gov_exp block).
    _Bond_ext_after_repayment = max(
        v.Bond_external[t - 1] - v.Bond_external_repayment[t] + v.Bond_external_routine_issuance[t],
        0.0
    )
    # Step 2: if Gov_ext_assets is still below the floor after amortisation, issue additional emergency bonds to top it back up.
    if _Gov_ext_assets_tentative < _Gov_ext_assets_floor:
        v.Bond_external[t] = _Bond_ext_after_repayment + (_Gov_ext_assets_floor - _Gov_ext_assets_tentative)
        v.Gov_ext_assets[t] = _Gov_ext_assets_floor
    else:
        v.Bond_external[t] = _Bond_ext_after_repayment
        v.Gov_ext_assets[t] = _Gov_ext_assets_tentative

    # BoP identity check: CAB + capital_account + financial_account should = 0
    # Gov_pif_investment is a FA outflow funded by the CA surplus → already absorbed in FA = −CA → BoP_check = 0 ✓
    v.BoP_check[t] = v.current_account[t] + v.capital_account[t] + v.financial_account[t]

    ########################################################################################################################################################################################################################
    # 12.5 Firm and bank accounting
    ########################################################################################################################################################################################################################
    # Loans = investment cost - own investment
    # Private firms take out loans to finance part of private investment
    # Firms finance Investments partly out of profits, part out of loans with an exogenous share invest_profit
    # Depending on scenario setup, pc.invest_profit can be endogenous, to enable households to manage their balance sheet so their wealth does not decrease.
    # The residual of what is not financed by profits is financed by loans
    #
    # FDI ADJUSTMENT: Only the domestically-financed portion of private investment
    # needs to be financed through domestic loans and profits.
    # I_private_dom_[t] excludes FDI (computed in investment_module), so the loan
    # equation reflects reduced domestic financing needs when FDI is active.
    # I_private_[t] (TOTAL, including FDI portion) is kept unchanged for GDP accounting.
    if p.Invest_loan_adaptation_to_HH_wealth:
        if v.gV[t-1] < 0:
            v.invest_profit[t] = v.invest_profit[t-1] * \
                p.decrease_invest_profit
            v.L_[t] = v.L_[t-1] * (1 - p.ρ) + \
                v.I_private_dom_[t] * (1 - v.invest_profit[t])
        else:
            # Stabilize investment out of profit rate in case HH wealth is increasing.
            v.invest_profit[t] = v.invest_profit[t-1]
            v.L_[t] = v.L_[t-1] * (1 - p.ρ) + \
                v.I_private_dom_[t] * (1 - v.invest_profit[t])
    else:
        v.invest_profit[t] = pc.invest_profit
        v.L_[t] = v.L_[t-1] * (1 - p.ρ) + v.I_private_dom_[t] * \
            (1 - v.invest_profit[t])

    # HH deposits = total loan stock (endogenous money: loans create deposits attributed to households).
    # D_HH = sum(L_): HH hold deposits equal to total loans outstanding — the asset counterpart of firm liabilities.
    # This makes the loan-deposit circuit explicit in the balance sheet and TFM.
    v.D_HH[t] = np.sum(v.L_[t])

    # Bank realized profits - calibrated to gross operating surplus of financial sector
    v.PB[t] = pc.rl * np.sum(v.L_[t-1]) + pc.interest_gov_bonds* v.Bond_domestic[t-1]  - (pc.rm*v.V[t-1]) 

    # total profit accounting for the financial sector for price formation
    v.P_tot_[t] = v.P_[t] + pc.s_bank_ * v.PB[t]

    ########################################################################################################################################################################################################################
    # 12.6 Household accounting
    ########################################################################################################################################################################################################################
    # HH disposable income: distinction profit and wage income, determines NET cons, VAT is paid in cons. funct.
    # Allocate TH according to the share of the respective
    # Fixed government expenditure residual is added to wage income; assumed to be primarily social spending (for lack of better data)
    v.YD_wage[t] = np.sum(v.W_[t]) - v.Tax_wages[t] + pc.gov_exp_resid 
    # Real disposable wage income, corrected by consumption price index
    v.yd_wage[t] = v.YD_wage[t] / v.price_index_cons[t]

    # Only private profits enter disposable profit income
    # FDI income payments are deducted here: they represent returns on FDI-financed capital
    # repatriated abroad, which flow through the firm (as after-profit transfers) and thus
    # reduce the profit income available to domestic households.
    # FDI_income_payments is computed in section 12.4b (capped, net-stock based).

    # P_distributed is the explicit firm cash-flow account: gross private profits after
    # all debt service (interest rl*L + principal repayment rho*L).  This is the amount
    # firms can distribute to households (before the retained-earnings deduction).
    v.P_distributed[t] = np.sum(pc.s_private_ * v.P_[t]) - np.sum((pc.rl + p.ρ) * v.L_[t-1])
    v.YD_profit[t] = v.P_distributed[t] + (1 - pc.gov_bank_share) * v.PB[t] + (pc.rm*v.V[t-1]) - \
        v.TH[t] - v.Tax_profits[t] - FDI_income_payments

    # HH Wealth = past wealth + income - consumption (net) - VAT - investment financed out of profits - remittances
    # I_private_dom_[t] excludes FDI: HH only bear the cost of domestically-financed
    # private investment out of their profits. FDI-financed investment does not
    # reduce domestic household wealth.
    # The V equation is the SFC stock-flow identity for the current model structure:
    #   - v.C[t] is NET consumption (producer prices, ex-VAT); see consumption function.
    #   - v.VAT[t] = v.C[t] * tau_vat is the separate VAT outflow that flows to government.
    #     Households pay C + VAT gross; only C reaches firms. VAT must be deducted here so
    #     TFM HH S/D ≈ ΔV (residual gap ≈ remittances only, which are also deducted).
    #   - The term -sum(I_private_dom*invest_profit) = -retained_earnings (firms self-finance from profits).
    #   - Firm loan service (rl+rho)*L is already netted in P_distributed (and thus in YD_profit).
    # NOTE on V vs total wealth:
    #   v.V[t] = HH LIQUID financial wealth (portfolio claims on banks). NOT total HH net worth.
    #   v.HH_total_wealth[t] = full HH net worth = V + D_HH + E_firm_HH + E_bank_HH + OFA_HH,
    #   consistent with the balance sheet matrix (balance_sheet.py create_balance_sheet).
    #   V is kept separate because (pc.rm * v.V) drives YD_profit → consumption.
    v.V[t] = v.V[t-1] + v.YD_wage[t] + v.YD_profit[t] - \
        v.C[t] - v.VAT[t] - np.sum(v.I_private_dom_[t] * v.invest_profit[t]) - v.remittances[t]
    if t > 1:
        v.gV[t] = (v.V[t] - v.V[t-1])/v.V[t-1]
    # HH net worth (balance-sheet consistent) ─────────────────────────────────
    # Equity components mirror balance_sheet.py
    # Capital-weighted public ownership share (constant; same as TFM / BSM computation).
    _s_pub_agg = np.sum(pc.s_public_ * pc.K_) / np.sum(pc.K_)
    # Firm equity HH share: E_firm_HH = (1 − s_pub) × (K − L − FDI_net_stock)
    _K_net_t   = np.sum(v.K_[t])   - np.sum(v.L_[t])   - v.FDI_net_stock_total[t]
    _K_net_tm1 = np.sum(v.K_[t-1]) - np.sum(v.L_[t-1]) - v.FDI_net_stock_total[t-1]
    _E_firm_HH = (1.0 - _s_pub_agg) * _K_net_t
    # Bank equity HH share: E_bank_HH = (1 − gov_bank_share) × (B_dom − V)
    _E_bank_HH = (1.0 - pc.gov_bank_share) * (v.Bond_domestic[t] - v.V[t])
    # OFA_HH accumulation: in this model S/D_HH = ΔV exactly (V captures the full HH
    # budget constraint), so ΔOFA_HH = ΔD_HH − ΔE_firm_HH  (TFM residual identity).
    v.OFA_HH[t] = v.OFA_HH[t-1] + (v.D_HH[t] - v.D_HH[t-1]) - (1.0 - _s_pub_agg) * (_K_net_t - _K_net_tm1)
    v.HH_total_wealth[t] = v.V[t] + v.D_HH[t] + _E_firm_HH + _E_bank_HH + v.OFA_HH[t]

    ########################################################################################################################################################################################################################
    # 12.6b Government net worth (BSM-consistent)
    ########################################################################################################################################################################################################################
    # Mirrors create_balance_sheet.  Reuses _s_pub_agg, _K_net_t, _K_net_tm1 from section 12.6a.
    # Government column BSM assets/liabilities:
    #   +Gov_ext_assets  −Bond_dom  −Bond_ext          ← liquid stock (already in old formula)
    #   +E_firm_gov = s_pub × (K − L − FDI)     ← public-firm equity (Aramco, utilities, desal)
    #   +E_bank_gov = gov_bank_share × (B_dom − V) ← bank equity gov share
    #   +OFA_Gov    ← clearing residual (all unmodelled gov financial instruments)
    # V[t] is now available (computed in section 12.6a).
    _E_firm_gov = _s_pub_agg * _K_net_t
    _E_bank_gov = pc.gov_bank_share * (v.Bond_domestic[t] - v.V[t])
    # ΔOFA_Gov = Gov_surplus − Δ(Gov_ext_assets − Bond_dom − Bond_ext) − ΔE_firm_gov − ΔE_bank_gov
    # This is the BSM residual that guarantees NLNB_Gov = −S/D_Gov identically.
    _gov_liquid_NW_t   = v.Gov_ext_assets[t]   - v.Bond_domestic[t]   - v.Bond_external[t]
    _gov_liquid_NW_tm1 = v.Gov_ext_assets[t-1] - v.Bond_domestic[t-1] - v.Bond_external[t-1]
    _Delta_E_firm_gov  = _s_pub_agg      * (_K_net_t - _K_net_tm1)
    _Delta_E_bank_gov  = pc.gov_bank_share * ((v.Bond_domestic[t] - v.V[t]) -
                                              (v.Bond_domestic[t-1] - v.V[t-1]))
    v.OFA_Gov[t] = (v.OFA_Gov[t-1]
                    + (v.Gov_rev[t] - v.Gov_exp[t])
                    - (_gov_liquid_NW_t - _gov_liquid_NW_tm1)
                    - _Delta_E_firm_gov
                    - _Delta_E_bank_gov)
    v.Gov_net_wealth[t] = v.Gov_ext_assets[t] - v.Bond_domestic[t] - v.Bond_external[t]
    v.Gov_net_wealth_FULL[t] = (_gov_liquid_NW_t
                           + _E_firm_gov
                           + _E_bank_gov
                           + v.OFA_Gov[t])
    v.Gov_net_wealth_FULL_plus_aramco[t] = v.Gov_net_wealth_FULL[t] + pc.aramco_value

    ########################################################################################################################################################################################################################
    # 12.7 GDP accounting — production and income approaches
    ########################################################################################################################################################################################################################
    v.Y_production_[t] = v.X_[t] - v.IntS_[t]
    v.Y_distribution_[t] = v.W_[t] + v.P_[t] + v.C_[t] * pc.tau_vat

    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################
    # 13. Biophysical accounting (except energy, handled in energy module)
    ########################################################################################################################################################################################################################
    ########################################################################################################################################################################################################################

    ########################################################################################################################################################################################################################
    # 13.1 Water module
    ########################################################################################################################################################################################################################
    water_accounting(v, p, pc, t, verbose)

    ########################################################################################################################################################################################################################
    # 13.2 Irrigation costs (investment, O&M, subsidy)
    ########################################################################################################################################################################################################################
    apply_irrigation_costs(v, pc, t)



############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
# RUN MODEL END
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################


############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
# IV. Prepare results for postprocessing
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
def prepare_results(T: int, v: ModelVariables, c: ModelConfig):
    """Create a ModelResults object from the simulation data."""
    # Round zero is discarded (initial values)
    # Sectoral results are stored in a dictionary
    # Sum over all sectors is added to the macro variables
    macro_vars_dict = {}
    sector_vars = {}
    iomatrix = {}  # New category for IO matrix
    trange = list(range(1, T + 1))
    srange = list(range(c.pc.S))
    macro_vars_dict["t"] = trange

    for field in fields(v):
        key = field.name
        data = getattr(v, key)
        if key.endswith("__"):
            # Handle dynamic IO table (matrix structure)
            key = key[:-2]  # Remove trailing "__" to get the field name
            iomatrix[key] = {
                t: pd.DataFrame(data[t], columns=srange, index=srange)
                for t in trange
            }
            # Set index and column names for each DataFrame
            for t, df in iomatrix[key].items():
                df.index.name = 's'  # Rows represent sectors
                df.columns.name = 's'  # Columns represent sectors
        elif key.endswith("_"):
            key = key[:-1]
            df = pd.DataFrame(
                data[1:], columns=srange, index=trange)
            df.index.name = 't'
            df.columns.name = 's'
            sector_vars[key] = df
            macro_vars_dict[key] = df.sum(axis=1)
        else:
            macro_vars_dict[key] = data[1:]

    macro_vars = pd.DataFrame(macro_vars_dict).set_index("t")

    return ModelResults(config=c, macro=macro_vars, sectoral=sector_vars, iomatrix=iomatrix)
#####################################################################################################################################################################################################################

#####################################################################################################################################################################################################################
#####################################################################################################################################################################################################################
# V. Test results
#####################################################################################################################################################################################################################
#####################################################################################################################################################################################################################


def test_results(results: ModelResults):
    """Test that the results are consistent with the model equations."""

    pc = results.config.pc
    v = results.macro
    assert np.allclose(pc.Y_, v.Y_[1]), "Final demand does not match"
    assert np.allclose(pc.X_, v.X_[1]), "Total output does not match"
